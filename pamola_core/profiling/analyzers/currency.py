"""
Currency Field Profiler Operation for the project.

This module defines operations for profiling currency fields in tabular datasets.
It extracts detailed statistical summaries, handles locale-aware parsing of currency
values, and produces artifacts to support subsequent anonymization, data quality
evaluation, and semantic transformation.

The module supports:
- Pandas and optionally Dask for large-scale data
- Input via DataFrame or CSV
- Descriptive statistics, outlier detection, distribution shape, and normality testing
- Robust handling of locale-specific formatting and common inconsistencies
- Multiple artifact outputs: stats in JSON, visualizations (PNG), and samples (CSV)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd

from pamola_core.profiling.commons.currency_utils import (
    is_currency_field, parse_currency_field, analyze_currency_stats,
    detect_currency_from_sample,
    generate_currency_samples, create_empty_currency_stats
)
from pamola_core.profiling.commons.numeric_utils import (
    calculate_percentiles, calculate_histogram
)
from pamola_core.utils.io import (
    load_data_operation, write_dataframe_to_csv, write_json, load_settings_operation
)
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.visualization import (
    create_histogram, create_boxplot, create_correlation_pair
)
from pamola_core.common.constants import Constants
# Configure logger
logger = logging.getLogger(__name__)


class CurrencyAnalyzer:
    """
    Analyzer for currency fields.

    This analyzer provides methods for handling, parsing, and analyzing
    currency fields in tabular datasets with support for locale-aware
    parsing and robust error handling.
    """

    def analyze(self,
                df: pd.DataFrame,
                field_name: str,
                locale: str = 'en_US',
                bins: int = 10,
                detect_outliers: bool = True,
                test_normality: bool = True,
                chunk_size: int = 10000,
                use_dask: bool = False,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a currency field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        locale : str
            Locale to use for parsing (default: 'en_US')
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        chunk_size : int
            Size of chunks for processing large datasets
        use_dask : bool
            Whether to use Dask for large datasets
        progress_tracker : ProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        logger.info(f"Analyzing currency field: {field_name}")

        # Validate input
        if field_name not in df.columns:
            logger.error(f"Field {field_name} not found in DataFrame")
            return {'error': f"Field {field_name} not found in DataFrame"}

        total_rows = len(df)

        # Initialize progress tracking
        if progress_tracker:
            progress_tracker.update(0, {
                "step": "Initializing currency analysis",
                "field": field_name
            })

        # Check if this is actually a currency field
        # This is both a validation step and may determine parsing approach
        is_detected_currency = is_currency_field(field_name)
        force_currency = kwargs.get('force_currency', False)

        if not is_detected_currency and not force_currency:
            logger.info(f"Field {field_name} does not appear to be a currency field. "
                        f"Use force_currency=True to override.")
            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Field validation",
                    "warning": "Not likely a currency field"
                })

        # Initialize result structure
        result = {
            'field_name': field_name,
            'total_rows': total_rows,
            'is_detected_currency': is_detected_currency,
            'locale_used': locale
        }

        # Handle large datasets with chunking if needed
        is_large_df = total_rows > chunk_size

        if use_dask and is_large_df:
            try:
                import dask.dataframe as dd
                logger.info(f"Using Dask for large dataset with {total_rows} rows")
                return self._analyze_with_dask(df, field_name, locale, bins,
                                               detect_outliers, test_normality,
                                               progress_tracker, **kwargs)
            except ImportError:
                logger.warning("Dask requested but not available. Falling back to chunked processing.")
                if progress_tracker:
                    progress_tracker.update(0, {
                        "step": "Dask fallback",
                        "warning": "Dask not available, using chunks"
                    })

        # Process in chunks for large datasets
        if is_large_df and not use_dask:
            return self._analyze_in_chunks(df, field_name, locale, bins,
                                           detect_outliers, test_normality,
                                           chunk_size, progress_tracker, **kwargs)

        # Standard processing for smaller datasets
        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Parsing currency values",
                "rows": total_rows
            })

        # Parse currency values
        try:
            normalized_values, currency_counts = parse_currency_field(df, field_name, locale)

            # Count valid, null, and invalid values
            valid_flags = getattr(normalized_values, 'valid_flags', [True] * len(normalized_values))
            valid_count = sum(1 for flag in valid_flags if flag)
            null_count = normalized_values.isna().sum()
            invalid_count = total_rows - valid_count - null_count

            result.update({
                'valid_count': valid_count,
                'null_count': null_count,
                'invalid_count': invalid_count,
                'null_percentage': (null_count / total_rows * 100) if total_rows > 0 else 0.0,
                'invalid_percentage': (invalid_count / total_rows * 100) if total_rows > 0 else 0.0,
                'currency_counts': currency_counts,
                'multi_currency': len(currency_counts) > 1
            })

            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Currency parsing complete",
                    "valid": valid_count,
                    "null": null_count,
                    "invalid": invalid_count
                })

        except Exception as e:
            logger.error(f"Error parsing currency field {field_name}: {e}", exc_info=True)
            return {
                'error': f"Error parsing currency field: {str(e)}",
                'field_name': field_name,
                'total_rows': total_rows
            }

        # Calculate statistics on valid values
        valid_values = normalized_values.dropna()

        if len(valid_values) == 0:
            stats = create_empty_currency_stats()
            result['stats'] = stats

            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Statistics calculation",
                    "warning": "No valid values for statistics"
                })

            return result

        # Calculate currency statistics
        try:
            stats = analyze_currency_stats(valid_values, currency_counts)

            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Basic statistics calculated",
                    "min": stats.get('min'),
                    "max": stats.get('max')
                })

            # Calculate percentiles
            stats['percentiles'] = calculate_percentiles(valid_values)

            # Calculate histogram data
            stats['histogram'] = calculate_histogram(valid_values, bins)

            # Detect outliers if requested
            if detect_outliers:
                from pamola_core.profiling.commons.numeric_utils import detect_outliers as detect_outliers_func
                outlier_results = detect_outliers_func(valid_values)
                stats['outliers'] = outlier_results

                if progress_tracker:
                    progress_tracker.update(1, {
                        "step": "Outlier detection",
                        "outliers_found": outlier_results.get('count', 0)
                    })

            # Test normality if requested and we have enough data
            if test_normality and len(valid_values) >= 8:
                try:
                    from pamola_core.profiling.commons.numeric_utils import test_normality as test_normality_func
                    normality_results = test_normality_func(valid_values)
                    stats['normality'] = normality_results

                    if progress_tracker:
                        progress_tracker.update(1, {
                            "step": "Normality testing",
                            "is_normal": normality_results.get('is_normal', False)
                        })
                except Exception as e:
                    logger.warning(f"Error during normality testing for {field_name}: {e}")
                    stats['normality'] = {
                        'error': str(e),
                        'is_normal': False
                    }
            else:
                stats['normality'] = {
                    'is_normal': False,
                    'message': 'Insufficient data for normality testing' if len(
                        valid_values) < 8 else 'Normality testing skipped'
                }

            # Generate samples for JSON output
            stats['samples'] = generate_currency_samples(stats)

            # Currency-specific semantic analysis
            negative_count = stats.get('negative_count', 0)
            if negative_count > 0:
                stats['semantic_notes'] = stats.get('semantic_notes', []) + [
                    f"Field contains {negative_count} negative values, possibly representing debits or expenses."
                ]

            zero_count = stats.get('zero_count', 0)
            if zero_count > 0:
                stats['semantic_notes'] = stats.get('semantic_notes', []) + [
                    f"Field contains {zero_count} zero values, possibly representing unpaid/free items or placeholder values."
                ]

            # Check for suspiciously large values (potential data entry errors)
            if stats.get('max', 0) > stats.get('mean', 0) * 100:
                stats['semantic_notes'] = stats.get('semantic_notes', []) + [
                    "Field contains extremely large values that may be data entry errors."
                ]

            # Add to result
            result['stats'] = stats

            # Finalize progress
            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Analysis complete",
                    "field": field_name
                })

            return result

        except Exception as e:
            logger.error(f"Error calculating statistics for {field_name}: {e}", exc_info=True)
            result['error'] = f"Error calculating statistics: {str(e)}"
            return result

    def _analyze_with_dask(self,
                           df: pd.DataFrame,
                           field_name: str,
                           locale: str,
                           bins: int,
                           detect_outliers: bool,
                           test_normality: bool,
                           progress_tracker: Optional[ProgressTracker] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Analyze a currency field using Dask for large datasets.

        Parameters match the main analyze method.
        """
        try:
            import dask.dataframe as dd

            # Convert to Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=kwargs.get('npartitions', None))

            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Dask initialization",
                    "partitions": ddf.npartitions
                })

            # Basic counts that can be done with Dask
            total_rows = len(df)
            null_count = ddf[field_name].isna().sum().compute()

            # For currency parsing, we need custom logic that may not work well with Dask
            # Do a sample-based analysis first
            sample_size = min(1000, total_rows)
            sample_df = df.sample(n=sample_size) if total_rows > sample_size else df

            # Detect currency from sample
            detected_currency = detect_currency_from_sample(sample_df, field_name)

            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Sample analysis",
                    "detected_currency": detected_currency
                })

            # Define a function to normalize values that can be applied to Dask partitions
            def normalize_partition(partition):
                normalized, _ = parse_currency_field(partition, field_name, locale)
                return normalized

            # Apply to Dask DataFrame
            normalized_values = ddf.map_partitions(normalize_partition, meta=pd.Series(dtype='float64'))

            # Compute basic statistics
            mean = normalized_values.mean().compute()
            median = normalized_values.quantile(0.5).compute()
            std = normalized_values.std().compute()
            min_val = normalized_values.min().compute()
            max_val = normalized_values.max().compute()

            # Create results structure
            valid_count = total_rows - null_count
            result = {
                'field_name': field_name,
                'total_rows': total_rows,
                'valid_count': valid_count,
                'null_count': null_count,
                'null_percentage': (null_count / total_rows * 100) if total_rows > 0 else 0.0,
                'is_detected_currency': True,
                'locale_used': locale,
                'detected_currency': detected_currency,
                'stats': {
                    'min': float(min_val),
                    'max': float(max_val),
                    'mean': float(mean),
                    'median': float(median),
                    'std': float(std),
                    'valid_count': int(valid_count),
                    'multi_currency': False,  # Simplified assumption with Dask
                    'currency_distribution': {detected_currency: valid_count} if detected_currency != 'UNKNOWN' else {}
                }
            }

            if progress_tracker:
                progress_tracker.update(1, {
                    "step": "Dask statistics calculated",
                    "min": result['stats']['min'],
                    "max": result['stats']['max']
                })

            # For more advanced statistics, we'd need to bring data back to pandas
            # which may defeat the purpose of using Dask for large datasets
            # Consider implementing with Dask's built-in histogram, etc.

            # Add note about Dask usage
            result[
                'note'] = "Analysis performed using Dask for large dataset. Some detailed metrics may be unavailable."

            return result

        except Exception as e:
            logger.error(f"Error in Dask analysis for {field_name}: {e}", exc_info=True)
            return {
                'error': f"Error in Dask analysis: {str(e)}",
                'field_name': field_name
            }

    def _analyze_in_chunks(self,
                           df: pd.DataFrame,
                           field_name: str,
                           locale: str,
                           bins: int,
                           detect_outliers: bool,
                           test_normality: bool,
                           chunk_size: int,
                           progress_tracker: Optional[ProgressTracker] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Analyze a currency field in chunks for large datasets.

        Parameters match the main analyze method, with chunk_size determining the size of each chunk.
        """
        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size

        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Chunked processing setup",
                "total_chunks": total_chunks,
                "chunk_size": chunk_size
            })

        # Initialize accumulators
        all_values = []
        all_currencies = {}
        null_count = 0
        invalid_count = 0

        # Process each chunk
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]

            # Parse chunk
            normalized_values, currency_counts = parse_currency_field(chunk, field_name, locale)

            # Count nulls and invalids
            valid_flags = getattr(normalized_values, 'valid_flags', [True] * len(normalized_values))
            chunk_valid_count = sum(1 for flag in valid_flags if flag)
            chunk_null_count = normalized_values.isna().sum()
            chunk_invalid_count = len(chunk) - chunk_valid_count - chunk_null_count

            # Update accumulators
            all_values.append(normalized_values.dropna())
            null_count += chunk_null_count
            invalid_count += chunk_invalid_count

            # Merge currency counts
            for currency, count in currency_counts.items():
                all_currencies[currency] = all_currencies.get(currency, 0) + count

            if progress_tracker:
                progress_tracker.update(0, {
                    "step": f"Processing chunk {i + 1}/{total_chunks}",
                    "valid": chunk_valid_count,
                    "null": chunk_null_count,
                    "invalid": chunk_invalid_count
                })

        # Combine values from all chunks
        if all_values:
            combined_values = pd.concat(all_values)
        else:
            combined_values = pd.Series()

        valid_count = len(combined_values)

        # Create result structure
        result = {
            'field_name': field_name,
            'total_rows': total_rows,
            'valid_count': valid_count,
            'null_count': null_count,
            'invalid_count': invalid_count,
            'null_percentage': (null_count / total_rows * 100) if total_rows > 0 else 0.0,
            'invalid_percentage': (invalid_count / total_rows * 100) if total_rows > 0 else 0.0,
            'currency_counts': all_currencies,
            'multi_currency': len(all_currencies) > 1,
            'is_detected_currency': is_currency_field(field_name),
            'locale_used': locale
        }

        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Chunks processed",
                "valid_total": valid_count,
                "currencies_detected": len(all_currencies)
            })

        # If we have no valid values, return early
        if valid_count == 0:
            result['stats'] = create_empty_currency_stats()
            return result

        # Calculate statistics on combined values
        stats = analyze_currency_stats(combined_values, all_currencies)

        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Statistics calculated",
                "min": stats.get('min'),
                "max": stats.get('max')
            })

        # Add percentiles and histogram
        stats['percentiles'] = calculate_percentiles(combined_values)
        stats['histogram'] = calculate_histogram(combined_values, bins)

        # Detect outliers if requested and we have enough data
        if detect_outliers and len(combined_values) >= 5:
            from pamola_core.profiling.commons.numeric_utils import detect_outliers as detect_outliers_func
            stats['outliers'] = detect_outliers_func(combined_values)
        else:
            stats['outliers'] = {
                'count': 0,
                'percentage': 0.0,
                'message': 'Outlier detection skipped or insufficient data'
            }

        # Test normality if requested and we have enough data
        if test_normality and len(combined_values) >= 8:
            try:
                from pamola_core.profiling.commons.numeric_utils import test_normality as test_normality_func
                stats['normality'] = test_normality_func(combined_values)
            except Exception as e:
                logger.warning(f"Error in normality testing for {field_name}: {e}")
                stats['normality'] = {
                    'is_normal': False,
                    'error': str(e)
                }
        else:
            stats['normality'] = {
                'is_normal': False,
                'message': 'Insufficient data for normality testing' if len(
                    combined_values) < 8 else 'Normality testing skipped'
            }

        # Generate samples for JSON output
        stats['samples'] = generate_currency_samples(stats)

        # Add to result
        result['stats'] = stats
        result['note'] = "Analysis performed in chunks due to large dataset size"

        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Chunked analysis complete",
                "field": field_name
            })

        return result


@register(version="1.0.0")
class CurrencyOperation(FieldOperation):
    """
    Operation for analyzing currency fields.

    This operation extends the FieldOperation base class and provides methods for
    executing currency field analysis, including visualization generation and result reporting.
    """

    def __init__(self,
                 field_name: str,
                 locale: str = 'en_US',
                 bins: int = 10,
                 detect_outliers: bool = True,
                 test_normality: bool = True,
                 description: str = "",
                 chunk_size: int = 10000,
                 use_dask: bool = False,
                 generate_plots: bool = True,
                 include_timestamp: bool = True,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 **kwargs):
        """
        Initialize a currency field operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to analyze
        locale : str
            Locale to use for parsing (default: 'en_US')
        bins : int
            Number of bins for histogram analysis
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to perform normality testing
        description : str
            Description of the operation (optional)
        **kwargs : dict
            Additional parameters passed to the base class
        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of currency field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )
        self.locale = locale
        self.bins = bins
        self.detect_outliers = detect_outliers
        self.test_normality = test_normality
        self.analyzer = CurrencyAnalyzer()
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.generate_plots = generate_plots
        self.include_timestamp = include_timestamp

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the currency field analysis operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : ProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation:
            - locale: str, locale to use for parsing
            - chunk_size: int, size of chunks for processing
            - use_dask: bool, whether to use Dask for large datasets
            - generate_plots: bool, whether to generate visualizations
            - include_timestamp: bool, whether to include timestamps in filenames

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs with defaults
        locale = kwargs.get('locale', self.locale)
        generate_plots = kwargs.get('generate_plots', self.generate_plots)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        use_dask = kwargs.get('use_dask', self.use_dask)
        encryption_key = kwargs.get('encryption_key', None)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        try:
            # Get DataFrame from data source
            dataset_name = kwargs.get('dataset_name', "main")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source"
                )

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame"
                )

            # Add operation to reporter
            reporter.add_operation(f"Analyzing currency field: {self.field_name}", details={
                "field_name": self.field_name,
                "locale": locale,
                "bins": self.bins,
                "detect_outliers": self.detect_outliers,
                "test_normality": self.test_normality,
                "operation_type": "currency_analysis"
            })

            # Execute the analyzer
            analysis_results = self.analyzer.analyze(
                df=df,
                field_name=self.field_name,
                locale=locale,
                bins=self.bins,
                detect_outliers=self.detect_outliers,
                test_normality=self.test_normality,
                chunk_size=chunk_size,
                use_dask=use_dask,
                progress_tracker=progress_tracker,
                **kwargs
            )

            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            # Save analysis results to JSON
            stats_filename = f"{self.field_name}_stats.json"
            stats_path = output_dir / stats_filename

            write_json(analysis_results, stats_path, encryption_key=encryption_key)
            result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis", category=Constants.Artifact_Category_Output)

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), f"{self.field_name} currency analysis")

            # Generate visualizations if requested
            if generate_plots:
                kwargs_encryption = {
                    "use_encryption": kwargs.get('use_encryption', False),
                    "encryption_key": encryption_key
                }
                self._generate_visualizations(
                    df,
                    analysis_results,
                    visualizations_dir,
                    result,
                    reporter,
                    **kwargs_encryption
                )

            # Save sample records with original currency values
            self._save_sample_records(
                df,
                analysis_results,
                dictionaries_dir,
                result,
                reporter,
                encryption_key=encryption_key
            )

            # Add metrics to the result
            self._add_metrics_to_result(analysis_results, result)

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of currency field {self.field_name} completed", details={
                "valid_values": analysis_results.get('valid_count', 0),
                "null_percentage": analysis_results.get('null_percentage', 0),
                "multi_currency": analysis_results.get('multi_currency', False),
                "currencies_detected": len(analysis_results.get('currency_counts', {}))
            })

            return result

        except Exception as e:
            logger.exception(f"Error in currency operation for {self.field_name}: {e}")

            # Add error to reporter
            reporter.add_operation(f"Error analyzing currency field {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing currency field {self.field_name}: {str(e)}"
            )

    def _generate_visualizations(self,
                                 df: pd.DataFrame,
                                 analysis_results: Dict[str, Any],
                                 vis_dir: Path,
                                 result: OperationResult,
                                 reporter: Any,
                                 **kwargs):
        """
        Generate visualizations for the currency field analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        stats_dict = analysis_results.get('stats', {})

        # Parse currency values for visualization
        normalized_values, _ = parse_currency_field(df, self.field_name, analysis_results.get('locale_used', 'en_US'))
        valid_values = normalized_values.dropna()

        if len(valid_values) == 0:
            logger.warning(f"No valid currency values for visualization in {self.field_name}")
            return

        # Generate distribution histogram
        if 'histogram' in stats_dict:
            try:
                hist_filename = f"{self.field_name}_distribution.png"
                hist_path = vis_dir / hist_filename

                # Create histogram using the visualization module
                min_value = stats_dict.get('min')
                max_value = stats_dict.get('max')

                title = f"Distribution of {self.field_name}"
                if min_value is not None and max_value is not None:
                    title += f" (min: {min_value:.2f}, max: {max_value:.2f})"

                # Add currency information if available
                currencies = analysis_results.get('currency_counts', {})
                if currencies:
                    main_currency = max(currencies.items(), key=lambda x: x[1])[0] if currencies else "Unknown"
                    if main_currency != "UNKNOWN":
                        title += f" ({main_currency})"

                # Create the histogram
                hist_result = create_histogram(
                    data=valid_values,
                    output_path=str(hist_path),
                    title=title,
                    x_label=f"{self.field_name} Value",
                    y_label="Frequency",
                    bins=self.bins,
                    **kwargs
                )

                if not hist_result.startswith("Error"):
                    result.add_artifact("png", hist_path, f"{self.field_name} distribution histogram", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(hist_path), f"{self.field_name} distribution histogram")
            except Exception as e:
                logger.warning(f"Error creating histogram for {self.field_name}: {e}")

        # Generate boxplot
        if len(valid_values) >= 5:
            try:
                boxplot_filename = f"{self.field_name}_boxplot.png"
                boxplot_path = vis_dir / boxplot_filename

                # Create boxplot using the visualization module
                boxplot_result = create_boxplot(
                    data={self.field_name: valid_values.tolist()},
                    output_path=str(boxplot_path),
                    title=f"Boxplot of {self.field_name}",
                    y_label=self.field_name,
                    show_points=True,
                    **kwargs
                )

                if not boxplot_result.startswith("Error"):
                    result.add_artifact("png", boxplot_path, f"{self.field_name} boxplot", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(boxplot_path), f"{self.field_name} boxplot")
            except Exception as e:
                logger.warning(f"Error creating boxplot for {self.field_name}: {e}")

        # Generate Q-Q plot for normality if requested
        if self.test_normality and 'normality' in stats_dict and len(valid_values) >= 10:
            try:
                qq_filename = f"{self.field_name}_qq_plot.png"
                qq_path = vis_dir / qq_filename

                # Generate synthetic normal data for comparison
                np.random.seed(42)  # For reproducibility
                normal_data = np.random.normal(loc=valid_values.mean(), scale=valid_values.std(),
                                               size=len(valid_values))
                normal_data.sort()

                # Sort the actual data for Q-Q plot
                sorted_data = valid_values.sort_values()

                # Create scatter plot comparing theoretical quantiles to actual data
                normality_info = stats_dict['normality']
                is_normal = normality_info.get('is_normal', False)
                p_value = normality_info.get('shapiro', {}).get('p_value', None)

                title = f"Q-Q Plot for {self.field_name}"
                if p_value is not None:
                    title += f" (Shapiro p-value: {p_value:.4f})"

                # Create the Q-Q plot
                qq_result = create_correlation_pair(
                    x_data=normal_data,
                    y_data=sorted_data,
                    output_path=str(qq_path),
                    title=title,
                    x_label="Theoretical Quantiles",
                    y_label="Sample Quantiles",
                    add_trendline=True,
                    **kwargs
                )

                if not qq_result.startswith("Error"):
                    result.add_artifact("png", qq_path, f"{self.field_name} Q-Q plot (normality test)", category=Constants.Artifact_Category_Visualization)
                    reporter.add_artifact("png", str(qq_path), f"{self.field_name} Q-Q plot")
            except Exception as e:
                logger.warning(f"Error creating Q-Q plot for {self.field_name}: {e}")

    def _save_sample_records(self,
                             df: pd.DataFrame,
                             analysis_results: Dict[str, Any],
                             dict_dir: Path,
                             result: OperationResult,
                             reporter: Any,
                             encryption_key: Optional[str] = None):
        """
        Save sample records with original currency values.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        dict_dir : Path
            Directory to save sample records
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        try:
            # Create a sample of records containing the field
            # Prioritize getting diverse examples including:
            # - Records with different currencies
            # - Records with high/low values
            # - Records with missing/invalid values

            # Get field values
            field_values = df[self.field_name]
            total_records = len(field_values)

            # Determine how many samples to take
            sample_size = min(100, total_records)
            if sample_size == 0:
                return

            # Create a sample index
            indices = []

            # Start with any outliers if available
            stats_dict = analysis_results.get('stats', {})
            outliers = stats_dict.get('outliers', {})

            if outliers and 'indices' in outliers and outliers['indices']:
                # Take up to 20% of the sample from outliers
                outlier_indices = outliers['indices']
                outlier_sample_size = min(len(outlier_indices), int(sample_size * 0.2))
                if outlier_sample_size > 0:
                    indices.extend(np.random.choice(outlier_indices, size=outlier_sample_size, replace=False))

            # Handle multi-currency examples if available
            if analysis_results.get('multi_currency', False):
                # Get normalized values and currencies
                normalized_values, _ = parse_currency_field(df, self.field_name,
                                                            analysis_results.get('locale_used', 'en_US'))
                currencies = getattr(normalized_values, 'currencies', [None] * len(normalized_values))

                # Get indices for each currency
                for currency in set(currencies):
                    if currency:
                        currency_indices = [i for i, c in enumerate(currencies) if c == currency]
                        if currency_indices:
                            # Take up to 10% of the sample from each currency
                            currency_sample_size = min(len(currency_indices), int(sample_size * 0.1))
                            if currency_sample_size > 0:
                                indices.extend(
                                    np.random.choice(currency_indices, size=currency_sample_size, replace=False))

            # Add some null values if available
            null_indices = df[df[self.field_name].isna()].index.tolist()
            if null_indices:
                # Take up to 10% of the sample from null values
                null_sample_size = min(len(null_indices), int(sample_size * 0.1))
                if null_sample_size > 0:
                    indices.extend(np.random.choice(null_indices, size=null_sample_size, replace=False))

            # Fill the rest with random samples
            if len(indices) < sample_size:
                remaining_indices = list(set(range(total_records)) - set(indices))
                remaining_sample_size = sample_size - len(indices)

                if remaining_indices and remaining_sample_size > 0:
                    indices.extend(
                        np.random.choice(remaining_indices, size=min(len(remaining_indices), remaining_sample_size),
                                         replace=False))

            # Deduplicate and sort indices
            indices = sorted(set(indices))

            # Create sample DataFrame
            if df.index.name:
                id_field = df.index.name
            else:
                id_field = "index"

            sample_df = df.loc[indices, [self.field_name]].copy()
            sample_df = sample_df.reset_index()

            # Save to CSV
            sample_filename = f"{self.field_name}_sample.csv"
            sample_path = dict_dir / sample_filename

            write_dataframe_to_csv(sample_df, sample_path, encryption_key=encryption_key)

            # Add to result
            result.add_artifact("csv", sample_path, f"{self.field_name} sample records", category=Constants.Artifact_Category_Dictionary)
            reporter.add_artifact("csv", str(sample_path), f"{self.field_name} sample records")

        except Exception as e:
            logger.warning(f"Error saving sample records for {self.field_name}: {e}")

    def _add_metrics_to_result(self, analysis_results: Dict[str, Any], result: OperationResult):
        """
        Add metrics from the analysis results to the operation result.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        result : OperationResult
            Operation result to add metrics to
        """
        # Add basic metrics
        result.add_metric("total_rows", analysis_results.get('total_rows', 0))
        result.add_metric("valid_count", analysis_results.get('valid_count', 0))
        result.add_metric("null_count", analysis_results.get('null_count', 0))
        result.add_metric("invalid_count", analysis_results.get('invalid_count', 0))
        result.add_metric("null_percentage", analysis_results.get('null_percentage', 0.0))
        result.add_metric("invalid_percentage", analysis_results.get('invalid_percentage', 0.0))
        result.add_metric("multi_currency", analysis_results.get('multi_currency', False))
        result.add_metric("currency_count", len(analysis_results.get('currency_counts', {})))

        # Add statistics metrics
        stats_dict = analysis_results.get('stats', {})

        if stats_dict:
            # Add basic statistics
            result.add_nested_metric("statistics", "min", stats_dict.get('min'))
            result.add_nested_metric("statistics", "max", stats_dict.get('max'))
            result.add_nested_metric("statistics", "mean", stats_dict.get('mean'))
            result.add_nested_metric("statistics", "median", stats_dict.get('median'))
            result.add_nested_metric("statistics", "std", stats_dict.get('std'))

            # Add distribution statistics
            result.add_nested_metric("statistics", "skewness", stats_dict.get('skewness'))
            result.add_nested_metric("statistics", "kurtosis", stats_dict.get('kurtosis'))

            # Add zero and negative counts
            result.add_nested_metric("statistics", "zero_count", stats_dict.get('zero_count', 0))
            result.add_nested_metric("statistics", "zero_percentage", stats_dict.get('zero_percentage', 0.0))
            result.add_nested_metric("statistics", "negative_count", stats_dict.get('negative_count', 0))
            result.add_nested_metric("statistics", "negative_percentage", stats_dict.get('negative_percentage', 0.0))

            # Add outlier metrics
            outliers = stats_dict.get('outliers', {})
            if outliers:
                result.add_nested_metric("outliers", "count", outliers.get('count', 0))
                result.add_nested_metric("outliers", "percentage", outliers.get('percentage', 0.0))
                result.add_nested_metric("outliers", "lower_bound", outliers.get('lower_bound'))
                result.add_nested_metric("outliers", "upper_bound", outliers.get('upper_bound'))

            # Add normality metrics
            normality = stats_dict.get('normality', {})
            if normality:
                result.add_nested_metric("normality", "is_normal", normality.get('is_normal', False))
                shapiro = normality.get('shapiro', {})
                if shapiro:
                    result.add_nested_metric("normality", "shapiro_stat", shapiro.get('statistic'))
                    result.add_nested_metric("normality", "shapiro_p_value", shapiro.get('p_value'))

        # Add currency metrics
        currency_counts = analysis_results.get('currency_counts', {})
        if currency_counts:
            for currency, count in currency_counts.items():
                result.add_nested_metric("currencies", currency, count)