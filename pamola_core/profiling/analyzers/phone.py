"""
Phone data analyzer for the anonymization project.

This module provides analyzers and operations for phone number fields,
following the new operation architecture. It includes phone number validation,
component parsing, and messenger detection capabilities.

It integrates with utility modules:
- io.py: For reading/writing data and managing directories
- visualization.py: For creating standardized plots
- progress.py: For tracking operation progress
- logging.py: For operation logging

Operations:
- PhoneAnalyzer: Static methods for phone analysis
- PhoneOperation: Main operation for phone field analysis
- analyze_phone_fields: Function for analyzing multiple phone fields
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from pamola_core.utils.helpers import filter_used_kwargs
from pamola_core.profiling.commons.phone_utils import (
    analyze_phone_field,
    analyze_phone_field_with_chunk,
    analyze_phone_field_with_joblib,
    analyze_phone_field_with_dask,
    create_country_code_dictionary,
    create_operator_code_dictionary,
    create_messenger_dictionary,
    estimate_resources
)
from pamola_core.utils.io import write_json, ensure_directory, get_timestamped_filename, load_data_operation, write_dataframe_to_csv, load_settings_operation
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_base import FieldOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.common.constants import Constants
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode
# Configure logger
logger = logging.getLogger(__name__)


class PhoneAnalyzer:
    """
    Analyzer for phone number fields.

    This analyzer provides static methods for validating phone numbers, extracting components,
    and detecting messenger references in comments.
    """

    @staticmethod
    def analyze(df: pd.DataFrame,
                field_name: str,
                patterns_csv: Optional[str] = None,
                chunk_size: int = 10000,
                use_dask: bool = False,
                npartitions: int = 2,
                use_vectorization: bool = False,
                parallel_processes: int = 2,
                **kwargs) -> Dict[str, Any]:
        """
        Analyze a phone field in the given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_name : str
            The name of the field to analyze
        patterns_csv : str, optional
            Path to CSV with messenger patterns
        chunk_size : int
            Size of data chunks for processing large datasets
        use_dask : bool, optional
            Whether to use Dask for processing (default: False).
        npartitions : int, optional
            Number of partitions use with Dask (default: 1).
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False).
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1).
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        Dict[str, Any]
            The results of the analysis
        """
        analysis_results = {}
        flag_processed = False
        try:
            if not flag_processed and use_dask and npartitions > 1:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Dask")
                logger.info(f"Parallel Workers: {npartitions}")
                logger.info(f"Using dask processing with chunk size {chunk_size}")

                analysis_results = analyze_phone_field_with_dask(
                    df=df,
                    field_name=field_name,
                    patterns_csv=patterns_csv,
                    npartitions=npartitions,
                    chunk_size=chunk_size,
                    **kwargs
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using dask processing: {e}")
            flag_processed = False

        try:
            if not flag_processed and use_vectorization and parallel_processes > 1:
                logger.info("Parallel Enabled")
                logger.info("Parallel Engine: Joblib")
                logger.info(f"Parallel Workers: {parallel_processes}")
                logger.info(f"Using vectorized processing with chunk size {chunk_size}")

                analysis_results = analyze_phone_field_with_joblib(
                    df=df,
                    field_name=field_name,
                    patterns_csv=patterns_csv,
                    n_jobs=parallel_processes,
                    chunk_size=chunk_size,
                    **kwargs
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using joblib processing: {e}")
            flag_processed = False

        try:
            if not flag_processed and len(df) > chunk_size:
                logger.info(f"Processing in chunks with chunk size {chunk_size}")
                total_chunks = (len(df) + chunk_size - 1) // chunk_size
                logger.info(f"Total chunks to process: {total_chunks}")

                analysis_results = analyze_phone_field_with_chunk(
                    df=df,
                    field_name=field_name,
                    patterns_csv=patterns_csv,
                    chunk_size=chunk_size,
                    **kwargs
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in using chunks processing: {e}")
            flag_processed = False

        try:
            if not flag_processed:
                logger.info("Fallback process as usual")

                analysis_results = analyze_phone_field(
                    df=df,
                    field_name=field_name,
                    patterns_csv=patterns_csv,
                    **kwargs
                )

                flag_processed = True
        except Exception as e:
            logger.exception(f"Error in processing: {e}")
            flag_processed = False

        if not flag_processed:
            logger.exception(f"Error in processing")

        return analysis_results

    @staticmethod
    def create_country_code_dictionary(df: pd.DataFrame,
                                       field_name: str,
                                       min_count: int = 1,
                                       **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for country codes.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the phone field
        min_count : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with country code frequency data and metadata
        """
        return create_country_code_dictionary(
            df=df,
            field_name=field_name,
            min_count=min_count,
            **kwargs
        )

    @staticmethod
    def create_operator_code_dictionary(df: pd.DataFrame,
                                        field_name: str,
                                        country_codes: Optional[List[str]] = None,
                                        min_count: int = 1,
                                        **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for operator codes.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the phone field
        country_codes : List[str], optional
            The country codes  to filter by (if None, use all)
        min_count : int
            Minimum frequency for inclusion in the dictionary
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with operator code frequency data and metadata
        """
        return create_operator_code_dictionary(
            df=df,
            field_name=field_name,
            country_codes=country_codes,
            min_count=min_count,
            **kwargs
        )

    @staticmethod
    def create_messenger_dictionary(df: pd.DataFrame,
                                    field_name: str,
                                    min_count: int = 1,
                                    patterns_csv: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Create a frequency dictionary for messenger mentions in phone comments.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the phone field
        min_count : int
            Minimum frequency for inclusion in the dictionary
        patterns_csv : str, optional
            Path to CSV with messenger patterns
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        Dict[str, Any]
            Dictionary with messenger frequency data and metadata
        """
        return create_messenger_dictionary(
            df=df,
            field_name=field_name,
            min_count=min_count,
            patterns_csv=patterns_csv,
            **kwargs
        )

    @staticmethod
    def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
        """
        Estimate resources needed for analyzing the phone field.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data
        field_name : str
            The name of the field to analyze

        Returns:
        --------
        Dict[str, Any]
            Estimated resource requirements
        """
        return estimate_resources(df, field_name)


@register(override=True)
class PhoneOperation(FieldOperation):
    """
    Operation for analyzing phone number fields.

    This operation wraps the PhoneAnalyzer and provides methods for
    executing analysis, saving results, and generating artifacts.
    """

    def __init__(self,
                field_name: str,
                min_frequency: int = 1,
                patterns_csv: Optional[str] = None,
                country_codes: List[str] = None,
                visualization_theme: Optional[str] = None,
                visualization_backend: Optional[str] = "plotly",
                visualization_strict: bool = False,
                visualization_timeout: int = 120,
                chunk_size: int = 10000,
                use_dask: bool = False,
                npartitions: int = 2,
                use_vectorization: bool = False,
                parallel_processes: int = 2,
                use_cache: bool = True,
                description: str = "",
                use_encryption: bool = False,
                encryption_key: Optional[Union[str, Path]] = None,
                encryption_mode: Optional[str] = None,
                include_timestamp: Optional[bool] = True,
                track_progress: Optional[bool] = True,
                generate_visualization: Optional[bool] = True):
        """
        Initialize the phone operation.

        Parameters:
        -----------
        field_name : str
            The name of the field to analyze
        min_frequency : int
            Minimum frequency for inclusion in the dictionary
        patterns_csv : str, optional
            Path to CSV file with messenger detection patterns
        country_codes : List[str], optional
            List of country codes to filter or analyze (default: None).
        visualization_theme : str, optional
            Theme to use for visualizations (default: None - uses system default)
        visualization_backend : str, optional
            Backend to use for visualizations: "plotly" or "matplotlib" (default: None - uses system default)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        visualization_timeout : int, optional
            Timeout in seconds for visualization generation (default: 120)
        chunk_size : int
            Size of data chunks for processing large datasets
        use_dask : bool, optional
            Whether to use Dask for processing (default: False).
        npartitions : int, optional
            Number of partitions use with Dask (default: 1).
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False).
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1).
        use_cache : bool
            Whether to use caching for intermediate results
        description : str
            Description of the operation (optional)
        use_encryption : bool, optional
            Whether to use encryption for output files (default: False).
        encryption_key : str or Path, optional
            Encryption key for output files (default: None).
        encryption_mode : str, optional
            Encryption mode for output files (default: None).

        """
        super().__init__(
            field_name=field_name,
            description=description or f"Analysis of phone field '{field_name}'",
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode
            )
        
        self.min_frequency = min_frequency
        self.patterns_csv = patterns_csv
        self.country_codes = country_codes
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        self.use_cache = use_cache

        # Set up performance tracking variables
        self.start_time = None
        self.end_time = None
        self.execution_time = 0
        self.process_count = 0

        self.include_timestamp = include_timestamp,
        self.track_progress = track_progress,
        self.generate_visualization = generate_visualization

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the phone analysis operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation:
            - generate_visualization: bool, whether to generate visualizations
            - dataset_name: str - Name of dataset - main
            - force_recalculation: bool - Force operation even if cached results exist - False
            - visualization_theme: str - Override theme for visualizations - None
            - visualization_backend: str - Override backend for visualizations - None
            - visualization_strict: bool - Override strict mode for visualizations - False
            - visualization_timeout: int - Override timeout for visualizations - 120

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        global logger
        if kwargs.get("logger"):
            logger = kwargs.get("logger")

        # Generate single timestamp for all artifacts
        operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Initialize timing and result
            self.start_time = time.time()
            self.process_count = 0
            result = OperationResult(status=OperationStatus.SUCCESS)

            # Decompose kwargs and introduce variables for clarity
            generate_visualization = kwargs.get('generate_visualization', True)
            force_recalculation = kwargs.get("force_recalculation", False)
            encryption_key = kwargs.get('encryption_key', None)

            # Extract visualization parameters
            self.visualization_theme = kwargs.get("visualization_theme", self.visualization_theme)
            self.visualization_backend = kwargs.get("visualization_backend", self.visualization_backend)
            self.visualization_strict = kwargs.get("visualization_strict", self.visualization_strict)
            self.visualization_timeout = kwargs.get("visualization_timeout", self.visualization_timeout)

            self.logger.info(
                f"Visualization settings: theme={self.visualization_theme}, backend={self.visualization_backend}, "
                f"strict={self.visualization_strict}, timeout={self.visualization_timeout}s"
            )

            # Set up directories
            dirs = self._prepare_directories(task_dir)
            output_dir = dirs['output']
            visualizations_dir = dirs['visualizations']
            dictionaries_dir = dirs['dictionaries']
            cache_dir = dirs["cache"]

            # Update progress if tracker provided
            if progress_tracker:
                progress_tracker.update(1, {"step": "Preparation", "field": self.field_name})

            # Get DataFrame from data source
            dataset_name = kwargs.get('dataset_name', "main")
            try:
                settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
                df = load_data_operation(data_source, dataset_name, **settings_operation)
                if df is None:
                    return OperationResult(
                        status=OperationStatus.ERROR,
                        error_message="No valid DataFrame found in data source"
                    )
            except Exception as e:
                error_message = f"Data loading error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message, exception=e)

            # Check if field exists
            if self.field_name not in df.columns:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=f"Field {self.field_name} not found in DataFrame"
                )

            # Add operation to reporter
            reporter.add_operation(f"Analyzing phone field: {self.field_name}", details={
                "field_name": self.field_name,
                "min_frequency": self.min_frequency,
                "operation_type": "phone_analysis"
            })

            # Check for cached results if caching is enabled
            if self.use_cache and not force_recalculation:
                try:
                    cached_result = self._check_cache(df, reporter, task_dir, **kwargs)
                except Exception as e:
                    error_message = f"Check cache error: {str(e)}"
                    self.logger.error(error_message)
                    return OperationResult(status=OperationStatus.ERROR, error_message=error_message, exception=e)
                
                if cached_result:
                    self.logger.info(f"Using cached results for {self.field_name}")

                    # Update progress if tracker provided
                    if progress_tracker:
                        progress_tracker.update(5, {"step": "Loaded from cache", "field": self.field_name})

                    return cached_result

            # Adjust progress tracker total steps if provided
            total_steps = 4  # Preparation, analysis, saving results, dictionaries
            if generate_visualization:
                total_steps += 1  # Add step for generating visualizations

            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "Analyzing field"})

            try:
                # Execute the analyzer
                safe_kwargs = filter_used_kwargs(kwargs, PhoneAnalyzer.analyze)
                analysis_results = PhoneAnalyzer.analyze(
                    df=df,
                    field_name=self.field_name,
                    patterns_csv=self.patterns_csv,
                    chunk_size=self.chunk_size,
                    use_dask=self.use_dask,
                    npartitions=self.npartitions,
                    use_vectorization=self.use_vectorization,
                    parallel_processes=self.parallel_processes,
                    **safe_kwargs
                )
            except Exception as e:
                error_message = f"Attribute analysis error: {str(e)}"
                self.logger.error(error_message)
                return OperationResult(status=OperationStatus.ERROR, error_message=error_message, exception=e)

            # Check for errors
            if 'error' in analysis_results:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis_results['error']
                )

            artifacts = []
            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Analysis complete", "field": self.field_name})

            # Save analysis results to JSON
            stats_filename = f"{self.field_name}_stats_{operation_timestamp}.json"
            stats_path = output_dir / stats_filename

            encryption_mode = get_encryption_mode(analysis_results, **kwargs)
            write_json(analysis_results, stats_path, encryption_key=encryption_key, encryption_mode=encryption_mode)
            result.add_artifact("json", stats_path, f"{self.field_name} statistical analysis", category=Constants.Artifact_Category_Output)

            # Add to reporter
            reporter.add_artifact("json", str(stats_path), f"{self.field_name} statistical analysis")
            artifacts.append({
                "artifact_type": "json",
                "path": str(stats_path),
                "description": f"{self.field_name} statistical analysis",
                "category": Constants.Artifact_Category_Output
            })

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Saved analysis results"})

            # Generate visualizations if requested
            if generate_visualization and self.visualization_backend is not None:
                # Update progress
                if progress_tracker:
                    progress_tracker.update(0, {"step": "Generating visualizations"})

                try:
                    kwargs_encryption = {
                        "use_encryption": self.use_encryption,
                        "encryption_key": self.encryption_key,
                    }

                    visualization_paths = self._handle_visualizations(
                        analysis_results=analysis_results,
                        visualizations_dir=visualizations_dir,
                        operation_timestamp=operation_timestamp,
                        result=result,
                        reporter=reporter,
                        vis_theme=self.visualization_theme,
                        vis_backend=self.visualization_backend,
                        vis_strict=self.visualization_strict,
                        vis_timeout=self.visualization_timeout,
                        progress_tracker=progress_tracker,
                        **kwargs_encryption,
                    )
                except Exception as e:
                    error_message = f"Error generating visualizations: {str(e)}"
                    self.logger.error(error_message)
                    # Continue execution - visualization failure is not critical
                artifacts.extend(visualization_paths)

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Created visualizations"})

            # Create and save country code dictionary
            country_dict = PhoneAnalyzer.create_country_code_dictionary(
                df=df,
                field_name=self.field_name,
                min_count=self.min_frequency,
                **kwargs
            )

            if 'error' not in country_dict:
                # Save dictionary to CSV
                dict_filename = f"{self.field_name}_country_codes_dictionary_{operation_timestamp}.csv"
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                dict_df = pd.DataFrame(country_dict['country_codes'])
                write_dataframe_to_csv(df=dict_df, file_path=dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Save detailed dictionary as JSON
                json_dict_filename = f"{self.field_name}_country_codes_dictionary_{operation_timestamp}.json"
                json_dict_path = output_dir / json_dict_filename
                encryption_mode_country_dict = get_encryption_mode(country_dict, **kwargs)
                write_json(country_dict, json_dict_path, encryption_key=encryption_key, encryption_mode=encryption_mode_country_dict)

                result.add_artifact("csv", dict_path, f"{self.field_name} country codes dictionary (CSV)", category=Constants.Artifact_Category_Dictionary)
                result.add_artifact("json", json_dict_path, f"{self.field_name} country codes dictionary (JSON)", category=Constants.Artifact_Category_Output)

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} country codes dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path), f"{self.field_name} country codes dictionary (JSON)")
                artifacts.append({
                    "artifact_type": "csv",
                    "path": str(dict_path),
                    "description": f"{self.field_name} country codes dictionary (CSV)",
                    "category": Constants.Artifact_Category_Dictionary
                })
                artifacts.append({
                    "artifact_type": "json",
                    "path": str(json_dict_path),
                    "description": f"{self.field_name} country codes dictionary (JSON)",
                    "category": Constants.Artifact_Category_Output
                })

            # Create and save operator code dictionary (for specific country or all)
            operator_dict = PhoneAnalyzer.create_operator_code_dictionary(
                df=df,
                field_name=self.field_name,
                country_codes=self.country_codes,
                min_count=self.min_frequency,
                **kwargs
            )

            if 'error' not in operator_dict:
                # Save dictionary to CSV
                dict_filename = f"{self.field_name}_operator_codes_dictionary_{operation_timestamp}.csv"
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                dict_df = pd.DataFrame(operator_dict['operator_codes'])
                write_dataframe_to_csv(df=dict_df, file_path=dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Save detailed dictionary as JSON
                json_dict_filename = f"{self.field_name}_operator_codes_dictionary_{operation_timestamp}.json"
                json_dict_path = output_dir / json_dict_filename
                encryption_mode_operator_dict = get_encryption_mode(operator_dict, **kwargs)
                write_json(operator_dict, json_dict_path, encryption_key=encryption_key, encryption_mode=encryption_mode_operator_dict)

                result.add_artifact("csv", dict_path, f"{self.field_name} operator codes dictionary (CSV)", category=Constants.Artifact_Category_Dictionary)
                result.add_artifact("json", json_dict_path, f"{self.field_name} operator codes dictionary (JSON)", category=Constants.Artifact_Category_Output)

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} operator codes dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path),
                                      f"{self.field_name} operator codes dictionary (JSON)")
                artifacts.append({
                    "artifact_type": "csv",
                    "path": str(dict_path),
                    "description": f"{self.field_name} operator codes dictionary (CSV)",
                    "category": Constants.Artifact_Category_Dictionary
                })
                artifacts.append({
                    "artifact_type": "json",
                    "path": str(json_dict_path),
                    "description": f"{self.field_name} operator codes dictionary (JSON)",
                    "category": Constants.Artifact_Category_Output
                })

            # Create and save messenger dictionary
            messenger_dict = PhoneAnalyzer.create_messenger_dictionary(
                df=df,
                field_name=self.field_name,
                min_count=self.min_frequency,
                patterns_csv=self.patterns_csv,
                **kwargs
            )

            if 'error' not in messenger_dict:
                # Save dictionary to CSV
                dict_filename = f"{self.field_name}_messenger_dictionary{operation_timestamp}.csv"
                dict_path = dictionaries_dir / dict_filename

                # Create DataFrame and save to CSV
                import pandas as pd
                dict_df = pd.DataFrame(messenger_dict['messengers'])
                write_dataframe_to_csv(df=dict_df, file_path=dict_path, index=False, encoding='utf-8', encryption_key=encryption_key)

                # Save detailed dictionary as JSON
                json_dict_filename = f"{self.field_name}_messenger_dictionary_{operation_timestamp}.json"
                json_dict_path = output_dir / json_dict_filename
                encryption_mode_messenger_dict = get_encryption_mode(messenger_dict, **kwargs)
                write_json(messenger_dict, json_dict_path, encryption_key=encryption_key, encryption_mode=encryption_mode_messenger_dict)

                result.add_artifact("csv", dict_path, f"{self.field_name} messenger dictionary (CSV)", category=Constants.Artifact_Category_Dictionary)
                result.add_artifact("json", json_dict_path, f"{self.field_name} messenger dictionary (JSON)", category=Constants.Artifact_Category_Output)

                reporter.add_artifact("csv", str(dict_path), f"{self.field_name} messenger dictionary (CSV)")
                reporter.add_artifact("json", str(json_dict_path), f"{self.field_name} messenger dictionary (JSON)")
                artifacts.append({
                    "artifact_type": "csv",
                    "path": str(dict_path),
                    "description": f"{self.field_name} messenger dictionary (CSV)",
                    "category": Constants.Artifact_Category_Dictionary
                })
                artifacts.append({
                    "artifact_type": "json",
                    "path": str(json_dict_path),
                    "description": f"{self.field_name} messenger dictionary (JSON)",
                    "category": Constants.Artifact_Category_Output
                })

            # Cache results if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        df=df,
                        analysis_results=analysis_results,
                        artifacts=artifacts,
                        task_dir=task_dir
                    )
                except Exception as e:
                    error_message = f"Failed to cache results: {str(e)}"
                    self.logger.error(error_message)
                    # Continue execution - cache failure is not critical

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Created dictionaries"})

            # Add metrics to the result
            result.add_metric("total_records", analysis_results.get('total_rows', 0))
            result.add_metric("null_count", analysis_results.get('null_count', 0))
            result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
            result.add_metric("valid_count", analysis_results.get('valid_count', 0))
            result.add_metric("valid_percentage", analysis_results.get('valid_percentage', 0))
            result.add_metric("format_error_count", analysis_results.get('format_error_count', 0))
            result.add_metric("has_comment_count", analysis_results.get('has_comment_count', 0))

            # Add normalization metrics if available
            if 'normalization_success_count' in analysis_results:
                result.add_metric("normalization_success_count", analysis_results.get('normalization_success_count', 0))
                result.add_metric("normalization_success_percentage",
                                  analysis_results.get('normalization_success_percentage', 0))

            # Add final operation status to reporter
            reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                "valid_phones": analysis_results.get('valid_count', 0),
                "format_errors": analysis_results.get('format_error_count', 0),
                "with_comments": analysis_results.get('has_comment_count', 0),
                "normalization_success": analysis_results.get('normalization_success_count', 0)
            })

            self.end_time = time.time()
            if self.end_time and self.start_time:
                self.execution_time = self.end_time - self.start_time

            return result
        except Exception as e:
            self.logger.exception(f"Error in phone operation for {self.field_name}: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            reporter.add_operation(f"Error analyzing {self.field_name}",
                                   status="error",
                                   details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error analyzing phone field {self.field_name}: {str(e)}",
                exception=e,
            )

    def _check_cache(
            self,
            df: pd.DataFrame,
            reporter: Any,
            task_dir: Path,
            **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation
        reporter : Any
            The reporter to log artifacts to
        task_dir : Path
            Task directory

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            operation_cache_dir = OperationCache(cache_dir=task_dir/"cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache_dir.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            if cached_data:
                logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Restore artifacts from cache
                artifacts_restored = 0

                # Add artifacts
                artifacts = cached_data.get("artifacts", [])
                for artifact in artifacts:
                    if artifact and isinstance(artifact, dict):
                        if Path(artifact.get("path")).exists():
                            artifacts_restored += 1
                            cached_result.add_artifact(
                                artifact.get("artifact_type"),
                                artifact.get("path"),
                                artifact.get("description"),
                                category=artifact.get("category")
                            )
                            reporter.add_artifact(
                                artifact.get("artifact_type"),
                                artifact.get("path"),
                                artifact.get("description")
                            )

                # Add cached metrics to result
                analysis_results = cached_data.get("analysis_results", {})

                cached_result.add_metric("total_records", analysis_results.get('total_rows', 0))
                cached_result.add_metric("null_count", analysis_results.get('null_count', 0))
                cached_result.add_metric("null_percentage", analysis_results.get('null_percentage', 0))
                cached_result.add_metric("valid_count", analysis_results.get('valid_count', 0))
                cached_result.add_metric("valid_percentage", analysis_results.get('valid_percentage', 0))
                cached_result.add_metric("format_error_count", analysis_results.get('format_error_count', 0))
                cached_result.add_metric("has_comment_count", analysis_results.get('has_comment_count', 0))

                # Add normalization metrics if available
                if 'normalization_success_count' in analysis_results:
                    cached_result.add_metric("normalization_success_count",
                                      analysis_results.get('normalization_success_count', 0))
                    cached_result.add_metric("normalization_success_percentage",
                                      analysis_results.get('normalization_success_percentage', 0))

                # Add final operation status to reporter
                reporter.add_operation(f"Analysis of {self.field_name} completed", details={
                    "valid_phones": analysis_results.get('valid_count', 0),
                    "format_errors": analysis_results.get('format_error_count', 0),
                    "with_comments": analysis_results.get('has_comment_count', 0),
                    "normalization_success": analysis_results.get('normalization_success_count', 0)
                })

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric("cache_timestamp", cached_data.get("timestamp", "unknown"))
                cached_result.add_metric("artifacts_restored", artifacts_restored)

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None

    def _save_to_cache(
            self,
            df: pd.DataFrame,
            analysis_results: Dict[str, Any],
            artifacts: List[Dict[str, str]],
            task_dir: Path
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation
        analysis_results : dict
            Analysis results to cache
        artifacts : list of dict
            Artifacts
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache, OperationCache

            # Generate operation cache
            operation_cache_dir = OperationCache(cache_dir=task_dir/"cache")

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "analysis_results": analysis_results,
                "artifacts": artifacts,
                "data_info": {
                    "df_length": len(df)
                }
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache_dir.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(
            self,
            df: pd.DataFrame
    ) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Unique cache key
        """
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get operation parameters
        parameters = self._get_operation_parameters()

        # Generate data hash based on key characteristics
        data_hash = self._generate_data_hash(df)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.__class__.__name__,
            parameters=parameters,
            data_hash=data_hash
        )

    def _get_operation_parameters(
            self
    ) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "field_name": self.field_name,
            "min_frequency": self.min_frequency,
            "patterns_csv": self.patterns_csv,
            "country_codes": self.country_codes,
            "version": self.version
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(
            self
    ) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

    def _generate_data_hash(
            self,
            df: pd.DataFrame
    ) -> str:
        """
        Generate a hash representing the key characteristics of the data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input data for the operation

        Returns:
        --------
        str
            Hash string representing the data
        """
        import json
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format='iso')
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _handle_visualizations(
            self,
            analysis_results: Dict[str, Any],
            visualizations_dir: Path,
            operation_timestamp: Optional[str],
            result: OperationResult,
            reporter: Any,
            vis_theme: Optional[str],
            vis_backend: Optional[str],
            vis_strict: bool,
            vis_timeout: int,
            progress_tracker: Optional[HierarchicalProgressTracker],
            **kwargs
    ) -> List[Dict[str, Path]]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results of the analysis
        visualizations_dir : Path
            Directory to save visualizations
        operation_timestamp : Optional[str]
            Timestamp for file naming
        result : OperationResult
            The operation result to add artifacts to
        reporter : Any
            The reporter to log artifacts to
        vis_theme : str, optional
            Theme to use for visualizations
        vis_backend : str, optional
            Backend to use: "plotly" or "matplotlib"
        vis_strict : bool, optional
            If True, raise exceptions for configuration errors
        vis_timeout : int, optional
            Timeout for visualization generation (default: 120 seconds)
        progress_tracker : Optional[HierarchicalProgressTracker]
            Optional progress tracker

        Returns:
        --------
        List[Dict[str, Path]]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        self.logger.info(f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s")

        try:
            import threading
            import contextvars

            visualization_paths = []
            visualization_error = None

            if operation_timestamp is None:
                operation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                self.logger.info(f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}")
                self.logger.info(f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}")

                start_time = time.time()

                try:
                    # Log context variables
                    self.logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        self.logger.info(f"[DIAG] Context vars count: {len(list(current_context))}")
                    except Exception as ctx_e:
                        self.logger.warning(f"[DIAG] Could not inspect context: {ctx_e}")

                    # Generate visualizations with visualization context parameters
                    self.logger.info(f"[DIAG] Calling _generate_visualizations...")
                    # Create child progress tracker for visualization if available
                    total_steps = 3  # prepare data, create viz, save
                    viz_progress = None
                    if progress_tracker and hasattr(progress_tracker, "create_subtask"):
                        try:
                            viz_progress = progress_tracker.create_subtask(
                                total=total_steps,
                                description="Generating visualizations",
                                unit="steps",
                            )
                        except Exception as e:
                            self.logger.debug(f"Could not create child progress tracker: {e}")

                    # Generate visualizations

                    # 1. Country code distribution visualization
                    if 'country_codes' in analysis_results and analysis_results['country_codes']:
                        # Create visualization filename with extension "png"
                        viz_filename = f"{self.field_name}_country_codes_distribution{operation_timestamp}.png"
                        viz_path = visualizations_dir / viz_filename

                        # Create visualization using the visualization module
                        from pamola_core.utils.visualization import plot_value_distribution
                        title = f"Country Code Distribution in {self.field_name}"

                        viz_result = plot_value_distribution(
                            data=analysis_results['country_codes'],
                            output_path=str(viz_path),
                            title=title,
                            max_items=15,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            **kwargs
                        )

                        if not viz_result.startswith("Error"):
                            visualization_paths.append({
                                "artifact_type": "png",
                                "path": str(viz_path),
                                "description": f"{self.field_name} country codes distribution",
                                "category": Constants.Artifact_Category_Visualization
                            })
                        else:
                            self.logger.warning(f"Error creating country code visualization: {viz_result}")

                    # 2. Operator code distribution visualization
                    if 'operator_codes' in analysis_results and analysis_results['operator_codes']:
                        # Create visualization filename with extension "png"
                        viz_filename = f"{self.field_name}_operator_codes_distribution_{operation_timestamp}.png"
                        viz_path = visualizations_dir / viz_filename

                        # Create visualization using the visualization module
                        from pamola_core.utils.visualization import plot_value_distribution
                        title = f"Operator Code Distribution in {self.field_name}"

                        viz_result = plot_value_distribution(
                            data=analysis_results['operator_codes'],
                            output_path=str(viz_path),
                            title=title,
                            max_items=15,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            **kwargs
                        )

                        if not viz_result.startswith("Error"):
                            visualization_paths.append({
                                "artifact_type": "png",
                                "path": str(viz_path),
                                "description": f"{self.field_name} operator codes distribution",
                                "category": Constants.Artifact_Category_Visualization
                            })
                        else:
                            self.logger.warning(f"Error creating operator code visualization: {viz_result}")

                    # 3. Messenger mentions visualization
                    if 'messenger_mentions' in analysis_results and any(
                            analysis_results['messenger_mentions'].values()):
                        # Create visualization filename with extension "png"
                        viz_filename = f"{self.field_name}_messenger_mentions_{operation_timestamp}.png"
                        viz_path = visualizations_dir / viz_filename

                        # Create visualization using the visualization module
                        from pamola_core.utils.visualization import plot_value_distribution
                        title = f"Messenger Mentions in {self.field_name}"

                        viz_result = plot_value_distribution(
                            data=analysis_results['messenger_mentions'],
                            output_path=str(viz_path),
                            title=title,
                            max_items=10,
                            theme=vis_theme,
                            backend=vis_backend,
                            strict=vis_strict,
                            **kwargs
                        )

                        if not viz_result.startswith("Error"):
                            visualization_paths.append({
                                "artifact_type": "png",
                                "path": str(viz_path),
                                "description": f"{self.field_name} messenger mentions",
                                "category": Constants.Artifact_Category_Visualization
                            })
                        else:
                            self.logger.warning(f"Error creating messenger mentions visualization: {viz_result}")

                    # Close visualization progress tracker
                    if viz_progress:
                        try:
                            viz_progress.close()
                        except:
                            pass

                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"[DIAG] Visualization completed in {elapsed:.2f}s, generated {len(visualization_paths)} files"
                    )
                except Exception as e:
                    elapsed = time.time() - start_time
                    visualization_error = e
                    self.logger.error(f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}")
                    self.logger.error(f"[DIAG] Stack trace:", exc_info=True)

            # Copy context for the thread
            self.logger.info(f"[DIAG] Preparing to launch visualization thread...")
            ctx = contextvars.copy_context()

            # Create thread with context
            viz_thread = threading.Thread(
                target=ctx.run,
                args=(generate_viz_with_diagnostics,),
                name=f"VizThread-",
                daemon=False,  # Changed from True to ensure proper cleanup
            )

            self.logger.info(f"[DIAG] Starting visualization thread with timeout={vis_timeout}s")
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    self.logger.info(f"[DIAG] Visualization thread still running after {elapsed:.1f}s...")

            if viz_thread.is_alive():
                self.logger.error(f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout")
                self.logger.error(f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}")
                visualization_paths = []
            elif visualization_error:
                self.logger.error(f"[DIAG] Visualization failed with error: {visualization_error}")
                visualization_paths = []
            else:
                total_time = time.time() - thread_start_time
                self.logger.info(f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s")
                self.logger.info(
                    f"[DIAG] Generated visualizations: {[viz['path'] for viz in visualization_paths]}"
                )
        except Exception as e:
            self.logger.error(f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}")
            self.logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = []

        # Register visualization artifacts
        for viz in visualization_paths:
            artifact_type = viz["artifact_type"]
            path = viz["path"]
            description = viz["description"]

            # Add to result
            result.add_artifact(
                artifact_type=artifact_type,
                path=path,
                description=description,
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type=artifact_type, path=path, description=description
                )

        return visualization_paths

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare required directories for artifacts.

        Parameters:
        -----------
        task_dir : Path
            Base directory for the task

        Returns:
        --------
        Dict[str, Path]
            Dictionary of directory paths
        """
        # Create required directories
        output_dir = task_dir / 'output'
        visualizations_dir = task_dir / 'visualizations'
        dictionaries_dir = task_dir / 'dictionaries'
        cache_dir = task_dir / 'cache'

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)
        ensure_directory(dictionaries_dir)
        ensure_directory(cache_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir,
            'dictionaries': dictionaries_dir,
            'cache': cache_dir
        }
