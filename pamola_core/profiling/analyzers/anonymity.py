"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        K-Anonymity Profiling Operation
Package:       pamola.pamola_core.profiling.analyzers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides K-anonymity profiling operations for privacy risk analysis.
  It evaluates quasi-identifier combinations to assess re-identification risks,
  generates comprehensive metrics, and optionally creates visualizations and reports.

Key Features:
  - K-anonymity metric calculation for single and multiple QI combinations
  - Optional DataFrame enrichment with k-values
  - Configurable visualization generation
  - Support for large datasets via chunked processing
  - Integration with PAMOLA visualization system
  - Comprehensive vulnerability analysis
  - Thread-safe operation execution

Design Principles:
  - Follows PAMOLA Operations Framework patterns
  - Leverages existing field utilities
  - Modular visualization generation
  - Efficient memory usage for large datasets
  - Clear separation of analysis and reporting

Changelog:
  2.0.0 - Major refactoring for improved modularity and performance
        - Added ENRICH mode for DataFrame extension
        - Made visualizations and metrics export optional
        - Integrated with op_field_utils for consistent naming
        - Added chunked processing support
        - Simplified API for common use cases
  1.0.0 - Initial implementation with visualization support

TODO:
  - Add approximate k-anonymity for very large datasets
  - Support for streaming data processing
  - Add l-diversity and t-closeness metrics
  - Implement parallel processing for multiple QI combinations
  - Add incremental analysis capabilities
"""

from collections import Counter
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from itertools import combinations

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from pamola_core.common.constants import Constants
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import (
    OperationArtifact,
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.ops.op_field_utils import (
    generate_ka_field_name,
    generate_output_field_name,
    get_field_statistics,
)
from pamola_core.utils.ops.op_data_processing import get_dataframe_chunks
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.io import (
    ensure_directory,
    load_data_operation,
    load_settings_operation,
    write_json,
    get_timestamped_filename,
    write_dataframe_to_csv,
)
from pamola_core.utils.visualization import create_bar_plot, create_spider_chart
from pamola_core.utils.vis_helpers.context import visualization_context
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

# Configure logger
logger = logging.getLogger(__name__)


class AnalysisMode(Enum):
    """K-anonymity analysis modes."""

    ANALYZE = "ANALYZE"  # Generate metrics and reports
    ENRICH = "ENRICH"  # Add k-values to DataFrame
    BOTH = "BOTH"  # Both analyze and enrich


class KAnonymityProfilerOperation(BaseOperation):
    """
    Operation for k-anonymity profiling with optional DataFrame enrichment.

    This operation can work in three modes:
    1. ANALYZE: Generate metrics, reports, and visualizations
    2. ENRICH: Add k-anonymity values as a new column
    3. BOTH: Perform both analysis and enrichment
    """

    def __init__(
        self,
        name: str = "KAnonymityProfiler",
        description: str = "K-anonymity profiling and risk assessment",
        quasi_identifiers: List[str] = None,
        mode: str = "ANALYZE",
        threshold_k: int = 5,
        generate_visualizations: bool = True,
        export_metrics: bool = True,
        visualization_theme: str = "professional",
        max_combinations: int = 50,
        chunk_size: int = 100000,
        output_field_suffix: str = "k_anon",
        quasi_identifier_sets=List[List[str]],
        id_fields=List[str],
        use_dask: bool = False,
        use_cache: bool = True,
        use_vectorization: bool = False,
        parallel_processes: int = 1,
        npartitions: int = 1,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None,
        encryption_mode: Optional[str] = None,
    ):
        """
        Initialize the k-anonymity profiler.

        Parameters:
        -----------
        name : str
            Operation name
        description : str
            Operation description
        quasi_identifiers : List[str]
            List of quasi-identifier fields to analyze
        mode : str
            Analysis mode: "ANALYZE", "ENRICH", or "BOTH"
        threshold_k : int
            Threshold for considering records vulnerable (k < threshold)
        generate_visualizations : bool
            Whether to generate visualization artifacts
        export_metrics : bool
            Whether to export metrics to JSON/CSV files
        visualization_theme : str
            Theme for visualizations
        max_combinations : int
            Maximum number of QI combinations to analyze
        chunk_size : int
            Size of chunks for processing large datasets
        output_field_suffix : str
            Suffix for the k-anonymity field in ENRICH mode
        """
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode,
        )

        self.quasi_identifiers = quasi_identifiers or []
        self.analysis_mode = AnalysisMode(mode)
        self.threshold_k = threshold_k
        self.generate_visualizations = generate_visualizations
        self.export_metrics = export_metrics
        self.visualization_theme = visualization_theme
        self.max_combinations = max_combinations
        self.chunk_size = chunk_size
        self.output_field_suffix = output_field_suffix
        self.mode = mode
        self.use_dask = use_dask
        self.use_cache = use_cache
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        self.npartitions = npartitions
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout
        self.quasi_identifier_sets = quasi_identifier_sets
        self.id_fields = id_fields

    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
        """
        Execute k-anonymity profiling operation.

        Parameters:
        -----------
        data_source : DataSource
            Input data source
        task_dir : Path
            Directory for operation artifacts
        reporter : Any
            Reporter for operation tracking
        progress_tracker : HierarchicalProgressTracker
            Optional progress tracker
        **kwargs : dict
            Additional parameters:
            - quasi_identifier_sets: List of QI combinations to analyze
            - id_fields: Fields to identify vulnerable records
            - include_timestamp: Whether to timestamp output files

        Returns:
        --------
        OperationResult
            Operation result with status and artifacts
        """
        # Save configuration
        self.save_config(task_dir)

        # Extract additional parameters
        qi_sets = kwargs.get("quasi_identifier_sets", self.quasi_identifier_sets)
        id_fields = kwargs.get("id_fields", self.id_fields)
        include_timestamp = kwargs.get("include_timestamp", True)
        force_recalculation = kwargs.get("force_recalculation", False)
        dataset_name = kwargs.get("dataset_name", "main")

        self.visualization_theme = kwargs.get(
            "visualization_theme", self.visualization_theme
        )
        self.visualization_backend = kwargs.get(
            "visualization_backend", self.visualization_backend
        )
        self.visualization_strict = kwargs.get(
            "visualization_strict", self.visualization_strict
        )
        self.visualization_timeout = kwargs.get(
            "visualization_timeout", self.visualization_timeout
        )

        # Initialize result
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Set up progress tracking
        # Initializing, Prepared QI combinations, Completed
        total_steps = (
            4
            + (1 if self.use_cache and not force_recalculation else 0)
            + (1 if self.analysis_mode == AnalysisMode.ANALYZE else 0)
        )
        current_steps = 0

        # Set up progress tracking
        # Step 1: Initializing
        if progress_tracker:
            progress_tracker.total = total_steps
            current_steps += 1
            progress_tracker.update(
                current_steps,
                {
                    "step": "Initializing",
                    "operation": self.name,
                    "total_steps": total_steps,
                },
            )

        # Step 2: Check Cache (if enabled and not forced to recalculate)
        if self.use_cache and not force_recalculation:
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Checking Cache"})

            self.logger.info("Checking operation cache...")
            cache_result = self._check_cache(data_source, dataset_name, **kwargs)

            if cache_result:
                self.logger.info("Cache hit! Using cached results.")

                # Update progress
                if progress_tracker:
                    progress_tracker.update(total_steps, {"step": "Complete (cached)"})

                # Report cache hit to reporter
                if reporter:
                    reporter.add_operation(
                        f"Clean invalid values (from cache)", details={"cached": True}
                    )
                return cache_result

        # Step 3: Data Loading
        if progress_tracker:
            current_steps += 1
            progress_tracker.update(current_steps, {"step": "Data Loading"})

        try:
            # Get DataFrame
            settings_operation = load_settings_operation(
                data_source, dataset_name, **kwargs
            )
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            if df is None:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No valid DataFrame found in data source",
                )

            # Check if field exists
            missing_fields = [field for field in id_fields if field not in df.columns]
            if missing_fields:
                raise ValueError(
                    f"The following id_fields are not in the DataFrame columns: {missing_fields}"
                )

            # Prepare QI combinations
            qi_combinations = self._prepare_qi_combinations(df, qi_sets, id_fields)
            if not qi_combinations:
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message="No quasi-identifiers specified or detected",
                )

            self.logger.info(
                f"Analyzing {len(qi_combinations)} quasi-identifier combinations"
            )

            # Update progress
            # Step 4: Prepared QI combinations
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(
                    current_steps,
                    {
                        "step": "Prepared QI combinations",
                        "combinations": len(qi_combinations),
                    },
                )

            kwargs_encryption = {
                "use_encryption": kwargs.get("use_encryption", False),
                "encryption_key": kwargs.get("encryption_key", None),
                "encryption_mode": kwargs.get("encryption_mode", "none"),
            }

            # Perform analysis based on mode
            if self.analysis_mode in [AnalysisMode.ANALYZE, AnalysisMode.BOTH]:
                analysis_results = self._perform_analysis(
                    df,
                    qi_combinations,
                    task_dir,
                    id_fields,
                    include_timestamp,
                    result,
                    reporter,
                    progress_tracker,
                    **kwargs_encryption,
                )

                # Add metrics to result
                for qi_name, metrics in analysis_results.items():
                    result.add_nested_metric(
                        "k_anonymity",
                        qi_name,
                        {
                            "min_k": metrics.get("min_k", 0),
                            "mean_k": metrics.get("mean_k", 0),
                            "vulnerable_percent": metrics.get("vulnerable_percent", 0),
                        },
                    )

            # Perform enrichment if requested
            if self.analysis_mode in [AnalysisMode.ENRICH, AnalysisMode.BOTH]:
                enriched_df = self._perform_enrichment(
                    df,
                    qi_combinations[0],
                    task_dir,
                    include_timestamp,
                    result,
                    reporter,
                    progress_tracker,
                    **kwargs_encryption,
                )

                # Update data source with enriched DataFrame
                if enriched_df is not None:
                    if hasattr(data_source, "dataframes"):
                        data_source.dataframes["main"] = enriched_df
                    else:
                        self.logger.warning(
                            f"Cannot update data source: expected DataSource object but got {type(data_source)}"
                        )

            # Final progress update
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(
                    current_steps, {"step": "Completed", "status": "success"}
                )

            # Add operation summary to reporter
            reporter.add_operation(
                "K-Anonymity Profiling Completed",
                details={
                    "mode": self.analysis_mode.value,
                    "combinations_analyzed": len(qi_combinations),
                    "threshold_k": self.threshold_k,
                },
            )

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        artifacts=result.artifacts,
                        original_df=df,
                        metrics=result.metrics,
                        task_dir=task_dir,
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            return result

        except Exception as e:
            self.logger.exception(f"Error in k-anonymity profiling: {e}")

            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            reporter.add_operation(
                "K-Anonymity Profiling", status="error", details={"error": str(e)}
            )

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"K-anonymity profiling failed: {str(e)}",
                exception=e,
            )

    def _prepare_qi_combinations(
        self, df: pd.DataFrame, qi_sets: Optional[List[List[str]]], id_fields: List[str]
    ) -> List[List[str]]:
        """
        Prepare quasi-identifier combinations for analysis.
        """
        if qi_sets:
            # Use provided combinations
            return qi_sets[: self.max_combinations]
        elif self.quasi_identifiers:
            # Use single combination from constructor
            return [self.quasi_identifiers]
        else:
            # Auto-detect potential quasi-identifiers
            return self._detect_quasi_identifiers(df, id_fields)

    def _detect_quasi_identifiers(
        self, df: pd.DataFrame, exclude_fields: List[str]
    ) -> List[List[str]]:
        """
        Auto-detect potential quasi-identifier combinations.
        """
        # Get all columns except excluded
        potential_qis = [col for col in df.columns if col not in exclude_fields]

        # Filter to categorical and low-cardinality numeric fields
        selected_qis = []
        for col in potential_qis:
            stats = get_field_statistics(df[col])

            # Include if categorical or numeric with reasonable cardinality
            if stats["dtype"] in ["object", "category"] or (
                stats["dtype"] in ["int64", "float64"] and stats["unique_count"] < 100
            ):
                selected_qis.append(col)

        # Generate combinations (limit to avoid explosion)
        qi_combinations = []
        for size in range(2, min(4, len(selected_qis) + 1)):
            for combo in combinations(selected_qis, size):
                qi_combinations.append(list(combo))
                if len(qi_combinations) >= self.max_combinations:
                    break
            if len(qi_combinations) >= self.max_combinations:
                break

        return qi_combinations

    def _perform_analysis(
        self,
        df: pd.DataFrame,
        qi_combinations: List[List[str]],
        task_dir: Path,
        id_fields: List[str],
        include_timestamp: bool,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform k-anonymity analysis for multiple QI combinations.
        """
        # Prepare directories
        dirs = self._prepare_directories(task_dir)

        # Analyze each combination
        all_metrics = {}
        vulnerable_records = {}

        for qi_fields in qi_combinations:
            # Generate name for this combination
            qi_name = generate_ka_field_name(qi_fields)

            self.logger.info(
                f"Analyzing combination: {qi_name} ({', '.join(qi_fields)})"
            )

            # Calculate k-anonymity metrics
            metrics = self._calculate_k_anonymity(
                df,
                qi_fields,
                self.use_dask,
                self.use_vectorization,
                self.chunk_size,
                self.parallel_processes,
                self.npartitions,
            )

            # Store metrics
            all_metrics[qi_name] = metrics

            # Find vulnerable records if ID fields provided
            if id_fields:
                vuln_records = self._find_vulnerable_records(
                    df, qi_fields, self.threshold_k, id_fields[0]
                )
                vulnerable_records[qi_name] = vuln_records

        # Export metrics if requested
        if self.export_metrics:
            self._export_metrics(
                all_metrics,
                vulnerable_records,
                dirs["output"],
                include_timestamp,
                result,
                reporter,
                **kwargs,
            )

        # Generate visualizations if requested
        if self.generate_visualizations and all_metrics:
            self._handle_visualizations(
                all_metrics=all_metrics,
                include_timestamp=include_timestamp,
                progress_tracker=progress_tracker,
                reporter=reporter,
                result=result,
                vis_dir=dirs["visualizations"],
                vis_backend=self.visualization_backend,
                vis_strict=self.visualization_strict,
                vis_theme=self.visualization_theme,
                vis_timeout=self.visualization_timeout,
                **kwargs,
            )

        # Update progress
        if progress_tracker:
            progress_tracker.update(1, {"step": "Analysis completed"})

        return all_metrics

    def _perform_enrichment(
        self,
        df: pd.DataFrame,
        qi_fields: List[str],
        task_dir: Path,
        include_timestamp: bool,
        result: OperationResult,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker],
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Enrich DataFrame with k-anonymity values.
        """
        try:
            # Generate output field name using standard naming convention
            output_field = generate_output_field_name(
                field_name=generate_ka_field_name(qi_fields),
                mode="ENRICH",
                operation_suffix=self.output_field_suffix,
            )

            self.logger.info(
                f"Enriching DataFrame with k-values in column: {output_field}"
            )

            # Calculate k-values for each record
            k_values = self._calculate_k_values(df, qi_fields)

            # Add to DataFrame
            enriched_df = df.copy()
            enriched_df[output_field] = k_values

            # Save enriched data
            output_dir = task_dir / "output"
            ensure_directory(output_dir)

            output_filename = get_timestamped_filename(
                "enriched_data", "csv", include_timestamp
            )
            output_path = output_dir / output_filename

            encryption_mode_enriched_df = get_encryption_mode(enriched_df, **kwargs)
            write_dataframe_to_csv(
                df=enriched_df,
                file_path=output_path,
                encryption_key=kwargs.get("encryption_key", None),
                use_encryption=kwargs.get("use_encryption", False),
                encryption_mode=encryption_mode_enriched_df,
            )

            # Register artifact
            result.add_artifact(
                "csv",
                output_path,
                f"Data enriched with {output_field}",
                category="output",
            )
            reporter.add_artifact("csv", str(output_path), "Enriched data")

            # Add enrichment metrics
            result.add_metric("enrichment_field", output_field)
            result.add_metric("enrichment_min_k", int(k_values.min()))
            result.add_metric("enrichment_max_k", int(k_values.max()))

            # Update progress
            if progress_tracker:
                progress_tracker.update(1, {"step": "Enrichment completed"})

            return enriched_df

        except Exception as e:
            self.logger.error(f"Error in enrichment: {e}")
            return None

    def _calculate_k_anonymity(
        self,
        df: pd.DataFrame,
        fields: List[str],
        use_dask: bool = False,
        use_vectorization: bool = False,
        chunk_size: int = 10000,
        parallel_processes: Optional[int] = 1,
        npartitions: Optional[int] = 1,
    ) -> Dict[str, Any]:
        """
        Calculate k-anonymity metrics for a field combination.
        """
        # --- 1. Calculate group sizes ---
        k_values = self._calculate_group_sizes(
            df,
            fields,
            use_dask,
            use_vectorization,
            chunk_size,
            parallel_processes,
            npartitions,
        )

        # Remove any zero or NaN values (if any) - using pure numpy
        k_values = k_values[k_values > 0]  # Remove zeros
        k_values = k_values[~np.isnan(k_values)]  # Remove NaN values

        total_records = len(df)
        if len(k_values) == 0:
            return {
                "min_k": 0,
                "max_k": 0,
                "mean_k": 0,
                "median_k": 0,
                "unique_groups": 0,
                "unique_percent": 0,
                "vulnerable_count": total_records,
                "vulnerable_percent": 100,
                "k_distribution": {},
                "threshold_compliance": 0,
                "entropy": 0,
                "normalized_entropy": 0,
            }

        # --- 2. Basic statistics ---
        unique_groups = len(k_values)
        vulnerable_mask = k_values < self.threshold_k
        # Sum the k-values, not count of groups
        vulnerable_count = int(k_values[vulnerable_mask].sum())

        metrics = {
            "min_k": int(k_values.min()),
            "max_k": int(k_values.max()),
            "mean_k": float(k_values.mean()),
            "median_k": float(np.median(k_values)),
            "unique_groups": unique_groups,
            "unique_percent": (unique_groups / total_records * 100),
            "vulnerable_count": vulnerable_count,
            "vulnerable_percent": (vulnerable_count / total_records * 100),
            "k_distribution": self._calculate_k_distribution(k_values),
            "threshold_compliance": self._calculate_threshold_compliance(
                k_values, total_records
            ),
        }

        # --- 3. Entropy calculation (with epsilon) ---
        probabilities = k_values / total_records
        safe_probs = probabilities[probabilities > 0]

        # Add epsilon inside log to prevent log(0), and ensure non-negative result
        epsilon = 1e-10
        entropy = -np.sum(safe_probs * np.log2(safe_probs + epsilon))
        entropy = max(entropy, 0.0)  # Ensure non-negative

        normalized_entropy = (
            entropy / np.log2(unique_groups) if unique_groups > 1 else 0
        )

        metrics["entropy"] = float(entropy)
        metrics["normalized_entropy"] = float(
            max(normalized_entropy, 0.0)
        )  # Ensure non-negative

        return metrics

    def _calculate_group_sizes(
        self,
        df: pd.DataFrame,
        fields: List[str],
        use_dask: bool = False,
        use_vectorization: bool = False,
        chunk_size: int = 10000,
        parallel_processes: Optional[int] = 1,
        npartitions: Optional[int] = 1,
    ) -> np.ndarray:
        """
        Calculate k-values representing group sizes for k-anonymity analysis
        based on specified combinations of fields.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing the data to analyze.
        fields : List[str]
            List of column names to group by (quasi-identifiers).
        use_dask : bool, default=False
            If True, use Dask for scalable parallel processing.
        use_vectorization : bool, default=False
            If True, use Joblib for parallel processing across chunks.
        chunk_size : int, default=10000
            Number of rows per chunk when splitting the DataFrame.
        parallel_processes : int, optional
            Number of parallel processes to use with Joblib.
        npartitions : int, optional
            Number of Dask partitions (used only if use_dask is True).

        Returns:
        --------
        np.ndarray
            An array of group sizes (frequencies) representing the k-values.
        """
        total_rows = len(df)
        is_large_df = total_rows > chunk_size

        # Step 1: Normalize columns used for grouping
        df = df.copy()
        for field in fields:
            if field in df.columns:  # Safety check
                df[field] = df[field].astype("object")

        group_sizes = None

        # Step 2: Dask-based group size calculation
        if use_dask:
            self.logger.info("Parallel Enabled")
            self.logger.info("Parallel Engine: Dask")
            self.logger.info(f"Parallel Workers: {npartitions}")

            import dask.dataframe as dd

            # Convert the DataFrame to a Dask DataFrame with the specified number of partitions
            ddf = dd.from_pandas(df, npartitions=npartitions)

            # Function to group by columns and count sizes within a partition
            def _group_sizes(partition, group_cols):
                return partition.groupby(group_cols, dropna=False).size()

            # Apply groupby-size on each partition
            partial = ddf.map_partitions(
                _group_sizes, fields, meta=pd.Series(dtype="int64")
            )

            # Aggregate the group sizes across all partitions
            group_sizes = partial.groupby(partial.index).sum().compute()

        # Step 3: Joblib-based parallelism
        elif use_vectorization and parallel_processes > 1:
            self.logger.info("Parallel Enabled")
            self.logger.info("Parallel Engine: Joblib")
            self.logger.info(f"Parallel Workers: {parallel_processes}")

            # Split the DataFrame into chunks
            chunks = list(get_dataframe_chunks(df, chunk_size=chunk_size))

            # Function to process each chunk: group by and count
            def process_chunk(chunk):
                return chunk.groupby(fields, dropna=False).size().to_dict()

            # Run the processing function in parallel across chunks
            partial_counts = Parallel(n_jobs=parallel_processes)(
                delayed(process_chunk)(chunk) for chunk in chunks
            )

            # Merge counts from all chunks using a Counter
            group_counter = Counter()
            for chunk_counts in partial_counts:
                group_counter.update(chunk_counts)

            group_sizes = pd.Series(list(group_counter.values()), dtype="int64")

        # Step 4: Manual chunked fallback
        elif is_large_df:
            # Manual chunked processing without parallelism
            group_counter = {}

            for chunk in get_dataframe_chunks(df, chunk_size=chunk_size):
                chunk_counts = chunk.groupby(fields, dropna=False).size()
                for group, count in chunk_counts.items():
                    group_counter[group] = group_counter.get(group, 0) + count

            # Convert to NumPy array
            group_sizes = pd.Series(list(group_counter.values()), dtype="int64")

        # Step 5: Default single groupby (for small data)
        else:
            # Small dataset: process directly without chunking or parallelism
            group_sizes = df.groupby(fields, dropna=False).size()

        # Step 6: Sanity checks
        if group_sizes is None or len(group_sizes) == 0:
            raise ValueError(
                "Group size calculation failed: No valid equivalence classes found."
            )

        total_sum = int(group_sizes.sum())
        if total_sum != total_rows:
            self.logger.warning(
                f"[WARN] Sum of group sizes ({total_sum}) != total records ({total_rows})."
            )

        # Remove zero/negative group sizes — these are invalid for k-anonymity
        group_sizes = group_sizes[group_sizes > 0]

        return group_sizes.values  # k-values

    def _calculate_k_anonymity_chunked(
        self, df: pd.DataFrame, fields: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate k-anonymity metrics using chunked processing.
        """
        # Aggregate counts across chunks
        group_counts = {}

        for chunk in get_dataframe_chunks(df, self.chunk_size):
            chunk_counts = chunk.groupby(fields, dropna=False).size()
            for group, count in chunk_counts.items():
                group_counts[group] = group_counts.get(group, 0) + count

        # Convert to array of k-values
        k_values = np.array(list(group_counts.values()))

        # Calculate metrics (same as non-chunked version)
        total_records = len(df)
        unique_groups = len(k_values)

        metrics = {
            "min_k": int(k_values.min()) if len(k_values) > 0 else 0,
            "max_k": int(k_values.max()) if len(k_values) > 0 else 0,
            "mean_k": float(k_values.mean()) if len(k_values) > 0 else 0,
            "median_k": float(np.median(k_values)) if len(k_values) > 0 else 0,
            "unique_groups": unique_groups,
            "unique_percent": (
                (unique_groups / total_records * 100) if total_records > 0 else 0
            ),
            "vulnerable_count": int(np.sum(k_values < self.threshold_k)),
            "vulnerable_percent": (
                float(np.sum(k_values < self.threshold_k) / total_records * 100)
                if total_records > 0
                else 0
            ),
            "k_distribution": self._calculate_k_distribution(k_values),
            "threshold_compliance": self._calculate_threshold_compliance(
                k_values, total_records
            ),
        }

        # Calculate entropy
        probabilities = k_values / total_records
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        metrics["entropy"] = float(entropy)
        metrics["normalized_entropy"] = (
            float(entropy / np.log2(unique_groups)) if unique_groups > 1 else 0
        )

        return metrics

    def _calculate_k_values(self, df: pd.DataFrame, fields: List[str]) -> pd.Series:
        """
        Calculate k-value for each record.
        """
        if len(df) > self.chunk_size:
            return self._calculate_k_values_chunked(df, fields)

        # Group and calculate sizes
        return df.groupby(fields, dropna=False)[fields[0]].transform("size")

    def _calculate_k_values_chunked(
        self, df: pd.DataFrame, fields: List[str]
    ) -> pd.Series:
        """
        Calculate k-values using chunked processing.
        """
        # First pass: count groups
        group_counts = {}

        for chunk in get_dataframe_chunks(df, self.chunk_size):
            chunk_counts = chunk.groupby(fields, dropna=False).size()
            for group, count in chunk_counts.items():
                group_counts[group] = group_counts.get(group, 0) + count

        # Second pass: assign k-values
        k_values = pd.Series(index=df.index, dtype="int64")

        for i, chunk in enumerate(get_dataframe_chunks(df, self.chunk_size)):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(df))

            # Get group keys for this chunk
            chunk_groups = chunk[fields].apply(tuple, axis=1)

            # Map to k-values
            chunk_k = chunk_groups.map(group_counts)
            k_values.iloc[start_idx:end_idx] = chunk_k

        return k_values

    def _calculate_k_distribution(self, k_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate distribution of k-values across ranges.
        """
        total = len(k_values)
        if total == 0:
            return {}

        ranges = {
            "k=1": np.sum(k_values == 1),
            "k=2-4": np.sum((k_values >= 2) & (k_values <= 4)),
            "k=5-9": np.sum((k_values >= 5) & (k_values <= 9)),
            "k=10-19": np.sum((k_values >= 10) & (k_values <= 19)),
            "k=20-49": np.sum((k_values >= 20) & (k_values <= 49)),
            "k=50-99": np.sum((k_values >= 50) & (k_values <= 99)),
            "k≥100": np.sum(k_values >= 100),
        }

        # Convert to percentages
        return {k: float(v / total * 100) for k, v in ranges.items()}

    def _calculate_threshold_compliance(
        self, k_values: np.ndarray, total_records: int
    ) -> Dict[str, float]:
        """
        Calculate percentage of records meeting various k-thresholds.
        """
        if total_records == 0:
            return {}

        thresholds = [2, 5, 10, 20, 50, 100]
        compliance = {}

        for t in thresholds:
            compliant = np.sum(k_values >= t)
            compliance[f"k≥{t}"] = float(compliant / total_records * 100)

        return compliance

    def _find_vulnerable_records(
        self, df: pd.DataFrame, fields: List[str], threshold: int, id_field: str
    ) -> Dict[str, Any]:
        """
        Identify records with k < threshold.
        """

        if isinstance(fields, str):
            fields = [fields]

        if df.empty or id_field not in df.columns:
            return {
                "total_vulnerable": 0,
                "vulnerable_percent": 0,
                "sample_ids": [],
                "vulnerable_groups": 0,
            }

        # Group by quasi-identifiers
        group_sizes = (
            df.groupby(fields, dropna=False)
            .size()
            .reset_index(name="k")
            .query("k > 0")  # ensure positive groups
        )

        # Identify vulnerable groups
        vulnerable_groups = group_sizes[group_sizes["k"] < threshold]

        if vulnerable_groups.empty:
            return {
                "total_vulnerable": 0,
                "vulnerable_percent": 0,
                "sample_ids": [],
                "vulnerable_groups": 0,
            }

        # Calculate total vulnerable records by summing k-values of vulnerable groups
        total_vulnerable_records = int(vulnerable_groups["k"].sum())

        # Join back to get vulnerable record IDs for sampling
        join_fields = fields.copy()
        df_subset = df[[id_field] + join_fields]
        vulnerable_df = pd.merge(
            df_subset, vulnerable_groups[join_fields], on=join_fields, how="inner"
        )

        return {
            "total_vulnerable": total_vulnerable_records,  # Use sum of k-values
            "vulnerable_percent": (
                (total_vulnerable_records / len(df) * 100) if len(df) > 0 else 0
            ),
            "sample_ids": vulnerable_df[id_field].drop_duplicates().head(100).tolist(),
            "vulnerable_groups": len(vulnerable_groups),
        }

    def _export_metrics(
        self,
        all_metrics: Dict[str, Dict[str, Any]],
        vulnerable_records: Dict[str, Dict[str, Any]],
        output_dir: Path,
        include_timestamp: bool,
        result: OperationResult,
        reporter: Any,
        **kwargs,
    ):
        """Export metrics to JSON files."""
        # Export summary metrics
        summary_filename = get_timestamped_filename(
            "k_anonymity_summary", "json", include_timestamp
        )
        summary_path = output_dir / summary_filename

        summary_data = {
            qi_name: {
                "fields": qi_name,
                "min_k": metrics["min_k"],
                "mean_k": metrics["mean_k"],
                "vulnerable_percent": metrics["vulnerable_percent"],
                "entropy": metrics["entropy"],
            }
            for qi_name, metrics in all_metrics.items()
        }

        encryption_mode_summary_data = get_encryption_mode(summary_data, **kwargs)
        write_json(
            summary_data,
            str(summary_path),
            encryption_key=kwargs.get("encryption_key", None),
            encryption_mode=encryption_mode_summary_data,
        )
        result.add_artifact(
            "json", summary_path, "K-anonymity summary", category="metrics"
        )
        reporter.add_artifact("json", str(summary_path), "K-anonymity summary")

        # Export detailed metrics if needed
        if vulnerable_records:
            vuln_filename = get_timestamped_filename(
                "vulnerable_records", "json", include_timestamp
            )
            vuln_path = output_dir / vuln_filename
            encryption_mode_records = get_encryption_mode(vulnerable_records, **kwargs)
            write_json(
                vulnerable_records,
                str(vuln_path),
                encryption_key=kwargs.get("encryption_key", None),
                encryption_mode=encryption_mode_records,
            )
            result.add_artifact(
                "json", vuln_path, "Vulnerable records", category="metrics"
            )
            reporter.add_artifact("json", str(vuln_path), "Vulnerable records")

    def _create_visualizations(
        self,
        all_metrics: Dict[str, Dict[str, Any]],
        vis_dir: Path,
        include_timestamp: bool,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        **kwargs,
    ) -> Dict[str, Path]:
        """Generate visualizations for k-anonymity metrics."""
        visualization_paths = {}
        try:
            # 1. Vulnerability comparison
            vuln_data = {
                qi: metrics["vulnerable_percent"] for qi, metrics in all_metrics.items()
            }

            if vuln_data:
                vuln_filename = get_timestamped_filename(
                    "vulnerability_comparison", "png", include_timestamp
                )
                vuln_path = vis_dir / vuln_filename

                result_vuln = create_bar_plot(
                    vuln_data,
                    str(vuln_path),
                    f"Vulnerability Comparison (k < {self.threshold_k})",
                    orientation="v",
                    y_label="Vulnerable Records (%)",
                    x_label="QI Combination",
                    sort_by="value",
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )

                if isinstance(result_vuln, str) and not result_vuln.startswith("Error"):
                    visualization_paths["vulnerability_comparison"] = result_vuln

            # 2. Multi-metric spider chart (if multiple QIs)
            if len(all_metrics) > 2:
                spider_data = {}
                for qi_name, metrics in all_metrics.items():
                    spider_data[qi_name] = {
                        "Min K": min(metrics["min_k"] / 10, 1),  # Normalize
                        "Mean K": min(metrics["mean_k"] / 50, 1),  # Normalize
                        "Vulnerability": metrics["vulnerable_percent"] / 100,
                        "Entropy": metrics["normalized_entropy"],
                        "Uniqueness": metrics["unique_percent"] / 100,
                    }

                spider_filename = get_timestamped_filename(
                    "k_anonymity_spider", "png", include_timestamp
                )
                spider_path = vis_dir / spider_filename

                result_k = create_spider_chart(
                    spider_data,
                    str(spider_path),
                    "K-Anonymity Multi-Metric Comparison",
                    normalize_values=False,  # Already normalized
                    theme=vis_theme,
                    backend=vis_backend,
                    strict=vis_strict,
                    **kwargs,
                )

                if isinstance(result_k, str) and not result_k.startswith("Error"):
                    visualization_paths["k_anonymity_spider"] = result_k

        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")
            # Don't fail the operation due to visualization errors

        return visualization_paths

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """Prepare required directories for artifacts."""
        dirs = {
            "output": task_dir / "output",
            "visualizations": task_dir / "visualizations",
            "dictionaries": task_dir / "dictionaries",
        }

        for dir_path in dirs.values():
            ensure_directory(dir_path)

        return dirs

    def _check_cache(
        self, data_source: DataSource, data_source_name: str = "main", **kwargs
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        task_dir : Path
            Task directory
        data_source_name: str
            Dataset name

        Returns:
        --------
        Optional[OperationResult]
            Cached result if found, None otherwise
        """
        if not self.use_cache:
            return None

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Get DataFrame from data source
            settings_operation = load_settings_operation(
                data_source, data_source_name, **kwargs
            )
            df = load_data_operation(
                data_source, data_source_name, **settings_operation
            )
            if df is None:
                self.logger.warning("No valid DataFrame found in data source")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key, operation_type=self.__class__.__name__
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        cached_result.add_metric(key, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        artifact_type = artifact.get("type", "")
                        artifact_path = artifact.get("path", "")
                        artifact_name = artifact.get("description", "")
                        artifact_category = artifact.get("category", "output")
                        cached_result.add_artifact(
                            artifact_type,
                            artifact_path,
                            artifact_name,
                            artifact_category,
                        )

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric(
                    "cache_timestamp", cached_data.get("timestamp", "unknown")
                )

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")

    def _save_to_cache(
        self,
        original_df: pd.DataFrame,
        artifacts: List[OperationArtifact],
        metrics: Dict[str, Any],
        task_dir: Path,
    ) -> bool:
        """
        Save operation results to cache.

        Parameters:
        -----------
        original_df : pd.DataFrame
            Original input data
        processed_df : pd.DataFrame
            Processed DataFrame
        metrics : dict
            The metrics of operation
        task_dir : Path
            Task directory

        Returns:
        --------
        bool
            True if successfully saved to cache, False otherwise
        """
        if not self.use_cache or (not artifacts and not metrics):
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            artifacts_for_cache = [artifact.to_dict() for artifact in artifacts]

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "artifacts": artifacts_for_cache,
                "metrics": metrics,
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)},
            )

            if success:
                self.logger.info(f"Successfully saved results to cache")
            else:
                self.logger.warning(f"Failed to save results to cache")

            return success
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
            return False

    def _generate_cache_key(self, df: pd.DataFrame) -> str:
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
            data_hash=data_hash,
        )

    def _get_operation_parameters(self) -> Dict[str, Any]:
        """
        Get operation parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Get basic operation parameters
        parameters = {
            "quasi_identifiers": self.quasi_identifiers,
            "mode": self.mode,
            "threshold_k": self.threshold_k,
            "max_combinations": self.max_combinations,
            "id_fields": self.id_fields,
            "threshold_k": self.threshold_k,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        return {}

    def _generate_data_hash(self, df: pd.DataFrame) -> str:
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
        import hashlib

        try:
            # Create data characteristics
            characteristics = df.describe(include="all")

            # Convert to JSON string and hash
            json_str = characteristics.to_json(date_format="iso")
        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")

            # Fallback to a simple hash of the data length and type
            json_str = f"{len(df)}_{json.dumps(df.dtypes.apply(str).to_dict())}"

        return hashlib.md5(json_str.encode()).hexdigest()

    def _handle_visualizations(
        self,
        all_metrics: Dict[str, Dict[str, Any]],
        vis_dir: Path,
        include_timestamp: bool,
        result: OperationResult,
        reporter: Any,
        vis_theme: Optional[str] = None,
        vis_backend: Optional[str] = None,
        vis_strict: bool = False,
        vis_timeout: int = 120,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> Dict[str, Path]:
        """
        Generate and save visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data
        analysis_results : Dict[str, Any]
            Results of the analysis
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in output filenames
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
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
        Dict[str, Path]
            Dictionary with visualization types and paths
        """
        if progress_tracker:
            progress_tracker.update(0, {"step": "Generating visualizations"})

        self.logger.info(
            f"Generating visualizations with backend: {vis_backend}, timeout: {vis_timeout}s"
        )

        try:
            import threading
            import contextvars

            visualization_paths = {}
            visualization_error = None

            def generate_viz_with_diagnostics():
                nonlocal visualization_paths, visualization_error
                thread_id = threading.current_thread().ident
                thread_name = threading.current_thread().name

                self.logger.info(
                    f"[DIAG] Visualization thread started - Thread ID: {thread_id}, Name: {thread_name}"
                )
                self.logger.info(
                    f"[DIAG] Backend: {vis_backend}, Theme: {vis_theme}, Strict: {vis_strict}"
                )

                start_time = time.time()

                try:
                    # Log context variables
                    self.logger.info(f"[DIAG] Checking context variables...")
                    try:
                        current_context = contextvars.copy_context()
                        self.logger.info(
                            f"[DIAG] Context vars count: {len(list(current_context))}"
                        )
                    except Exception as ctx_e:
                        self.logger.warning(
                            f"[DIAG] Could not inspect context: {ctx_e}"
                        )

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
                            self.logger.debug(
                                f"Could not create child progress tracker: {e}"
                            )

                    # Generate visualizations
                    visualization_paths = self._create_visualizations(
                        all_metrics,
                        vis_dir,
                        include_timestamp,
                        vis_theme,
                        vis_backend,
                        vis_strict,
                        **kwargs,
                    )

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
                    self.logger.error(
                        f"[DIAG] Visualization failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                    )
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

            self.logger.info(
                f"[DIAG] Starting visualization thread with timeout={vis_timeout}s"
            )
            thread_start_time = time.time()
            viz_thread.start()

            # Log periodic status while waiting
            check_interval = 5  # seconds
            elapsed = 0
            while viz_thread.is_alive() and elapsed < vis_timeout:
                viz_thread.join(timeout=check_interval)
                elapsed = time.time() - thread_start_time
                if viz_thread.is_alive():
                    self.logger.info(
                        f"[DIAG] Visualization thread still running after {elapsed:.1f}s..."
                    )

            if viz_thread.is_alive():
                self.logger.error(
                    f"[DIAG] Visualization thread still alive after {vis_timeout}s timeout"
                )
                self.logger.error(
                    f"[DIAG] Thread state: alive={viz_thread.is_alive()}, daemon={viz_thread.daemon}"
                )
                visualization_paths = {}
            elif visualization_error:
                self.logger.error(
                    f"[DIAG] Visualization failed with error: {visualization_error}"
                )
                visualization_paths = {}
            else:
                total_time = time.time() - thread_start_time
                self.logger.info(
                    f"[DIAG] Visualization thread completed successfully in {total_time:.2f}s"
                )
                self.logger.info(
                    f"[DIAG] Generated visualizations: {list(visualization_paths.keys())}"
                )
        except Exception as e:
            self.logger.error(
                f"[DIAG] Error in visualization thread setup: {type(e).__name__}: {e}"
            )
            self.logger.error(f"[DIAG] Stack trace:", exc_info=True)
            visualization_paths = {}

        # Register visualization artifacts
        for viz_type, path in visualization_paths.items():
            # Add to result
            result.add_artifact(
                artifact_type="png",
                path=path,
                description=f"{viz_type} visualization",
                category=Constants.Artifact_Category_Visualization,
            )

            # Report to reporter
            if reporter:
                reporter.add_artifact(
                    artifact_type="png",
                    path=str(path),
                    description=f"{viz_type} visualization",
                )

        return visualization_paths


# Register the operation
register_operation(KAnonymityProfilerOperation)

# Module metadata
__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main classes
__all__ = ["KAnonymityProfilerOperation", "AnalysisMode"]
