"""
K-anonymity profiling operation for the project.

This module provides operations for analyzing k-anonymity in data, identifying
quasi-identifiers that may compromise privacy, and generating visualizations
and reports about data anonymization risks.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd

from pamola_core.profiling.commons.anonymity_utils import (
    get_field_combinations,
    create_ka_index_map,
    calculate_k_anonymity,
    find_vulnerable_records,
    prepare_metrics_for_spider_chart,
    prepare_field_uniqueness_data,
    save_ka_index_map,
    save_ka_metrics,
    save_vulnerable_records
)
from pamola_core.utils.io import ensure_directory, write_json, get_timestamped_filename, load_data_operation, load_settings_operation
from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.visualization import (
    create_bar_plot,
    create_line_plot,
    create_spider_chart,
    create_combined_chart
)
from pamola_core.common.constants import Constants

# Configure logger
logger = logging.getLogger(__name__)

class PreKAnonymityAnalyzer:
    """
    Analyzer for k-anonymity profiling on tabular data.

    This class provides methods to analyze k-anonymity for a given DataFrame, identify quasi-identifier field combinations,
    compute k-anonymity metrics, and detect vulnerable records. It supports both chunked and Dask-based processing for scalability.
    The results include detailed metrics, mapping of field combinations, and field uniqueness statistics for further reporting or visualization.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        min_combination_size: int = 2,
        max_combination_size: int = 4,
        threshold_k: int = 5,
        id_fields: List[str] = [],
        fields_combinations: List = None,
        excluded_combinations: List = None,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        chunk_size: int = 10000,
        use_dask: bool = False,
        use_vectorization: bool = False,
        current_steps: int = 3,
        parallel_processes: int = 1,
        **kwargs
    ) -> dict:
        """
        Analyze k-anonymity on the provided DataFrame and return profiling results.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        min_combination_size : int, optional
            Minimum size of field combinations to consider as quasi-identifiers
        max_combination_size : int, optional
            Maximum size of field combinations to consider as quasi-identifiers
        threshold_k : int, optional
            The k threshold below which records are considered vulnerable
        id_fields : List[str], optional
            List of identifier fields to help detect vulnerable records
        fields_combinations : List[List[str]], optional
            List of field combinations to analyze. If None, will be generated automatically
        excluded_combinations : List[List[str]], optional
            List of field combinations to exclude from analysis
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for reporting analysis progress
        chunk_size : int, optional
            Number of combinations to process per chunk (for large datasets)
        use_dask : bool, optional
            Whether to use Dask for parallel processing (recommended for large datasets)
        **kwargs : dict
            Additional keyword arguments for advanced configuration
        
        Returns:
        --------
        dict
            Dictionary containing k-anonymity metrics, vulnerable records, field combination mapping, and field uniqueness statistics
        """
        # Ensure id_fields is not None to prevent errors
        if id_fields is None:
            id_fields = []
        
        # Generate field combinations if not provided
        if not fields_combinations:
            all_fields = list(df.columns)
            # Filter out ID fields to create quasi-identifier fields
            quasi_identifier_fields = [field for field in all_fields if field not in id_fields]
            
            # Generate all possible combinations within size limits
            field_combinations = get_field_combinations(
                quasi_identifier_fields,
                min_size=min_combination_size,
                max_size=max_combination_size,
                excluded_combinations=excluded_combinations
            )
        else:
            # Use provided field combinations
            field_combinations = fields_combinations
            
        # Validate that we have combinations to analyze
        if not field_combinations:
            return {"error": "No valid field combinations to analyze"}
          # Get dataset size for processing strategy decisions
        total_rows = len(df)
        
        # Initialize progress tracking for the analysis
        if progress_tracker:
            progress_tracker.update(current_steps, {
                "step": "Initializing k-anonymity analysis",
                "field": "\n".join([str(combo) for combo in field_combinations])
            })
        
        # Determine if dataset is large enough to warrant chunked processing
        # Analyzer processes entire dataset, so bypass parallel and chunked processing
        is_large_df = total_rows > chunk_size
        if use_dask and is_large_df:
            # TODO: Implement Dask-based processing for large datasets
            logger.warning("Dask processing not yet implemented, falling back to normal processing")
            pass

        # Chunked processing with vectorization and parallel processing
        if use_vectorization and parallel_processes > 1:
            # TODO: Implement parallel vectorized processing
            logger.warning("Parallel vectorized processing not yet implemented, falling back to normal processing")
            pass
        elif use_vectorization and not use_dask:
            # TODO: Implement vectorized processing without Dask
            logger.warning("Vectorized processing not yet implemented, falling back to normal processing")
            pass
        
        return self._analyze_normal(
            df,
            field_combinations,
            threshold_k,
            id_fields,
            progress_tracker=progress_tracker,
            **kwargs
        )
    
    def _analyze_normal(
        self,
        df: pd.DataFrame,
        field_combinations: List,
        threshold_k: int,
        id_fields: List[str] = [],
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        current_steps: int = 3,
        **kwargs
    ) -> dict:
        """
        Analyze k-anonymity for combinations without Dask or chunking.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data to analyze
        field_combinations : list
            List of field combinations to analyze
        threshold_k : int
            The k threshold below which records are considered vulnerable
        id_fields : list
            List of identifier fields to help detect vulnerable records
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the analysis

        Returns:
        --------
        dict
            Dictionary containing k-anonymity metrics, vulnerable records, field combination mapping, and field uniqueness statistics
        """
        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Generated field combinations",
                "combinations_count": len(field_combinations)
            })

        # Create KA index map
        ka_index_map = create_ka_index_map(field_combinations)

        # Log the index map
        logger.info(f"Created {len(ka_index_map)} KA indices")

        if progress_tracker:
            progress_tracker.update(current_steps, {"step": "Created KA index map"})

        ka_metrics = {}
        vulnerable_records = {}
        total_combinations = len(ka_index_map)
        combination_tracker = None
        if progress_tracker:
            combination_tracker = HierarchicalProgressTracker(
                total=total_combinations,
                description="Analyzing field combinations",
                unit="combinations"
            )

        for i, (ka_index, fields) in enumerate(ka_index_map.items()):
            logger.info(f"Analyzing combination {i + 1}/{total_combinations}: {ka_index} ({', '.join(fields)})")
            # Note: calculate_k_anonymity expects ProgressTracker, not HierarchicalProgressTracker
            metrics = calculate_k_anonymity(df, fields, progress_tracker=None)

            # Add to results
            if "error" not in metrics:
                ka_metrics[ka_index] = metrics

                # Find vulnerable records
                vuln_records = find_vulnerable_records(
                    df,
                    fields,
                    k_threshold=threshold_k,
                    id_field=id_fields[0] if id_fields else None
                )
                vulnerable_records[ka_index] = {
                    **vuln_records,
                    "min_k": metrics.get("min_k", 0)
                }
            else:
                logger.warning(f"Error analyzing {ka_index}: {metrics.get('error')}")

            # Update main progress
            if combination_tracker:
                combination_tracker.update(current_steps, {
                    "combination": ka_index,
                    "fields": ", ".join(fields),
                    "progress": f"{i + 1}/{total_combinations}"
                })

        # Close combination tracker
        if combination_tracker:
            combination_tracker.close()

        # Update progress
        if progress_tracker:
            progress_tracker.update(1, {
                "step": "Calculated k-anonymity metrics",
                "metrics_count": len(ka_metrics)
            })

        all_fields = set()
        for fields in field_combinations:
            all_fields.update(fields)

        # Prepare data for field uniqueness visualization
        field_uniqueness = prepare_field_uniqueness_data(df, list(all_fields))

        return {
            "ka_metrics": ka_metrics,
            "vulnerable_records": vulnerable_records,
            "ka_index_map": ka_index_map,
            "field_combinations": field_combinations,
            "field_uniqueness": field_uniqueness
        }

@register()
class PreKAnonymityProfilingOperation(BaseOperation):
    """
    Operation for preliminary profiling of data for k-anonymity analysis.

    This operation analyzes combinations of fields to assess their potential
    as quasi-identifiers and calculates k-anonymity metrics for each combination.
    It generates visualizations and reports about anonymization risks.
    """

    def __init__(self,
                 name: str = "PreKAnonymityProfiling",
                 description: str = "Preliminary profiling for k-anonymity analysis",
                 min_combination_size: int = 2,
                 max_combination_size: int = 4,
                 threshold_k: int = 5,
                 fields_combinations: List = None,
                 excluded_combinations: List = None,
                 id_fields: List = [],
                 include_timestamp: bool = True,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 chunk_size: int = 10000,
                 use_dask: bool = False,
                 npartitions: int = 1,
                 use_cache: bool = True,
                 use_vectorization: bool = False,
                 parallel_processes: int = 1):
        """
        Initialize the k-anonymity profiling operation.

        Parameters:
        -----------
        name : str
            Name of the operation
        description : str
            Description of the operation
        min_combination_size : int
            Minimum size of field combinations to analyze
        max_combination_size : int
            Maximum size of field combinations to analyze
        threshold_k : int
            Threshold for vulnerability (k < threshold_k is considered vulnerable)
        fields_combinations : Optional[List[List[str]]]
            Specific field combinations to analyze. If None, combinations will be generated automatically
        excluded_combinations : Optional[List[List[str]]]
            List of field combinations to exclude from analysis
        id_fields : List[str]
            List of identifier fields to help detect vulnerable records
        include_timestamp : bool
            Whether to include timestamps in output filenames
        use_encryption : bool
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]]
            Key or path to key file used for encryption
        chunk_size : int
            Batch size for processing large datasets (default: 10000)
        use_dask : bool
            Whether to use Dask for processing (default: False)
        npartitions : int, optional
            Number of partitions use with Dask (default: 1)
        use_cache : bool
            Whether to use operation caching (default: True)
        use_vectorization : bool, optional
            Whether to use vectorized (parallel) processing (default: False)
        parallel_processes : int, optional
            Number of processes use with vectorized (parallel) (default: 1)

        """
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )
        self.min_combination_size = min_combination_size
        self.max_combination_size = max_combination_size
        self.threshold_k = threshold_k
        self.fields_combinations = fields_combinations
        self.excluded_combinations = excluded_combinations
        self.id_fields = id_fields
        self.include_timestamp = include_timestamp
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        self.use_cache = use_cache
        self.use_vectorization = use_vectorization

        self.analyzer = PreKAnonymityAnalyzer()

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the k-anonymity profiling operation.

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
            Additional parameters:
            - fields_combinations: List of fields combinations to analyze
            - id_fields: List of ID fields for vulnerable records identification
            - excluded_combinations: List of combinations to exclude
            - threshold_k: Threshold for vulnerability (overrides constructor value)
            - include_timestamp: Whether to include timestamps in filenames
            - chunk_size: Number of records to process in each chunk (for large datasets)
            - npartitions: Number of partitions for Dask
            - force_recalculation: bool - Skip cache check

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Extract parameters from kwargs, defaulting to instance variables
        dataset_name = kwargs.get('dataset_name', "main")
        fields_combinations = kwargs.get('fields_combinations', self.fields_combinations)
        excluded_combinations = kwargs.get('excluded_combinations', self.excluded_combinations)
        id_fields = kwargs.get('id_fields', self.id_fields)
        threshold_k = kwargs.get('threshold_k', self.threshold_k)
        include_timestamp = kwargs.get('include_timestamp', self.include_timestamp)
        encryption_key = kwargs.get('encryption_key', None)
        min_combination_size = kwargs.get('min_combination_size', self.min_combination_size)
        max_combination_size = kwargs.get('max_combination_size', self.max_combination_size)
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        use_dask = kwargs.get('use_dask', self.use_dask)
        use_vectorization= kwargs.get('use_dask', self.use_vectorization)
        force_recalculation = kwargs.get("force_recalculation", False)
        parallel_processes = kwargs.get("parallel_processes", 1)
        encryption_key = kwargs.get('encryption_key', self.encryption_key)
        npartitions = kwargs.get('npartitions', 1)

        # Set up directories
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

        # Create the main result object with initial status
        result = OperationResult(status=OperationStatus.SUCCESS)

        # Set up progress tracking
        # Preparation, Cache Check, Data Loading, Analysis, Metrics, Visualizations, Finalization
        total_steps = 5 + (1 if self.use_cache and not force_recalculation else 0)
        current_steps = 0

        # Update progress if tracker provided
        if progress_tracker:
            progress_tracker.total = total_steps
            progress_tracker.update(current_steps, {
                "step": "Preparation", 
                "operation": self.name,
                "total_steps": total_steps
            })

        # Step 1: Check Cache (if enabled and not forced to recalculate)
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
                    progress_tracker.update(total_steps,{"step": "Complete (cached)"})

                # Report cache hit to reporter
                if reporter:
                    reporter.add_operation(
                        f"Clean invalid values (from cache)",
                        details={"cached": True}
                    )
                return cache_result

        # Step 2: Data Loading
        if progress_tracker:
            current_steps += 1
            progress_tracker.update(current_steps, {"step": "Data Loading"}) 

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

            # Log basic information
            logger.info(
                f"Starting k-anonymity profiling on DataFrame with {len(df)} rows and {len(df.columns)} columns")

            # Add operation to reporter
            if reporter:
                reporter.add_operation(f"K-Anonymity Profiling", details={
                    "records_count": len(df),
                    "columns_count": len(df.columns),
                    "threshold_k": threshold_k,
                    "operation_type": "ka_profiling"
                })

            # Step 3: Analysis
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "K-Anonymity Analysis"}) 
                
            analysis = self.analyzer.analyze(
                df=df,
                min_combination_size=min_combination_size,
                max_combination_size=max_combination_size,
                threshold_k=threshold_k,
                id_fields=id_fields,
                fields_combinations=fields_combinations,
                excluded_combinations=excluded_combinations,
                progress_tracker=progress_tracker,
                chunk_size=chunk_size,
                use_dask=use_dask,
                use_vectorization=use_vectorization,
                current_steps=current_steps,
                parallel_processes = parallel_processes,
            )
            
            if 'error' in analysis:
                if progress_tracker:
                    progress_tracker.update(current_steps, {"step": "Error", "error": analysis['error']})
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=analysis['error']
                )
            
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Created KA index map"})
            
            # Create KA index map
            ka_metrics = analysis['ka_metrics']
            vulnerable_records = analysis['vulnerable_records']
            ka_index_map = analysis['ka_index_map']
            field_combinations = analysis['field_combinations']
            field_uniqueness = analysis['field_uniqueness']
            
            # Step 4: Saving Metrics
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Saving Metrics"})

            # Log the index map
            logger.info(f"Created {len(ka_index_map)} KA indices")

            # Save KA index map to CSV
            ka_index_map_filename = get_timestamped_filename("ka_index_map", "csv", include_timestamp)
            ka_index_map_path = dictionaries_dir / ka_index_map_filename
            save_ka_index_map(ka_index_map, str(ka_index_map_path), encryption_key=encryption_key)
            
            # Add to result and reporter
            result.add_artifact("csv", ka_index_map_path, "KA Index Map", category="dictionary")
            if reporter:
                reporter.add_artifact("csv", str(ka_index_map_path), "KA Index Map")

            # Save metrics to CSV
            metrics_filename = get_timestamped_filename("ka_metrics", "csv", include_timestamp)
            metrics_path = output_dir / metrics_filename
            save_ka_metrics(ka_metrics, str(metrics_path), ka_index_map, **kwargs)

            # Add to result and reporter
            result.add_artifact("csv", metrics_path, "KA Metrics", category=Constants.Artifact_Category_Metrics)
            if reporter:
                reporter.add_artifact("csv", str(metrics_path), "KA Metrics")

            # Save vulnerable records to JSON
            vulnerable_filename = get_timestamped_filename("ka_vulnerable_records", "json", include_timestamp)
            vulnerable_path = output_dir / vulnerable_filename
            save_vulnerable_records(vulnerable_records, str(vulnerable_path), encryption_key)

            # Add to result and reporter
            result.add_artifact("json", vulnerable_path, "KA Vulnerable Records", category=Constants.Artifact_Category_Metrics)
            if reporter:
                reporter.add_artifact("json", str(vulnerable_path), "KA Vulnerable Records")

            # Update progress
            if progress_tracker:
                progress_tracker.update(current_steps, {"step": "Generated metrics files"})

            # Step 5: Creating Visualizations
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Creating Visualizations"})

            # Create visualizations if metrics are available
            if ka_metrics:
                kwargs_encryption = {
                    "use_encryption": kwargs.get('use_encryption', False),
                    "encryption_key": encryption_key
                }
                self._create_visualizations(
                    ka_metrics,
                    field_uniqueness,
                    field_combinations,
                    visualizations_dir,
                    include_timestamp,
                    result,
                    reporter,
                    **kwargs_encryption
                )

            # Calculate individual field uniqueness
            all_fields = set()
            for fields in field_combinations:
                all_fields.update(fields)

            field_uniqueness = prepare_field_uniqueness_data(df, list(all_fields))

            # Save field uniqueness data
            uniqueness_filename = get_timestamped_filename("field_uniqueness", "json", include_timestamp)
            uniqueness_path = output_dir / uniqueness_filename
            write_json(field_uniqueness, str(uniqueness_path), encryption_key=encryption_key)

            # Add to result and reporter
            result.add_artifact("json", uniqueness_path, "Field Uniqueness Metrics", category=Constants.Artifact_Category_Metrics)
            if reporter:
                reporter.add_artifact("json", str(uniqueness_path), "Field Uniqueness Metrics")

            # Step 6: Finalization
            if progress_tracker:
                current_steps += 1
                progress_tracker.update(current_steps, {"step": "Operation complete", "status": "success"})

            # Add metrics to the result for easy access
            for ka_index, metrics in ka_metrics.items():
                result.add_nested_metric("ka_metrics", ka_index, {
                    "min_k": metrics.get("min_k", 0),
                    "mean_k": metrics.get("mean_k", 0),
                    "unique_percentage": metrics.get("unique_percentage", 0),
                    "vulnerable_percentage": 100 - metrics.get("threshold_metrics", {}).get("k≥5", 0)
                })

            # Add operation summary to reporter
            top_risks = sorted(
                [(ka, metrics.get("min_k", float('inf'))) for ka, metrics in ka_metrics.items()],
                key=lambda x: x[1]
            )[:3]
            
            if reporter:
                reporter.add_operation("K-Anonymity Profiling Completed", details={
                    "analyzed_combinations": len(ka_metrics),
                    "top_risk_combinations": [f"{ka} (min_k={k})" for ka, k in top_risks],
                    "threshold_k": threshold_k
                })

            # Cache the result if caching is enabled
            if self.use_cache:
                try:
                    self._save_to_cache(
                        artifacts=result.artifacts,
                        original_df=df,
                        metrics=result.metrics,
                        task_dir=task_dir
                    )
                except Exception as e:
                    # Failure to cache is non-critical
                    self.logger.warning(f"Failed to cache results: {str(e)}")

            return result

        except Exception as e:
            logger.exception(f"Error in k-anonymity profiling operation: {e}")

            # Update progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            # Add error to reporter
            if reporter:
                reporter.add_operation("K-Anonymity Profiling",
                                    status="error",
                                    details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Error in k-anonymity profiling: {str(e)}"
            )

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

        ensure_directory(output_dir)
        ensure_directory(visualizations_dir)
        ensure_directory(dictionaries_dir)

        return {
            'output': output_dir,
            'visualizations': visualizations_dir,
            'dictionaries': dictionaries_dir
        }

    @staticmethod
    def _create_visualizations(ka_metrics: Dict[str, Dict[str, Any]],
                               field_uniqueness:  Dict[str, Dict[str, Any]],
                               field_combinations: List[List[str]],
                               vis_dir: Path,
                               include_timestamp: bool,
                               result: OperationResult,
                               reporter: Any,
                               **kwargs):
        """
        Create visualizations for k-anonymity metrics.

        Parameters:
        -----------
        ka_metrics : Dict[str, Dict[str, Any]]
            Dictionary mapping KA indices to their metrics
        ka_index_map : Dict[str, List[str]]
            Mapping from KA indices to field lists
        df : pd.DataFrame
            Original DataFrame
        field_combinations : List[List[str]]
            List of field combinations
        vis_dir : Path
            Directory to save visualizations
        include_timestamp : bool
            Whether to include timestamps in filenames
        threshold_k : int
            Threshold for vulnerability
        result : OperationResult
            Operation result to add artifacts to
        reporter : Any
            Reporter to add artifacts to
        """
        if not ka_metrics:
            logger.warning("No metrics available for visualization")
            return

        try:
            # 1. Create k-range distribution visualization
            k_range_data = {}
            for ka_index, metrics in ka_metrics.items():
                if "k_range_distribution" in metrics:
                    k_range_data[ka_index] = metrics["k_range_distribution"]

            if k_range_data:
                # Convert to DataFrame for easier plotting
                df_k_range = pd.DataFrame(k_range_data)

                # Create bar plot
                k_range_filename = get_timestamped_filename("ka_k_distribution", "png", include_timestamp)
                k_range_path = vis_dir / k_range_filename

                # Convert DataFrame to dictionary format expected by create_bar_plot
                k_range_result = create_bar_plot(
                    data={str(k): v for k, v in df_k_range.to_dict().items()},  # Convert to dictionary format
                    output_path=str(k_range_path),
                    title="K-Anonymity Range Distribution",
                    orientation="h",
                    y_label="K Range",
                    x_label="Percentage of Records (%)",
                    **kwargs
                )

                if not k_range_result.startswith("Error"):
                    result.add_artifact("png", k_range_path, "K-anonymity range distribution", category=Constants.Artifact_Category_Visualization)
                    if reporter:
                        reporter.add_artifact("png", str(k_range_path), "K-anonymity range distribution")

            # 2. Create threshold curve visualization
            threshold_data = {}
            for ka_index, metrics in ka_metrics.items():
                if "threshold_metrics" in metrics:
                    threshold_data[ka_index] = metrics["threshold_metrics"]

            if threshold_data:
                # Convert to DataFrame for easier plotting
                df_threshold = pd.DataFrame(threshold_data)

                # Create line plot
                threshold_filename = get_timestamped_filename("ka_vulnerable_curve", "png", include_timestamp)
                threshold_path = vis_dir / threshold_filename

                threshold_result = create_line_plot(
                    data=df_threshold.T,  # Transpose for better format
                    output_path=str(threshold_path),
                    title="Records Meeting K-Anonymity Thresholds",
                    x_label="K Threshold",
                    y_label="Percentage of Records (%)",
                    add_markers=True,
                    **kwargs
                )

                if not threshold_result.startswith("Error"):
                    result.add_artifact("png", threshold_path, "K-anonymity threshold compliance", category=Constants.Artifact_Category_Visualization)
                    if reporter:
                        reporter.add_artifact("png", str(threshold_path), "K-anonymity threshold compliance")

            # 3. Create spider chart for multi-metric comparison
            spider_data = prepare_metrics_for_spider_chart(ka_metrics)

            if spider_data:
                spider_filename = get_timestamped_filename("ka_comparison_spider", "png", include_timestamp)
                spider_path = vis_dir / spider_filename

                spider_result = create_spider_chart(
                    data=spider_data,
                    output_path=str(spider_path),
                    title="K-Anonymity Metrics Comparison",
                    normalize_values=False,  # Values are already normalized
                    fill_area=True
                )

                if not spider_result.startswith("Error"):
                    result.add_artifact("png", spider_path, "K-anonymity metrics comparison", category=Constants.Artifact_Category_Visualization)
                    if reporter:
                        reporter.add_artifact("png", str(spider_path), "K-anonymity metrics comparison")

            # 4. Create field uniqueness visualization
            # Get all individual fields
            if field_uniqueness:
                # Prepare data for the combined chart
                fields = list(field_uniqueness.keys())
                unique_counts = [field_uniqueness[f].get("unique_values", 0) for f in fields]
                uniqueness_percent = [field_uniqueness[f].get("uniqueness_percentage", 0) for f in fields]

                # Create combined chart
                uniqueness_filename = get_timestamped_filename("ka_field_uniqueness", "png", include_timestamp)
                uniqueness_path = vis_dir / uniqueness_filename

                uniqueness_result = create_combined_chart(
                    primary_data=dict(zip(fields, unique_counts)),
                    secondary_data=dict(zip(fields, uniqueness_percent)),
                    output_path=str(uniqueness_path),
                    title="Quasi-Identifier Analysis",
                    primary_type="bar",
                    secondary_type="line",
                    x_label="Field",
                    primary_y_label="Unique Values Count",
                    secondary_y_label="Uniqueness (%)",
                    primary_color="steelblue",
                    secondary_color="crimson"
                )

                if not uniqueness_result.startswith("Error"):
                    result.add_artifact("png", uniqueness_path, "Field uniqueness analysis", category=Constants.Artifact_Category_Visualization)
                    if reporter:
                        reporter.add_artifact("png", str(uniqueness_path), "Field uniqueness analysis")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}", exc_info=True)
            if reporter:
                reporter.add_operation("Creating visualizations", status="warning",
                                   details={"warning": f"Error creating some visualizations: {str(e)}"})
                
    def _check_cache(
            self,
            data_source: DataSource,
            dataset_name: str = "main"
    ) -> Optional[OperationResult]:
        """
        Check if a cached result exists for operation.

        Parameters:
        -----------
        data_source : DataSource
            Data source for the operation
        task_dir : Path
            Task directory
        dataset_name: str
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
            df = load_data_operation(data_source, dataset_name)
            if df is None:
                self.logger.warning("No valid DataFrame found in data source")
                return None

            # Generate cache key
            cache_key = self._generate_cache_key(df)

            # Check for cached result
            self.logger.debug(f"Checking cache for key: {cache_key}")
            cached_data = operation_cache.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            if cached_data:
                self.logger.info(f"Using cached result.")

                # Create result object from cached data
                cached_result = OperationResult(status=OperationStatus.SUCCESS)

                # Add cached metrics to result
                metrics = cached_data.get("metrics", {})
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float, str, bool)):
                            cached_result.add_metric(key, value)

                # Add cached artifacts to result
                artifacts = cached_data.get("artifacts", [])
                if isinstance(artifacts, list):
                    for artifact in artifacts:
                        if isinstance(artifact, dict):
                            artifact_type = artifact.get("artifact_type", "")
                            artifact_path = artifact.get("path", "")
                            artifact_name = artifact.get("description", "")
                            artifact_category = artifact.get("category", "output")
                            cached_result.add_artifact(artifact_type, artifact_path, artifact_name, artifact_category) 

                # Add cache information to result
                cached_result.add_metric("cached", True)
                cached_result.add_metric("cache_key", cache_key)
                cached_result.add_metric("cache_timestamp", cached_data.get("timestamp", "unknown"))

                return cached_result

            self.logger.debug(f"No cache found for key: {cache_key}")
            return None
        except Exception as e:
            self.logger.warning(f"Error checking cache: {str(e)}")
            return None
        
    def _save_to_cache(
            self,
            original_df: pd.DataFrame,
            artifacts: List[Any],
            metrics: Dict[str, Any],
            task_dir: Path
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
        if not self.use_cache:
            return False

        try:
            # Import and get global cache manager
            from pamola_core.utils.ops.op_cache import operation_cache

            # Generate cache key
            cache_key = self._generate_cache_key(original_df)

            # Prepare metadata for cache
            operation_parameters = self._get_operation_parameters()

            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "parameters": operation_parameters,
                "artifacts": artifacts,
                "metrics": metrics,
            }

            # Save to cache
            self.logger.debug(f"Saving to cache with key: {cache_key}")
            success = operation_cache.save_cache(
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
            "fields_combinations": self.fields_combinations,
            "excluded_combinations": self.excluded_combinations,
            "id_fields": self.id_fields,
            "min_combination_size": self.min_combination_size,
            "max_combination_size": self.max_combination_size,
            "threshold_k": self.threshold_k
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