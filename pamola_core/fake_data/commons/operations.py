"""
Operations module for fake data generation.

This module provides standardized operation classes for fake data generation
that integrate with PAMOLA's operation system to generate synthetic data while
maintaining statistical properties of the original data.
"""
import psutil
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.fake_data.commons.base import (
    BaseGenerator,
    FieldOperation as BaseFieldOperation,
    NullStrategy,
    ValidationError
)
# Import metrics collector
from pamola_core.fake_data.commons.metrics import create_metrics_collector, generate_metrics_report
from pamola_core.utils.io import (
    ensure_directory,
    write_dataframe_to_csv,
    write_json,
    get_timestamped_filename,
    load_data_operation,
    load_settings_operation
)
from pamola_core.utils.ops.op_base import BaseOperation as PAMOLABaseOperation
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.progress import ProgressTracker
import dask.dataframe as dd
from joblib import Parallel, delayed
from pamola_core.common.constants import Constants
# Configure logger
logger = logging.getLogger(__name__)


class BaseOperation(PAMOLABaseOperation):
    """
    Base class for fake data operations.

    Provides common functionality for operations that generate or replace
    data with fake values.
    """

    name: str = "base_fake_data_operation"
    description: str = "Base operation for fake data generation"

    def __init__(self):
        """
        Initializes the operation.
        """
        super().__init__(
            name=self.name,
            description=self.description
        )

    def execute(self, data_source: Any, task_dir: Path, reporter: Any, progress_tracker: Optional[ProgressTracker] = None, **kwargs) -> OperationResult:
        """
        Executes the operation.

        Parameters:
        -----------
        data_source : Any
            Source of data for processing
        task_dir : Path
            Directory for storing operation artifacts
        reporter : Any
            Reporter for operation progress and results
        **kwargs : dict
            Additional operation parameters

        Returns:
        --------
        OperationResult
            Operation result
        """
        logger.info(f"Executing {self.name} operation")

        # Initialize result with default error status
        result = OperationResult(
            status=OperationStatus.ERROR,
            error_message="Operation not implemented",
            execution_time=0
        )

        return result


class FieldOperation(BaseFieldOperation, BaseOperation):
    """
    Base class for operations on specific fields.

    Provides common functionality for operations that process
    specific fields in a dataset.
    """

    name: str = "field_operation"
    description: str = "Base operation for field processing"

    def __init__(self, field_name: str, mode: str = "REPLACE", output_field_name: Optional[str] = None,
                 batch_size: int = 10000, null_strategy: Union[str, NullStrategy] = NullStrategy.PRESERVE,
                 use_cache: bool = True, force_recalculation: bool = False,
                 use_dask: bool = False, npartitions: int = 1,
                 use_vectorization: bool = False, parallel_processes: int = 1,
                 use_encryption: bool = False, encryption_key: Optional[Union[str, Path]] = None,
                 visualization_backend: Optional[str] = None, vis_theme: Optional[str] = None, vis_strict: bool = False):
        """
        Initializes the field operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to process
        mode : str
            Operation mode: "REPLACE" or "ENRICH"
        output_field_name : str, optional
            Name of the output field (used in ENRICH mode)
        batch_size : int
            Size of batches for processing
        null_strategy : str or NullStrategy
            Strategy for handling NULL values: "PRESERVE", "REPLACE", "EXCLUDE", "ERROR"
        use_cache : bool
            force_recalculation (bool): If True, bypass cached results and force full reprocessing.
        force_recalculation : bool
            force_recalculation (bool): If True, bypass cached results and force full reprocessing.
        use_dask : bool
            Whether to use Dask for large datasets
        npartitions : int, optional
            Number of partitions for Dask processing (if use_dask=True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]], optional
            Encryption key for securing outputs (default: None)
        visualization_backend : str, optional
            Backend to use for visualizations: "plotly" or "matplotlib" (default: None - uses system default)
        visualization_theme : str, optional
            Theme to use for visualizations (default: None - uses system default)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        """
        # Initialize BaseFieldOperation
        if isinstance(null_strategy, str):
            try:
                null_strategy = NullStrategy(null_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown NULL strategy: {null_strategy}. Using PRESERVE.")
                null_strategy = NullStrategy.PRESERVE

        BaseFieldOperation.__init__(
            self,
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            null_strategy=null_strategy
        )

        # Initialize BaseOperation
        BaseOperation.__init__(self)

        # Additional field operation parameters
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.force_recalculation = force_recalculation
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        # Store original data for metrics comparison
        self._original_df = None

        # Create a metrics collector
        self._metrics_collector = create_metrics_collector()
        
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key

        self.visualization_backend = visualization_backend
        self.vis_theme = vis_theme
        self.vis_strict = vis_strict

    def execute(self, data_source: Any, task_dir: Path, reporter: Any, progress_tracker: Optional[HierarchicalProgressTracker] = None, **kwargs) -> OperationResult:
        """
        Executes the field operation.

        Parameters:
        -----------
        data_source : Any
            Source of data for processing
        task_dir : Path
            Directory for storing operation artifacts
        reporter : Any
            Reporter for operation progress and results
        **kwargs : dict
            Additional operation parameters

        Returns:
        --------
        OperationResult
            Operation result
        """
        start_time = time.time()

        logger.info(f"Executing {self.name} operation on field {self.field_name}")

        try:
            if reporter:
                reporter.add_operation(f"Operation {self.name}", status="info",
                                       details={"message": "Starting operation execution"})

            step_count = 0
            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Preparation", "operation": self.name})
                step_count += 1

            # Create output directory for data output and maps
            dirs = self._prepare_directories(task_dir)
            output_dir = dirs['output']
            visualizations_dir = dirs['visualizations']

            if reporter:
                reporter.add_operation(f"Operation {self.name}", status="info",
                                       details={"message": "Prepared output directories"})

            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Loading dataset", "operation": self.name})
                step_count += 1

            # Load data
            dataset_name = kwargs.get('dataset_name', "main")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)

            # Store original data for metrics comparison
            self._original_df = df.copy()

            # Check if field exists
            if self.field_name not in df.columns:
                error_message = f"Field {self.field_name} not found in the data"
                logger.error(error_message)

                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message,
                    execution_time=time.time() - start_time
                )


            # Progress bar for batch processing
            total_records = len(df)
            total_batches = (total_records + self.batch_size - 1) // self.batch_size
           
            # Preprocess data
            df = self.preprocess_data(df)

            # Handle NULL values
            df = self.handle_null_values(df)

            if reporter:
                reporter.add_operation(f"Operation {self.name}", status="info",
                                       details={"message": f"Dataset '{dataset_name}' loaded and verify successfully"})

            # Handle caching
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Use cache", "operation": self.name})
                    step_count += 1

                cached_result = self._get_cache(df.copy())  # _get_cache now returns OperationResult or None
                if cached_result is not None and isinstance(cached_result, OperationResult):
                    if reporter:
                        reporter.add_operation(f"Operation {self.name}", status="info",
                                               details={"message": "Result loaded from cache – execution skipped"})
                    return cached_result
                else:
                    self.logger.info("No valid cached result found — proceeding with execution.")

            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Processing data", "operation": self.name})
                step_count += 1

            # Handle dask
            if self.use_dask and self.npartitions > 1:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Use dask", "operation": self.name})
                    step_count += 1

                ddf = dd.from_pandas(df, npartitions=self.npartitions)
                processed_ddf = ddf.map_partitions(self.process_batch)
                df = processed_ddf.compute()

            # Handle joblib
            elif self.use_vectorization:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Use joblib", "operation": self.name})
                    step_count += 1

                # Split data into batches
                batches = [
                    (i, df.iloc[batch_start:batch_start + self.batch_size].copy())
                    for i, batch_start in enumerate(range(0, total_records, self.batch_size))
                ]

                # Process batches in parallel using joblib
                results = Parallel(n_jobs=self.parallel_processes)(
                    delayed(self.process_batch)(batch) for _, batch in batches
                )

                # Merge the results
                for (i, batch), batch_result in zip(batches, results):
                    batch_start = i * self.batch_size
                    batch_end = min(batch_start + self.batch_size, total_records)
                    if self.mode == "REPLACE":
                        df.iloc[batch_start:batch_end] = batch_result
                    elif self.mode == "ENRICH":
                        for col in batch_result.columns:
                            if col not in df.columns or col == self.output_field_name:
                                df.loc[batch_start:batch_end - 1, col] = batch_result[col].values

            # Handle chunk
            else:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Chunk processing", "operation": self.name})
                    step_count += 1

                for i, batch_start in enumerate(range(0, total_records, self.batch_size)):
                    batch_end  = min(batch_start + self.batch_size, total_records)
                    batch = df.iloc[batch_start:batch_end].copy()

                    # Process batch
                    batch = self.process_batch(batch)
                    
                    # Update processed data
                    if self.mode == "REPLACE":
                        df.iloc[batch_start:batch_end] = batch
                    elif self.mode == "ENRICH":
                        for col in batch.columns:
                            if col not in df.columns or col == self.output_field_name:
                                df.loc[batch_start:batch_end - 1, col] = batch[col].values

            # Calculate execution time
            execution_time = time.time() - start_time

            # Update performance metrics
            records_per_second = int(total_records / execution_time) if execution_time > 0 else 0

            # Collect metrics
            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Collecting metrics", "operation": self.name})
                step_count += 1

            metrics = self._collect_metrics(df)
            # Add performance metrics
            metrics["performance"]["generation_time"] = execution_time
            metrics["performance"]["records_per_second"] = records_per_second

            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                metrics["performance"]["memory_usage_mb"] = memory_info.rss / (1024 * 1024)

            except Exception as e:
                logger.warning(f"Error collecting memory usage: {str(e)}")

            # Generate visualizations
            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Generating visualizations", "operation": self.name})
                step_count += 1

            try:
                if self.mode == "REPLACE":
                    gen_series = df[self.field_name]
                elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                    gen_series = df[self.output_field_name]
                else:
                    gen_series = None

                kwargs_visualization = {
                    "use_encryption": kwargs.get("use_encryption", False),
                    "encryption_key": kwargs.get("encryption_key", None),
                    "backend": kwargs.get("visualization_backend", self.visualization_backend),
                    "theme": kwargs.get("vis_theme", self.vis_theme),
                    "strict": kwargs.get("vis_strict", self.vis_strict)
                }

                if gen_series is not None:
                    visualizations = self._metrics_collector.visualize_metrics(
                        metrics=metrics,
                        field_name=self.field_name,
                        output_dir=visualizations_dir,
                        op_type=self.name,
                        **kwargs_visualization
                    )

                    # Add visualizations to metrics
                    metrics["visualizations"] = {name: str(path) for name, path in visualizations.items()}
            except Exception as e:
                logger.warning(f"Error generating visualizations: {str(e)}")

            # Save result
            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Saving results", "operation": self.name})
                step_count += 1

            # Save data if required
            save_data = kwargs.get("save_data", True)
            if save_data:
                output_path = self._save_result(df, output_dir, **kwargs)
            else:
                output_path = None

            if reporter:
                reporter.add_operation(f"Operation {self.name} ", status="info",
                    details={
                        "message": "save data successfully",
                        "path": str(output_path),
                    },
                )

            # Save metrics
            metrics_path = self._save_metrics(metrics, task_dir, **kwargs)

            if reporter:
                reporter.add_operation(f"Operation {self.name}", status="info",
                                       details={"message": "Metrics saved successfully", "path": str(metrics_path)})

            # Generate and save metrics report
            try:
                report_dir = task_dir / "reports"
                ensure_directory(report_dir)
                generate_metrics_report(
                    metrics,
                    report_dir,
                    op_type=self.name,
                    field_name=self.field_name
                )
            except Exception as e:
                logger.warning(f"Error generating metrics report: {str(e)}")

            # Create result
            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Create OperationResult", "operation": self.name})
                step_count += 1

            result = OperationResult(
                status=OperationStatus.SUCCESS,
                execution_time=execution_time
            )

            if hasattr(result, 'set_output') and callable(getattr(result, 'set_output')):
                result.set_output(df)
            elif hasattr(result, 'data') or hasattr(type(result), 'data'):
                result.data = df
            # Add artifacts
            if output_path:
                result.add_artifact(
                    path=output_path,                   
                    artifact_type="csv",
                    description=f"Processed data with {self.mode.lower()}d {self.field_name} field",
                    category=Constants.Artifact_Category_Output
                )

            result.add_artifact(
                path=metrics_path,
                artifact_type="json",
                description=f"Metrics for {self.name} operation on {self.field_name}",
                category=Constants.Artifact_Category_Metrics
            )

            # Add visualization artifacts if available
            if "visualizations" in metrics:
                for name, path_str in metrics["visualizations"].items():
                    result.add_artifact(
                        path=Path(path_str),
                        artifact_type="png",
                        description=f"{name.replace('_', ' ').title()} visualization for {self.field_name}",
                        category=Constants.Artifact_Category_Visualization
                    )

            # Add metrics
            result.metrics = metrics

            if self.use_cache:
                self._save_cache(task_dir, result)

            if reporter:
                reporter.add_operation(f"Operation {self.name} ", status="info",
                                       details={f"message: {self.name} cached result saved successfully"})

            return result

        except Exception as e:
            print(f"Operations execute failed: {e}")
            logger.error(f"Operations execute failed: {e}")
            if reporter:
                reporter.add_operation(f"Operation {self.name}", status="error",
                                       details={"message": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    @staticmethod
    def _load_data(data_source: Any) -> pd.DataFrame:
        """
        Loads data from the data source.

        Parameters:
        -----------
        data_source : Any
            Source of data

        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        if hasattr(data_source, "get_dataframe"):
            return data_source.get_dataframe("maim")
        elif isinstance(data_source, pd.DataFrame):
            return data_source
        elif isinstance(data_source, str) or isinstance(data_source, Path):
            # Load from file path
            try:
                from pamola_core.utils.io import read_full_csv
                return read_full_csv(data_source)
            except Exception as e:
                raise ValueError(f"Unable to load data from path {data_source}: {str(e)}")
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

    def _save_result(self, df: pd.DataFrame, output_dir: Path, **kwargs) -> Path:
        """
        Saves the result to a file.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        output_dir : Path
            Directory to save the file

        Returns:
        --------
        Path
            Path to the saved file
        """
        file_name = get_timestamped_filename(f"{self.name}_{self.field_name}", "csv")
        output_path = output_dir / file_name

        use_encryption = kwargs.get('use_encryption', False)
        encryption_key= kwargs.get('encryption_key', None) if use_encryption else None
        write_dataframe_to_csv(df=df, file_path=output_path, encryption_key=encryption_key)
        return output_path

    def _save_metrics(self, metrics: Dict[str, Any], task_dir: Path, **kwargs) -> Path:
        """
        Saves the metrics to a file.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Metrics to save
        task_dir : Path
            Directory to save the file

        Returns:
        --------
        Path
            Path to the saved file
        """
        # Create metrics file name directly in the task directory
        metrics_path = task_dir / f"{self.name}_{self.field_name}_metrics.json"
        use_encryption = kwargs.get('use_encryption', False)
        encryption_key= kwargs.get('encryption_key', None) if use_encryption else None
        write_json(metrics, metrics_path, encryption_key=encryption_key)
        return metrics_path

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collects metrics for the operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Processed DataFrame

        Returns:
        --------
        Dict[str, Any]
            Metrics for the operation
        """
        # Basic metrics
        metrics = {
            "total_records": len(df),
            "non_null_records": df[self.field_name].notna().sum(),
            "execution_time": 0,  # Will be updated by execute method
            "performance": {}
        }

        # Output field metrics if in ENRICH mode
        if self.mode == "ENRICH" and self.output_field_name in df.columns:
            metrics["output_field"] = {
                "name": self.output_field_name,
                "non_null_records": df[self.output_field_name].notna().sum()
            }

        # Use metrics collector to get detailed metrics
        try:
            # Extract original and generated data series
            orig_series = self._original_df[self.field_name] if self._original_df is not None else None

            if self.mode == "REPLACE":
                gen_series = df[self.field_name]
            elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                gen_series = df[self.output_field_name]
            else:
                gen_series = None

            # Collect metrics using the metrics collector
            if orig_series is not None and gen_series is not None:
                collector_metrics = self._metrics_collector.collect_metrics(
                    orig_data=orig_series,
                    gen_data=gen_series,
                    field_name=self.field_name,
                    operation_params={"field_name": self.field_name}
                )

                # Merge metrics
                for key, value in collector_metrics.items():
                    if key not in metrics:
                        metrics[key] = value
                    elif isinstance(metrics[key], dict) and isinstance(value, dict):
                        metrics[key].update(value)
        except Exception as e:
            logger.warning(f"Error collecting detailed metrics: {str(e)}")

        return metrics

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a batch of data.

        Parameters:
        -----------
        batch : pd.DataFrame
            Batch of data to process

        Returns:
        --------
        pd.DataFrame
            Processed batch
        """
        raise NotImplementedError("Subclasses must implement process_batch method")

    def _get_cache(self, df: pd.DataFrame) -> Optional[OperationResult]:
        """
        Retrieve cached result if available and valid.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame used to generate the cache key.

        Returns
        -------
        Optional[OperationResult]
            The cached OperationResult if available, otherwise None.
        """
        try:
            cache_key = operation_cache.generate_cache_key(
                operation_name=self.__class__.__name__,
                parameters=self._get_cache_parameters(),
                data_hash=self._generate_data_hash(df)
            )

            cached = operation_cache.get_cache(
                cache_key=cache_key,
                operation_type=self.__class__.__name__
            )

            result_data = cached.get("result")
            if not isinstance(result_data, dict):
                return None

            # Parse enum safely
            status_str = result_data.get("status", OperationStatus.ERROR.name)
            status = OperationStatus[status_str] if isinstance(status_str,
                                                               str) and status_str in OperationStatus.__members__ else OperationStatus.ERROR

            # Rebuild artifacts
            artifacts = []
            for art_dict in result_data.get("artifacts", []):
                if isinstance(art_dict, dict):
                    try:
                        artifacts.append(OperationArtifact(
                            artifact_type=art_dict.get("type"),
                            path=art_dict.get("path"),
                            description=art_dict.get("description", ""),
                            category=art_dict.get("category", "output"),
                            tags=art_dict.get("tags", []),
                        ))
                    except Exception as e:
                        self.logger.warning(f"Failed to deserialize artifact: {e}")

            return OperationResult(
                status=status,
                artifacts=artifacts,
                metrics=result_data.get("metrics", {}),
                error_message=result_data.get("error_message"),
                execution_time=result_data.get("execution_time"),
                error_trace=result_data.get("error_trace"),
            )

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters required for generating a cache key.

        These parameters define the behavior of the transformation and are used
        to determine cache uniqueness.

        Returns
        -------
        Dict[str, Any]
            Dictionary of relevant parameters to identify the operation configuration.
        """
        params = {
            "operation": self.__class__.__name__,
            "version": self.version,  # To support invalidation on version changes
            "field_name": self.field_name,
            "mode": self.mode,
            "null_strategy": self.null_strategy
        }
        return params

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Generate a hash that represents key characteristics of the input DataFrame.

        The hash is based on structure and summary statistics to detect changes
        for caching purposes.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to generate a representative hash from.

        Returns
        -------
        str
            A hash string representing the structure and key properties of the data.
        """
        try:
            characteristics = {
                "columns": list(data.columns),
                "shape": data.shape,
                "summary": {}
            }

            for col in data.columns:
                col_data = data[col]
                col_info = {
                    "dtype": str(col_data.dtype),
                    "null_count": int(col_data.isna().sum()),
                    "unique_count": int(col_data.nunique())
                }

                if pd.api.types.is_numeric_dtype(col_data):
                    non_null = col_data.dropna()
                    if not non_null.empty:
                        col_info.update({
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "mean": float(non_null.mean()),
                            "median": float(non_null.median()),
                            "std": float(non_null.std())
                        })
                elif pd.api.types.is_object_dtype(col_data) or isinstance(col_data.dtype, pd.CategoricalDtype):
                    top_values = col_data.value_counts(dropna=True).head(5)
                    col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

                characteristics["summary"][col] = col_info

            json_str = json.dumps(characteristics, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Error generating data hash: {str(e)}")
            fallback = f"{data.shape}_{list(data.dtypes)}"
            return hashlib.md5(fallback.encode()).hexdigest()

    def _save_cache(self, task_dir: Path, result: OperationResult) -> None:
        """
        Save the operation result to cache.

        Parameters
        ----------
        task_dir : Path
            Root directory for the task.
        result : OperationResult
            The result object to be cached.
        """
        try:
            result_data = {
                "status": result.status.name if isinstance(result.status, OperationStatus) else str(result.status),
                "metrics": result.metrics,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "error_trace": result.error_trace,
                "artifacts": [artifact.to_dict() for artifact in result.artifacts]
            }

            cache_data = {
                "result": result_data,
                "parameters": self._get_cache_parameters(),
            }

            cache_key = operation_cache.generate_cache_key(
                operation_name=self.__class__.__name__,
                parameters=self._get_cache_parameters(),
                data_hash=self._generate_data_hash(self._original_df.copy())
            )

            operation_cache.save_cache(
                data=cache_data,
                cache_key=cache_key,
                operation_type=self.__class__.__name__,
                metadata={"task_dir": str(task_dir)}
            )

            self.logger.info(f"Saved result to cache with key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")


class GeneratorOperation(FieldOperation):
    """
    Operation that uses a generator to create fake data.

    This operation integrates with the PAMOLA operation system and uses a generator
    to create synthetic data for specific fields.
    """

    name: str = "generator_operation"
    description: str = "Operation for generating fake data"

    def __init__(self, field_name: str, generator: BaseGenerator,
                 mode: str = "REPLACE", output_field_name: Optional[str] = None,
                 batch_size: int = 10000, null_strategy: Union[str, NullStrategy] = NullStrategy.PRESERVE,
                 consistency_mechanism: str = "prgn", mapping_store: Optional[MappingStore] = None,
                 generator_params: Optional[Dict[str, Any]] = None,
                 use_cache: bool = True, force_recalculation: bool = False,
                 use_dask: bool = False, npartitions: int = 1,
                 use_vectorization: bool = False, parallel_processes: int = 1,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 visualization_backend: Optional[str] = None,
                 vis_theme: Optional[str] = None,
                 vis_strict: bool = False):
        """
        Initializes the generator operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to process
        generator : BaseGenerator
            Generator for creating fake data
        mode : str
            Operation mode: "REPLACE" or "ENRICH"
        output_field_name : str, optional
            Name of the output field (used in ENRICH mode)
        batch_size : int
            Size of batches for processing
        null_strategy : str or NullStrategy
            Strategy for handling NULL values: "PRESERVE", "REPLACE", "EXCLUDE", "ERROR"
        consistency_mechanism : str
            Mechanism for ensuring consistent replacements: "prgn" or "mapping"
        mapping_store : MappingStore, optional
            Store for mappings between original and synthetic values
        generator_params : Dict[str, Any], optional
            Additional parameters for the generator
        use_cache : bool
            force_recalculation (bool): If True, bypass cached results and force full reprocessing.
        force_recalculation : bool
            force_recalculation (bool): If True, bypass cached results and force full reprocessing.
        use_dask : bool
            Whether to use Dask for large datasets
        npartitions : int, optional
            Number of partitions for Dask processing (if use_dask=True)
        use_encryption : bool, optional
            Whether to encrypt output files (default: False)
        encryption_key : Optional[Union[str, Path]], optional
            Encryption key for securing outputs (default: None)
        visualization_backend : str, optional
            Backend to use for visualizations: "plotly" or "matplotlib" (default: None - uses system default)
        visualization_theme : str, optional
            Theme to use for visualizations (default: None - uses system default)
        visualization_strict : bool, optional
            If True, raise exceptions for visualization config errors (default: False)
        """
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            batch_size=batch_size,
            null_strategy=null_strategy,
            use_cache=use_cache,
            force_recalculation=force_recalculation,
            use_dask=use_dask,
            npartitions=npartitions,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            visualization_backend=visualization_backend,
            vis_theme=vis_theme,
            vis_strict=vis_strict
        )

        self.generator = generator
        self.consistency_mechanism = consistency_mechanism.lower()
        self.mapping_store = mapping_store or MappingStore()
        self.generator_params = generator_params or {}

        # Update operation name and description
        self.name = f"{generator.__class__.__name__}_operation"
        self.description = f"Operation for generating fake {field_name} data using {generator.__class__.__name__}"

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a batch of data using the generator.

        Parameters:
        -----------
        batch : pd.DataFrame
            Batch of data to process

        Returns:
        --------
        pd.DataFrame
            Processed batch with synthetic data
        """
        result_batch = batch.copy()
        field_values = batch[self.field_name]

        # Skip processing if the field has no non-null values
        if field_values.notna().sum() == 0:
            if self.mode == "ENRICH" and self.output_field_name:
                result_batch[self.output_field_name] = field_values
            return result_batch

        # Generate synthetic values
        synthetic_values = []

        # Process each value
        for value in field_values:
            if pd.isna(value):
                # Handle NULL values based on strategy
                if self.null_strategy == NullStrategy.PRESERVE:
                    synthetic_values.append(None)
                elif self.null_strategy == NullStrategy.REPLACE:
                    # Generate a random value
                    synthetic_value = self._process_value(None)
                    synthetic_values.append(synthetic_value)
                elif self.null_strategy == NullStrategy.EXCLUDE:
                    # Should have been filtered out in the handle_null_values method
                    synthetic_values.append(None)
                elif self.null_strategy == NullStrategy.ERROR:
                    # Should have been caught in the handle_null_values method
                    synthetic_values.append(None)
            else:
                # Process non-NULL value
                synthetic_value = self._process_value(value)
                synthetic_values.append(synthetic_value)

        # Update the batch
        if self.mode == "REPLACE":
            result_batch[self.field_name] = synthetic_values
        elif self.mode == "ENRICH":
            result_batch[self.output_field_name] = synthetic_values

        return result_batch

    def _process_value(self, value: Any) -> Any:
        """
        Processes a single value using the generator.

        Parameters:
        -----------
        value : Any
            Original value to process

        Returns:
        --------
        Any
            Synthetic value
        """
        if self.consistency_mechanism == "mapping":
            # Check if mapping exists
            existing_mapping = self.mapping_store.get_mapping(self.field_name, value)
            if existing_mapping is not None:
                return existing_mapping

            # Generate new value
            synthetic_value = self.generator.generate_like(value, **self.generator_params)

            # Store mapping
            self.mapping_store.add_mapping(self.field_name, value, synthetic_value)

            return synthetic_value
        else:  # Using PRGN (default)
            # Generate consistent value based on original
            return self.generator.generate_like(value, **self.generator_params)

    def _collect_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Collects metrics for the generator operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Processed DataFrame

        Returns:
        --------
        Dict[str, Any]
            Metrics for the operation
        """
        metrics = super()._collect_metrics(df)

        # Add generator-specific metrics
        metrics["generator"] = {
            "type": self.generator.__class__.__name__,
            "consistency_mechanism": self.consistency_mechanism,
        }

        # Add mapping metrics if using mapping mechanism
        if self.consistency_mechanism == "mapping":
            field_mappings = self.mapping_store.get_all_mappings_for_field(self.field_name)
            metrics["mapping"] = {
                "total_mappings": len(field_mappings),
            }

        # Add generator-specific metrics to dictionary_metrics
        if hasattr(self.generator, "get_dictionary_info"):
            try:
                dictionary_info = self.generator.get_dictionary_info()
                if dictionary_info:
                    metrics["dictionary_metrics"].update(dictionary_info)
            except Exception as e:
                logger.warning(f"Error getting dictionary info: {str(e)}")

        # Add operation parameters
        operation_params = {
            "field_name": self.field_name,
            "mode": self.mode,
            "consistency_mechanism": self.consistency_mechanism,
            "mapping_store": self.mapping_store if self.consistency_mechanism == "mapping" else None
        }

        # Get additional metrics from collector
        if self._original_df is not None:
            # Extract original and generated data series
            orig_series = self._original_df[self.field_name]

            if self.mode == "REPLACE":
                gen_series = df[self.field_name]
            elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                gen_series = df[self.output_field_name]
            else:
                gen_series = None

            # Update transformation_metrics with additional parameters
            if "transformation_metrics" in metrics and orig_series is not None and gen_series is not None:
                compare_metrics = self._metrics_collector.compare_distributions(orig_series, gen_series)

                if "transformation_metrics" not in metrics:
                    metrics["transformation_metrics"] = {}

                metrics["transformation_metrics"].update(compare_metrics)

        return metrics

    def execute(self, data_source: Any, task_dir: Path, reporter: Any, progress_tracker: Optional[HierarchicalProgressTracker] = None, **kwargs) -> OperationResult:
        """
        Executes the generator operation.

        Parameters:
        -----------
        data_source : Any
            Source of data for processing
        task_dir : Path
            Directory for storing operation artifacts
        reporter : Any
            Reporter for operation progress and results
        **kwargs : dict
            Additional operation parameters

        Returns:
        --------
        OperationResult
            Operation result
        """
        result = super().execute(data_source, task_dir, reporter, progress_tracker, **kwargs)

        # If operation was successful and we're using mapping mechanism, save mappings
        if result.status == OperationStatus.SUCCESS and self.consistency_mechanism == "mapping":
            try:
                # Preserve the maps directory for detailed mappings
                maps_dir = ensure_directory(task_dir / "maps")

                # Create a copy of mappings in maps directory for detailed analysis
                maps_detail_file = maps_dir / f"{self.field_name}_mappings.json"

                # Convert mappings to serializable format
                field_mappings = self.mapping_store.get_all_mappings_for_field(self.field_name)

                serializable_mappings = []
                for original, synthetic in field_mappings.items():
                    serializable_mappings.append({
                        "original": original,
                        "synthetic": synthetic,
                        "field": self.field_name
                    })

                # Save mappings directly to task_dir
                mappings_file = task_dir / f"{self.name}_{self.field_name}_mappings.json"
                write_json(serializable_mappings, mappings_file)

                # Add artifact to result
                result.add_artifact(
                    path=mappings_file,
                    artifact_type="json",
                    description=f"Mappings for {self.field_name} field",
                    category=Constants.Artifact_Category_Mapping
                )

                # Also save to detailed maps directory for analysis
                write_json(serializable_mappings, maps_detail_file)
                logger.info(f"Saved {len(serializable_mappings)} mappings to {mappings_file} and {maps_detail_file}")

            except Exception as e:
                logger.warning(f"Failed to save mappings: {str(e)}")

        return result

    def handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles NULL values in the data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with NULL values

        Returns:
        --------
        pd.DataFrame
            DataFrame with handled NULL values
        """
        # For GeneratorOperation, we handle NULLs during batch processing
        # except for the ERROR strategy which requires immediate handling
        if self.null_strategy == NullStrategy.ERROR:
            null_count = df[self.field_name].isna().sum()
            if null_count > 0:
                raise ValidationError(f"Found {null_count} NULL values in field {self.field_name}")

        elif self.null_strategy == NullStrategy.EXCLUDE:
            # Filter out NULL values
            return df[df[self.field_name].notna()].copy()

        # For PRESERVE and REPLACE, handle during batch processing
        return df


try:
    from pamola_core.utils.ops.op_registry import register

    # Использование декоратора без параметра 'category', так как он не поддерживается
    @register()
    class RegisteredBaseOperation(BaseOperation):
        name = "fake_data_base"
        description = "Base operation for fake data operations"
        # Устанавливаем категорию как атрибут класса вместо параметра декоратора
        category = "fake_data"

    @register()
    class RegisteredFieldOperation(FieldOperation):
        name = "fake_data_field"
        description = "Base operation for fake data field operations"
        category = "fake_data"

    @register()
    class RegisteredGeneratorOperation(GeneratorOperation):
        name = "fake_data_generator"
        description = "Operation for generating fake data using generators"
        category = "fake_data"

except ImportError:
    logger.warning("PAMOLA operation registry not available. Operations will not be registered.")