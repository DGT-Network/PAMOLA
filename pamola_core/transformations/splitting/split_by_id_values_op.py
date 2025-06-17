from datetime import datetime
import hashlib
import json
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import numpy as np
import pandas as pd
import matplotlib

# Set the backend to 'Agg' to avoid GUI issues
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.io import load_data_operation, ensure_directory, load_settings_operation, write_json, write_dataframe_to_csv
from pamola_core.utils.logging import configure_task_logging
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.ops.op_registry import register
from pamola_core.common.constants import Constants
from pamola_core.utils.visualization import _save_figure
from pamola_core.utils.io_helpers import crypto_utils, directory_utils

class PartitionMethod(Enum):
    EQUAL_SIZE = "equal_size"
    RANDOM = "random"
    MODULO = "modulo"

class OutputFormat(Enum):
    CSV = "csv"
    JSON = "json"

@register(version="1.0.0")
class SplitByIDValuesOperation(TransformationOperation):

    def __init__(self,
                 name: str = "split_by_id_values_operation",
                 description: str = "Split dataset by ID values",
                 id_field: str = None,
                 value_groups: Optional[Dict[str, List[Any]]] = None,
                 number_of_partitions: int = 0,
                 partition_method: str = PartitionMethod.EQUAL_SIZE.value,  # "equal_size", "random", "modulo"
                 output_format: str = OutputFormat.CSV.value,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 **kwargs):
        """
        Initialize a SplitByIDValuesOperation instance.

        This constructor sets up the configuration for splitting a dataset either by
        specific ID values (value_groups) or by automatic partitioning methods.

        Parameters
        ----------
        name : str, optional
            The name of the operation (default: "split_by_id_values_operation").
        description : str, optional
            A brief description of the operation (default: "Split dataset by ID values").
        id_field : str, optional
            The name of the field used to identify records uniquely.
        value_groups : dict[str, list], optional
            A dictionary that maps group names to lists of ID values to be included in each group.
            Example: {"group1": [1, 2, 3], "group2": [4, 5, 6]}.
        number_of_partitions : int, optional
            The number of partitions to divide the dataset into when `value_groups` is not provided.
            If 0, no automatic partitioning is applied (default: 0).
        partition_method : str, optional
            Method for partitioning data when `number_of_partitions` > 0.
            Available options:
                - "equal_size": even distribution of records across partitions.
                - "random": records are randomly assigned to partitions.
                - "modulo": partition assignment is based on ID % number_of_partitions.
            Default is "equal_size".
        output_format : str, optional
            Output format for the resulting files (e.g., "csv", "json"). Default is "csv".
        **kwargs : dict
        """

        # Initialize attributes specific to SplitByIDValuesOperation
        self.id_field = id_field
        self.value_groups = value_groups or {}
        self.number_of_partitions = number_of_partitions
        self.partition_method = partition_method
        self.output_format = output_format
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        
        # Call the parent constructor with extracted arguments
        super().__init__(
            name=name,
            description=description,
            use_encryption=self.use_encryption,
            encryption_key=self.encryption_key,
            )

    def execute(self, data_source: DataSource, task_dir: Path, reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None, **kwargs):
        """
        Execute the SplitByIDValuesOperation to split a dataset based on ID values.

        This method supports two main splitting modes:
        1. Explicit splitting using predefined groups of ID values (`value_groups`).
        2. Automatic partitioning based on the specified number of partitions and partition method.

        The execution flow includes logging, caching, optional encryption,
        parallel processing with Dask (if enabled), visualization generation,
        and saving the output partitions to disk.

        Parameters
        ----------
        data_source : DataSource
            The data source object used to load the input dataset.
        task_dir : Path
            Directory path where outputs, logs, visualizations, and other artifacts are stored.
        reporter : Any
            Reporting or logging object to record execution status and messages.
        progress_tracker : ProgressTracker, optional
            An optional progress tracker to update stepwise progress status.
        **kwargs : dict, optional
            Overrides for instance attributes and execution configuration. Supported keys:
                - id_field (str): Column name used for identifying records.
                - value_groups (dict[str, list]): Mapping of group names to lists of ID values.
                - number_of_partitions (int): Number of partitions for automatic splitting.
                - partition_method (str): Partitioning strategy; one of
                    "equal_size" (default), "random", or "modulo".
                - output_format (str): Output file format, e.g., "csv" or "json".
                - force_recalculation (bool): If True, ignores cached results and recomputes.
                - generate_visualization (bool): Whether to generate output visualizations.
                - include_timestamp (bool): Whether to append a timestamp to output filenames.
                - save_output (bool): Whether to save partitioned outputs to disk.
                - parallel_processes (int): Number of processes to use for saving output.
                - use_cache (bool): Enable caching of intermediate results.
                - dataset_name (str): Name of the dataset to load from the data source.
                - use_dask (bool): If True, enable Dask for parallel/distributed processing.
                - use_encryption (bool): If True, encrypt output files.
                - encryption_key (str or Path): Encryption key or path for encrypting outputs.
                - batch_size (int): Number of records processed per batch during processing.

        Returns
        -------
        OperationResult
            An object summarizing the execution outcome, including:
                - Status (success/failure)
                - Execution duration
                - Paths to saved files and visualizations
                - Collected metrics
                - Error messages, if any
        """

        try:
            self.logger = kwargs.get('logger', self.logger)
            self.logger.info("Starting SplitByIDValuesOperation execution")

            step_count = 0
            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Preparation", "operation": self.name})
                step_count += 1

            self.start_time = time.time()

            self._set_common_operation_parameters(**kwargs)

            directories = self._prepare_directories(task_dir)
            self.logger.info(f"Output directories prepared: {directories}")

            self.save_config(task_dir)
            self.logger.info("Configuration file saved")

            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Loading dataset", "operation": self.name})
                step_count += 1

            # Load data
            dataset_name = kwargs.get('dataset_name', "main")
            self.logger.info(f"Loading dataset '{dataset_name}' from data source")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)

            if df is None or df.empty:
                self.logger.error("Loaded DataFrame is None or empty — aborting")
                return OperationResult(status=OperationStatus.ERROR, error_message="Input data frame is None or empty")

            self.logger.info(f"Loaded dataset '{dataset_name}' with shape {df.shape}")

            self.input_dataset = dataset_name
            self.original_df = df.copy(deep=True)

            # Handle caching
            if self.use_cache and not self.force_recalculation:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Use cache", "operation": self.name})
                    step_count += 1

                cached_result = self._get_cache(df.copy())  # _get_cache now returns OperationResult or None
                if cached_result is not None and isinstance(cached_result, OperationResult):
                    self.logger.info("Loaded result from cache — skipping execution.")
                    return cached_result
                else:
                    self.logger.info("No valid cached result found — proceeding with execution.")

            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Validating parameters", "operation": self.name})
                step_count += 1

            if not self._validate_parameters(df):
                self.logger.error("Validation failed: Input parameters are invalid")
                return OperationResult(status=OperationStatus.ERROR, error_message="Input parameters are invalid")

            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Processing data", "operation": self.name})
                step_count += 1

            result = OperationResult(status=OperationStatus.SUCCESS,
                                     artifacts=[],
                                     metrics={},
                                     error_message=None, execution_time=0,
                                     error_trace=None)

            self.logger.info("Processing data into field-based subsets")
            transformed_df = self._process_data(df, **kwargs)
            self.logger.info(f"Data split into {len(transformed_df)} subset(s)")

            self.end_time = time.time()
            result.execution_time = self.end_time - self.start_time

            if progress_tracker:
                progress_tracker.update(step_count, {"step": "Collecting metrics", "operation": self.name})
                step_count += 1

            metrics = self._collect_metrics(df, transformed_df)
            result.metrics = metrics
            self._save_metrics(metrics, task_dir, result, **kwargs)

            if self.save_output:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Saving output", "operation": self.name})
                    step_count += 1

                self.logger.info("Saving output subsets to disk")
                self._save_output(transformed_df, task_dir, result, **kwargs)
                self.logger.info("Output files saved")

            if self.generate_visualization:
                if progress_tracker:
                    progress_tracker.update(step_count, {"step": "Generating visualizations", "operation": self.name})
                    step_count += 1

                self.logger.info("Generating visualizations")
                self._generate_visualizations(df, transformed_df, task_dir, result, **kwargs)
                self.logger.info("Visualizations completed")

            self.logger.info("SplitByIDValuesOperation completed successfully")

            if reporter:
                reporter.add_operation("SplitByIDValuesOperation", status="info",
                                       details={"message": "SplitByIDValuesOperation completed successfully"})

            if self.use_cache:
                self._save_cache(task_dir, result)
                self.logger.info("Cached result saved successfully in task directory.")

            return result

        except Exception as e:
            error_msg = f"Error in SplitByIDValuesOperation: {e}"
            self.logger.exception(error_msg)

            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            if reporter:
                reporter.add_operation("SplitByIDValuesOperation", status="error", details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_msg
            )

    def _process_data(self, df: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Split dataset horizontally based on values in the id_field.

        Two modes:
        1. If value_groups is specified: split by explicit ID value lists.
        2. If number_of_partitions > 0: split into partitions by method.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Mapping from subset name to subset DataFrame.
        """
        subsets = {}

        # Mode 1: value_groups splitting
        if self.value_groups:
            for group_name, values in self.value_groups.items():
                mask = df[self.id_field].isin(values)
                subset_df = df[mask].copy()
                subsets[group_name] = subset_df
            # Optional: group for "others" if some IDs don't belong to any group
            all_values = [v for vals in self.value_groups.values() for v in vals]
            others_mask = ~df[self.id_field].isin(all_values)
            if others_mask.any():
                subsets["others"] = df[others_mask].copy()

        # Mode 2: partition splitting
        elif self.number_of_partitions > 0:
            if self.partition_method == PartitionMethod.EQUAL_SIZE.value:
                # Sort by id_field, split roughly equal-sized chunks
                sorted_df = df.sort_values(by=self.id_field)
                partition_sizes = np.full(self.number_of_partitions, len(sorted_df) // self.number_of_partitions)
                partition_sizes[:len(sorted_df) % self.number_of_partitions] += 1
                start_idx = 0
                for i, size in enumerate(partition_sizes):
                    end_idx = start_idx + size
                    subset_df = sorted_df.iloc[start_idx:end_idx].copy()
                    subsets[f"partition_{i}"] = subset_df
                    start_idx = end_idx

            elif self.partition_method == PartitionMethod.RANDOM.value:
                # Randomly assign rows to partitions
                np.random.seed(42)
                partitions = np.random.choice(self.number_of_partitions, size=len(df))
                for i in range(self.number_of_partitions):
                    subset_df = df[partitions == i].copy()
                    subsets[f"partition_{i}"] = subset_df

            elif self.partition_method == PartitionMethod.MODULO.value:
                # Partition by modulo of hash of id_field values
                def modulo_partition(val):
                    h = hash(val)
                    return h % self.number_of_partitions

                partitions = df[self.id_field].apply(modulo_partition)
                for i in range(self.number_of_partitions):
                    subset_df = df[partitions == i].copy()
                    subsets[f"partition_{i}"] = subset_df

        else:
            # If no splitting requested, return entire dataframe as one subset
            subsets["all_data"] = df.copy()

        return subsets

    def _collect_metrics(self,
                         input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                         output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Collect operation-specific metrics for SplitByIDValuesOperation and return in structured format.
        """
        if not isinstance(output_data, dict):
            self.logger.warning("Transformed data is not in expected dictionary format.")
            return {}

        input_rows = input_data.shape[0] if isinstance(input_data, pd.DataFrame) else 0
        input_fields = input_data.shape[1] if isinstance(input_data, pd.DataFrame) else 0

        total_output_records = sum(len(df) for df in output_data.values())
        total_output_fields = sum(len(df.columns) for df in output_data.values())

        split_info = {
            name: {
                "field_count": len(df.columns),
                "included_fields": list(df.columns)
            }
            for name, df in output_data.items()
        }

        return {
            "operation_type": self.__class__.__name__,
            "input_dataset": self.input_dataset or "unknown.csv",
            "total_input_records": input_rows,
            "total_input_fields": input_fields,
            "total_output_records": total_output_records,
            "total_output_fields": total_output_fields,
            "id_field": self.id_field,
            "number_of_splits": len(output_data),
            "split_info": split_info,
            "execution_time_seconds": round(self.end_time - self.start_time, 2),
            "processing_date": datetime.now().isoformat()
        }

    def _save_metrics(self, metrics: Dict[str, Any], task_dir: Path, result: OperationResult, **kwargs) -> Path:
        """
        Save the structured metrics dictionary to a JSON file in the task directory.
        """
        metrics_dir = task_dir / "metrics"
        ensure_directory(metrics_dir)

        operation_name = self.__class__.__name__
        metrics_filename = f"{operation_name}_metrics_{self.timestamp}.json" if self.timestamp else f"{operation_name}_metrics.json"
        metrics_path = metrics_dir / metrics_filename

        try:
            use_encryption = kwargs.get('use_encryption', False)
            encryption_key= kwargs.get('encryption_key', None) if use_encryption else None
            write_json(metrics, metrics_path, encryption_key=encryption_key)
            
            result.add_artifact(
                artifact_type="json",
                path=metrics_path,
                description=f"Metrics for {operation_name} saved at {self.timestamp}",
                category=Constants.Artifact_Category_Metrics
            )

            self.logger.info(f"Structured metrics saved successfully to {metrics_path}")
            return metrics_path

        except Exception as e:
            self.logger.error(f"Error saving structured metrics to {metrics_path}: {e}")
            raise

    def _save_output(self, result_subsets: dict[str, pd.DataFrame], task_dir: Path, result: OperationResult, **kwargs):
        """
        Save the processed subsets to output files and record artifact paths.

        Parameters
        ----------
        result_subsets : dict[str, pd.DataFrame]
            Dictionary mapping dataset names to DataFrames.
        task_dir : Path
            Root task directory for saving outputs.
        result : OperationResult
            Result object to append artifact file paths.
        """
        output_dir = task_dir / "output"
        ensure_directory(output_dir)

        for dataset_name, df in result_subsets.items():
            filename = f"{dataset_name}_{self.timestamp}.{self.output_format}" if self.timestamp else f"{dataset_name}.{self.output_format}"
            output_path = output_dir / filename

            try:
                use_encryption = kwargs.get('use_encryption', False)
                encryption_key= kwargs.get('encryption_key', None) if use_encryption else None
                if self.output_format == OutputFormat.CSV.value:
                    write_dataframe_to_csv(df=df, file_path=output_path, encryption_key=encryption_key)
                elif self.output_format == OutputFormat.JSON.value:
                    if encryption_key:
                        file_path = Path(output_path)
                        temp_dir = file_path.parent / "temp"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        temp_destination_path = temp_dir / f"decrypted_{file_path.name}"
                        
                        df.to_json(temp_destination_path, orient="records", lines=True)
                        
                        crypto_utils.encrypt_file(
                            source_path=temp_destination_path,
                            destination_path=output_path,
                            key=encryption_key
                        )
                        directory_utils.safe_remove_temp_file(temp_destination_path)
                    else:
                        df.to_json(temp_destination_path, orient="records", lines=True)
                else:
                    self.logger.warning(f"Unsupported output format: {self.output_format}")
                    continue

                self.logger.info(f"Saved output: {output_path}")
                result.add_artifact(
                    artifact_type=self.output_format,
                    path=output_path,
                    description=f"Output for {dataset_name} saved at {self.timestamp}" if self.timestamp else f"Output for {dataset_name}",
                    category=Constants.Artifact_Category_Output
                )

            except Exception as e:
                self.logger.error(f"Failed to save {dataset_name} to {output_path}: {e}")

    def _generate_visualizations(self,
                                 input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                 output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                 task_dir: Path,
                                 result: OperationResult,
                                 **kwargs) -> None:
        """
        Generate visualizations for SplitByIDValuesOperation, only if output_data is a dictionary of DataFrames.
        """
        if not isinstance(output_data, dict) or not output_data:
            self.logger.warning("Skipping visualization: output_data is not a non-empty dictionary of DataFrames.")
            return

        vis_dir = task_dir / "visualizations"
        ensure_directory(vis_dir)

        suffix = f"_{self.timestamp}" if self.timestamp else ""
        operation = self.__class__.__name__

        bar_path = self._plot_record_count_bar_chart(output_data, vis_dir, suffix, operation, **kwargs)
        if bar_path:
            result.add_artifact(
                artifact_type="png",
                path=bar_path,
                description="Bar chart showing record count per subset",
                category=Constants.Artifact_Category_Visualization
            )

        pie_path = self._plot_record_distribution_pie_chart(output_data, vis_dir, suffix, operation, **kwargs)
        if pie_path:
            result.add_artifact(
                artifact_type="png",
                path=pie_path,
                description="Pie chart showing record distribution across subsets",
                category=Constants.Artifact_Category_Visualization
            )

        id_dist_path = self._plot_id_value_distribution(output_data, vis_dir, suffix, operation, **kwargs)
        if id_dist_path:
            result.add_artifact(
                artifact_type="png",
                path=id_dist_path,
                description="Distribution of ID values across subsets (stacked bar chart)",
                category=Constants.Artifact_Category_Visualization
            )

    def _plot_record_count_bar_chart(self, output_data: Dict[str, pd.DataFrame], vis_dir: Path, suffix: str, operation: str, **kwargs) -> Path:
        counts = {k: len(df) for k, df in output_data.items()}

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax)
        ax.set_title("Record Count per Subset")
        ax.set_xlabel("Subset")
        ax.set_ylabel("Number of Records")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=45)

        path = vis_dir / f"{operation}_record_count_bar_chart{suffix}.png"
        plt.tight_layout()
        _save_figure(fig, path, **kwargs)

        return path

    def _plot_record_distribution_pie_chart(self, output_data: Dict[str, pd.DataFrame], vis_dir: Path, suffix: str, operation: str, **kwargs) -> Path:
        counts = {k: len(df) for k, df in output_data.items()}
        labels = list(counts.keys())
        sizes = list(counts.values())

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title("Record Distribution Across Subsets")

        path = vis_dir / f"{operation}_record_distribution_pie_chart{suffix}.png"
        plt.tight_layout()
        _save_figure(fig, path, **kwargs)

        return path

    def _plot_id_value_distribution(self, output_data: Dict[str, pd.DataFrame], vis_dir: Path, suffix: str, operation: str, **kwargs) -> Path:
        id_values_per_subset = {
            subset_name: df[self.id_field].value_counts()
            for subset_name, df in output_data.items()
        }

        combined_df = pd.DataFrame(id_values_per_subset).fillna(0).sort_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        combined_df.T.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title("Distribution of ID Values Across Subsets")
        ax.set_xlabel("Subset")
        ax.set_ylabel("Count of ID Values")
        ax.legend(title="ID Values", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        path = vis_dir / f"{operation}_id_value_distribution{suffix}.png"
        _save_figure(fig, path, **kwargs)

        return path

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
                data_hash=self._generate_data_hash(self.original_df.copy())
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
            "value_groups": self.value_groups,
            "partition_method": self.partition_method,
            "number_of_partitions": self.number_of_partitions,
            "id_field": self.id_field,
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

    def _validate_parameters(self, df: pd.DataFrame) -> bool:
        all_columns = set(df.columns)

        # Check if the ID field exists
        if not self.id_field:
            self.logger.error("No ID field specified.")
            return False

        if self.id_field not in all_columns:
            self.logger.error(f"ID field '{self.id_field}' not found in the DataFrame.")
            return False

        # Validate value_groups or number_of_partitions
        if self.value_groups:
            # Ensure all values in value_groups exist in the DataFrame column
            id_values_in_df = set(df[self.id_field].unique())
            for group_name, values in self.value_groups.items():
                missing = [v for v in values if v not in id_values_in_df]
                if missing:
                    self.logger.warning(
                        f"Group '{group_name}' contains ID values not present in the DataFrame: {missing}"
                    )
        elif self.number_of_partitions <= 0:
            self.logger.error(
                "Either 'value_groups' must be specified or 'number_of_partitions' must be a positive integer."
            )
            return False

        # Validate partition_method
        if self.partition_method not in PartitionMethod._value2member_map_:
            self.logger.error(
                f"Unsupported partition method: {self.partition_method}. "
                f"Choose from {[m.value for m in PartitionMethod]}."
            )
            return False

        return True

    def _initialize_logger(self, task_dir: Path):
        """
        Initialize the logger for the operation using the class name as task_id.

        Parameters
        ----------
        task_dir : Path
            The base directory for the task, used to determine log directory.
        """
        task_id = self.__class__.__name__
        log_dir = task_dir / "logs"

        self.logger = configure_task_logging(
            task_id=task_id,
            log_level="INFO",
            log_dir=log_dir,
        )

    def _set_common_operation_parameters(self, **kwargs):
        """
        Set common configurable operation parameters from keyword arguments.
        """

        self.id_field = kwargs.get("id_field", getattr(self, "id_field", None))
        self.value_groups = kwargs.get("value_groups", getattr(self, "value_groups", None))
        self.number_of_partitions = kwargs.get("number_of_partitions", getattr(self, "number_of_partitions", 0))
        self.partition_method = kwargs.get("partition_method",
                                           getattr(self, "partition_method", PartitionMethod.EQUAL_SIZE.value))
        self.output_format = kwargs.get("output_format", getattr(self, "output_format", OutputFormat.CSV.value))

        self.force_recalculation = kwargs.get("force_recalculation", getattr(self, "force_recalculation", False))
        self.generate_visualization = kwargs.get("generate_visualization",
                                                 getattr(self, "generate_visualization", True))
        self.include_timestamp = kwargs.get("include_timestamp", getattr(self, "include_timestamp", True))
        self.save_output = kwargs.get("save_output", getattr(self, "save_output", True))

        self.parallel_processes = kwargs.get("parallel_processes", getattr(self, "parallel_processes", 1))
        self.batch_size = kwargs.get("batch_size", getattr(self, "batch_size", 10000))
        self.use_cache = kwargs.get("use_cache", getattr(self, "use_cache", False))
        self.use_dask = kwargs.get("use_dask", getattr(self, "use_dask", False))

        self.use_encryption = kwargs.get("use_encryption", getattr(self, "use_encryption", False))
        self.encryption_key = kwargs.get("encryption_key", getattr(self, "encryption_key", None))

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.include_timestamp else ""