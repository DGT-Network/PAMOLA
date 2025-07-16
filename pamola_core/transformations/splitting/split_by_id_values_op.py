from datetime import datetime
import hashlib
import json
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib
# Set the backend to 'Agg' to avoid GUI issues
matplotlib.use('Agg')
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.io import load_data_operation, ensure_directory, load_settings_operation, write_json, write_dataframe_to_csv
from pamola_core.utils.logging import configure_task_logging
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_registry import register
from pamola_core.common.constants import Constants
from pamola_core.utils.visualization import create_bar_plot, create_pie_chart, create_heatmap
from pamola_core.utils.io_helpers import crypto_utils, directory_utils
import dask.dataframe as dd
from joblib import Parallel, delayed
from pamola_core.utils.io_helpers.crypto_utils import get_encryption_mode

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
                 id_field: Optional[str] = None,
                 value_groups: Optional[Dict[str, List[Any]]] = None,
                 number_of_partitions: int = 0,
                 partition_method: str = PartitionMethod.EQUAL_SIZE.value,  # "equal_size", "random", "modulo"
                 output_format: str = OutputFormat.CSV.value,
                 save_output: bool = True,
                 use_cache: bool = True,
                 force_recalculation: bool = False,
                 use_dask: bool = False,
                 npartitions: int = 1,
                 use_vectorization: bool = False,
                 parallel_processes: int = 1,
                 visualization_backend: Optional[str] = None,
                 visualization_theme: Optional[str] = None,
                 visualization_strict: bool = False,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 encryption_mode: Optional[str] = None,
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
        
        # Call the parent constructor with extracted arguments
        super().__init__(
            name=name,
            description=description,
            use_cache=use_cache,
            use_dask=use_dask,
            npartitions=npartitions,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            visualization_backend=visualization_backend,
            visualization_theme=visualization_theme,
            visualization_strict=visualization_strict,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            encryption_mode=encryption_mode
        )

        # Initialize attributes specific to SplitByIDValuesOperation
        self.id_field = id_field
        self.value_groups = value_groups or {}
        self.number_of_partitions = number_of_partitions
        self.partition_method = partition_method
        self.output_format = output_format
        self.save_output = save_output
        self.force_recalculation = force_recalculation

    def execute(self, data_source: DataSource, task_dir: Path, reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None, **kwargs):
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

        self.start_time = time.time()

        caller_operation = self.__class__.__name__
        self.logger = kwargs.get('logger', self.logger)

        try:
            # Start operation
            self.logger.info(f"Operation: {caller_operation}, Start operation")
            if progress_tracker:
                progress_tracker.total = self._compute_total_steps(**kwargs)
                progress_tracker.update(1, {"step": "Start operation - Preparation", "operation": caller_operation})

            dirs = self._prepare_directories(task_dir)

            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Preparation",
                                                "message": "Preparation successfully",
                                                "directories": {k: str(v) for k, v in dirs.items()}
                                       })

            # Load data and validate input parameters
            self.logger.info(f"Operation: {caller_operation}, Load data and validate input parameters")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Load data and validate input parameters", "operation": caller_operation})

            df, is_valid = self._load_data_and_validate_input_parameters(data_source, **kwargs)

            if is_valid:
                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Load data and validate input parameters",
                                                    "message": "Load data and validate input parameters successfully",
                                                    "shape": df.shape
                                           })
            else:
                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Load data and validate input parameters",
                                                    "message": "Load data and validate input parameters failed"
                                           })
                    return OperationResult(status=OperationStatus.ERROR,
                                           error_message="Load data and validate input parameters failed")

            # Handle cache if required
            if self.use_cache and not self.force_recalculation:
                self.logger.info(f"Operation: {caller_operation}, Load result from cache")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Load result from cache", "operation": caller_operation})

                cached_result = self._get_cache(df.copy(), **kwargs)  # _get_cache now returns OperationResult or None
                if cached_result is not None and isinstance(cached_result, OperationResult):
                    if reporter:
                        reporter.add_operation(f"Operation {caller_operation}", status="info",
                                               details={"step": "Load result from cache",
                                                        "message": "Load result from cache successfully"
                                               })
                    return cached_result
                else:
                    self.logger.info(f"Operation: {caller_operation}, Load result from cache failed — proceeding with execution.")
                    if reporter:
                        reporter.add_operation(f"Operation {caller_operation}", status="info",
                                               details={"step": "Load result from cache",
                                                        "message": "Load result from cache failed - proceeding with execution"
                                               })

            # Process data
            self.logger.info(f"Operation: {caller_operation}, Process data")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Process data", "operation": caller_operation})

            transformed_df = self._process_data(df, **kwargs)

            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Process data",
                                                "message": "Process data successfully",
                                                "num_subsets": len(transformed_df)
                                       })

            result = OperationResult(status=OperationStatus.SUCCESS,
                                     artifacts=[],
                                     metrics={},
                                     error_message=None, execution_time=0,
                                     error_trace=None)

            self.end_time = time.time()
            result.execution_time = self.end_time - self.start_time

            # Collect metric
            self.logger.info(f"Operation: {caller_operation}, Collect metric")
            if progress_tracker:
                progress_tracker.update(1, {"step": "Collect metric", "operation": caller_operation})

            metrics = self._collect_metrics(df, transformed_df)
            result.metrics = metrics
            self._save_metrics(metrics, task_dir, result, **kwargs)

            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Collect metric",
                                                "message": "Collect metric successfully",
                                                "summary": {
                                                    "input_dataset": metrics.get("input_dataset"),
                                                    "input_records": metrics.get("total_input_records"),
                                                    "input_fields": metrics.get("total_input_fields"),
                                                    "output_records": metrics.get("total_output_records"),
                                                    "output_fields": metrics.get("total_output_fields"),
                                                    "id_field": metrics.get("id_field"),
                                                    "splits": metrics.get("number_of_splits"),
                                                    "execution_time_seconds": metrics.get("execution_time_seconds")
                                                }
                                       })

            # Save output if required
            if self.save_output:
                self.logger.info(f"Operation: {caller_operation}, Save output")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Save output", "operation": caller_operation})

                self._save_output(transformed_df, task_dir, result, **kwargs)

                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Save output",
                                                    "message": "Save output successfully",
                                                    "files_saved": len(result.artifacts)
                                           })

            # Generate visualizations if required
            if self.generate_visualization:
                self.logger.info(f"Operation: {caller_operation}, Generate visualizations")
                if progress_tracker:
                    progress_tracker.update(1,{"step": "Generate visualizations", "operation": caller_operation})

                self._generate_visualizations(df, transformed_df, task_dir, result, **kwargs)

                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Generate visualizations",
                                                    "message": "Generate visualizations successfully",
                                                    "num_images": len([a for a in result.artifacts if a.get("artifact_type") == "png"])
                                           })

            # Save cache if required
            if self.use_cache:
                self.logger.info(f"Operation: {caller_operation}, Save cache")
                if progress_tracker:
                    progress_tracker.update(1, {"step": "Save cache", "operation": caller_operation})

                self._save_cache(task_dir, result, **kwargs)

                if reporter:
                    reporter.add_operation(f"Operation {caller_operation}", status="info",
                                           details={"step": "Save cache",
                                                    "message": "Save cache successfully"
                                           })

            # Operation completed successfully
            self.logger.info(f"Operation: {caller_operation}, Operation completed successfully.")
            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="info",
                                       details={"step": "Return result",
                                                "message": "Operation completed successfully"
                                       })

            return result

        except Exception as e:
            self.logger.error(f"Operation: {caller_operation}, error occurred: {e}")
            if reporter:
                reporter.add_operation(f"Operation {caller_operation}", status="error",
                                       details={
                                            "step": "Exception",
                                            "message": "Operation failed due to an exception",
                                            "error": str(e)
                                       })

            return OperationResult(status=OperationStatus.ERROR, error_message=str(e))


    def _process_data(self, df: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Main entry point to split a DataFrame into subsets based on ID values or partition strategy.

        The method automatically chooses an execution strategy depending on:
        - use_dask: for distributed processing with Dask.
        - use_vectorization: for parallel batch processing with Joblib.
        - Fallback to standard Pandas processing otherwise.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to be split.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping partition names to DataFrame subsets.
        """

        # Dask should only be used with partition methods like MODULO or RANDOM.
        # Other methods (e.g., EQUAL_SIZE or value_groups) are better handled with Pandas or Joblib
        # Dask is not efficient for index-based slicing or group-based filtering
        if self.use_dask and self.npartitions > 1 and self.partition_method in {
            PartitionMethod.MODULO.value, PartitionMethod.RANDOM.value
        }:
            return self._process_with_dask(df)

        # Joblib should only be used when value_groups are defined
        # It is not suitable for partition methods like MODULO, RANDOM, or EQUAL_SIZE
        # Since those can be handled more efficiently with Pandas or Dask
        elif self.use_vectorization and self.parallel_processes > 0 and self.value_groups:
            return self._process_with_joblib(df)

        else:
            return self._process_with_pandas(df)

    def _process_with_dask(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the DataFrame into partitions using Dask for scalable processing.

        Supported partition methods:
        - MODULO: partition based on hash(id) % number_of_partitions
        - RANDOM: random distribution into partitions

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to be partitioned.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary where keys are partition names (e.g., "partition_0") and values are DataFrame subsets.
        """
        subsets = {}
        ddf = dd.from_pandas(df, npartitions=self.npartitions)

        if self.partition_method == PartitionMethod.MODULO.value:
            def apply_modulo(part):
                part["_partition"] = part[self.id_field].apply(lambda x: hash(x) % self.number_of_partitions)
                return part

            ddf = ddf.map_partitions(apply_modulo)

        elif self.partition_method == PartitionMethod.RANDOM.value:
            def apply_random_partition(part):
                np.random.seed(42)
                part["_partition"] = np.random.choice(self.number_of_partitions, size=len(part))
                return part

            ddf = ddf.map_partitions(apply_random_partition)

        for i in range(self.number_of_partitions):
            part_df = ddf[ddf["_partition"] == i].drop(columns=["_partition"]).compute()
            subsets[f"partition_{i}"] = part_df

        return subsets

    def _process_with_joblib(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the DataFrame based on explicit value_groups using Joblib for parallel filtering.

        Each group is defined by a list of values from the id_field.
        Any unmatched rows will be grouped under "others".

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to be split.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping group names to DataFrame subsets.
        """

        def filter_group(group_name, values):
            mask = df[self.id_field].isin(values)
            return group_name, df[mask].copy()

        results = Parallel(n_jobs=self.parallel_processes)(
            delayed(filter_group)(group_name, values)
            for group_name, values in self.value_groups.items()
        )
        subsets = dict(results)

        # Handle "others"
        all_values = [v for vals in self.value_groups.values() for v in vals]
        others_mask = ~df[self.id_field].isin(all_values)
        if others_mask.any():
            subsets["others"] = df[others_mask].copy()

        return subsets

    def _process_with_pandas(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the DataFrame using pure Pandas, based on value_groups or partition method.

        - If value_groups is provided: splits by defined groups and an optional "others" group.
        - If number_of_partitions is set: supports EQUAL_SIZE, RANDOM, or MODULO strategies.
        - If none of the above: returns full DataFrame under "all_data".

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to be split.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of split DataFrames by group or partition.
        """
        subsets = {}

        if self.value_groups:
            for group_name, values in self.value_groups.items():
                mask = df[self.id_field].isin(values)
                subsets[group_name] = df[mask].copy()

            all_values = [v for vals in self.value_groups.values() for v in vals]
            others_mask = ~df[self.id_field].isin(all_values)
            if others_mask.any():
                subsets["others"] = df[others_mask].copy()

        elif self.number_of_partitions > 0:
            if self.partition_method == PartitionMethod.EQUAL_SIZE.value:
                sorted_df = df.sort_values(by=self.id_field)
                partition_sizes = np.full(self.number_of_partitions, len(sorted_df) // self.number_of_partitions)
                partition_sizes[:len(sorted_df) % self.number_of_partitions] += 1

                start_idx = 0
                for i, size in enumerate(partition_sizes):
                    end_idx = start_idx + size
                    subsets[f"partition_{i}"] = sorted_df.iloc[start_idx:end_idx].copy()
                    start_idx = end_idx

            elif self.partition_method == PartitionMethod.RANDOM.value:
                np.random.seed(42)
                partitions = np.random.choice(self.number_of_partitions, size=len(df))
                for i in range(self.number_of_partitions):
                    subsets[f"partition_{i}"] = df[partitions == i].copy()

            elif self.partition_method == PartitionMethod.MODULO.value:
                partitions = df[self.id_field].apply(lambda x: hash(x) % self.number_of_partitions)
                for i in range(self.number_of_partitions):
                    subsets[f"partition_{i}"] = df[partitions == i].copy()

        else:
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
            "input_dataset": self._input_dataset or "unknown.csv",
            "total_input_records": input_rows,
            "total_input_fields": input_fields,
            "total_output_records": total_output_records,
            "total_output_fields": total_output_fields,
            "id_field": self.id_field,
            "number_of_splits": len(output_data),
            "split_info": split_info,
            "execution_time_seconds": self.end_time - self.start_time,
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
            encryption_key = kwargs.get('encryption_key', None) if use_encryption else None
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
                encryption_key = kwargs.get('encryption_key', None)
                encryption_mode = get_encryption_mode(df, **kwargs)
                if self.output_format == OutputFormat.CSV.value:
                    write_dataframe_to_csv(df=df, file_path=output_path, encryption_key=encryption_key, use_encryption=use_encryption, encryption_mode=encryption_mode)
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
                            key=encryption_key,
                            mode=encryption_mode
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

        if not isinstance(output_data, dict) or not output_data:
            self.logger.warning("Skipping visualization: output_data is not a non-empty dictionary of DataFrames.")
            return

        vis_dir = task_dir / "visualizations"
        ensure_directory(vis_dir)

        suffix = f"_{self.timestamp}" if self.timestamp else ""
        operation = self.__class__.__name__

        kwargs["backend"] = kwargs.pop("visualization_backend", self.visualization_backend)
        kwargs["theme"] = kwargs.pop("visualization_theme", self.visualization_theme)
        kwargs["strict"] = kwargs.pop("visualization_strict", self.visualization_strict)

        # 1. Bar chart: Record count per subset
        try:
            bar_data = {k: len(df) for k, df in output_data.items()}
            bar_path = vis_dir / f"{operation}_record_count_bar_chart{suffix}.png"
            bar_result = create_bar_plot(
                data=bar_data,
                output_path=bar_path,
                title="Record Count per Subset",
                orientation="v",
                x_label="Subset",
                y_label="Number of Records",
                **kwargs
            )
            if not bar_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=bar_result,
                    description="Bar chart showing record count per subset",
                    category=Constants.Artifact_Category_Visualization
                )
        except Exception as e:
            self.logger.error(f"Failed to create record count bar chart: {e}")

        # 2. Pie chart: Distribution across subsets
        try:
            pie_path = vis_dir / f"{operation}_record_distribution_pie_chart{suffix}.png"
            pie_result = create_pie_chart(
                data=bar_data,
                output_path=pie_path,
                title="Record Distribution Across Subsets",
                show_percentages=True,
                show_values=True,
                **kwargs
            )
            if not pie_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=pie_result,
                    description="Pie chart showing record distribution across subsets",
                    category=Constants.Artifact_Category_Visualization
                )
        except Exception as e:
            self.logger.error(f"Failed to create record distribution pie chart: {e}")

        # 3. Heatmap: ID value distribution across subsets
        try:
            id_values_per_subset = {
                subset_name: df[self.id_field].value_counts()
                for subset_name, df in output_data.items()
            }
            combined_df = pd.DataFrame(id_values_per_subset).fillna(0).sort_index()

            id_dist_path = vis_dir / f"{operation}_id_value_distribution{suffix}.png"
            heatmap_result = create_heatmap(
                data=combined_df.T,
                output_path=id_dist_path,
                title="Distribution of ID Values Across Subsets",
                x_label="ID Value",
                y_label="Subset",
                annotate=True,
                annotation_format=".0f",
                **kwargs
            )
            if not heatmap_result.startswith("Error"):
                result.add_artifact(
                    artifact_type="png",
                    path=heatmap_result,
                    description="Distribution of ID values across subsets (heatmap)",
                    category=Constants.Artifact_Category_Visualization
                )
        except Exception as e:
            self.logger.error(f"Failed to create ID value distribution heatmap: {e}")

    def _save_cache(self, task_dir: Path, result: OperationResult, **kwargs) -> None:
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
                "parameters": self._get_cache_parameters(**kwargs),
            }

            cache_key = operation_cache.generate_cache_key(
                operation_name=self.__class__.__name__,
                parameters=self._get_cache_parameters(**kwargs),
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

    def _get_cache(self, df: pd.DataFrame, **kwargs) -> Optional[OperationResult]:
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
                parameters=self._get_cache_parameters(**kwargs),
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

    def _get_cache_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Get operation-specific parameters for SplitByIDValuesOperation using external kwargs.

        Parameters
        ----------
        **kwargs : dict
            Dictionary of parameters that may override instance attributes.

        Returns
        -------
        Dict[str, Any]
            Dictionary of relevant parameters to identify the operation configuration.
        """
        return {
            "operation": self.__class__.__name__,
            "version": self.version,
            "id_field": kwargs.get("id_field"),
            "value_groups": kwargs.get("value_groups"),
            "number_of_partitions": kwargs.get("number_of_partitions"),
            "partition_method": kwargs.get("partition_method"),
            "output_format": kwargs.get("output_format"),
            "save_output": kwargs.get("save_output"),
            "use_cache": kwargs.get("use_cache"),
            "force_recalculation": kwargs.get("force_recalculation"),
            "use_dask": kwargs.get("use_dask"),
            "npartitions": kwargs.get("npartitions"),
            "use_vectorization": kwargs.get("use_vectorization"),
            "parallel_processes": kwargs.get("parallel_processes"),
            "visualization_backend": kwargs.get("visualization_backend"),
            "visualization_theme": kwargs.get("visualization_theme"),
            "visualization_strict": kwargs.get("visualization_strict"),
            "use_encryption": kwargs.get("use_encryption"),
            "encryption_key": str(kwargs.get("encryption_key")) if kwargs.get("encryption_key") else None
        }

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

    def _set_input_parameters(self, **kwargs):
        """
        Set common configurable operation parameters from keyword arguments.
        """

        self.id_field = kwargs.get("id_field", getattr(self, "id_field", None))
        self.value_groups = kwargs.get("value_groups", getattr(self, "value_groups", None))
        self.number_of_partitions = kwargs.get("number_of_partitions", getattr(self, "number_of_partitions", 0))
        self.partition_method = kwargs.get("partition_method",
                                           getattr(self, "partition_method", PartitionMethod.EQUAL_SIZE.value))

        self.generate_visualization = kwargs.get("generate_visualization",
                                                 getattr(self, "generate_visualization", True))
        self.save_output = kwargs.get("save_output", getattr(self, "save_output", True))
        self.output_format = kwargs.get("output_format", getattr(self, "output_format", OutputFormat.CSV.value))
        self.include_timestamp = kwargs.get("include_timestamp", getattr(self, "include_timestamp", True))

        self.use_cache = kwargs.get("use_cache", getattr(self, "use_cache", True))
        self.force_recalculation = kwargs.get("force_recalculation", getattr(self, "force_recalculation", False))

        self.use_dask = kwargs.get("use_dask", getattr(self, "use_dask", False))
        self.npartitions = kwargs.get("npartitions", getattr(self, "npartitions", 1))

        self.use_vectorization = kwargs.get("use_vectorization", getattr(self, "use_vectorization", False))
        self.parallel_processes = kwargs.get("parallel_processes", getattr(self, "parallel_processes", 1))

        self.visualization_backend = kwargs.get("visualization_backend", getattr(self, "visualization_backend", False))
        self.visualization_theme = kwargs.get("visualization_theme", getattr(self, "visualization_theme", None))
        self.visualization_strict = kwargs.get("visualization_strict", getattr(self, "visualization_strict", None))

        self.use_encryption = kwargs.get("use_encryption", getattr(self, "use_encryption", False))
        self.encryption_key = kwargs.get("encryption_key",
                                         getattr(self, "encryption_key", None)) if self.use_encryption else None

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.include_timestamp else ""

    def _validate_input_parameters(self, df: pd.DataFrame) -> bool:
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
            id_values_in_df = set(df[self.id_field].unique())
            for group_name, values in self.value_groups.items():
                missing = [v for v in values if v not in id_values_in_df]
                if missing:
                    self.logger.warning(
                        f"Group '{group_name}' contains ID values not present in the DataFrame: {missing}"
                    )
        else:
            if self.number_of_partitions <= 0:
                self.logger.error(
                    "Either 'value_groups' must be specified or 'number_of_partitions' must be a positive integer."
                )
                return False

            if self.partition_method not in PartitionMethod._value2member_map_:
                self.logger.error(
                    f"Unsupported partition method: {self.partition_method}. "
                    f"Choose from {[m.value for m in PartitionMethod]}."
                )
                return False

        return True

    def _load_data_and_validate_input_parameters(self, data_source: DataSource, **kwargs) -> Tuple[Optional[pd.DataFrame], bool]:
        self._set_input_parameters(**kwargs)

        dataset_name = kwargs.get('dataset_name', "main")
        settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
        df = load_data_operation(data_source, dataset_name, **settings_operation)

        if df is None or df.empty:
            self.logger.error("Error data frame is None or empty")
            return None, False

        self._input_dataset = dataset_name
        self._original_df = df.copy(deep=True)

        return df, self._validate_input_parameters(df)

    def _compute_total_steps(self, **kwargs) -> int:
        use_cache = kwargs.get("use_cache", self.use_cache)
        force_recalculation = kwargs.get("force_recalculation", self.force_recalculation)
        save_output = kwargs.get("save_output", self.save_output)
        generate_visualization = kwargs.get("generate_visualization", self.generate_visualization)

        steps = 0

        steps += 1  # Step 1: Preparation
        steps += 1  # Step 2: Load data and validate input

        if use_cache and not force_recalculation:
            steps += 1  # Step 3: Try to load from cache

        steps += 1  # Step 4: Process data
        steps += 1  # Step 5: Collect metrics

        if save_output:
            steps += 1  # Step 6: Save output

        if generate_visualization:
            steps += 1  # Step 7: Generate visualizations

        if use_cache:
            steps += 1  # Step 8: Save cache

        return steps