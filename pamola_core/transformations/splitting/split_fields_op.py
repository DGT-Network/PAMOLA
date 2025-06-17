import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import pandas as pd
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.io import load_data_operation, ensure_directory, load_settings_operation, write_json, write_dataframe_to_csv
from pamola_core.utils.logging import configure_task_logging
from pamola_core.utils.ops.op_cache import operation_cache
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
from pamola_core.utils.progress import ProgressTracker
from pamola_core.common.constants import Constants
from pamola_core.utils.visualization import _save_figure
from pamola_core.utils.io_helpers import crypto_utils, directory_utils

import matplotlib

# Set the backend to 'Agg' to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
import hashlib
from pamola_core.utils.ops.op_registry import register



class OutputFormat(Enum):
    CSV = "csv"
    JSON = "json"

@register(version="1.0.0")
class SplitFieldsOperation(TransformationOperation):

    def __init__(self,
                 name: str = "split_fields_operation",
                 description: str = "Split dataset by fields",
                 id_field: str = None,
                 field_groups: Optional[Dict[str, List[str]]] = None,
                 include_id_field: bool = True,
                 output_format: str = OutputFormat.CSV.value,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 **kwargs):
        """
        Initialize a SplitFieldsOperation instance.

        This operation splits a dataset into multiple groups of fields, optionally including
        an ID field in each group. Output files are saved in the specified format.

        Parameters
        ----------
        name : str, optional
            The name of the operation (default is "split_fields_operation").
        description : str, optional
            A short description of the operation (default is "Split dataset by fields").
        id_field : str, optional
            The field used to uniquely identify records in the dataset.
        field_groups : dict[str, list[str]], optional
            A dictionary mapping group names to lists of field names to be grouped together.
        include_id_field : bool, optional
            Whether to include the `id_field` in each output group (default is True).
        output_format : str, optional
            The format to save the output files (e.g., "csv", "json"). Default is "csv".
        **kwargs : dict
        """

        # Call the parent constructor with extracted arguments
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key
            )

        # Initialize attributes specific to SplitFieldsOperation
        self.id_field = id_field
        self.field_groups = field_groups or {}
        self.include_id_field = include_id_field
        self.output_format = output_format

    def execute(self, data_source: DataSource, task_dir: Path, reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None, **kwargs):
        """
        Execute the SplitFieldsOperation.

        This method performs the logic to split a dataset into multiple groups based on specified field groups.
        It supports optional features including caching, encryption, visualization, and progress tracking.
        Parameters can be overridden dynamically via keyword arguments to customize behavior per execution.

        Parameters
        ----------
        data_source : DataSource
            The input data source from which the dataset will be loaded.
        task_dir : Path
            The directory where output files, logs, visualizations, and other artifacts will be saved.
        reporter : Any
            A reporter or logger object used to report status, messages, or errors.
        progress_tracker : ProgressTracker, optional
            An optional object to track and update progress step-by-step, typically for UI or monitoring purposes.
        **kwargs : dict, optional
            Optional overrides for instance attributes and execution parameters:
                - id_field (str): Field name used to uniquely identify records.
                - field_groups (dict[str, list[str]]): Mapping of group names to lists of field names for splitting.
                - include_id_field (bool): Whether to include the id_field in each output group.
                - output_format (str): Format for saving output files (e.g., "csv", "json").
                - force_recalculation (bool): If True, bypass cached results and force full reprocessing.
                - generate_visualization (bool): Whether to generate visualizations for the output data.
                - include_timestamp (bool): Whether to append a timestamp to output filenames.
                - save_output (bool): Whether to save the resulting split datasets to disk.
                - parallel_processes (int): Number of parallel processes to use for output saving.
                - use_cache (bool): Whether to use cached results when available.
                - dataset_name (str): Optional name of the dataset to load from the data source.
                - use_dask (bool): Whether to use Dask for parallel/distributed processing.
                - use_encryption (bool): Whether to encrypt output files.
                - encryption_key (str or Path): Key or path to key file used for encryption.
                - batch_size (int): Number of records to process per batch during operation.

        Returns
        -------
        OperationResult
            An object summarizing the outcome of the operation, including:
                - Execution status (success or failure)
                - Execution time duration
                - Paths to saved output files and visualizations
                - Collected metrics and logs
                - Any errors encountered during processing
        """

        try:
            self.logger = kwargs.get('logger', self.logger)
            self.logger.info("Starting SplitFieldsOperation execution")

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

            self.logger.info("SplitFieldsOperation completed successfully")

            if reporter:
                reporter.add_operation("SplitFieldsOperation", status="info",
                                       details={"message": "SplitFieldsOperation completed successfully"})

            if self.use_cache:
                self._save_cache(task_dir, result)
                self.logger.info("Cached result saved successfully in task directory.")

            return result

        except Exception as e:
            error_msg = f"Error in SplitFieldsOperation: {e}"
            self.logger.exception(error_msg)

            if progress_tracker:
                progress_tracker.update(0, {"step": "Error", "error": str(e)})

            if reporter:
                reporter.add_operation("SplitFieldsOperation", status="error", details={"error": str(e)})

            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=error_msg
            )

    def _process_data(self, df: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Process data according to operation-specific logic.
        Split the dataset into multiple subsets based on field_groups.
        Each subset contains its specified columns, and optionally the ID field.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset to process.
        **kwargs : dict
            Additional parameters (not used here directly).

        Returns:
        --------
        Dict[str, pd.DataFrame]
            A dictionary where keys are group names and values are corresponding dataframes.
        """

        result_subsets = {}

        for group_name, fields in self.field_groups.items():
            selected_columns = fields.copy()

            # Add ID field if required and not already present
            if self.include_id_field and self.id_field and self.id_field not in selected_columns:
                selected_columns.insert(0, self.id_field)

            # Subset the DataFrame
            subset_df = df[selected_columns].copy()
            result_subsets[group_name] = subset_df

        return result_subsets

    def _collect_metrics(self,
                         input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                         output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Collect operation-specific metrics for SplitFieldsOperation and return in structured format.
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
            
            # Save metrics to file
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
        """Generate visualizations specific to this operation."""
        if not isinstance(output_data, dict) or not output_data:
            self.logger.warning("Skipping visualization: output_data is not a non-empty dictionary of DataFrames.")
            return

        vis_dir = task_dir / "visualizations"
        ensure_directory(vis_dir)

        suffix = f"_{self.timestamp}" if self.timestamp else ""
        operation = self.__class__.__name__

        bar_chart_path = self._plot_fields_per_subset_bar_chart(output_data, vis_dir, suffix, operation, **kwargs)
        if bar_chart_path:
            result.add_artifact(
                artifact_type="png",
                path=bar_chart_path,
                description="Bar chart showing number of fields in each subset",
                category=Constants.Artifact_Category_Visualization
            )

        network_path = self._plot_field_subset_network(output_data, vis_dir, suffix, operation, **kwargs)
        if network_path:
            result.add_artifact(
                artifact_type="png",
                path=network_path,
                description="Network diagram showing field distribution across subsets (Plotly)",
                category=Constants.Artifact_Category_Visualization
            )

        if isinstance(input_data, pd.DataFrame):
            schema_path = self._plot_schema_comparison(input_data, output_data, vis_dir, suffix, operation, **kwargs)
            if schema_path:
                result.add_artifact(
                    artifact_type="png",
                    path=schema_path,
                    description="Schema visualization of original vs. split datasets",
                    category=Constants.Artifact_Category_Visualization
                )

    def _plot_fields_per_subset_bar_chart(self, output_data: Dict[str, pd.DataFrame], vis_dir: Path, suffix: str, operation: str, **kwargs) -> Path:
        """Create a bar chart showing the number of fields in each subset."""
        subset_field_counts = {k: len(df.columns) for k, df in output_data.items()}

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(subset_field_counts.keys()), y=list(subset_field_counts.values()), ax=ax)
        ax.set_title("Number of Fields per Subset")
        ax.set_xlabel("Subset")
        ax.set_ylabel("Number of Fields")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=45)

        bar_chart_path = vis_dir / f"{operation}_viz_fields_per_subset{suffix}.png"
        plt.tight_layout()
        
        _save_figure(fig, bar_chart_path, **kwargs)
        return bar_chart_path

    def _plot_field_subset_network(self, output_data: Dict[str, pd.DataFrame], vis_dir: Path, suffix: str, operation: str, **kwargs) -> Path:
        """Create a network diagram showing how fields are distributed across subsets using Plotly."""
        nodes = set()
        edges = []
        node_types = {}

        for subset_name, df in output_data.items():
            nodes.add(subset_name)
            node_types[subset_name] = "subset"
            for col in df.columns:
                nodes.add(col)
                node_types[col] = "field"
                edges.append((subset_name, col))

        subset_nodes = [n for n in nodes if node_types[n] == "subset"]
        field_nodes = [n for n in nodes if node_types[n] == "field"]

        positions = {n: (0, i * -1.5) for i, n in enumerate(subset_nodes)}
        positions.update({n: (3, i * -1.2) for i, n in enumerate(field_nodes)})

        edge_x, edge_y = [], []
        for src, tgt in edges:
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in nodes:
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append("skyblue" if node_types[node] == "subset" else "lightgreen")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=20, color=node_color, line=dict(width=1, color='black')),
            text=node_text,
            textposition="top center",
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text='Field Distribution Across Subsets (Network Diagram)', font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white'
                        ))

        network_path = vis_dir / f"{operation}_viz_field_subset_network{suffix}.png"
        _save_figure(fig, network_path, **kwargs)
        return network_path

    def _plot_schema_comparison(self, input_data: pd.DataFrame, output_data: Dict[str, pd.DataFrame], vis_dir: Path, suffix: str, operation: str, **kwargs) -> Path:
        """Create a bar chart comparing the schema of the original dataset to the split subsets."""
        fig, ax = plt.subplots(figsize=(10, 6))
        original_columns = input_data.columns
        sns.barplot(x=["Original"] + list(output_data.keys()),
                    y=[len(original_columns)] + [len(df.columns) for df in output_data.values()],
                    palette="pastel", ax=ax)
        ax.set_title("Schema Comparison: Original vs. Split Datasets")
        ax.set_ylabel("Number of Fields")
        ax.set_xlabel("Dataset")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        schema_path = vis_dir / f"{operation}_viz_schema_comparison{suffix}.png"
        plt.tight_layout()
        _save_figure(fig, schema_path, **kwargs)

        return schema_path

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
            "field_groups": self.field_groups,
            "include_id_field": self.include_id_field,
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
        """
        Validate that all specified fields in field_groups exist in the DataFrame.
        Optionally check if the ID field exists.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset to validate.

        Returns:
        --------
        bool
            True if all fields are valid; False otherwise.
        """
        all_columns = set(df.columns)

        # Ensure field_groups is not empty
        if not self.field_groups:
            self.logger.error("field_groups must not be empty.")
            return False

        # Check that every field in each group exists in the DataFrame
        for group_name, fields in self.field_groups.items():
            for field in fields:
                if field not in all_columns:
                    self.logger.error(f"Field '{field}' in group '{group_name}' not found in DataFrame.")
                    return False

        # If ID field is to be included, check that it exists in the DataFrame
        if self.include_id_field and self.id_field:
            if self.id_field not in all_columns:
                self.logger.error(f"ID field '{self.id_field}' not found in DataFrame.")
                return False

        # All validations passed
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
        self.field_groups = kwargs.get("field_groups", getattr(self, "field_groups", None))
        self.include_id_field = kwargs.get("include_id_field", getattr(self, "include_id_field", True))
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