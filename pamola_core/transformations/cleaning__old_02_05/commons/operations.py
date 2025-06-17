"""
Standardized operations for clean data generation.

This module defines reusable operation classes that integrate with PAMOLA's transformation
framework, supporting clean data generation and synthetic data creation while preserving
key statistical properties of the original datasets.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from pamola_core.transformations.cleaning__old_02_05.commons.base import (
    FieldOperation as BaseFieldOperation,
)

# Import metrics collector
from pamola_core.transformations.cleaning__old_02_05.commons.metrics import (
    create_metrics_collector,
)
from pamola_core.transformations.cleaning__old_02_05.commons.metrics_report import (
    generate_metrics_report,
)
from pamola_core.utils.io import (
    ensure_directory,
    write_dataframe_to_csv,
    write_json,
    get_timestamped_filename,
    load_data_operation,
    load_settings_operation
)
from pamola_core.utils.ops.op_base import BaseOperation as PamolaBaseOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressBar
from pamola_core.common.constants import Constants
# Configure logger
logger = logging.getLogger(__name__)

class BaseOperation(PamolaBaseOperation):
    """
    Base class for clean data operations.

    Provides common functionality for operations that generate or replace
    data with clean values.
    """

    name: str = "base_clean_data_operation"
    description: str = "Base operation for clean data generation"

    def __init__(self):
        """
        Initializes the operation.
        """
        super().__init__(name=self.name, description=self.description)

    def execute(
        self, data_source: Any, task_dir: Path, reporter: Any, **kwargs
    ) -> OperationResult:
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
            execution_time=0,
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

    def __init__(
        self,
        field_name: str,
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        batch_size: int = 10000,
        use_encryption: bool = False,
        encryption_key: Optional[Union[str, Path]] = None
    ):
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
        """

        BaseFieldOperation.__init__(
            self,
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
        )

        # Initialize BaseOperation
        BaseOperation.__init__(self)

        # Additional field operation parameters
        self.batch_size = batch_size

        # Store original data for metrics comparison
        self._original_df = None

        # Create a metrics collector
        self._metrics_collector = create_metrics_collector()
        
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key

    def execute(self, data_source: Any, task_dir: Path, reporter: Any, **kwargs) -> OperationResult:
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
        logger.info(f"Starting {self.name} operation on field {self.field_name}")

        try:
            # Setup and validation
            dataset_name = kwargs.get('dataset_name', "main")
            settings_operation = load_settings_operation(data_source, dataset_name, **kwargs)
            df = load_data_operation(data_source, dataset_name, **settings_operation)
            self._original_df = df.copy()

            if self.field_name not in df.columns:
                return self._create_error_result(
                    f"Field {self.field_name} not found in the data", 
                    start_time
                )

            # Process the data
            df = self._process_data(df, reporter)
            
            # Finalize results
            duration = time.time() - start_time
            mem_usage = self._get_memory_usage()
            operation_params = {"execution_time_sec": duration, "mem_usage": mem_usage}
            
            # Generate metrics and artifacts
            result = self._generate_output(df, task_dir, duration, operation_params, **kwargs)
            
            logger.info(f"Operation {self.name} completed successfully")
            return result

        except Exception as e:
            logger.exception(f"Error executing {self.name} operation: {str(e)}")
            return self._create_error_result(str(e), start_time)

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
            return data_source.get_dataframe()
        elif isinstance(data_source, pd.DataFrame):
            return data_source
        elif isinstance(data_source, str) or isinstance(data_source, Path):
            # Load from file path
            try:
                from pamola_core.utils.io import read_full_csv

                return read_full_csv(data_source)
            except Exception as e:
                raise ValueError(
                    f"Unable to load data from path {data_source}: {str(e)}"
                )
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
        
    def _process_data(self, df: pd.DataFrame, reporter: Any) -> pd.DataFrame:
        """
        Process the data through preprocessing, batch processing, and postprocessing phases.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        reporter : Any
            Reporter for logging operation metrics and progress
            
        Returns:
        --------
        pd.DataFrame
            The processed dataframe
        """
        # Preprocess
        df = self.preprocess_data(df)
        reporter.add_operation(
            "Preprocess Data",
            {
                "records_count": len(df),
                "columns_count": len(df.columns),
                "operation_type": "preprocess",
            },
        )

        # Batch processing
        if self.mode is None:
            df = self._process_without_batching(df, reporter)
        else:
            df = self._process_in_batches(df, reporter)

        # Postprocess
        df = self.postprocess_data(df)
        reporter.add_operation(
            "Postprocessing Data",
            {
                "records_count": len(df),
                "columns_count": len(df.columns),
                "operation_type": "postprocess",
            },
        )
        
        return df

    def _process_in_batches(self, df: pd.DataFrame, reporter: Any) -> pd.DataFrame:
        """
        Process the dataframe in batches using the specified mode.

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process in batches
        reporter : Any
            Reporter for logging operation metrics and progress

        Returns:
        --------
        pd.DataFrame
            The batch-processed dataframe
        """
        total = len(df)
        progress_bar = ProgressBar(
            total=total, description=f"Processing {self.field_name}", unit="records"
        )

        # Accumulate indices of processed rows
        all_processed_indices = set()

        for i in range(0, total, self.batch_size):
            batch = df.iloc[i : i + self.batch_size].copy()
            processed = self.process_batch(batch)

            # Add processed indices to the accumulated set
            all_processed_indices.update(processed.index)

            if self.mode == "REPLACE":
                # For REPLACE mode, just replace the rows in df with the processed values
                if self.field_name in processed.columns:
                    df.loc[processed.index, :] = processed
                else:
                    df.drop(columns=[self.field_name], inplace=True)

            elif self.mode == "ENRICH":
                # For ENRICH mode, apply enrichment logic
                self._apply_enrichment(df, processed, i, default_fill_value=self.default_fill_value)

            progress_bar.update(
                len(batch),
                {"message": f"Processed {i + len(batch)}/{total} records"},
            )

        progress_bar.close()

        if self.mode == "REPLACE":
            # Filter df to only keep the rows that were processed (REPLACE mode)
            df = df.loc[list(all_processed_indices)]

        return df
    
    def _process_without_batching(self, df: pd.DataFrame, reporter: Any) -> pd.DataFrame:
        """
        Process the entire dataframe at once, without batching.
        Used for operations that do not require row-wise batching (e.g., removing columns).

        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to process
        reporter : Any
            Reporter for logging operation metrics and progress

        Returns:
        --------
        pd.DataFrame
            The processed dataframe
        """
        original_size = len(df)
        progress_bar = ProgressBar(
            total=original_size,
            description=f"Processing {self.field_name}",
            unit="records"
        )

        processed_df = self.process_without_batching(df)

        processed_size = len(processed_df)
        progress_bar.update(
            processed_size,
            {"message": f"Processed {processed_size}/{original_size} records"}
        )
        progress_bar.close()

        return processed_df


    def _apply_enrichment(self, df: pd.DataFrame, processed: pd.DataFrame, start_idx: int, default_fill_value: Any = np.nan) -> None:
        """
        Apply enrichment by adding the processed output field to the original dataframe. 
        Fills missing rows with the default value if rows were removed (in 'remove row' case).
        """
        output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

        # Initialize the output field in df if not exists
        if output_field not in df.columns:
            df[output_field] = np.nan
            
        # Get the end index of the current batch
        end_idx = start_idx + len(processed) - 1 if processed.empty else start_idx + self.batch_size - 1
        end_idx = min(end_idx, len(df) - 1)  # Ensure we don't go beyond dataframe bounds
        
        # Get indices of rows that should be in this batch
        batch_indices = df.index[start_idx:end_idx + 1]
        
        # Case when rows are replaced (just update the present rows in df)
        df.loc[processed.index, output_field] = processed[output_field].values
        
        # Fill missing rows ONLY within the current batch's range
        missing_rows = [idx for idx in batch_indices if idx not in processed.index]
        if missing_rows:
            df.loc[missing_rows, output_field] = default_fill_value

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage of the process.
        
        Returns:
        --------
        float
            Memory usage in MB, or 0.0 if measurement fails
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Error collecting memory usage: {str(e)}")
            return 0.0

    def _generate_output(self, df: pd.DataFrame, task_dir: Path, 
                        duration: float, operation_params: dict,
                        **kwargs) -> OperationResult:
        """
        Generate metrics, visualizations, and final output.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The fully processed dataframe
        task_dir : Path
            Directory to save outputs and artifacts
        duration : float
            Execution time in seconds
        operation_params : dict
            Additional operation parameters and metrics
            
        Returns:
        --------
        OperationResult
            Operation result with metrics and artifacts
        """
        dirs = self._prepare_directories(task_dir)
        output_dir = dirs['output']
        visualizations_dir = dirs['visualizations']
        dictionaries_dir = dirs['dictionaries']

        # Collect metrics
        df_metrics = (
                df
                if self.mode is not None
                else self._original_df
        )
        metrics = self._collect_metrics(df_metrics, operation_params)
        
        # Generate visualizations
        metrics = self._generate_visualizations(metrics, visualizations_dir, **kwargs)
        
        # Save results
        output_path = self._save_result(df, output_dir, **kwargs)
        metrics_path = self._save_metrics(metrics, output_dir, **kwargs)
        
        # Generate report
        self._generate_report(metrics, output_dir)
        
        # Create result object with artifacts
        result = OperationResult(
            status=OperationStatus.SUCCESS, 
            execution_time=duration, 
            metrics=metrics
        )
        
        self._add_artifacts_to_result(result, output_path, metrics_path, metrics)
        
        return result

    def _generate_visualizations(self, metrics: dict, output_dir: Path, **kwargs) -> dict:
        """
        Generate visualizations for the metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing operation metrics
        output_dir : Path
            Directory to save generated visualizations
            
        Returns:
        --------
        dict
            Updated metrics dictionary with visualization references
        """
        try:
            kwargs_encryption = {
                "use_encryption": kwargs.get('use_encryption', False),
                "encryption_key": kwargs.get('encryption_key', None)
            }
            visuals = self._metrics_collector.visualize_metrics(
                metrics=metrics,
                field_name=self.field_name,
                output_dir=output_dir,
                op_type=self.name,
                **kwargs_encryption
            )
            metrics["visualizations"] = {k: str(v) for k, v in visuals.items()}
        except Exception as e:
            logger.warning(f"Error generating visualizations: {str(e)}")
            metrics["visualizations"] = {}
        
        return metrics

    def _generate_report(self, metrics: dict, output_dir: Path) -> None:
        """
        Generate a metrics report.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary containing operation metrics
        output_dir : Path
            Directory to save the generated report
        """
        try:
            generate_metrics_report(
                metrics, output_dir, op_type=self.name, field_name=self.field_name
            )
        except Exception as e:
            logger.warning(f"Error generating metrics report: {str(e)}")

    def _add_artifacts_to_result(self, result: OperationResult, 
                            output_path: Path, metrics_path: Path, metrics: dict) -> None:
        """
        Add all artifacts to the operation result.
        
        Parameters:
        -----------
        result : OperationResult
            The operation result object to add artifacts to
        output_path : Path
            Path to the output data file
        metrics_path : Path
            Path to the metrics JSON file
        metrics : dict
            Dictionary containing metrics with visualization references
        """
        if output_path:
            result.add_artifact(
                "csv", output_path, f"Processed data for {self.field_name}", Constants.Artifact_Category_Output
            )
        
        result.add_artifact(
            "json", metrics_path, f"Metrics for {self.name} on {self.field_name}", Constants.Artifact_Category_Metrics
        )

        for name, path in metrics.get("visualizations", {}).items():
            result.add_artifact(
                "png",
                Path(path),
                f"{name.replace('_', ' ').title()} for {self.field_name}",
                Constants.Artifact_Category_Visualization
            )

    def _create_error_result(self, error_message: str, start_time: float) -> OperationResult:
        """
        Create an error result with the given message.
        
        Parameters:
        -----------
        error_message : str
            Description of the error that occurred
        start_time : float
            Timestamp when operation execution started
            
        Returns:
        --------
        OperationResult
            Error result with appropriate status and message
        """
        logger.error(f"Field operation error: {error_message}")
        return OperationResult(
            status=OperationStatus.ERROR,
            error_message=error_message,
            execution_time=time.time() - start_time,
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

    def _collect_metrics(
        self, df: pd.DataFrame, operation_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
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
            "non_null_records": df[self.field_name].notna().sum() if self.field_name in df.columns else 0,
            "execution_time": 0,  # Will be updated by execute method
            "performance": {},
        }

        # Output field metrics if in ENRICH mode
        if self.mode == "ENRICH" and self.output_field_name in df.columns:
            metrics["output_field"] = {
                "name": self.output_field_name,
                "non_null_records": df[self.output_field_name].notna().sum(),
            }

        # Use metrics collector to get detailed metrics
        try:
            # Extract original and generated data series
            orig_series = (
                self._original_df[self.field_name]
                if self._original_df is not None
                else None
            )

            if self.mode == "REPLACE":
                gen_series = df[self.field_name] if self.field_name in df.columns else None
            elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                gen_series = df[self.output_field_name]
            else:
                gen_series = None

            # Check if mapping exists
            existing_mapping = self.mapping_store.get_mapping(self.field_name, orig_series)
            if existing_mapping is not None:
                return existing_mapping
            
            # Store mapping
            if gen_series is not None:
                self.mapping_store.add_mappings_from_series(
                    field_name=self.field_name,
                    originals=orig_series,
                    synthetics=gen_series,
                )

            # Collect metrics using the metrics collector
            collector_metrics = self._metrics_collector.collect_metrics(
                    orig_data=orig_series,
                    gen_data=gen_series,
                    field_name=self.field_name,
                    operation_params=operation_params,
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
    
    def process_without_batching(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire dataframe at once, without batching.
        Used for operations like removing a field (column).

        Parameters:
        -----------
        df : pd.DataFrame
            The full dataframe to process

        Returns:
        --------
        pd.DataFrame
            Processed dataframe
        """
        raise NotImplementedError("Subclasses must implement process_without_batching method")

# Fixed version of operation registration
try:
    from pamola_core.utils.ops.op_registry import register

    # Using the decorator without the 'category' parameter, since it's not supported
    @register()
    class RegisteredBaseOperation(BaseOperation):
        name = "clean_data_base"
        description = "Base operation for clean data operations"
        # Set the category as a class attribute instead of using the decorator parameter
        category = "clean_data"

    @register()
    class RegisteredFieldOperation(FieldOperation):
        name = "clean_data_field"
        description = "Base operation for clean data field operations"
        category = "clean_data"


except ImportError:
    logger.warning(
        "Operation registry not available. Operations will not be registered."
    )
