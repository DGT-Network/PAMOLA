"""
Operations module for fake data generation.

This module provides standardized operation classes for fake data generation
that integrate with HHR's operation system to generate synthetic data while
maintaining statistical properties of the original data.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from pamola_core.fake_data.commons.base import (
    BaseGenerator,
    MappingStore,
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
    get_timestamped_filename
)
from pamola_core.utils.ops.op_base import BaseOperation as HHRBaseOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressBar

# Configure logger
logger = logging.getLogger(__name__)


class BaseOperation(HHRBaseOperation):
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

    def execute(self, data_source: Any, task_dir: Path, reporter: Any, **kwargs) -> OperationResult:
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
                 batch_size: int = 10000, null_strategy: Union[str, NullStrategy] = NullStrategy.PRESERVE):
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

        # Store original data for metrics comparison
        self._original_df = None

        # Create a metrics collector
        self._metrics_collector = create_metrics_collector()

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

        logger.info(f"Executing {self.name} operation on field {self.field_name}")
        if reporter:
            reporter.update_progress(0, f"Starting {self.name} operation")

        try:
            # Create output directory for data output and maps
            output_dir = ensure_directory(task_dir / "output")
            maps_dir = ensure_directory(task_dir / "maps")

            # Load data
            df = self._load_data(data_source)

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

            # Calculate total records and batches
            total_records = len(df)
            total_batches = (total_records + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing {total_records} records in {total_batches} batches")
            if reporter:
                reporter.update_progress(5, f"Preprocessing data")

            # Preprocess data
            df = self.preprocess_data(df)

            # Handle NULL values
            df = self.handle_null_values(df)

            # Progress bar for batch processing
            progress_bar = ProgressBar(
                total=total_records,
                description=f"Processing {self.field_name}",
                unit="records"
            )

            # Process data in batches
            processed_batches = 0
            processed_records = 0

            for batch_start in range(0, total_records, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_records)
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

                # Update progress
                processed_batches += 1
                processed_records += len(batch)
                progress = 5 + int(90 * processed_batches / total_batches)

                # Update progress
                batch_progress_message = f"Processed {processed_records}/{total_records} records"
                # Fix: Use dictionary for postfix
                progress_bar.update(len(batch), {"message": batch_progress_message})

                if reporter:
                    reporter.update_progress(
                        progress,
                        batch_progress_message
                    )

            # Close progress bar
            progress_bar.close()

            # Postprocess data
            if reporter:
                reporter.update_progress(95, "Postprocessing data")
            df = self.postprocess_data(df)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Update performance metrics
            records_per_second = int(total_records / execution_time) if execution_time > 0 else 0

            # Collect metrics
            if reporter:
                reporter.update_progress(96, "Collecting metrics")
            metrics = self._collect_metrics(df)

            # Add performance metrics
            metrics["performance"]["generation_time"] = execution_time
            metrics["performance"]["records_per_second"] = records_per_second

            # Импорт psutil вне блока try для доступности в except блоке
            import psutil
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                metrics["performance"]["memory_usage_mb"] = memory_info.rss / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Error collecting memory usage: {str(e)}")

            # Generate visualizations
            if reporter:
                reporter.update_progress(97, "Generating visualizations")

            try:
                # Extract original and generated data series for visualization
                orig_series = self._original_df[self.field_name]

                if self.mode == "REPLACE":
                    gen_series = df[self.field_name]
                elif self.mode == "ENRICH" and self.output_field_name in df.columns:
                    gen_series = df[self.output_field_name]
                else:
                    gen_series = None

                if gen_series is not None:
                    visualizations = self._metrics_collector.visualize_metrics(
                        metrics=metrics,
                        field_name=self.field_name,
                        output_dir=task_dir,
                        op_type=self.name
                    )

                    # Add visualizations to metrics
                    metrics["visualizations"] = {name: str(path) for name, path in visualizations.items()}
            except Exception as e:
                logger.warning(f"Error generating visualizations: {str(e)}")

            # Save result
            if reporter:
                reporter.update_progress(98, "Saving results")

            # Save data if required
            save_data = kwargs.get("save_data", True)
            if save_data:
                output_path = self._save_result(df, output_dir)
            else:
                output_path = None

            # Save metrics
            metrics_path = self._save_metrics(metrics, task_dir)

            # Generate and save metrics report
            try:
                generate_metrics_report(
                    metrics,
                    task_dir,
                    op_type=self.name,
                    field_name=self.field_name
                )
            except Exception as e:
                logger.warning(f"Error generating metrics report: {str(e)}")

            # Create result
            if reporter:
                reporter.update_progress(100, "Operation completed")

            # Исправленное создание OperationResult, т.к. 'output' не является ожидаемым аргументом
            result = OperationResult(
                status=OperationStatus.SUCCESS,
                execution_time=execution_time
            )

            # Добавляем выходные данные другим способом, если это поддерживается классом
            if hasattr(result, 'set_output') and callable(getattr(result, 'set_output')):
                result.set_output(df)
            # Альтернативный вариант, если 'set_output' не доступен
            elif hasattr(result, 'data') or hasattr(type(result), 'data'):
                result.data = df

            # Add artifacts
            if output_path:
                result.add_artifact(
                    path=output_path,
                    artifact_type="csv",
                    description=f"Processed data with {self.mode.lower()}d {self.field_name} field"
                )

            result.add_artifact(
                path=metrics_path,
                artifact_type="json",
                description=f"Metrics for {self.name} operation on {self.field_name}"
            )

            # Add visualization artifacts if available
            if "visualizations" in metrics:
                for name, path_str in metrics["visualizations"].items():
                    result.add_artifact(
                        path=Path(path_str),
                        artifact_type="png",
                        description=f"{name.replace('_', ' ').title()} visualization for {self.field_name}"
                    )

            # Add metrics
            result.metrics = metrics

            logger.info(f"Operation {self.name} completed successfully")
            return result

        except Exception as e:
            logger.exception(f"Error executing {self.name} operation: {str(e)}")

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
            return data_source.get_dataframe()
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

    def _save_result(self, df: pd.DataFrame, output_dir: Path) -> Path:
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

        write_dataframe_to_csv(df, output_path)
        return output_path

    def _save_metrics(self, metrics: Dict[str, Any], task_dir: Path) -> Path:
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

        write_json(metrics, metrics_path)
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


class GeneratorOperation(FieldOperation):
    """
    Operation that uses a generator to create fake data.

    This operation integrates with the HHR operation system and uses a generator
    to create synthetic data for specific fields.
    """

    name: str = "generator_operation"
    description: str = "Operation for generating fake data"

    def __init__(self, field_name: str, generator: BaseGenerator,
                 mode: str = "REPLACE", output_field_name: Optional[str] = None,
                 batch_size: int = 10000, null_strategy: Union[str, NullStrategy] = NullStrategy.PRESERVE,
                 consistency_mechanism: str = "prgn", mapping_store: Optional[MappingStore] = None,
                 generator_params: Optional[Dict[str, Any]] = None):
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
        """
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            batch_size=batch_size,
            null_strategy=null_strategy
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

    def execute(self, data_source: Any, task_dir: Path, reporter: Any, **kwargs) -> OperationResult:
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
        result = super().execute(data_source, task_dir, reporter, **kwargs)

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
                    description=f"Mappings for {self.field_name} field"
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


# Исправленная версия регистрации операций
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
    logger.warning("HHR operation registry not available. Operations will not be registered.")