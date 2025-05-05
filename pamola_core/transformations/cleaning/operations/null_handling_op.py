import logging
import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from pamola_core.transformations.cleaning.cleaners.null_handling_cleaner import NullHandlingCleaner
from pamola_core.transformations.cleaning.commons.data_cleaner_operation import DataCleanerOperation
from pamola_core.transformations.cleaning.commons.mapping_store import MappingStore
from pamola_core.utils import io
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_registry import register
# Configure logger
logger = logging.getLogger(__name__)

@register()
class NullHandlingOperation(DataCleanerOperation):
    """
    Operation for handling missing values by removing records or enriching fields based on thresholds.
    """

    name = "null_handling"
    category = "remove_null_data"
    def __init__(self,
                field_name: str,
                action: str,
                threshold_type: str = "count",
                row_threshold: float = 1,
                field_threshold: float = 0.8,
                column_prefix: str = "removed_",
                mode: str = "ENRICH",
                batch_size: int = 10000,
                output_field_name: Optional[str] = None,
                mapping_store_path: Optional[str] = None,
                save_mapping: bool = False,
                detailed_metrics: bool = False,
                error_logging_level: str = "WARNING",
                max_retries: int = 3,
                default_fill_value: str = "row removed",
    ):
        """
        Initialize a NullHandlingOperation instance.

        Parameters:
        -----------
        field_name : str
            The name of the field (column) to apply null handling on.
        action : str
            The action to perform: 'remove_rows' to delete rows, 'remove_fields' to delete fields, 'flag' to filter missing values.
        threshold_type : str, default="count"
            The type of threshold to apply ('count' or 'percentage').
        row_threshold : float, default=1
            Threshold for row nulls (number or percentage depending on threshold_type).
        field_threshold : float, default=0.8
            Threshold for field null percentage to trigger removal.
        column_prefix : str, default="removed_"
            Prefix to add to output column name when mode is 'ENRICH'.
        mode : str, default="ENRICH"
            Processing mode: 'ENRICH' creates a new field; 'REPLACE' modifies the existing field.
        batch_size : int, default=10000
            Number of records to process in each batch.
        output_field_name : Optional[str], default=None
            Custom name for the output field if mode is 'ENRICH'.
        mapping_store_path : Optional[str], default=None
            Path to store mappings if saving mappings is enabled.
        save_mapping : bool, default=False
            Whether to save transformation mappings after processing.
        detailed_metrics : bool, default=False
            Whether to collect detailed performance and error metrics.
        error_logging_level : str, default="WARNING"
            Logging level for errors ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        max_retries : int, default=3
            Maximum number of retries on processing failure.
        default_fill_value : str, default="row removed"
            Default value to fill if no value is generated during processing.
        """
        self.output_field_name = output_field_name

        # Initialize in null_removal
        self.action = action
        self.field_name = field_name
        self.threshold_type = threshold_type
        self.row_threshold = row_threshold
        self.field_threshold = field_threshold
        self.mapping_store_path = mapping_store_path
        self.default_fill_value = default_fill_value

        # Save new parameters
        self.detailed_metrics = detailed_metrics
        self.error_logging_level = error_logging_level.upper()
        self.max_retries = max_retries

        # Store attributes locally that we need to access directly
        self.column_prefix = column_prefix
        self.save_mapping = save_mapping

        # Configure logging
        self._configure_logging()

        # Create Null handling cleaner
        cleaner = NullHandlingCleaner(
            config={
                "action": action,
                "field_threshold": self.field_threshold,
                "target_field": self.field_name,
                'threshold_type': self.threshold_type,
                "row_threshold": self.row_threshold,
            }
        )

        # Initialize parent class first with the base generator
        super().__init__(
            field_name=field_name,
            cleaner=cleaner,
            mode=mode,
            output_field_name=output_field_name,
            batch_size=batch_size,
        )
        # Set up performance metrics
        self.start_time = None
        self.process_count = 0
        self.error_count = 0
        self.retry_count = 0

        # Ensure we have a reference to the original DataFrame for metrics collection
        self._original_df = None

        # Initialize mapping store if path is provided
        if mapping_store_path:
            self._initialize_mapping_store(mapping_store_path)

        # Update operation name and description
        self.name = f"null_handling_{action}"
        self.description = f"Null handling for {field_name} using {action} action"


        # For detailed metrics
        if self.detailed_metrics:
            self._generation_times = []
            self._error_fields = []

    def execute(self,
                data_source: Any,
                task_dir: Path,
                reporter: Any,
                **kwargs) -> OperationResult:
        try:
            # Start timing for performance metrics
            self.start_time = time.time()
            self.process_count = 0

            # Load data if needed
            if isinstance(data_source, pd.DataFrame):
                self._original_df = data_source.copy()
            else:
                # If data_source is a path, it will be loaded by the parent class
                pass

            # Call parent execute method
            result = super().execute(data_source, task_dir, reporter, **kwargs)

            # Add additional metrics
            if result.status == OperationStatus.SUCCESS and hasattr(result, "metrics"):
                execution_time = time.time() - self.start_time

                # Add performance metrics
                result.metrics["performance"].update(
                    {"generation_time": execution_time}
                )

                # Add detailed metrics if enabled
                if (
                    self.detailed_metrics
                    and hasattr(self, "_generation_times")
                    and self._generation_times
                ):
                    result.metrics["detailed_performance"] = {
                        "min_time": min(self._generation_times),
                        "max_time": max(self._generation_times),
                        "avg_time": sum(self._generation_times)
                        / len(self._generation_times),
                        "total_times": len(self._generation_times),
                    }

                    if hasattr(self, "_error_fields") and self._error_fields:
                        result.metrics["error_details"] = {
                            "error_fields": self._error_fields[
                                :100
                            ]  # Limit to first 100
                        }

                # Save mapping if requested
                if (
                    self.save_mapping
                    and hasattr(self, "mapping_store")
                    and self.mapping_store
                ):
                    mapping_dir = Path(task_dir) / "maps"
                    io.ensure_directory(mapping_dir)
                    mapping_path = (
                        mapping_dir / f"{self.name}_{self.field_name}_mappings.json"
                    )
                    self.mapping_store.save_json(mapping_path)
                    logger.info(f"Saved mapping to {mapping_path}")

            return result

        except Exception as e:
            logger.exception("Error during Null handling operation.")
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                execution_time=time.time() - (self.start_time or time.time()),
            )

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data to handle missing values by removing records or enriching fields,
        depending on the specified action and mode.

        Args:
            batch: DataFrame batch to process.

        Returns:
            Processed DataFrame batch.
        """
        # Create result dataframe
        result = batch.copy()

        # Determine output field based on mode
        if self.mode == "REPLACE":
            output_field = self.field_name
        else:  # ENRICH
            output_field = self.output_field_name or f"{self.column_prefix}{self.field_name}"

        params = {
            "field_name": self.field_name,
            "action": self.action,
            "threshold_type": self.threshold_type,
            "row_threshold": self.row_threshold,
            "field_threshold": self.field_threshold,
        }

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Run the cleaner
                cleaned_df = self.cleaner.execute(result.copy(), **params)

                if cleaned_df is None or cleaned_df.empty:
                    logger.warning(f"Cleaner returned empty dataframe for field '{self.field_name}'.")
                    return result

                # Align result with cleaned_df index if rows were removed
                if len(cleaned_df) != len(result):
                    logger.debug(f"Detected removed rows after cleaning for field '{self.field_name}'.")

                    # Match result with cleaned_df (drop rows that were removed)
                    result = result.loc[cleaned_df.index]

                # Update the field values
                if self.field_name in cleaned_df.columns:
                    result[output_field] = cleaned_df[self.field_name].values
                else:
                    result.drop(columns=[self.field_name], inplace=True)
                
                # Update metrics
                self.process_count += batch[self.field_name].isna().sum()

                return result

            except Exception as e:
                last_error = e
                retries += 1
                self.retry_count += 1

                if retries <= self.max_retries:
                    logger.debug(
                        f"Retry {retries}/{self.max_retries} - Error during processing field '{self.field_name}': {str(e)}"
                    )
                else:
                    logger.error(
                        f"Failed processing field '{self.field_name}' after {self.max_retries} retries: {str(e)}"
                    )
                    self.error_count += 1
                    self._error_fields.append(self.field_name)

        # All retries failed — return original batch unmodified
        return result

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
        params = {
            "field_name": self.field_name,
            "action": self.action,
            "threshold_type": self.threshold_type,
            "row_threshold": self.row_threshold,
            "field_threshold": self.field_threshold,
        }

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                cleaned_df = self.cleaner.execute(df.copy(), **params)

                if cleaned_df is None or cleaned_df.empty:
                    logger.warning(f"Cleaner returned empty dataframe for field '{self.field_name}'.")
                    return df

                # If the field was removed, return the cleaned dataframe directly
                if self.field_name not in cleaned_df.columns:
                    logger.info(f"Field '{self.field_name}' has been removed from the dataframe.")
                    return cleaned_df

                return cleaned_df

            except Exception as e:
                last_error = e
                retries += 1
                self.retry_count += 1

                if retries <= self.max_retries:
                    logger.debug(
                        f"Retry {retries}/{self.max_retries} - Error during processing field '{self.field_name}': {str(e)}"
                    )
                else:
                    logger.error(
                        f"Failed processing field '{self.field_name}' after {self.max_retries} retries: {str(e)}"
                    )
                    self.error_count += 1
                    self._error_fields.append(self.field_name)

        # All retries failed — return original df unmodified
        return df

    def _process_value(self, value, **params):
        """
        Process a single value using the appropriate generation method with retry logic.

        Args:
            value: Original value
            **params: Additional parameters

        Returns:
            Processed value
        """
        # Add time for detailed metrics
        return value
    
    def _collect_metrics(self, df: pd.DataFrame, operation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Collect metrics for the Null handling operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Processed DataFrame

        Returns:
        --------
        Dict[str, Any]
            Metrics dictionary
        """
        # Get base metrics from parent class
        metrics = super()._collect_metrics(df, operation_params)

        # Add Null handling specific metrics
        metrics["model"] = {
            "field_name": self.field_name,
            "action": self.action,
            "threshold_type": self.threshold_type,
        }

        # Initialize the counts outside the conditional branches to avoid UnboundLocalError
        removed_count = 0

        # Calculate specific metrics for removing null values
        if self._original_df is not None:
            original_null_count = self._original_df[self.field_name].isna().sum()

            if self.mode == "REPLACE":
                # For REPLACE mode, calculate removed count
                current_null_value_count = df[self.field_name].isna().sum() if self.field_name in df else 0
                removed_count = original_null_count - current_null_value_count
                if self.field_name in df:
                    current_null_value_count = df[self.field_name].isna().sum()
                    removed_count = original_null_count - current_null_value_count
                else:
                    current_null_value_count = 0
                    removed_count = self._original_df[self.field_name].count()

            elif self.mode == "ENRICH":
                # For ENRICH mode, calculate removed count
                if self.output_field_name in df.columns:
                    current_null_value_count = df[self.output_field_name].isna().sum()
                    removed_count = original_null_count - current_null_value_count

            removal_rate = float(removed_count / original_null_count) if original_null_count > 0 else 1.0
            # Calculate the null removal rate
            metrics["null_removal"] = {
                "original_null_count": int(original_null_count),
                "removed_count": int(removed_count),
                "removal_rate": removal_rate,
            }

        return metrics
    
    def _configure_logging(self):
        """
        Configure logging based on error_logging_level.
        """
        log_level = getattr(logging, self.error_logging_level, logging.WARNING)
        logger.setLevel(log_level)

    def _initialize_mapping_store(self, path: Union[str, Path]) -> None:
        """
        Initialize the mapping store if needed.

        Args:
            path: Path to mapping store file
        """
        try:
            self.mapping_store = MappingStore()

            # Load existing mappings if the file exists
            path_obj = Path(path)
            if path_obj.exists():
                self.mapping_store.load(path_obj)
                logger.info(f"Loaded mapping store from {path}")
        except Exception as e:
            logger.warning(f"Failed to initialize mapping store: {str(e)}")
            self.mapping_store = None
