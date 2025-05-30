"""
This module defines the BasicStatisticalImputationOperation class, which is a data cleaning operation
for imputing missing values using common statistical methods based on the distribution of existing values.

The operation supports batch processing and can operate in two modes:
- REPLACE: Overwrites the original column with imputed values.
- ENRICH: Adds a new column with imputed values, leaving the original data intact.

The operation is configurable, allowing users to:
- Select the common statistical methods.
- Specify which fields to use as group by.
- Control logging, retry behavior, and metric reporting.
- Collect detailed performance metrics.

This class is registered as a DataCleanerOperation and uses BasicStatisticalImputationCleaner
as the pamola core imputation engine.

Typical use cases include cleaning datasets with missing numerical values
in preparation for downstream data analysis or machine learning pipelines.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import pandas as pd
from pamola_core.utils import io
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.transformations.cleaning__old_02_05.commons.mapping_store import MappingStore
from pamola_core.transformations.cleaning__old_02_05.commons.data_cleaner_operation import DataCleanerOperation
from pamola_core.transformations.cleaning__old_02_05.cleaners.statistical_imputation_cleaner import (
    BasicStatisticalImputationCleaner
)

# Configure logger
logger = logging.getLogger(__name__)

@register()
class BasicStatisticalImputationOperation(DataCleanerOperation):
    """
    Operation for imputing missing values using common statistical methods.

    Uses mean, median, mode to imputing missing values.
    """

    name = "statistical_imputation"
    category = "clean_data"

    def __init__(
            self,
            field_name: str,
            grouping_fields: List[str] = None,
            method: str = "mean",
            mode: str = "ENRICH",
            output_field_name: Optional[str] = None,
            batch_size: int = 10000,
            column_prefix: str = "imputed_",
            detailed_metrics: bool = False,
            mapping_store_path: Optional[str] = None,
            error_logging_level: str = "WARNING",
            max_retries: int = 3,
            save_mapping: bool = True,
            default_fill_value: str = "row removed",
            use_encryption: bool = False,
            encryption_key: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the Basic Statistical Imputation Operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to impute missing values for
        grouping_fields : List[str]
            List of field names are used to group by before imputing
        method : str
            Common statistical methods to use: 'mean', 'median', 'mode'
        mode : str
            Operation mode: "REPLACE" or "ENRICH"
        output_field_name : str, optional
            Name of the output field (for ENRICH mode)
        batch_size : int
            Size of batches for processing
        column_prefix : str
            Prefix to add to output field name if not specified
        detailed_metrics : bool
            Whether to collect detailed performance metrics
        mapping_store_path: Optional[str] = None
            Path to store mappings
        error_logging_level : str
            Logging level for errors: "DEBUG", "INFO", "WARNING", "ERROR"
        max_retries : int
            Maximum number of retry attempts on error
        save_mapping : bool
            Whether to save the mapping of original to imputed values
        default_fill_value : str
            Whether to default fill value for null
        """
        # Store configuration parameters
        self.grouping_fields = grouping_fields
        self.method = method
        self.column_prefix = column_prefix
        self.detailed_metrics = detailed_metrics
        self.mapping_store_path = mapping_store_path
        self.error_logging_level = error_logging_level.upper()
        self.max_retries = max_retries
        self.save_mapping = save_mapping
        self.default_fill_value = default_fill_value

        # Set default output field name if not provided
        if not output_field_name and mode == "ENRICH":
            output_field_name = f"{column_prefix}{field_name}"

        # Configure logging
        self._configure_logging()

        cleaner_params = {
            "target_field": field_name,
            "grouping_fields": grouping_fields,
            "method": method
        }

        # Create basic statistical imputation cleaner
        cleaner = BasicStatisticalImputationCleaner(
            config = cleaner_params
        )

        # Initialize parent class
        super().__init__(
            field_name = field_name,
            cleaner = cleaner,
            mode = mode,
            output_field_name = output_field_name,
            batch_size = batch_size,
            cleaner_params = cleaner_params,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        self._original_df = None

        # Initialize mapping store if path is provided
        if mapping_store_path:
            self._initialize_mapping_store(mapping_store_path)

        # Update operation name and description
        self.name = f"statistical_imputation_{method}"
        self.description = f"Basic statistical imputation for {field_name} using {method}"

        # Performance tracking
        self.start_time = None
        self.process_count = 0
        self.error_count = 0
        self.retry_count = 0

        # For detailed metrics
        if self.detailed_metrics:
            self._generation_times = []
            self._error_fields = []

    def execute(
            self,
            data_source: Any,
            task_dir: Path,
            reporter: Any,
            **kwargs
    ) -> OperationResult:
        """
        Execute the basic statistical imputation operation.

        Parameters:
        -----------
        data_source : Any
            Source of data for processing (DataFrame or file path)
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
        try:
            # Start timing for performance metrics
            self.start_time = time.time()

            # Check if grouping fields exist in the data
            if isinstance(data_source, pd.DataFrame):
                if self.grouping_fields:
                    missing_grouping_fields = [
                        f for f in self.grouping_fields if f not in data_source.columns
                    ]
                    if missing_grouping_fields:
                        error_msg = (
                            f"Grouping fields not found in data: {missing_grouping_fields}"
                        )
                        logger.error(error_msg)
                        return OperationResult(
                            status = OperationStatus.ERROR,
                            error_message = error_msg
                        )

            # Call parent execute method
            logger.info(
                f"Starting basic statistical imputation using {self.method} for field {self.field_name}"
            )

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
                        "avg_time": sum(self._generation_times) / len(self._generation_times),
                        "total_times": len(self._generation_times)
                    }

                    if hasattr(self, "_error_fields") and self._error_fields:
                        result.metrics["error_details"] = {
                            "error_fields": self._error_fields[:100] # Limit to first 100
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
            logger.exception(f"Error during basic statistical imputation operation: {str(e)}")
            return OperationResult(
                status = OperationStatus.ERROR,
                error_message = str(e),
                execution_time = time.time() - (self.start_time or time.time())
            )

    def process_batch(
            self,
            batch: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process a batch of data using the configured basic statistical imputation cleaner.

        Args:
            batch (pd.DataFrame): Input batch of data.

        Returns:
            pd.DataFrame: Batch with the imputed field.
        """
        result = batch.copy()

        # Determine output field
        output_field = (
            self.field_name if self.mode == "REPLACE"
            else self.output_field_name or f"{self.column_prefix}{self.field_name}"
        )

        params = {
            "target_field": self.field_name,
            "grouping_fields": self.grouping_fields,
            "method": self.method
        }

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                # Run basic statistical imputation cleaner
                imputed_df = self.cleaner.execute(result.copy(), **params)

                # Update result with imputed values
                result[output_field] = imputed_df[self.field_name]

                # Count how many missing values were filled
                self.process_count += batch[self.field_name].isna().sum()

                return result
            except Exception as e:
                last_error = e
                retries += 1
                self.retry_count += 1

                if retries <= self.max_retries:
                    logger.debug(
                        f"Retry {retries}/{self.max_retries} - Error during imputation for field '{self.field_name}': {str(e)}"
                    )
                else:
                    logger.error(
                        f"Failed imputation for field '{self.field_name}' after {self.max_retries} retries: {str(e)}"
                    )
                    self.error_count += 1
                    self._error_fields.append(self.field_name) # Track field that failed

        # All retries failed — return original batch unmodified
        return result

    def _process_value(
            self,
            value: Any,
            **params
    ) -> Any:
        """
        Deprecated in batch mode. Not used in basic statistical imputation anymore.
        """
        logger.warning("_process_value is deprecated and not used in basic statistical based batch processing.")
        return value

    def _collect_metrics(
            self,
            df: pd.DataFrame,
            operation_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Collect metrics for the basic statistical imputation operation.

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

        # Add basic statistical imputation specific metrics
        metrics["model"] = {
            "field_name": self.field_name,
            "grouping_fields": self.grouping_fields,
            "method": self.method
        }

        # Calculate imputation-specific metrics
        if self._original_df is not None:
            original_missing = self._original_df[self.field_name].isna().sum()

            if self.mode == "REPLACE":
                current_missing = df[self.field_name].isna().sum()
                imputed_count = original_missing - current_missing
            else:  # ENRICH
                if self.output_field_name in df.columns:
                    current_missing = df[self.output_field_name].isna().sum()
                    imputed_count = original_missing - current_missing
                else:
                    imputed_count = 0

            metrics["imputation"] = {
                "original_missing_count": int(original_missing),
                "imputed_count": int(imputed_count),
                "imputation_rate": (
                    float(imputed_count / original_missing) if original_missing > 0 else 1.0
                )
            }

        return metrics

    def _configure_logging(
            self
    ):
        """
        Configure logging based on error_logging_level.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        log_level = getattr(logging, self.error_logging_level, logging.WARNING)
        logger.setLevel(log_level)

    def _initialize_mapping_store(
            self,
            path: Union[str, Path]
    ) -> None:
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
