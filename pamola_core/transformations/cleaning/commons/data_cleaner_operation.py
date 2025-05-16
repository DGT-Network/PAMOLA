import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
# Import the base class differently to avoid circular import
from pamola_core.transformations.cleaning.cleaners.base_data_cleaner import BaseDataCleaner
from pamola_core.transformations.cleaning.commons.mapping_store import MappingStore
from pamola_core.transformations.cleaning.commons.operations import FieldOperation
from pamola_core.utils.io import ensure_directory, write_json
from pamola_core.utils.ops.op_registry import register
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


@register()
class DataCleanerOperation(FieldOperation):
    """
    Operation for clean data generation using a cleaner.

    This operation integrates with PAMOLA's transformation framework and applies a configured
    cleaner to generate synthetic or imputed data for specific fields while preserving
    data consistency and structure.
    """

    name: str = "data_cleaner_operation"
    description: str = "Operation for create clean data"

    def __init__(
        self,
        field_name: str,
        cleaner: BaseDataCleaner,
        mode: str = "REPLACE",
        output_field_name: Optional[str] = None,
        batch_size: int = 10000,
        mapping_store: Optional[MappingStore] = None,
        cleaner_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the cleaner operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to process
        cleaner : BaseDataCleaner
            BaseDataCleaner for creating clean data
        mode : str
            Operation mode: "REPLACE" or "ENRICH"
        output_field_name : str, optional
            Name of the output field (used in ENRICH mode)
        batch_size : int
            Size of batches for processing
        mapping_store : MappingStore, optional
            Store for mappings between original and synthetic values
        cleaner_params : Dict[str, Any], optional
            Additional parameters for the cleaner
        """
        
        # Initialize the base class
        self.field_name = field_name
        self.mode = mode
        self.output_field_name = output_field_name
        self.batch_size = batch_size
        self.column_prefix = "clean_"  # Assuming this is a default from FieldOperation
        self._original_df = None
        self.process_count = 0
        self._metrics_collector = None  # Will be initialized in FieldOperation
        
        # Call the init method of FieldOperation to properly set up
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            batch_size=batch_size,
        )

        self.cleaner = cleaner
        self.mapping_store = mapping_store or MappingStore()
        self.cleaner_params = cleaner_params or {}
        self.logger = logging.getLogger(__name__)

        # Update operation name and description
        self.name = f"{cleaner.__class__.__name__}_operation"
        self.description = f"Operation for generating clean {field_name} data using {cleaner.__class__.__name__}"

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a batch of data using the configured cleaner.

        This method applies the process data clean in the target field
        for the entire batch. Supports REPLACE and ENRICH modes.

        Parameters:
        -----------
        batch : pd.DataFrame
            Batch of data to process.

        Returns:
        --------
        pd.DataFrame
            Batch with data values in-place or enriched as a new column.
        """
        result_batch = batch.copy()

        # Validate input field
        if not self._validate_input_field(result_batch):
            return result_batch

        # Determine target field and validate for ENRICH mode
        target_field = self._get_target_field_name()
        if self.mode == "ENRICH" and not target_field:
            raise ValueError("Output field name must be specified in ENRICH mode.")

        field_values = result_batch[self.field_name]
        
        # Skip if there are no nulls to clean
        if field_values.isna().sum() == 0:
            if self.mode == "ENRICH":
                result_batch[target_field] = field_values
            return result_batch

        return self._apply_cleaning(result_batch, target_field)

    def _validate_input_field(self, df: pd.DataFrame) -> bool:
        """
        Validates if the input field exists in the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to validate
            
        Returns:
        --------
        bool
            True if field exists, False otherwise
        """
        if self.field_name not in df.columns:
            self.logger.warning(f"Field '{self.field_name}' not found in batch, skipping.")
            return False
        return True

    def _get_target_field_name(self) -> str:
        """
        Determines the target field name based on operation mode.
        
        Returns:
        --------
        str
            Name of the target field for cleaned data
        """
        if self.mode == "REPLACE":
            return self.field_name
        else:
            return self.output_field_name or f"{self.column_prefix}{self.field_name}"

    def _apply_cleaning(self, df: pd.DataFrame, target_field: str) -> pd.DataFrame:
        """
        Applies the cleaner to the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe to clean
        target_field : str
            Field to store cleaned values
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with cleaned values
        """
        try:
            # Execute cleaner on full DataFrame
            cleaned_df = self.cleaner.execute(df.copy(), **self.cleaner_params)

            # Inject cleaned values into the correct column
            df[target_field] = cleaned_df[self.field_name]
            self.process_count += df[self.field_name].isna().sum()
            
            return df

        except Exception as e:
            self.logger.exception(f"Error during data clean for field '{self.field_name}': {str(e)}")

            if self.mode == "ENRICH":
                df[target_field] = df[self.field_name]

            return df

    def _process_value(self, value: Any, **params) -> Any:
        """
        Deprecated in batch mode. Not used in data clean anymore.
        
        Parameters:
        -----------
        value : Any
            Value to process
        **params : dict
            Additional parameters
            
        Returns:
        --------
        Any
            The input value unchanged
        """
        self.logger.warning("_process_value is deprecated and not used in data clean batch processing.")
        return value

    def _collect_metrics(self, df: pd.DataFrame, operation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Collects metrics for the cleaner operation.

        Parameters:
        -----------
        df : pd.DataFrame
            Processed DataFrame with cleaned data
        operation_params : Dict[str, Any], optional
            Additional operation parameters for metrics collection

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing various metrics about the operation
        """
        # Get base metrics from parent class
        metrics = super()._collect_metrics(df, operation_params)

        # Add cleaner-specific information
        metrics["cleaner"] = self._get_cleaner_metrics()
        
        # Add operation parameters
        metrics["operation"] = self._get_operation_metrics()

        # Compare distributions only if we have original data
        if self._original_df is not None:
            self._add_distribution_metrics(df, metrics)

        return metrics

    def _get_cleaner_metrics(self) -> Dict[str, Any]:
        """
        Gets metrics about the cleaner component.
        
        Returns:
        --------
        Dict[str, Any]
            Cleaner metrics
        """
        return {
            "type": self.cleaner.__class__.__name__,
            "parameters": getattr(self, "cleaner_params", {}),
        }

    def _get_operation_metrics(self) -> Dict[str, Any]:
        """
        Gets metrics about the operation configuration.
        
        Returns:
        --------
        Dict[str, Any]
            Operation configuration metrics
        """
        return {
            "field_name": self.field_name,
            "mode": self.mode,
            "output_field": self.output_field_name if self.mode == "ENRICH" else None,
            "mapping_count": self._get_mapping_count(),
        }

    def _get_mapping_count(self) -> int:
        """
        Gets the count of field mappings.
        
        Returns:
        --------
        int
            Number of mappings for the field
        """
        if hasattr(self.mapping_store, "get_field_mappings"):
            return len(self.mapping_store.get_field_mappings(self.field_name))
        return 0

    def _add_distribution_metrics(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        """
        Adds distribution comparison metrics to the metrics dictionary.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        metrics : Dict[str, Any]
            Metrics dictionary to update
        """
        try:
            # Get original data series
            orig_series = self._original_df[self.field_name]

            # Determine which field contains generated data
            gen_series = self._get_generated_series(df)

            # Calculate distribution comparison metrics if both series are available
            if gen_series is not None:
                # Ensure transformation_metrics section exists
                if "transformation_metrics" not in metrics:
                    metrics["transformation_metrics"] = {}

                # Get comparison metrics
                compare_metrics = self._metrics_collector.compare_distributions(
                    orig_series, gen_series
                )
                metrics["distribution_metrics"] = compare_metrics

                # Add categorical data metrics if applicable
                if self._is_categorical_data(orig_series):
                    self._add_value_preservation_metrics(orig_series, gen_series, metrics)
                    
        except Exception as e:
            # Log error but don't fail the entire metrics collection
            self.logger.warning(f"Error collecting comparison metrics: {str(e)}")
            metrics["errors"] = {"comparison": str(e)}

    def _get_generated_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Gets the series containing generated data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
            
        Returns:
        --------
        Optional[pd.Series]
            Series with generated data or None
        """
        if self.mode == "REPLACE":
            return df[self.field_name]
        elif self.mode == "ENRICH" and self.output_field_name in df.columns:
            return df[self.output_field_name]
        return None

    def _is_categorical_data(self, series: pd.Series) -> bool:
        """
        Checks if a series contains categorical data.
        
        Parameters:
        -----------
        series : pd.Series
            Series to check
            
        Returns:
        --------
        bool
            True if series contains categorical data
        """
        return series.dtype == "object" or series.dtype.name == "category"

    def _add_value_preservation_metrics(
        self, orig_series: pd.Series, gen_series: pd.Series, metrics: Dict[str, Any]
    ) -> None:
        """
        Adds value preservation metrics for categorical data.
        
        Parameters:
        -----------
        orig_series : pd.Series
            Original data series
        gen_series : pd.Series
            Generated data series
        metrics : Dict[str, Any]
            Metrics dictionary to update
        """
        # Calculate value preservation (what percentage of values remain the same)
        matching_values = (orig_series == gen_series).sum()
        total_values = len(orig_series)
        preservation_rate = (
            float(matching_values) / total_values
            if total_values > 0
            else 0.0
        )

        metrics["value_preservation"] = {
            "matching_values": int(matching_values),
            "total_values": int(total_values),
            "preservation_rate": preservation_rate,
        }

    def _save_field_mappings(self, task_dir: Path, result: OperationResult) -> None:
        """
        Saves the field mappings to both the task directory and a detailed maps subdirectory.

        Parameters:
        -----------
        task_dir : Path
            Directory for storing operation artifacts
        result : OperationResult
            The result object to which artifacts will be added
        """
        try:
            maps_dir = ensure_directory(task_dir / "maps")
            maps_detail_file = maps_dir / f"{self.field_name}_mappings.json"
            mappings_file = maps_dir / f"{self.name}_{self.field_name}_mappings.json"

            serializable_mappings = self._prepare_serializable_mappings()

            write_json(serializable_mappings, mappings_file)
            write_json(serializable_mappings, maps_detail_file)

            result.add_artifact(
                path=mappings_file,
                artifact_type="json",
                description=f"Mappings for {self.field_name} field",
            )

            self.logger.info(
                f"Saved {len(serializable_mappings)} mappings to {mappings_file} and {maps_detail_file}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to save mappings: {str(e)}")

    def _prepare_serializable_mappings(self) -> List[Dict[str, Any]]:
        """
        Prepares field mappings as serializable list of dictionaries.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of mapping dictionaries
        """
        field_mappings = self.mapping_store.get_field_mappings(self.field_name)
        return [
            {"original": original, "synthetic": synthetic, "field": self.field_name}
            for original, synthetic in field_mappings.items()
        ]

    def execute(
        self, data_source: Any, task_dir: Path, reporter: Any, **kwargs
    ) -> OperationResult:
        """
        Executes the cleaner operation.

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
        # Call the parent's execute method
        result = super().execute(data_source, task_dir, reporter, **kwargs)

        if result.status == OperationStatus.SUCCESS:
            self._save_field_mappings(task_dir, result)

        return result