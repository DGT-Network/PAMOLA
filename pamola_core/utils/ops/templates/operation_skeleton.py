"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: MyOperation
Description: [REPLACE WITH YOUR OPERATION DESCRIPTION]
Author: [YOUR NAME]
Created: [DATE]
License: BSD 3-Clause

This operation [BRIEFLY DESCRIBE WHAT IT DOES]

Key features:
- [FEATURE 1]
- [FEATURE 2]
- [FEATURE 3]
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd

from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_data_writer import DataWriter
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker


class MyOperationConfig(OperationConfig):
    """Configuration schema for MyOperation."""

    # Define JSON Schema for configuration validation
    schema = {
        "type": "object",
        "properties": {
            "operation_name": {"type": "string"},
            "version": {"type": "string"},
            # TODO: Add your operation-specific parameters here
            "column_name": {"type": "string"},
            "threshold": {"type": "number", "minimum": 0, "maximum": 1.0},
            "output_suffix": {"type": "string"}
        },
        "required": ["operation_name", "version", "column_name"]
    }


class MyOperation(BaseOperation):
    """
    [REPLACE WITH YOUR OPERATION DESCRIPTION]

    This class implements [DESCRIBE WHAT THE OPERATION DOES].
    """

    def __init__(self,
                 name: str = "my_operation",
                 description: str = "My custom operation",
                 column_name: str = None,
                 threshold: float = 0.5,
                 output_suffix: str = "_processed",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None):
        """
        Initialize the operation.

        Parameters:
        -----------
        name : str
            Name of the operation
        description : str
            Description of what the operation does
        column_name : str
            Name of the column to process
        threshold : float
            Threshold value for processing (between 0 and 1)
        output_suffix : str
            Suffix to append to output column names
        use_encryption : bool
            Whether to encrypt output files
        encryption_key : str or Path, optional
            The encryption key or path to a key file
        """
        super().__init__(
            name=name,
            description=description,
            use_encryption=use_encryption,
            encryption_key=encryption_key
        )

        # Store operation-specific parameters
        self.column_name = column_name
        self.threshold = threshold
        self.output_suffix = output_suffix

        # Set operation version (follow semantic versioning)
        self.version = "1.0.0"

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : ProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        self.logger.info(f"Starting {self.name} operation")

        # 1. Create result and writer objects
        result = OperationResult(status=OperationStatus.PENDING)
        writer = DataWriter(task_dir=task_dir, logger=self.logger, progress_tracker=progress_tracker)

        # 2. Save operation configuration
        self.save_config(task_dir)

        try:
            # 3. Set up progress tracking
            total_steps = 4  # Update this based on your operation's steps
            if progress_tracker:
                progress_tracker.total = total_steps
                progress_tracker.update(0, {"status": "starting"})

            # 4. Get input data
            self.logger.info("Loading input data")
            df, error_info = data_source.get_dataframe("main")

            if df is None:
                error_message = f"Failed to load input data: {error_info['message'] if error_info else 'Unknown error'}"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message
                )

            if progress_tracker:
                progress_tracker.update(1, {"status": "data_loaded", "rows": len(df)})

            # 5. Validate inputs
            if self.column_name not in df.columns:
                error_message = f"Column '{self.column_name}' not found in input data"
                self.logger.error(error_message)
                return OperationResult(
                    status=OperationStatus.ERROR,
                    error_message=error_message
                )

            # 6. Process data - TODO: Replace with your actual processing logic
            self.logger.info(f"Processing column: {self.column_name}")

            # --- START OF YOUR BUSINESS LOGIC ---
            # Example: Add a new column with processed values
            output_column = f"{self.column_name}{self.output_suffix}"

            # TODO: Replace this placeholder logic with your actual transformation
            df[output_column] = df[self.column_name].apply(
                lambda x: x * self.threshold if isinstance(x, (int, float)) else x)
            # --- END OF YOUR BUSINESS LOGIC ---

            if progress_tracker:
                progress_tracker.update(2, {"status": "processing_complete"})

            # 7. Calculate metrics - TODO: Replace with your actual metrics
            metrics = {
                "row_count": len(df),
                "processed_column": self.column_name,
                "output_column": output_column,
                "threshold_used": self.threshold,
                # Add your operation-specific metrics here
                "null_values_count": df[self.column_name].isna().sum(),
                "processed_values_count": (~df[output_column].isna()).sum()
            }

            # Add metrics to result
            for key, value in metrics.items():
                result.add_metric(key, value)

            # Write metrics to file
            writer.write_metrics(metrics, "operation_metrics")

            if progress_tracker:
                progress_tracker.update(3, {"status": "metrics_calculated"})

            # 8. Write output data
            self.logger.info("Writing output data")
            output_result = writer.write_dataframe(
                df,
                name="processed_data",
                format="csv",
                index=False,
                encryption_key=self.encryption_key if self.use_encryption else None
            )

            # Register the output file as an artifact
            result.add_artifact(
                artifact_type="csv",
                path=output_result.path,
                description=f"Processed data with {self.column_name} transformation",
                category="output"
            )

            if progress_tracker:
                progress_tracker.update(4, {"status": "complete"})

            # 9. Set result status to success
            result.status = OperationStatus.SUCCESS
            self.logger.info(f"Operation {self.name} completed successfully")

            return result

        except Exception as e:
            # Handle any errors
            error_message = f"Error in {self.name} operation: {str(e)}"
            self.logger.exception(error_message)

            result.status = OperationStatus.ERROR
            result.error_message = error_message

            return result


# Register the operation so it's discoverable
register_operation(MyOperation)