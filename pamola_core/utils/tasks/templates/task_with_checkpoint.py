"""
PAMOLA Project - PAMOLA Core
------------------------
Checkpoint-Enabled Task Template
Version: 1.0.0
Created: 2025-05-10
Author: PAMOLA Core Team

Description:
This template provides a structure for creating tasks that support checkpoints and resumable 
execution for processing large datasets or long-running operations. It demonstrates how to use 
the context_manager for creating checkpoints and recovering from interruptions.

Usage:
1. Rename this file to match your task ID (e.g., "t_2C_resumable_process.py")
2. Update the task_id, task_type, and description
3. Configure your operations in the configure_operations() method
4. Add checkpoint creation points in your operations where appropriate
5. Run the task with the appropriate arguments

Example:
python t_2C_resumable_process.py --data_repository /path/to/data
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import PAMOLA Core modules
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationResult


class CheckpointTask(BaseTask):
    """
    Task template with checkpoint support for long-running or interruptible processes.

    This template demonstrates how to create tasks that can be safely interrupted
    and resumed later, which is essential for long-running processes or tasks
    that process very large datasets. It uses the context_manager to create
    checkpoints at strategic points during execution.

    @author: PAMOLA Core Team
    @version: 1.0.0
    """

    def __init__(self):
        """Initialize the task with basic information and checkpoint capability."""
        super().__init__(
            task_id="t_2C_checkpoint_task",  # Replace with your task ID
            task_type="batch_processing",  # Replace with appropriate type
            description="Checkpoint-enabled task for processing large datasets",

            # Define your input datasets
            input_datasets={
                "large_dataset": "DATA/raw/large_dataset.csv",
                # Add more datasets if needed
            },

            # Optional: Add reference datasets
            auxiliary_datasets={
                # "reference_data": "DATA/dictionaries/reference.csv",
            },

            version="1.0.0"
        )

        # Additional checkpoint-specific attributes
        self._batch_size = 10000  # Number of records to process per batch
        self._checkpoint_frequency = 5  # Create checkpoint every N batches
        self._last_checkpoint_time = None
        self._total_batches_processed = 0

    def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the task with checkpoint recovery capability.

        This method extends the base initialization to check for existing
        checkpoints and set up for resumable execution.

        Args:
            args: Command line arguments to override configuration

        Returns:
            True if initialization is successful, False otherwise
        """
        # Call the parent initialization first
        if not super().initialize(args):
            return False

        # Log checkpoint status if resuming from checkpoint
        checkpoint_status = self.get_checkpoint_status()
        if checkpoint_status["resuming_from_checkpoint"]:
            self.logger.info(f"Resuming from checkpoint: {checkpoint_status['checkpoint_name']}")
            self.logger.info(f"Last completed operation index: {checkpoint_status['operation_index']}")

            # Load custom state if available
            if checkpoint_status["has_restored_state"]:
                restored_state = self.context_manager._restored_state
                if 'custom_state' in restored_state:
                    custom_state = restored_state['custom_state']
                    self._total_batches_processed = custom_state.get('batches_processed', 0)
                    self.logger.info(f"Restored {self._total_batches_processed} previously processed batches")
        else:
            self.logger.info("Starting new task execution (no checkpoint found)")

        # Apply any batch size override from arguments
        if args and "batch_size" in args:
            try:
                self._batch_size = int(args["batch_size"])
                self.logger.info(f"Using batch size: {self._batch_size}")
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid batch size: {args['batch_size']}, using default: {self._batch_size}")

        # Apply any checkpoint frequency override from arguments
        if args and "checkpoint_frequency" in args:
            try:
                self._checkpoint_frequency = int(args["checkpoint_frequency"])
                self.logger.info(f"Creating checkpoints every {self._checkpoint_frequency} batches")
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid checkpoint frequency: {args['checkpoint_frequency']}, "
                    f"using default: {self._checkpoint_frequency}"
                )

        return True

    def configure_operations(self) -> None:
        """
        Configure the operations to be executed by this task.

        Each operation should be designed to support checkpointing and
        resumable execution, especially for long-running operations.
        """
        self.logger.info("Configuring checkpoint-enabled operations")

        # Operation 1: Initial data loading and validation
        # This operation loads the data and performs validation
        self.add_operation(
            "DataLoadOperation",
            dataset_name="large_dataset",
            validate_schema=True,
            detect_encoding=True,
            output_prefix="validated_data"
        )

        # Operation 2: Batch processing operation
        # This demonstrates how to process data in batches with checkpoints
        self.add_operation(
            "BatchProcessingOperation",
            dataset_name="large_dataset",
            batch_size=self._batch_size,
            checkpoint_frequency=self._checkpoint_frequency,
            processing_steps=[
                {"type": "clean", "params": {"remove_nulls": True}},
                {"type": "transform", "params": {"fields": ["column1", "column2"]}},
                {"type": "validate", "params": {"rules": ["rule1", "rule2"]}}
            ],
            output_file="processed_data.csv",
            create_summary=True
        )

        # Operation 3: Aggregation operation
        # This operation aggregates the processed data
        self.add_operation(
            "AggregationOperation",
            input_file="processed_data.csv",  # Uses output from previous operation
            group_by_fields=["region", "category"],
            aggregate_fields=[
                {"field": "value", "functions": ["sum", "avg", "count"]}
            ],
            output_file="aggregated_data.csv"
        )

        # Operation 4: Report generation
        # This creates the final report after processing
        self.add_operation(
            "ReportGenerationOperation",
            input_files=["processed_data.csv", "aggregated_data.csv"],
            report_type="comprehensive",
            include_visualizations=True,
            output_prefix="final_report"
        )

        self.logger.info(f"Configured {len(self.operations)} checkpoint-enabled operations")

    def execute(self) -> bool:
        """
        Execute the task with checkpoint support.

        This method runs operations with support for interruption and resumption,
        creating checkpoints at strategic points in the execution.

        Returns:
            True if execution was successful, False otherwise
        """
        # Start or resume execution based on checkpoint status
        checkpoint_status = self.get_checkpoint_status()
        if checkpoint_status["resuming_from_checkpoint"]:
            self.logger.info(f"Resuming execution from operation {checkpoint_status['operation_index'] + 1}")
        else:
            self.logger.info("Starting execution from the beginning")

        # Execute operations using the standard implementation
        # (The BaseTask.execute method handles checkpoint resumption automatically)
        execution_success = super().execute()

        # Create a final checkpoint if execution was successful
        if execution_success:
            self._create_custom_checkpoint("final")
            self.logger.info("Created final execution checkpoint")

        return execution_success

    def _create_custom_checkpoint(self, name: str) -> None:
        """
        Create a custom checkpoint with task-specific state information.

        Args:
            name: Name for the checkpoint
        """
        try:
            # Get current state from context manager
            current_state = self.context_manager.get_current_state().copy() if self.context_manager else {}

            # Add custom state information
            if 'custom_state' not in current_state:
                current_state['custom_state'] = {}

            # Update custom state with task-specific information
            current_state['custom_state'].update({
                'batches_processed': self._total_batches_processed,
                'last_checkpoint_time': datetime.now().isoformat() if not self._last_checkpoint_time else self._last_checkpoint_time,
                'checkpoint_name': name
            })

            # Save the checkpoint
            if self.context_manager:
                checkpoint_name = f"custom_{name}_{self.task_id}"
                self.context_manager.save_execution_state(current_state, checkpoint_name)
                self._last_checkpoint_time = datetime.now().isoformat()
                self.logger.info(f"Created custom checkpoint: {checkpoint_name}")
            else:
                self.logger.warning("Context manager not available, skipping checkpoint creation")

        except Exception as e:
            self.logger.error(f"Error creating custom checkpoint: {str(e)}")

    def update_batch_progress(self, batches_processed: int) -> None:
        """
        Update the batch processing progress and create checkpoints when appropriate.

        This method should be called from batch processing operations to update
        progress and create checkpoints at regular intervals.

        Args:
            batches_processed: Number of batches processed in this update
        """
        # Update total batches processed
        self._total_batches_processed += batches_processed

        # Check if we should create a checkpoint
        if self._total_batches_processed % self._checkpoint_frequency == 0:
            checkpoint_name = f"batch_{self._total_batches_processed}"
            self._create_custom_checkpoint(checkpoint_name)
            self.logger.info(f"Created checkpoint after processing {self._total_batches_processed} batches")

    def process_batch(self, batch_data: Any, batch_index: int) -> Dict[str, Any]:
        """
        Process a single batch of data with automatic checkpointing.

        This is a utility method that can be used by operations to process
        batches of data with consistent checkpoint creation.

        Args:
            batch_data: Data for this batch to process
            batch_index: Index of this batch

        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Processing batch {batch_index}")

        # Simulate batch processing
        start_time = time.time()

        # Here you would actually process the batch
        # This is just a placeholder for the actual processing logic
        batch_result = {
            "batch_index": batch_index,
            "records_processed": len(batch_data) if hasattr(batch_data, "__len__") else 0,
            "processing_time": 0.0
        }

        # Simulate some processing time
        # In a real task, this would be replaced with actual data processing
        processing_time = 0.1  # seconds
        time.sleep(processing_time)

        # Finish processing
        end_time = time.time()
        batch_result["processing_time"] = end_time - start_time

        # Update progress and create checkpoint if needed
        self.update_batch_progress(1)

        return batch_result

    def finalize(self, success: bool) -> bool:
        """
        Finalize the task and clean up checkpoints if appropriate.

        This method extends the base finalization to handle checkpoint
        cleanup and final reporting.

        Args:
            success: Whether the task executed successfully

        Returns:
            True if finalization was successful, False otherwise
        """
        self.logger.info(f"Finalizing task (success: {success})")

        # Add checkpoint-specific metrics
        if self.reporter:
            checkpoint_metrics = {
                "total_batches_processed": self._total_batches_processed,
                "checkpoint_frequency": self._checkpoint_frequency,
                "resumable": True
            }
            self.reporter.add_metric("checkpoint_info", checkpoint_metrics)

        # If successful, clean up intermediate checkpoints
        if success and self.context_manager:
            try:
                # Keep only the final checkpoint
                removed_count = self.context_manager.cleanup_old_checkpoints(max_checkpoints=1)
                self.logger.info(f"Cleaned up {removed_count} intermediate checkpoints")
            except Exception as e:
                self.logger.warning(f"Error cleaning up checkpoints: {str(e)}")

        # Call parent finalize method
        return super().finalize(success)


def parse_arguments():
    """Parse command line arguments for checkpoint-enabled task."""
    parser = argparse.ArgumentParser(description="Checkpoint-Enabled Task")

    # Standard task arguments
    parser.add_argument("--data_repository", help="Path to data repository")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Log level")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue execution if an operation fails")

    # Checkpoint-specific arguments
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of records to process in each batch")
    parser.add_argument("--checkpoint_frequency", type=int, default=5,
                        help="Create checkpoint every N batches")
    parser.add_argument("--resume_from_checkpoint", choices=["auto", "none", "latest"],
                        default="auto", help="Checkpoint resumption mode")

    return parser.parse_args()


def main():
    """Main entry point for the checkpoint-enabled task."""
    # Parse command line arguments
    args = parse_arguments()

    # Create task instance
    task = CheckpointTask()

    # Convert arguments to dictionary for task configuration
    arg_dict = vars(args)

    # Run the task with the provided arguments
    success = task.run(arg_dict)

    # Handle task completion
    if success:
        print(f"Task completed successfully. Report saved to: {task.reporter.report_path}")
        sys.exit(0)
    else:
        print(f"Task failed: {task.error_info}")
        sys.exit(1)


if __name__ == "__main__":
    main()