"""
PAMOLA Project - PAMOLA Core
------------------------
Simple Data Analysis Task Template
Version: 1.0.0
Created: 2025-05-10
Author: PAMOLA Core Team

Description:
This template provides a lightweight structure for simple data parsing and analysis tasks.
It demonstrates basic task lifecycle, configuration loading, and standard operation execution
without the complexity of encryption or checkpointing.

Usage:
1. Rename this file to match your task ID (e.g., "t_1A_profile_data.py")
2. Update the task_id, task_type, and description
3. Configure your input datasets and operations in the configure_operations() method
4. Run the task from command line with appropriate arguments

Example:
python t_1A_profile_data.py --data_repository /path/to/data
"""

import argparse
import sys

# Import PAMOLA Core modules
from pamola_core.utils.tasks.base_task import BaseTask


class SimpleAnalysisTask(BaseTask):
    """
    Simple data analysis task template for quick data processing and insights.

    This template demonstrates how to create a basic task that loads data,
    performs simple analysis operations, and generates a structured report.
    It's designed for straightforward data processing without complex
    dependencies or security requirements.

    @author: PAMOLA Core Team
    @version: 1.0.0
    """

    def __init__(self):
        """Initialize the task with basic information."""
        super().__init__(
            task_id="t_1A_simple_analysis",  # Replace with your task ID
            task_type="analysis",  # Replace with appropriate type
            description="Simple data analysis template with basic operations",

            # Define your input datasets
            input_datasets={
                "primary_data": "DATA/raw/sample_data.csv",
                # Add more datasets if needed
            },

            # Optional: Add reference datasets
            auxiliary_datasets={
                # "lookup_table": "DATA/dictionaries/reference.csv",
            },

            version="1.0.0"
        )

    def configure_operations(self) -> None:
        """
        Configure the operations to be executed by this task.

        This method defines the sequence of operations that will be executed
        in order. Each operation performs a specific data processing function
        and outputs results for subsequent operations or final reporting.
        """
        self.logger.info("Configuring task operations")

        # Operation 1: Data summary operation
        # This operation generates basic statistics for the dataset
        self.add_operation(
            "DataSummaryOperation",
            dataset_name="primary_data",
            calculate_statistics=True,
            include_null_analysis=True,
            output_prefix="data_summary"
        )

        # Operation 2: Field distribution analysis
        # This analyzes value distributions for specified fields
        self.add_operation(
            "DistributionAnalysisOperation",
            dataset_name="primary_data",
            fields=["age", "income", "region"],  # Replace with your actual fields
            create_visualizations=True,
            bins=10,
            output_prefix="distributions"
        )

        # Operation 3: Outlier detection
        # This identifies outliers in the numeric fields
        self.add_operation(
            "OutlierDetectionOperation",
            dataset_name="primary_data",
            method="iqr",  # Options: "iqr", "zscore", "dbscan"
            numeric_fields=["age", "income"],  # Replace with your fields
            output_file="outliers_report.json"
        )

        # Operation 4: Simple data filtering
        # This filters the dataset based on criteria and saves the result
        self.add_operation(
            "FilterOperation",
            dataset_name="primary_data",
            conditions=[
                {"field": "age", "operator": ">=", "value": 18},
                {"field": "income", "operator": ">", "value": 0}
            ],
            output_file="filtered_data.csv"
        )

        # Add more operations as needed for your specific analysis

        self.logger.info(f"Configured {len(self.operations)} operations")

    def execute(self) -> bool:
        """
        Execute the task operations in sequence.

        This method runs all configured operations and collects their results.
        It's called automatically by the run() method but can be customized
        for specific execution requirements.

        Returns:
            bool: True if all operations completed successfully, False otherwise
        """
        self.logger.info(f"Executing {len(self.operations)} operations")

        # Use the parent class implementation for standard execution
        execution_success = super().execute()

        # You can add custom handling here if needed
        if execution_success:
            self.logger.info("All operations completed successfully")
        else:
            self.logger.error(f"Task execution failed: {self.error_info}")

        return execution_success

    def finalize(self, success: bool) -> bool:
        """
        Finalize the task by processing results and generating additional outputs.

        This method is called after execution completes. It processes operation
        results, generates summary metrics, and prepares the final report.

        Args:
            success: Whether the task executed successfully

        Returns:
            bool: True if finalization was successful, False otherwise
        """
        self.logger.info("Finalizing task and processing results")

        # Process results if execution was successful
        if success:
            # Aggregate metrics from all operations
            aggregated_metrics = {}
            for op_name, result in self.results.items():
                if hasattr(result, 'metrics') and result.metrics:
                    aggregated_metrics[op_name] = result.metrics

            # Add aggregated metrics to the report
            if self.reporter:
                for category, metrics in aggregated_metrics.items():
                    self.reporter.add_nested_metric(category, "summary", metrics)

            # Add count of artifacts by type
            artifact_counts = {}
            for artifact in self.artifacts:
                artifact_type = artifact.artifact_type
                if artifact_type not in artifact_counts:
                    artifact_counts[artifact_type] = 0
                artifact_counts[artifact_type] += 1

            if self.reporter:
                self.reporter.add_metric("artifact_counts", artifact_counts)

        # Call parent finalize method to complete standard finalization
        return super().finalize(success)


def parse_arguments():
    """Parse command line arguments for task configuration."""
    parser = argparse.ArgumentParser(description="Simple Data Analysis Task")

    # Standard task arguments
    parser.add_argument("--data_repository", help="Path to data repository")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Log level")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue execution if an operation fails")

    # Task-specific arguments
    parser.add_argument("--fields", help="Comma-separated list of fields to analyze")
    parser.add_argument("--output_format", choices=["json", "csv"], default="json",
                        help="Output format for reports")

    return parser.parse_args()


def main():
    """Main entry point for the task."""
    # Parse command line arguments
    args = parse_arguments()

    # Create task instance
    task = SimpleAnalysisTask()

    # Convert arguments to dictionary for task configuration
    arg_dict = vars(args)

    # Add custom processing for specific arguments if needed
    if args.fields:
        arg_dict["fields"] = [f.strip() for f in args.fields.split(",")]

    # Run the task with the provided arguments
    success = task.run(arg_dict)

    # Handle task completion
    if success:
        print(f"Task completed successfully. Report saved to: {task.reporter.report_path}")
        sys.exit(0)
    else:
        print(f"Task failed: {task.error_info.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()