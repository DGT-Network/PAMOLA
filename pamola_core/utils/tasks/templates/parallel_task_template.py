"""
PAMOLA Task Template - Parallel Processing
------------------------------------------
This template demonstrates how to create a PAMOLA task that processes data in parallel
to improve performance for compute-intensive or large data operations.

Instructions:
1. Rename this file to match your task ID (e.g., "t_1X_parallel_task.py")
2. Update the task_id, task_type, and description
3. Configure the parallel operations in configure_operations()
4. Adjust the number of parallel processes as needed

Example usage:
python t_1X_parallel_task.py --data_repository /path/to/repository --parallel_processes 4
"""

import argparse
import sys

# Import PAMOLA Core modules
from pamola_core.utils.tasks.base_task import BaseTask


class ParallelTask(BaseTask):
    """
    Example task template demonstrating parallel processing in PAMOLA.

    This template shows how to implement parallel execution of operations
    for improved performance when processing large datasets or running
    computationally intensive operations.

    @author: PAMOLA Team
    @version: 1.0.0
    """

    def __init__(self):
        """Initialize the task with basic information."""
        super().__init__(
            task_id="t_1X_parallel_task",
            task_type="parallel_processing",
            description="Example task for parallel data processing",

            # Large dataset(s) to process in parallel
            input_datasets={
                "large_dataset": "DATA/raw/large_dataset.csv",
                # Add more datasets as needed
            },

            # Optional reference datasets
            auxiliary_datasets={
                "reference_data": "DATA/dictionaries/reference.csv",
            },

            version="1.0.0"
        )

    def configure_operations(self):
        """
        Configure the operations to be executed in parallel.

        This method sets up a list of operations that can be executed in parallel,
        each processing a subset of the data or performing a different computation.
        """
        self.logger.info("Configuring parallel operations for task")

        # Example: Split input data into partitions
        self.add_operation(
            "DataPartitionOperation",
            dataset_name="large_dataset",
            partition_count=self.parallel_processes,  # Use the configured number of parallel processes
            partition_method="equal_rows",  # Split into roughly equal chunks by row count
            output_prefix="partition"
        )

        # Process each partition in parallel
        # Each instance of the operation will process a different partition
        for i in range(self.parallel_processes):
            self.add_operation(
                "ProcessPartitionOperation",
                dataset_name=f"partition_{i}",  # Reference the partition created above
                partition_id=i,
                transformation_rules={
                    "field1": "normalize",
                    "field2": "standardize",
                    "field3": "binning"
                },
                output_file=f"processed_partition_{i}.csv"
            )

        # Merge the processed partitions
        self.add_operation(
            "MergePartitionsOperation",
            partition_files_pattern="processed_partition_*.csv",
            output_file="merged_results.csv"
        )

        # Example: Execute different operations in parallel
        # These will be run concurrently rather than sequentially
        parallel_operations = [
            {
                "name": "StatisticalAnalysisOperation",
                "params": {
                    "dataset_name": "large_dataset",
                    "analysis_type": "descriptive",
                    "output_file": "statistical_analysis.json"
                }
            },
            {
                "name": "CorrelationOperation",
                "params": {
                    "dataset_name": "large_dataset",
                    "method": "pearson",
                    "output_file": "correlation_results.csv"
                }
            },
            {
                "name": "ClusteringOperation",
                "params": {
                    "dataset_name": "large_dataset",
                    "algorithm": "kmeans",
                    "n_clusters": 5,
                    "output_file": "clustering_results.csv"
                }
            },
            {
                "name": "OutlierDetectionOperation",
                "params": {
                    "dataset_name": "large_dataset",
                    "method": "isolation_forest",
                    "output_file": "outliers.csv"
                }
            }
        ]

        # Add all parallel operations
        for op in parallel_operations:
            self.add_operation(op["name"], **op["params"])

        self.logger.info(f"Configured {len(self.operations)} operations")

    def execute(self) -> bool:
        """
        Override the execute method to use parallel execution.

        This method demonstrates how to use the operation_executor's
        execute_operations_parallel method for parallel execution.
        """
        try:
            self.logger.info(f"Executing task: {self.task_id} with {self.parallel_processes} processes")

            # Check if we have operations configured
            if not self.operations:
                self.logger.error("No operations configured for this task")
                self.error_info = {
                    "type": "configuration_error",
                    "message": "No operations configured for this task"
                }
                self.status = "configuration_error"
                return False

            # Use the operation executor's parallel execution method
            self.results = self.operation_executor.execute_operations_parallel(
                operations=self.operations,
                common_params={
                    "data_source": self.data_source,
                    "task_dir": self.task_dir,
                    "reporter": self.reporter
                },
                max_workers=self.parallel_processes
            )

            # Check for overall success
            failed_operations = [name for name, result in self.results.items()
                                 if result.status.name != 'SUCCESS']

            # Доступ к атрибуту из config, а не напрямую
            if failed_operations and not self.config.continue_on_error:
                self.logger.error(f"Operations failed: {', '.join(failed_operations)}")
                self.status = "operation_error"
                return False

            # Process the results
            for name, result in self.results.items():
                # Collect artifacts
                if hasattr(result, 'artifacts') and result.artifacts:
                    self.artifacts.extend(result.artifacts)

                # Collect metrics
                if hasattr(result, 'metrics') and result.metrics:
                    self.metrics[name] = result.metrics

            self.status = "success"
            return True

        except Exception as e:
            self.logger.exception(f"Error executing task: {str(e)}")
            self.error_info = {
                "type": "execution_error",
                "message": str(e)
            }
            self.status = "execution_error"
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parallel Processing PAMOLA Task")

    # Add common arguments
    parser.add_argument("--data_repository", help="Path to data repository")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="Log level")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue execution if an operation fails")

    # Parallel processing specific arguments
    parser.add_argument("--parallel_processes", type=int, default=4,
                        help="Number of parallel processes to use")
    parser.add_argument("--use_vectorization", action="store_true",
                        help="Enable vectorized operations where supported")

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()

    # Create task instance
    task = ParallelTask()

    # Convert arguments to dictionary for task initialization
    arg_dict = vars(args)

    # Run the task
    success = task.run(arg_dict)

    if success and task.status == "success":
        print(f"Task completed successfully. Report saved to: {task.reporter.report_path}")
        sys.exit(0)
    else:
        print(f"Task failed: {task.error_info}")
        sys.exit(1)


if __name__ == "__main__":
    main()