import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class TestDataAttributeProfilerOperation(unittest.TestCase):
    def setUp(self):
        self.operation = DataAttributeProfilerOperation(
            name="TestAttributeProfiler",
            description="Test profiling of dataset attributes",
            dictionary_path=None,
            language="en",
            sample_size=10,
            max_columns=None,
            id_column=None,
            include_timestamp=True
        )
        self.mock_data_source = MagicMock()
        self.mock_task_dir = Path("/tmp/test_task_dir")  # Use a temporary directory
        self.mock_reporter = MagicMock()
        self.mock_progress_tracker = MagicMock()

    @patch("pamola_core.profiling.analyzers.attribute.load_data_operation")
    @patch("pamola_core.profiling.analyzers.attribute.load_attribute_dictionary")
    @patch("pamola_core.profiling.analyzers.attribute.analyze_dataset_attributes")
    def test_execute_success(self, mock_analyze_dataset_attributes, mock_load_attribute_dictionary, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock attribute dictionary
        mock_load_attribute_dictionary.return_value = {"mock_key": "mock_value"}

        # Mock analysis results
        mock_analyze_dataset_attributes.return_value = {
            "summary": {
                "DIRECT_IDENTIFIER": 1,
                "QUASI_IDENTIFIER": 2,
                "SENSITIVE_ATTRIBUTE": 1,
                "INDIRECT_IDENTIFIER": 0,
                "NON_SENSITIVE": 1
            },
            "columns": {
                "Name": {
                    "role": "QUASI_IDENTIFIER",
                    "statistics": {
                        "entropy": 0.8,
                        "normalized_entropy": 0.5,
                        "uniqueness_ratio": 0.6,
                        "missing_rate": 0.0,
                        "inferred_type": "string",
                        "samples": ["Alice", "Bob"]
                    }
                }
            },
            "column_groups": {
                "QUASI_IDENTIFIER": ["Name"]
            },
            "dataset_metrics": {
                "avg_entropy": 0.7,
                "avg_uniqueness": 0.6
            }
        }

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.mock_reporter.add_operation.assert_called_with(
            "Attribute Profiling Completed",
            details={
                "direct_identifiers": 1,
                "quasi_identifiers": 2,
                "sensitive_attributes": 1,
                "indirect_identifiers": 0,
                "non_sensitive": 1,
                "conflicts": 0
            }
        )
        self.mock_progress_tracker.update.assert_any_call(0, {"step": "Preparation", "operation": self.operation.name})
        self.mock_progress_tracker.update.assert_any_call(1, {"step": "Loading dictionary and preparing analysis"})
        self.mock_progress_tracker.update.assert_any_call(1, {"step": "Analyzing dataset attributes"})
        self.mock_progress_tracker.update.assert_any_call(1, {"step": "Saving analysis results"})
        self.mock_progress_tracker.update.assert_any_call(1, {"step": "Creating visualizations"})
        self.mock_progress_tracker.update.assert_any_call(1, {"step": "Operation complete", "status": "success"})

    @patch("pamola_core.profiling.analyzers.attribute.create_pie_chart")
    @patch("pamola_core.profiling.analyzers.attribute.create_scatter_plot")
    @patch("pamola_core.profiling.analyzers.attribute.create_bar_plot")
    def test_create_visualizations(self, mock_create_bar_plot, mock_create_scatter_plot, mock_create_pie_chart):
        # Mock analysis results
        analysis_results = {
            "summary": {
                "DIRECT_IDENTIFIER": 1,
                "QUASI_IDENTIFIER": 2,
                "SENSITIVE_ATTRIBUTE": 1,
                "INDIRECT_IDENTIFIER": 0,
                "NON_SENSITIVE": 1
            },
            "columns": {
                "Name": {
                    "role": "QUASI_IDENTIFIER",
                    "statistics": {
                        "entropy": 0.8,
                        "normalized_entropy": 0.5,
                        "uniqueness_ratio": 0.6,
                        "missing_rate": 0.0,
                        "inferred_type": "string",
                        "samples": ["Alice", "Bob"]
                    }
                }
            }
        }

        # Mock paths
        vis_dir = Path("/tmp/test_visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Execute visualization creation
        self.operation._create_visualizations(
            analysis_results=analysis_results,
            vis_dir=vis_dir,
            include_timestamp=True,
            result=MagicMock(),
            reporter=self.mock_reporter
        )

        # Assertions
        mock_create_pie_chart.assert_called_once()
        mock_create_scatter_plot.assert_called_once()
        mock_create_bar_plot.assert_called_once()

    @patch("pamola_core.profiling.analyzers.attribute.load_data_operation")
    @patch("pamola_core.profiling.analyzers.attribute.load_attribute_dictionary")
    @patch("pamola_core.profiling.analyzers.attribute.analyze_dataset_attributes")
    def test_execute_with_conflicts(self, mock_analyze_dataset_attributes, mock_load_attribute_dictionary, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock attribute dictionary
        mock_load_attribute_dictionary.return_value = {"mock_key": "mock_value"}

    @patch("pamola_core.profiling.analyzers.attribute.load_data_operation")
    @patch("pamola_core.profiling.analyzers.attribute.load_attribute_dictionary")
    @patch("pamola_core.profiling.analyzers.attribute.analyze_dataset_attributes")
    def test_execute_exception_handling(self, mock_analyze_dataset_attributes, mock_load_attribute_dictionary, mock_load_data_operation):
        # Arrange: cause an exception in analyze_dataset_attributes
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"]
        })
        mock_load_data_operation.return_value = mock_df
        mock_load_attribute_dictionary.return_value = {"mock_key": "mock_value"}
        mock_analyze_dataset_attributes.side_effect = Exception("Test exception in analysis")

        # Execute
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Test exception in analysis", result.error_message)
        self.mock_progress_tracker.update.assert_any_call(0, {"step": "Error", "error": "Test exception in analysis"})
        self.mock_reporter.add_operation.assert_any_call(
            "Attribute Profiling",
            status="error",
            details={"error": "Test exception in analysis"}
        )

    @patch("pamola_core.profiling.analyzers.attribute.load_data_operation")
    @patch("pamola_core.profiling.analyzers.attribute.load_attribute_dictionary")
    @patch("pamola_core.profiling.analyzers.attribute.analyze_dataset_attributes")
    def test_execute_exception_handling_no_progress_tracker(self, mock_analyze_dataset_attributes, mock_load_attribute_dictionary, mock_load_data_operation):
        # Arrange: cause an exception in analyze_dataset_attributes
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"]
        })
        mock_load_data_operation.return_value = mock_df
        mock_load_attribute_dictionary.return_value = {"mock_key": "mock_value"}
        mock_analyze_dataset_attributes.side_effect = Exception("Another test exception")

        # Execute with progress_tracker=None
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=None
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Another test exception", result.error_message)
        self.mock_reporter.add_operation.assert_any_call(
            "Attribute Profiling",
            status="error",
            details={"error": "Another test exception"}
        )

    @patch("pamola_core.profiling.analyzers.attribute.load_data_operation")
    def test_execute_no_dataframe(self, mock_load_data_operation):
        # Mock load_data_operation to return None
        mock_load_data_operation.return_value = None

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "No valid DataFrame found in data source")
    
    @patch("pamola_core.profiling.analyzers.attribute.logger")
    def test_create_visualizations_exception(self, mock_logger):
        # Arrange
        analysis_results = {
            "summary": {
                "DIRECT_IDENTIFIER": 1,
                "QUASI_IDENTIFIER": 1,
                "SENSITIVE_ATTRIBUTE": 0,
                "INDIRECT_IDENTIFIER": 0,
                "NON_SENSITIVE": 1
            },
            "columns": {}
        }
        vis_dir = Path("/tmp/test_visualizations_exception")
        vis_dir.mkdir(parents=True, exist_ok=True)
        mock_result = MagicMock()
        mock_reporter = MagicMock()

        # Patch a visualization function to raise exception
        with patch("pamola_core.profiling.analyzers.attribute.create_pie_chart", side_effect=Exception("Visualization error")):
            # Act
            self.operation._create_visualizations(
                analysis_results=analysis_results,
                vis_dir=vis_dir,
                include_timestamp=True,
                result=mock_result,
                reporter=mock_reporter
            )

        # Assert logger and reporter are called as expected
        mock_logger.error.assert_called()
        mock_reporter.add_operation.assert_any_call(
            "Creating visualizations",
            status="warning",
            details={"warning": "Error creating some visualizations: Visualization error"}
        )
 
if __name__ == "__main__":
    unittest.main()