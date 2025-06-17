import shutil
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

    # For utils tests
        self.test_dir = Path("test_task_dir")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        # For utils tests
        if hasattr(self, "test_dir") and self.test_dir.exists():
            shutil.rmtree(self.test_dir)


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

    @patch("pamola_core.profiling.analyzers.attribute.load_data_operation")
    @patch("pamola_core.profiling.analyzers.attribute.load_attribute_dictionary")
    @patch("pamola_core.profiling.analyzers.attribute.analyze_dataset_attributes")
    def test_conflicts_metric(self, mock_analyze, mock_load_dict, mock_load_data):
        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        mock_df.columns = ["a", "b"]

        # Mock data loading
        mock_load_data.return_value = mock_df
        mock_load_dict.return_value = {}

        # analysis_results with conflicts
        analysis_results = {
            "columns": {},
            "summary": {
                "DIRECT_IDENTIFIER": 1,
                "QUASI_IDENTIFIER": 1,
                "SENSITIVE_ATTRIBUTE": 1,
                "INDIRECT_IDENTIFIER": 1,
                "NON_SENSITIVE": 1
            },
            "column_groups": {"QUASI_IDENTIFIER": []},
            "conflicts": [{"col": "a", "reason": "test"}],
            "dataset_metrics": {"avg_entropy": 0.5, "avg_uniqueness": 0.5}
        }
        mock_analyze.return_value = analysis_results

        # Prepare operation
        op = DataAttributeProfilerOperation()
        reporter = MagicMock()
        task_dir = Path("test_dir")

        # Run
        result = op.execute(
            data_source=MagicMock(),
            task_dir=task_dir,
            reporter=reporter
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertIn("conflicts_count", result.metrics)
        self.assertEqual(result.metrics["conflicts_count"], 1)

    def test_prepare_directories(self):
            dirs = self.operation._prepare_directories(self.test_dir)
            self.assertTrue((self.test_dir / "output").exists())
            self.assertTrue((self.test_dir / "visualizations").exists())
            self.assertTrue((self.test_dir / "dictionaries").exists())
            self.assertIn("output", dirs)
            self.assertIn("visualizations", dirs)
            self.assertIn("dictionaries", dirs)

    @patch("pamola_core.profiling.analyzers.attribute.create_pie_chart")
    @patch("pamola_core.profiling.analyzers.attribute.create_scatter_plot")
    @patch("pamola_core.profiling.analyzers.attribute.create_bar_plot")
    @patch("pamola_core.profiling.analyzers.attribute.get_timestamped_filename", side_effect=lambda prefix, ext, ts: f"{prefix}.{ext}")
    def test_create_visualizations(self, mock_get_filename, mock_bar, mock_scatter, mock_pie):
        # Setup mock return values
        mock_pie.return_value = "OK"
        mock_scatter.return_value = "OK"
        mock_bar.return_value = "OK"

        # Fake analysis_results
        analysis_results = {
            "summary": {
                "DIRECT_IDENTIFIER": 1,
                "QUASI_IDENTIFIER": 2,
                "SENSITIVE_ATTRIBUTE": 1,
                "INDIRECT_IDENTIFIER": 0,
                "NON_SENSITIVE": 1
            },
            "columns": {
                "col1": {
                    "role": "DIRECT_IDENTIFIER",
                    "statistics": {
                        "entropy": 0.5,
                        "normalized_entropy": 0.5,
                        "uniqueness_ratio": 0.5,
                        "missing_rate": 0.1,
                        "inferred_type": "string"
                    }
                },
                "col2": {
                    "role": "QUASI_IDENTIFIER",
                    "statistics": {
                        "entropy": 0.7,
                        "normalized_entropy": 0.7,
                        "uniqueness_ratio": 0.7,
                        "missing_rate": 0.2,
                        "inferred_type": "int"
                    }
                }
            }
        }
        vis_dir = self.test_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        result = MagicMock()
        reporter = MagicMock()

        self.operation._create_visualizations(
            analysis_results=analysis_results,
            vis_dir=vis_dir,
            include_timestamp=False,
            result=result,
            reporter=reporter
        )

        # Check that the drawing functions have been called
        self.assertTrue(mock_pie.called)
        self.assertTrue(mock_scatter.called)
        self.assertTrue(mock_bar.called)

    @patch("pamola_core.profiling.analyzers.attribute.create_pie_chart")
    @patch("pamola_core.profiling.analyzers.attribute.logger")
    def test_create_visualizations_exception(self, mock_logger, mock_create_pie):
        # Mock create_pie_chart to raise exception
        mock_create_pie.side_effect = Exception("Test visualization error")

        vis_dir = Path("test_vis_dir")
        vis_dir.mkdir(exist_ok=True)
        result = MagicMock()
        reporter = MagicMock()

        # Minimal valid analysis_results to trigger pie chart
        analysis_results = {
            "summary": {"DIRECT_IDENTIFIER": 1, "QUASI_IDENTIFIER": 0, "SENSITIVE_ATTRIBUTE": 0, "INDIRECT_IDENTIFIER": 0, "NON_SENSITIVE": 0},
            "columns": {}
        }

        self.operation._create_visualizations(
            analysis_results=analysis_results,
            vis_dir=vis_dir,
            include_timestamp=False,
            result=result,
            reporter=reporter
        )

        # Check logger.error was called
        self.assertTrue(mock_logger.error.called)
        # Check reporter.add_operation was called with warning
        reporter.add_operation.assert_called_with(
            "Creating visualizations",
            status="warning",
            details={"warning": "Error creating some visualizations: Test visualization error"}
        )

if __name__ == "__main__":
    unittest.main()