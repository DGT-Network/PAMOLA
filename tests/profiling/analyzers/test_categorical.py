import unittest
from unittest import mock
from unittest.mock import PropertyMock, patch, MagicMock
from pathlib import Path
import pandas as pd
from pamola_core.profiling.analyzers.categorical import CategoricalAnalyzer, CategoricalOperation, analyze_categorical_fields
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class TestCategoricalOperation(unittest.TestCase):
    def setUp(self):
        self.operation = CategoricalOperation(
            field_name="Category",
            top_n=10,
            min_frequency=1,
            include_timestamp=True,
            generate_plots=True,
            profile_type="categorical",
            analyze_anomalies=True
        )
        self.mock_data_source = MagicMock()
        self.mock_task_dir = Path("/tmp/test_task_dir")  # Use a temporary directory
        self.mock_reporter = MagicMock()
        self.mock_progress_tracker = MagicMock()

    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_no_dataframe(self, mock_load_data_operation):
        # Mock load_data_operation to return None
        mock_load_data_operation.return_value = None

        # Execute the operation
        result = self.operation.execute(
            data_source=None,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )
        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "No valid DataFrame found in data source")


    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_field_not_found(self, mock_load_data_operation):
        # Mock DataFrame without the target field
        mock_df = pd.DataFrame({
            "OtherField": [1, 2, 3]
        })
        mock_load_data_operation.return_value = mock_df

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "Field Category not found in DataFrame")


    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_success(self, mock_load_data_operation, mock_analyze):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A", "C", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results
        mock_analyze.return_value = {
            "total_records": 6,
            "unique_values": 3,
            "null_values": 0,
            "null_percent": 0.0,
            "entropy": 1.5,
            "cardinality_ratio": 0.5,
            "top_values": [{"value": "A", "count": 3}, {"value": "B", "count": 2}, {"value": "C", "count": 1}]
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
        self.assertEqual(result.metrics["total_records"], 6)
        self.assertEqual(result.metrics["null_count"], 0)
        self.assertEqual(result.metrics["null_percent"], 0.0)
        self.assertEqual(result.metrics["unique_values"], 3)
        self.assertEqual(result.metrics["entropy"], 1.5)
        self.assertEqual(result.metrics["cardinality_ratio"], 0.5)
        self.mock_reporter.add_operation.assert_called_with(
            f"Analysis of {self.operation.field_name} completed",
            details={
                "unique_values": 3,
                "null_percent": 0.0,
                "entropy": 1.5,
                "anomalies_found": 0
            }
        )

    @patch("pamola_core.profiling.analyzers.categorical.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.categorical.pd.DataFrame.to_csv")
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_with_value_dictionary(self, mock_load_data_operation, mock_analyze, mock_to_csv, mock_get_timestamped_filename):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A", "C", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results with value dictionary
        mock_analyze.return_value = {
            "value_dictionary": {
                "dictionary_data": [
                    {"value": "A", "count": 3},
                    {"value": "B", "count": 2},
                    {"value": "C", "count": 1}
                ]
            }
        }

        # Mock timestamped filename
        mock_get_timestamped_filename.side_effect = ["Category_stats.json", "Category_dictionary.csv"]

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )
    
        # Assertions
        mock_get_timestamped_filename.assert_any_call("Category_dictionary", "csv", True)
        mock_get_timestamped_filename.assert_any_call("Category_stats", "json", True)
        mock_to_csv.assert_called_once()  # Ensure the CSV file is created
        self.mock_reporter.add_artifact.assert_called_with(
            "csv",
            str(self.mock_task_dir / "dictionaries" / "Category_dictionary.csv"),
            "Category value dictionary"
        )
        self.assertTrue(any(artifact.artifact_type == "csv" for artifact in result.artifacts))

    @patch("pamola_core.profiling.analyzers.categorical.logger.warning")
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_with_empty_value_dictionary(self, mock_load_data_operation, mock_analyze, mock_logger_warning):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A", "C", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results with empty dictionary data
        mock_analyze.return_value = {
            "value_dictionary": {
                "dictionary_data": []
            }
        }

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Debugging: Print all calls to add_artifact
        print(self.mock_reporter.add_artifact.mock_calls)

        # Assertions
        mock_logger_warning.assert_called_once_with("Empty dictionary data for Category")
        self.mock_reporter.add_artifact.assert_any_call(
            "json",
            unittest.mock.ANY,  # Accept any path for the JSON file
            "Category statistical analysis"
        )
        calls = self.mock_reporter.add_artifact.call_args_list
        for call in calls:
            self.assertNotEqual(
                call,
                mock.call(
                    "csv",
                    str(self.mock_task_dir / "dictionaries" / "Category_dictionary.csv"),
                    "Category value dictionary"
                )
            )

    @patch("pamola_core.profiling.analyzers.categorical.logger.warning")
    @patch("pamola_core.profiling.analyzers.categorical.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_visualization_error(self, mock_load_data_operation, mock_create_visualization, mock_logger_warning):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A", "C", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock visualization result with an error
        mock_create_visualization.return_value = "Error: Unable to create visualization"

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        mock_logger_warning.assert_called_once_with("Error creating visualization: Error: Unable to create visualization")

        # Ensure no visualization artifact is added
        calls = self.mock_reporter.add_artifact.call_args_list
        for call in calls:
            self.assertNotEqual(
                call,
                mock.call(
                    "png",
                    unittest.mock.ANY,  # Accept any path for the PNG file
                    f"{self.operation.field_name} distribution visualization"
                )
            )
    @patch("pamola_core.profiling.analyzers.categorical.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.categorical.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_visualization_success(self, mock_load_data_operation, mock_create_visualization, mock_get_timestamped_filename):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A", "C", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock visualization result
        mock_create_visualization.return_value = "Visualization created successfully"
        mock_get_timestamped_filename.return_value = "Category_distribution.png"

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Debugging: Print all calls to add_artifact
        print(self.mock_reporter.add_artifact.mock_calls)

        # Assertions
        mock_create_visualization.assert_called_once()
        self.mock_reporter.add_artifact.assert_any_call(
            "png",
            str(self.mock_task_dir / "visualizations" / "Category_distribution.png"),
            "Category distribution visualization"
        )

    
    @patch("pamola_core.profiling.analyzers.categorical.logger.exception")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_exception_handling(self, mock_load_data_operation, mock_logger_exception):
        # Mock load_data_operation to raise an exception
        mock_load_data_operation.side_effect = Exception("Test exception")

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Debugging: Print all calls to progress_tracker.update
        print(self.mock_progress_tracker.update.mock_calls)

        # Assertions
        # Ensure logger.exception is called with the correct error message
        mock_logger_exception.assert_called_once_with(
            f"Error in categorical operation for {self.operation.field_name}: Test exception"
        )

        # Ensure progress tracker is updated with the error
        self.mock_progress_tracker.update.assert_any_call(
            0, {"step": "Error", "error": "Test exception"}
        )

        # Ensure reporter adds the error operation
        self.mock_reporter.add_operation.assert_called_once_with(
            f"Error analyzing {self.operation.field_name}",
            status="error",
            details={"error": "Test exception"}
        )

        # Ensure the result is an error with the correct message
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, f"Error analyzing categorical field {self.operation.field_name}: Test exception")

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_no_anomalies(self, mock_load_data_operation, mock_analyze):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A", "C", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results with no anomalies
        mock_analyze.return_value = {
            "total_records": 6,
            "unique_values": 3,
            "anomalies": {}  # No anomalies
        }

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        # Ensure no anomalies artifact is added
        calls = self.mock_reporter.add_artifact.call_args_list
        for call in calls:
            self.assertNotEqual(
                call,
                mock.call(
                    "csv",
                    unittest.mock.ANY,  # Accept any path for the CSV file
                    f"{self.operation.field_name} anomalies"
                )
            )

        # Ensure the result is not an error
        self.assertEqual(result.status, OperationStatus.SUCCESS)

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_with_numeric_like_strings(self, mock_load_data_operation, mock_analyze):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["123", "456", "123", "789", "456", "123"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results with numeric-like strings in anomalies
        mock_analyze.return_value = {
            "total_records": 6,
            "unique_values": 3,
            "anomalies": {
                "numeric_like_strings": {
                    "123": 3,
                    "456": 2,
                    "789": 1
                }
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
        # Ensure anomalies are processed correctly
        self.mock_reporter.add_artifact.assert_any_call(
            "csv",
            unittest.mock.ANY,
            f"{self.operation.field_name} anomalies"
        )

        # Check the anomaly records
        anomaly_records = [
            {"value": "123", "frequency": 3, "anomaly_type": "numeric_like_string", "similar_to": "", "similar_count": 0},
            {"value": "456", "frequency": 2, "anomaly_type": "numeric_like_string", "similar_to": "", "similar_count": 0},
            {"value": "789", "frequency": 1, "anomaly_type": "numeric_like_string", "similar_to": "", "similar_count": 0}
        ]

        # Verify that the anomalies are correctly added to the result
        for record in anomaly_records:
            self.assertEqual(result.metrics["anomalies_count"], 1)
    
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_tuple_result_with_error(self, mock_load_data_operation):
        # Mock load_data_operation to return a tuple (None, error_info)
        mock_load_data_operation.return_value = (None, "Some error info")

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)

    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_tuple_result_with_error_dict(self, mock_load_data_operation):
        # Mock load_data_operation to return a tuple (None, error_info as dict)
        error_info = {"message": "Data source connection failed"}
        mock_load_data_operation.return_value = (None, error_info)

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Data source connection failed", result.error_message)

    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_dataframe_missing_columns_attribute(self, mock_load_data_operation):
        # Mock object without 'columns' attribute
        class DummyObj:
            pass
        dummy_df = DummyObj()
        mock_load_data_operation.return_value = dummy_df

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(
            result.error_message,
            "DataFrame does not have expected structure (missing columns attribute)"
        )

    @patch("pamola_core.profiling.analyzers.categorical.logger.error")
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_analyzer_exception(self, mock_load_data_operation, mock_analyze, mock_logger_error):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analyzer to raise an exception
        mock_analyze.side_effect = Exception("Analyzer failed")

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        mock_logger_error.assert_called_once_with(
            f"Error in analyzer for {self.operation.field_name}: Analyzer failed"
        )
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(
            result.error_message,
            f"Error analyzing field {self.operation.field_name}: Analyzer failed"
        )

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_analysis_results_with_error(self, mock_load_data_operation, mock_analyze):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            "Category": ["A", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results containing an error
        mock_analyze.return_value = {
            "error": "Analysis failed due to invalid data"
        }

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "Analysis failed due to invalid data")

    def test_save_anomalies_to_csv_no_anomalies(self):
        op = CategoricalOperation(field_name="Category")
        mock_result = MagicMock()
        mock_reporter = MagicMock()
        dict_dir = Path("/tmp/test_task_dir/dictionaries")
        # Case when anomalies is None
        op._save_anomalies_to_csv(
            analysis_results={},  # No anomalies key
            dict_dir=dict_dir,
            include_timestamp=True,
            result=mock_result,
            reporter=mock_reporter
        )
        # No artifact or other function should be called
        mock_result.add_artifact.assert_not_called()
        mock_reporter.add_artifact.assert_not_called()

        # Case when anomalies is {}
        op._save_anomalies_to_csv(
            analysis_results={"anomalies": {}},
            dict_dir=dict_dir,
            include_timestamp=True,
            result=mock_result,
            reporter=mock_reporter
        )
        mock_result.add_artifact.assert_not_called()
        mock_reporter.add_artifact.assert_not_called()
    @patch("pamola_core.profiling.analyzers.categorical.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.categorical.pd.DataFrame.to_csv")
    def test_save_anomalies_to_csv_typo_and_numeric(self, mock_to_csv, mock_get_filename):
        op = CategoricalOperation(field_name="Category")
        mock_result = MagicMock()
        mock_reporter = MagicMock()
        dict_dir = Path("/tmp/test_task_dir/dictionaries")
        mock_get_filename.return_value = "Category_anomalies.csv"
        analysis_results = {
            "anomalies": {
                "potential_typos": {
                    "Aaa": {"count": 2, "similar_to": "A", "similar_count": 10}
                },
                "numeric_like_strings": {
                    "123": 3
                }
            }
        }
        op._save_anomalies_to_csv(
            analysis_results, dict_dir, include_timestamp=True,
            result=mock_result, reporter=mock_reporter
        )
        # Check DataFrame.to_csv was called
        mock_to_csv.assert_called_once()
        # Check artifact added
        mock_result.add_artifact.assert_called_with(
            "csv", dict_dir / "Category_anomalies.csv", "Category anomalies"
        )
        mock_reporter.add_artifact.assert_called_with(
            "csv", str(dict_dir / "Category_anomalies.csv"), "Category anomalies"
        )

    @patch("pamola_core.profiling.analyzers.categorical.logger.warning")
    def test_save_anomalies_to_csv_handles_exception(self, mock_logger_warning):
        op = CategoricalOperation(field_name="Category")
        mock_result = MagicMock()
        mock_reporter = MagicMock()
        dict_dir = Path("/tmp/test_task_dir/dictionaries")
        # Patch pd.DataFrame to raise exception
        with patch("pamola_core.profiling.analyzers.categorical.pd.DataFrame", side_effect=Exception("df error")):
            op._save_anomalies_to_csv(
                {"anomalies": {"numeric_like_strings": {"123": 1}}},
                dict_dir, True, mock_result, mock_reporter
            )
        mock_logger_warning.assert_called_once()
        mock_reporter.add_operation.assert_called_once()

    @patch("pamola_core.profiling.analyzers.categorical.ensure_directory")
    def test_prepare_directories_creates_and_returns_paths(self, mock_ensure_directory):
        op = CategoricalOperation(field_name="Category")
        base_dir = Path("/tmp/test_task_dir")
        dirs = op._prepare_directories(base_dir)
        self.assertIn("output", dirs)
        self.assertIn("visualizations", dirs)
        self.assertIn("dictionaries", dirs)
        self.assertEqual(dirs["output"], base_dir / "output")
        self.assertEqual(dirs["visualizations"], base_dir / "visualizations")
        self.assertEqual(dirs["dictionaries"], base_dir / "dictionaries")
        # ensure_directory should be called for each dir
        self.assertEqual(mock_ensure_directory.call_count, 3)
        
class TestCategoricalAnalyzer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'E', 'C', 'A']
        })
        self.field_name = 'category'

    @patch("pamola_core.profiling.analyzers.categorical.analyze_categorical_field")
    def test_analyze(self, mock_analyze_categorical_field):
        mock_result = {
            'top_values': {'A': 4, 'B': 2},
            'cardinality': 5,
            'field': self.field_name
        }
        mock_analyze_categorical_field.return_value = mock_result

        result = CategoricalAnalyzer.analyze(self.df, self.field_name, top_n=2, min_frequency=1)
        self.assertEqual(result, mock_result)

        mock_analyze_categorical_field.assert_called_once_with(
            df=self.df,
            field_name=self.field_name,
            top_n=2,
            min_frequency=1
        )

    @patch("pamola_core.profiling.analyzers.categorical.estimate_resources")
    def test_estimate_resources(self, mock_estimate_resources):
        mock_result = {
            'memory': '10MB',
            'time': '0.01s',
            'field': self.field_name
        }
        mock_estimate_resources.return_value = mock_result

        result = CategoricalAnalyzer.estimate_resources(self.df, self.field_name)
        self.assertEqual(result, mock_result)

        mock_estimate_resources.assert_called_once_with(self.df, self.field_name)


class TestAnalyzeCategoricalFields(unittest.TestCase):
    def setUp(self):
        self.mock_reporter = MagicMock()
        self.mock_task_dir = Path("/tmp/test_task_dir")
        self.df = pd.DataFrame({
            "cat1": ["A", "B", "A"],
            "cat2": ["X", "Y", "Z"],
            "num": [1, 2, 3]
        })
        self.mock_data_source = MagicMock()
        self.mock_data_source.get_dataframe.return_value = self.df

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_analyze_fields_with_explicit_fields(self, mock_execute):
        # Mock operation result
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)
        results = analyze_categorical_fields(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            cat_fields=["cat1", "cat2"],
            top_n=5
        )
        self.assertIn("cat1", results)
        self.assertIn("cat2", results)
        self.assertEqual(results["cat1"].status, OperationStatus.SUCCESS)
        self.assertEqual(results["cat2"].status, OperationStatus.SUCCESS)
        self.mock_reporter.add_operation.assert_any_call(
            "Categorical fields analysis completed",
            details={"fields_analyzed": 2, "successful": 2, "failed": 0}
        )

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_analyze_fields_auto_detect(self, mock_execute):
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)
        results = analyze_categorical_fields(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )
        # Should auto-detect cat1 and cat2
        self.assertIn("cat1", results)
        self.assertIn("cat2", results)

    def test_analyze_fields_no_dataframe(self):
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = None
        results = analyze_categorical_fields(
            data_source=mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )
        self.assertEqual(results, {})
        self.mock_reporter.add_operation.assert_any_call(
            "Categorical fields analysis", status="error", details=unittest.mock.ANY
        )

    def test_analyze_fields_tuple_with_error(self):
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = (None, {"message": "Test error"})
        results = analyze_categorical_fields(
            data_source=mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )
        self.assertEqual(results, {})
        self.mock_reporter.add_operation.assert_any_call(
            "Categorical fields analysis", status="error", details=unittest.mock.ANY
        )

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_analyze_fields_operation_exception(self, mock_execute):
        # Simulate exception in operation.execute
        mock_execute.side_effect = Exception("Operation failed")
        results = analyze_categorical_fields(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            cat_fields=["cat1"]
        )
        self.assertNotIn("cat1", [k for k, v in results.items() if v.status == OperationStatus.SUCCESS])
        self.mock_reporter.add_operation.assert_any_call(
            "Analyzing cat1 field", status="error", details={"error": "Operation failed"}
        )

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_auto_detect_numeric_low_cardinality(self, mock_execute):
    # DataFrame with a numeric column that has few unique values and enough rows to satisfy the condition
        df = pd.DataFrame({
            "num_cat": [1, 1, 2, 2, 3, 3] * 5,  # 30 rows, 3 unique values
            "other": list(range(30))
        })
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = df
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)

        results = analyze_categorical_fields(
            data_source=mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )
        # num_cat must be automatically detected as categorical
        self.assertIn("num_cat", results)

   
    @patch("pamola_core.profiling.analyzers.categorical.logger.warning")
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_auto_detect_column_dtype_exception(self, mock_execute, mock_logger_warning):
        # Normal DataFrame
        df = pd.DataFrame({"bad_col": [1, 2, 3]})
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = df
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)

        # Patch Series.dtype to raise exception when accessed
        with patch("pandas.Series.dtype", new_callable=PropertyMock) as mock_dtype:
            mock_dtype.side_effect = ValueError("broken dtype")
            results = analyze_categorical_fields(
                data_source=mock_data_source,
                task_dir=self.mock_task_dir,
                reporter=self.mock_reporter
            )

        self.assertEqual(results, {})
        found = any(
            "Error checking column bad_col: broken dtype" in str(call.args[0])
            for call in mock_logger_warning.call_args_list
        )
        self.assertTrue(found, "logger.warning was not called with the expected message")

    @patch("pamola_core.profiling.analyzers.categorical.logger.error")
    def test_auto_detect_no_columns_attribute(self, mock_logger_error):
        # Create object without columns attribute
        class DummyObj:
            pass
        dummy_df = DummyObj()
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = dummy_df

        results = analyze_categorical_fields(
            data_source=mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )
        self.assertEqual(results, {})
        mock_logger_error.assert_called_once_with("DataFrame does not have columns attribute")
        self.mock_reporter.add_operation.assert_any_call(
            "Categorical fields detection",
            status="error",
            details={"error": "DataFrame doesn't have expected structure"}
        )

    @patch("pamola_core.profiling.analyzers.categorical.ProgressTracker")
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_overall_tracker_update_on_error(self, mock_execute, mock_progress_tracker_cls):
        # Mock tracker instance
        mock_tracker = MagicMock()
        mock_progress_tracker_cls.return_value = mock_tracker

        # Mock operation result with ERROR status
        error_message = "Some error"
        mock_execute.return_value = OperationResult(status=OperationStatus.ERROR, error_message=error_message)

        # DataFrame with 1 categorical column
        df = pd.DataFrame({"cat1": ["A", "B", "A"]})
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = df

        analyze_categorical_fields(
            data_source=mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            cat_fields=["cat1"]
        )

        # Ensure overall_tracker.update is called with status="error" and correct error message
        mock_tracker.update.assert_any_call(1, {"field": "cat1", "status": "error", "error": error_message})
            
if __name__ == "__main__":
    unittest.main()