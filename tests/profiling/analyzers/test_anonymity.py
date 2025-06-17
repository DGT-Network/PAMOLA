import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import pandas as pd
from pamola_core.profiling.analyzers.anonymity import PreKAnonymityProfilingOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class TestPreKAnonymityProfilingOperation(unittest.TestCase):
    def setUp(self):
        self.operation = PreKAnonymityProfilingOperation(
            min_combination_size=2,
            max_combination_size=3,
            treshold_k=5,
            fields_combinations=None,
            excluded_combinations=None,
            id_fields=["ID"],
            include_timestamp=True
        )
        self.mock_data_source = MagicMock()
        self.mock_task_dir = Path(tempfile.mkdtemp())  # Use a real temporary directory
        self.mock_reporter = MagicMock()

        self.task_dir = Path("test_task_dir")

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_success(self, mock_load_data_operation):
        # Create a realistic DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            fields_combinations=[["Name", "Age"], ["Age", "City"]],
            id_fields=["ID"]
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.mock_reporter.add_operation.assert_called_with(
            "K-Anonymity Profiling Completed",
            details={
                "analyzed_combinations": 2,
                "top_risk_combinations": ["KA_na_ag (min_k=1)", "KA_ag_ci (min_k=1)"],
                "threshold_k": 5
            }
        )

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_no_dataframe(self, mock_load_data_operation):
        # Mock no DataFrame returned
        mock_load_data_operation.return_value = None

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "No valid DataFrame found in data source")

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_no_field_combinations(self, mock_load_data_operation):
        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.columns = ["ID", "Name", "Age", "City"]
        mock_df.__len__.return_value = 100
        mock_load_data_operation.return_value = mock_df

        # Execute the operation with no field combinations
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            fields_combinations=[]
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "No valid field combinations to analyze")

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_exception_handling(self, mock_load_data_operation):
        # Mock exception during data loading
        mock_load_data_operation.side_effect = Exception("Test exception")

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Error in k-anonymity profiling", result.error_message)
        self.mock_reporter.add_operation.assert_called_with(
            "K-Anonymity Profiling",
            status="error",
            details={"error": "Test exception"}
        )

    def test_sorting_data(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]


        sorted_data = sorted(data, key=lambda x: x["age"])
        self.assertEqual(sorted_data, [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}])

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_with_progress_tracker(self, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock progress tracker
        mock_progress_tracker = MagicMock()

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=mock_progress_tracker,
            fields_combinations=[["Name", "Age"], ["Age", "City"]],
            id_fields=["ID"]
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)


        print("Mock calls to progress_tracker.update:")
        for call in mock_progress_tracker.update.mock_calls:
            print(call)

        # Verify progress tracker updates
        mock_progress_tracker.update.assert_any_call(0, {"step": "Preparation", "operation": self.operation.name})
        mock_progress_tracker.update.assert_any_call(1, {"step": "Generated field combinations", "combinations_count": 2})
        mock_progress_tracker.update.assert_any_call(1, {"step": "Created KA index map"})
        mock_progress_tracker.update.assert_any_call(1, {
            "step": "Calculated k-anonymity metrics",
            "metrics_count": 2 
        })
        self.assertEqual(mock_progress_tracker.total, 6)  # Ensure total steps are set correctly

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.anonymity.get_field_combinations")
    def test_execute_generate_field_combinations_when_none(self, mock_get_field_combinations, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
            "Age": [25, 30, 25, 30, 35],
            "City": ["NY", "LA", "NY", "LA", "SF"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock get_field_combinations to return a known value
        mock_get_field_combinations.return_value = [["Name", "Age"], ["Age", "City"]]

        # Do not pass fields_combinations (default value is None)
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            id_fields=["ID"]
        )

        # Ensure get_field_combinations is called correctly with non-ID fields
        mock_get_field_combinations.assert_called_once_with(
            ["Name", "Age", "City"],  # ID has been removed
            min_size=self.operation.min_combination_size,
            max_size=self.operation.max_combination_size,
            excluded_combinations=None
        )

        self.assertEqual(result.status, OperationStatus.SUCCESS)


    @patch("pamola_core.profiling.analyzers.anonymity.calculate_k_anonymity")
    @patch("pamola_core.profiling.analyzers.anonymity.create_ka_index_map")
    @patch("pamola_core.profiling.analyzers.anonymity.get_field_combinations")
    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.anonymity.save_ka_index_map")
    @patch("pamola_core.profiling.analyzers.anonymity.save_ka_metrics")
    @patch("pamola_core.profiling.analyzers.anonymity.save_vulnerable_records")
    @patch("pamola_core.profiling.analyzers.anonymity.get_timestamped_filename", return_value="dummy.csv")
    @patch("pamola_core.profiling.analyzers.anonymity.write_json")
    @patch("pamola_core.profiling.analyzers.anonymity.prepare_field_uniqueness_data", return_value={})
    @patch("pamola_core.profiling.analyzers.anonymity.logger")
    def test_execute_error_metrics(
        self, mock_logger, mock_uniqueness, mock_write_json, mock_get_filename,
        mock_save_vuln, mock_save_metrics, mock_save_index, mock_load_data,
        mock_get_combinations, mock_create_index, mock_calc_ka
    ):
        # Arrange
        op = PreKAnonymityProfilingOperation()
        data_source = MagicMock()
        reporter = MagicMock()
        progress_tracker = MagicMock()
        df = MagicMock()
        mock_load_data.return_value = df
        mock_get_combinations.return_value = [["a", "b"]]
        mock_create_index.return_value = {"ka1": ["a", "b"]}
        mock_calc_ka.return_value = {"error": "test error"}

        # Act
        result = op.execute(
            data_source=data_source,
            task_dir=Path("."),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Assert
        mock_logger.warning.assert_any_call("Error analyzing ka1: test error")
        self.assertEqual(result.status.name, "SUCCESS")

    @patch("pamola_core.profiling.analyzers.anonymity.load_data_operation")
    def test_execute_exception_updates_progress_tracker(self, mock_load_data_operation):
        # Simulate exception when load_data_operation
        mock_load_data_operation.side_effect = Exception("Test error for progress tracker")

        mock_progress_tracker = MagicMock()

        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=mock_progress_tracker
        )

        # Check that progress_tracker.update is called with error info
        mock_progress_tracker.update.assert_any_call(
            0, {"step": "Error", "error": "Test error for progress tracker"}
        )
        self.assertEqual(result.status, OperationStatus.ERROR)
        
    @patch("pamola_core.profiling.analyzers.anonymity.ensure_directory")
    def test_prepare_directories(self, mock_ensure):
        dirs = self.operation._prepare_directories(self.task_dir)
        self.assertIn('output', dirs)
        self.assertIn('visualizations', dirs)
        self.assertIn('dictionaries', dirs)
        self.assertTrue(str(dirs['output']).endswith('output'))
        self.assertTrue(str(dirs['visualizations']).endswith('visualizations'))
        self.assertTrue(str(dirs['dictionaries']).endswith('dictionaries'))
        self.assertEqual(mock_ensure.call_count, 3)

    @patch("pamola_core.profiling.analyzers.anonymity.prepare_field_uniqueness_data", return_value={"f1": {"unique_values": 1, "uniqueness_percentage": 100}})
    @patch("pamola_core.profiling.analyzers.anonymity.create_combined_chart", return_value="ok")
    @patch("pamola_core.profiling.analyzers.anonymity.create_spider_chart", return_value="ok")
    @patch("pamola_core.profiling.analyzers.anonymity.create_line_plot", return_value="ok")
    @patch("pamola_core.profiling.analyzers.anonymity.create_bar_plot", return_value="ok")
    @patch("pamola_core.profiling.analyzers.anonymity.get_timestamped_filename", return_value="file.png")
    @patch("pamola_core.profiling.analyzers.anonymity.prepare_metrics_for_spider_chart", return_value={"ka1": {"a": 1}})
    def test_create_visualizations(self, mock_spider_data, mock_get_filename, mock_bar, mock_line, mock_spider, mock_combined, mock_field_uni):
        ka_metrics = {
            "ka1": {
                "k_range_distribution": {"1-5": 50, "6-10": 50},
                "threshold_metrics": {"k≥5": 80},
            }
        }
        ka_index_map = {"ka1": ["f1", "f2"]}
        df = MagicMock()
        field_combinations = [["f1", "f2"]]
        vis_dir = Path(".")
        include_timestamp = True
        treshold_k = 5
        result = MagicMock()
        reporter = MagicMock()

        # Should not raise and should call all chart functions
        self.operation._create_visualizations(
            ka_metrics, ka_index_map, df, field_combinations, vis_dir,
            include_timestamp, treshold_k, result, reporter
        )
        self.assertTrue(mock_bar.called)
        self.assertTrue(mock_line.called)
        self.assertTrue(mock_spider.called)
        self.assertTrue(mock_combined.called)
        self.assertTrue(mock_field_uni.called)
        self.assertTrue(mock_spider_data.called)

    @patch("pamola_core.profiling.analyzers.anonymity.logger")
    def test_create_visualizations_empty_metrics(self, mock_logger):
        # Arrange
        ka_metrics = {}
        ka_index_map = {}
        df = MagicMock()
        field_combinations = []
        vis_dir = Path(".")
        include_timestamp = True
        treshold_k = 5
        result = MagicMock()
        reporter = MagicMock()

        # Act
        PreKAnonymityProfilingOperation._create_visualizations(
            ka_metrics, ka_index_map, df, field_combinations, vis_dir,
            include_timestamp, treshold_k, result, reporter
        )

        # Assert
        mock_logger.warning.assert_called_once_with("No metrics available for visualization")

    @patch("pamola_core.profiling.analyzers.anonymity.logger")
    @patch("pamola_core.profiling.analyzers.anonymity.create_bar_plot", side_effect=Exception("plot error"))
    @patch("pamola_core.profiling.analyzers.anonymity.get_timestamped_filename", return_value="file.png")
    def test_create_visualizations_exception(self, mock_get_filename, mock_create_bar, mock_logger):
        ka_metrics = {"ka1": {"k_range_distribution": {"1-5": 50}}}
        ka_index_map = {"ka1": ["f1", "f2"]}
        df = MagicMock()
        field_combinations = [["f1", "f2"]]
        vis_dir = Path(".")
        include_timestamp = True
        treshold_k = 5
        result = MagicMock()
        reporter = MagicMock()

        PreKAnonymityProfilingOperation._create_visualizations(
            ka_metrics, ka_index_map, df, field_combinations, vis_dir,
            include_timestamp, treshold_k, result, reporter
        )

        mock_logger.error.assert_called()
        reporter.add_operation.assert_called_with(
            "Creating visualizations",
            status="warning",
            details={"warning": "Error creating some visualizations: plot error"}
        )



if __name__ == "__main__":
    unittest.main()