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


if __name__ == "__main__":
    unittest.main()