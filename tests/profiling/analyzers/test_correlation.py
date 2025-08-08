from pprint import pprint
import unittest
from unittest.mock import Mock, call, patch, MagicMock
from pathlib import Path
import pandas as pd
from pamola_core.profiling.analyzers.correlation import (
    CorrelationAnalyzer,
    CorrelationOperation,
    CorrelationMatrixOperation,
    analyze_correlations
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class TestCorrelationAnalyzer(unittest.TestCase):

    @patch("pamola_core.profiling.analyzers.correlation.analyze_correlation")
    def test_analyze(self, mock_analyze_correlation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "Field1": [1, 2, 3, 4],
            "Field2": [2, 4, 6, 8]
        })

        # Mock correlation analysis result
        mock_analyze_correlation.return_value = {
            "method": "pearson",
            "correlation_coefficient": 1.0,
            "p_value": 0.01,
            "sample_size": 4
        }

        # Call analyze function
        result = CorrelationAnalyzer.analyze(
            df=mock_df,
            field1="Field1",
            field2="Field2",
            method="pearson"
        )

        # Check result
        mock_analyze_correlation.assert_called_once_with(
            df=mock_df,
            field1="Field1",
            field2="Field2",
            method="pearson"
        )
        self.assertEqual(result["method"], "pearson")
        self.assertEqual(result["correlation_coefficient"], 1.0)
        self.assertEqual(result["p_value"], 0.01)
        self.assertEqual(result["sample_size"], 4)

    @patch("pamola_core.profiling.analyzers.correlation.analyze_correlation_matrix")
    def test_analyze_matrix(self, mock_analyze_correlation_matrix):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "Field1": [1, 2, 3, 4],
            "Field2": [2, 4, 6, 8],
            "Field3": [1, 3, 5, 7]
        })

        # Mock correlation matrix result
        mock_analyze_correlation_matrix.return_value = {
            "correlation_matrix": {
                "Field1": {"Field1": 1.0, "Field2": 1.0, "Field3": 0.99},
                "Field2": {"Field1": 1.0, "Field2": 1.0, "Field3": 0.98},
                "Field3": {"Field1": 0.99, "Field2": 0.98, "Field3": 1.0}
            }
        }

        # Call analyze_matrix function
        result = CorrelationAnalyzer.analyze_matrix(
            df=mock_df,
            fields=["Field1", "Field2", "Field3"]
        )

        # Check result
        mock_analyze_correlation_matrix.assert_called_once_with(
            df=mock_df,
            fields=["Field1", "Field2", "Field3"]
        )
        self.assertIn("correlation_matrix", result)
        self.assertEqual(result["correlation_matrix"]["Field1"]["Field2"], 1.0)


class TestCorrelationOperation(unittest.TestCase):

    @patch("pamola_core.profiling.analyzers.correlation.write_json")
    @patch("pamola_core.utils.io.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.correlation.analyze_correlation")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute(self, mock_load_data_operation, mock_analyze_correlation, mock_get_timestamped_filename, mock_write_json):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "Field1": [1, 2, 3, 4],
            "Field2": [2, 4, 6, 8]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock correlation analysis result
        mock_analyze_correlation.return_value = {
            "method": "pearson",
            "correlation_coefficient": 1.0,
            "p_value": 0.01,
            "sample_size": 4
        }

        # Mock filename
        mock_get_timestamped_filename.return_value = "Field1_Field2_correlation.json"

        # Create CorrelationOperation object
        operation = CorrelationOperation(
            field1="Field1",
            field2="Field2",
            method="pearson",
            use_cache=False
        )

        # Mock reporter and progress tracker
        mock_reporter = MagicMock()
        mock_progress_tracker = MagicMock()

        # Call execute function
        result = operation.execute(
            data_source=MagicMock(),
            task_dir=Path("/tmp/test_task_dir"),
            reporter=mock_reporter,
            progress_tracker=mock_progress_tracker
        )

        # Debugging: Print all calls to write_json
        print(mock_write_json.mock_calls)

        # Check result
        mock_load_data_operation.assert_called_once()
        mock_analyze_correlation.assert_called_once_with(
            df=mock_df,
            field1="Field1",
            field2="Field2",
            method="pearson",
            null_handling="drop"
        )
        mock_write_json.assert_called_once_with(
            mock_analyze_correlation.return_value,
            unittest.mock.ANY,
            encryption_key=unittest.mock.ANY,
            encryption_mode=unittest.mock.ANY,
        )
        
        assert len(result.artifacts) > 0
        
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        
        self.assertEqual(result.metrics["correlation_coefficient"], 1.0)

    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_error(self, mock_load_data_operation):
        # Mock error when loading DataFrame
        mock_load_data_operation.side_effect = Exception("Test exception")

        # Create CorrelationOperation object
        operation = CorrelationOperation(
            field1="Field1",
            field2="Field2",
            method="pearson"
        )

        # Mock reporter and progress tracker
        mock_reporter = MagicMock()
        mock_progress_tracker = MagicMock()

        # Call execute function
        result = operation.execute(
            data_source=MagicMock(),
            task_dir=Path("/tmp/test_task_dir"),
            reporter=mock_reporter,
            progress_tracker=mock_progress_tracker
        )

        # Check result
        mock_reporter.add_operation.assert_called_with(
            "Operation CorrelationOperation",
            status="error",
            details={"step": "Exception", "message": "Operation failed due to an exception", "error": "Test exception"}
        )
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Test exception", result.error_message)

    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_no_dataframe(self, mock_load_data_operation):
        # Mock load_data_operation to return None
        mock_load_data_operation.return_value = None

        # Create CorrelationOperation object
        operation = CorrelationOperation(
            field1="Field1",
            field2="Field2",
            method="pearson"
        )

        # Mock reporter and progress tracker
        mock_reporter = MagicMock()
        mock_progress_tracker = MagicMock()

        # Call execute function
        result = operation.execute(
            data_source=MagicMock(),
            task_dir=Path("/tmp/test_task_dir"),
            reporter=mock_reporter,
            progress_tracker=mock_progress_tracker
        )

        # Check result
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "Load data and validate input parameters failed")

        # Ensure no artifact is added to reporter
        mock_reporter.add_artifact.assert_not_called()

        # Ensure no operation is added to reporter
        mock_reporter.add_operation.assert_called_with(
            "Operation CorrelationOperation",
            status="info",
            details={"step": "Load data and validate input parameters", "message": "Load data and validate input parameters failed"}
        )

    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_field_not_found(self, mock_load_data_operation):
        # Mock DataFrame without required fields
        mock_df = pd.DataFrame({
            "OtherField1": [1, 2, 3, 4],
            "OtherField2": [2, 4, 6, 8]
        })
        mock_load_data_operation.return_value = mock_df

        # Create CorrelationOperation object
        operation = CorrelationOperation(
            field1="Field1",  # Field does not exist in DataFrame
            field2="Field2",  # Field does not exist in DataFrame
            method="pearson"
        )

        # Mock reporter and progress tracker
        mock_reporter = MagicMock()
        mock_progress_tracker = MagicMock()

        # Call execute function
        result = operation.execute(
            data_source=MagicMock(),
            task_dir=Path("/tmp/test_task_dir"),
            reporter=mock_reporter,
            progress_tracker=mock_progress_tracker
        )

        # Check result
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "Load data and validate input parameters failed")

        # Ensure no artifact is added to reporter
        mock_reporter.add_artifact.assert_not_called()

        # Ensure no operation is added to reporter
        mock_reporter.add_operation.assert_called_with(
            "Operation CorrelationOperation",
            status="info",
            details={"step": "Load data and validate input parameters", "message": "Load data and validate input parameters failed"}
        )

    @patch("pamola_core.profiling.analyzers.correlation.create_scatter_plot")
    @patch("pamola_core.profiling.analyzers.correlation.create_boxplot")
    @patch("pamola_core.profiling.analyzers.correlation.create_heatmap")
    @patch("pamola_core.utils.io.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_with_visualizations(self, mock_load_data_operation, mock_get_timestamped_filename, mock_create_heatmap, mock_create_boxplot, mock_create_scatter_plot):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "Field1": [1, 2, 3, 4],
            "Field2": [2, 4, 6, 8]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock filename
        mock_get_timestamped_filename.side_effect = [
            "Field1_Field2_correlation_20250512_173810.json",
            "Field1_Field2_correlation_plot_20250512_173810.png"
        ]

        # Mock scatter plot creation
        mock_create_scatter_plot.return_value = "Scatter plot created successfully"

        # Create CorrelationOperation object
        operation = CorrelationOperation(
            field1="Field1",
            field2="Field2",
            method="pearson",
            generate_visualization=True,
            use_cache=False
        )

        # Mock reporter and progress tracker
        mock_reporter = MagicMock()
        mock_progress_tracker = MagicMock()

        # Mock analysis results
        analysis_results = {
            "method": "pearson",
            "correlation_coefficient": 0.85,
            "plot_data": {
                "type": "scatter",
                "x_values": [1, 2, 3, 4],
                "y_values": [2, 4, 6, 8],
                "x_label": "Field1",
                "y_label": "Field2"
            }
        }

        # Call execute function
        result = operation.execute(
            data_source=MagicMock(),
            task_dir=Path("/tmp/test_task_dir"),
            reporter=mock_reporter,
            progress_tracker=mock_progress_tracker,
            # analysis_results=analysis_results
        )

        # Check scatter plot is created
        mock_create_scatter_plot.assert_called_once_with(
            x_data=[1, 2, 3, 4],
            y_data=[2, 4, 6, 8],
            output_path=unittest.mock.ANY,
            title="Correlation between Field1 and Field2",
            x_label="Field1",
            y_label="Field2",
            add_trendline=True,
            correlation=0.9999999999999999,
            method="Pearson",
            use_encryption=unittest.mock.ANY,
            encryption_key=unittest.mock.ANY,
            backend=unittest.mock.ANY,
            theme=unittest.mock.ANY,
            strict=unittest.mock.ANY
        )

        # Check number of add_artifact calls
        assert len(result.artifacts) > 0
        
        output = next(
            (x for x in result.artifacts if x.category == 'output' and x.description == 'Correlation analysis between Field1 and Field2'),
            None
        )
        
        visualization = next(
            (x for x in result.artifacts if x.category == 'visualization' and x.description == 'Field1 distribution visualization'),
            None
        )
        
        self.assertIsNotNone(output)
        self.assertIsNotNone(visualization)

        # Check progress tracker updates
        mock_progress_tracker.update.assert_any_call(1, {'step': 'Start operation - Preparation', 'operation': 'CorrelationOperation'})
        mock_progress_tracker.update.assert_any_call(1, {'step': 'Generate visualizations', 'operation': 'CorrelationOperation'})

    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze")
    def test_execute_returns_error_on_analysis_error(self, mock_analyze, mock_load_data):
        # Mock DataFrame with required columns
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_load_data.return_value = df

        # Mock analyze to return error
        mock_analyze.return_value = {"error": "Some error occurred"}

        # Mock reporter
        reporter = MagicMock()

        # Instantiate operation
        op = CorrelationOperation(field1="A", field2="B")

        # Execute
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter
        )

        # Assert result is error
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(result.error_message, "Some error occurred")

    @patch("pamola_core.profiling.analyzers.correlation.create_boxplot")
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_boxplot_branch(self, mock_load_data, mock_analyze, mock_create_boxplot):
        # Mock DataFrame
        df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
        mock_load_data.return_value = df

        # Mock analysis result to trigger boxplot
        mock_analyze.return_value = {
            "method": "correlation_ratio",
            "correlation_coefficient": 0.5,
            "sample_size": 4,
            "plot_data": {
                "type": "boxplot",
                "x_label": "cat",
                "y_label": "val",
                "categories": ["A", "B"],
                "values": [[1, 3], [2, 4]]
            }
        }

        # Mock create_boxplot to simulate successful plot creation
        mock_create_boxplot.return_value = "visualization.png"

        # Mock reporter
        reporter = MagicMock()

        # Instantiate and execute operation
        op = CorrelationOperation(field1="cat", field2="val", use_cache=False)
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter
        )

        mock_create_boxplot.assert_called_once()
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertTrue(any(str(artifact.path).endswith(".png") for artifact in result.artifacts))

    @patch("pamola_core.profiling.analyzers.correlation.create_heatmap")
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_heatmap_branch(self, mock_load_data, mock_analyze, mock_create_heatmap):
        # Mock DataFrame
        df = pd.DataFrame({"cat1": ["A", "B", "A", "B"], "cat2": ["X", "Y", "X", "Y"]})
        mock_load_data.return_value = df

        # Mock analysis result to trigger heatmap
        mock_analyze.return_value = {
            "method": "cramers_v",
            "correlation_coefficient": 0.7,
            "sample_size": 4,
            "plot_data": {
                "type": "heatmap",
                "x_label": "cat1",
                "y_label": "cat2",
                "matrix": [[1, 2], [3, 4]]
            }
        }

        # Mock create_heatmap to simulate successful plot creation
        mock_create_heatmap.return_value = "visualization.png"

        # Mock reporter
        reporter = MagicMock()

        # Instantiate and execute operation
        op = CorrelationOperation(field1="cat1", field2="cat2", use_cache=False)
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter
        )

        # Check that the heatmap branch was executed
        mock_create_heatmap.assert_called_once()
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        # Check that a png artifact was added to the result
        self.assertTrue(any(str(getattr(artifact, "path", "" )).endswith(".png") for artifact in result.artifacts))

    @patch("pamola_core.profiling.analyzers.correlation.create_boxplot")
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_visualization_error_branch(self, mock_load_data, mock_analyze, mock_create_boxplot):
        # Mock DataFrame
        df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
        mock_load_data.return_value = df

        # Mock analysis result to trigger boxplot
        mock_analyze.return_value = {
            "method": "correlation_ratio",
            "correlation_coefficient": 0.5,
            "sample_size": 4,
            "plot_data": {
                "type": "boxplot",
                "x_label": "cat",
                "y_label": "val",
                "categories": ["A", "B"],
                "values": [[1, 3], [2, 4]]
            }
        }

        # Mock create_boxplot to simulate error
        mock_create_boxplot.return_value = "Error: failed to create plot"

        # Mock reporter
        reporter = MagicMock()

        # Instantiate and execute operation
        op = CorrelationOperation(field1="cat", field2="val")
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter
        )

        # No png artifact should be added to result
        self.assertFalse(any(str(getattr(artifact, "path", "" )).endswith(".png") for artifact in result.artifacts))
        self.assertEqual(result.status, OperationStatus.SUCCESS)


class TestCorrelationMatrixOperation(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [3, 2, 1],
            'C': [4, 5, 6]
        })
        self.mock_data_source = Mock()
        self.mock_reporter = Mock()
        self.mock_progress_tracker = Mock()
        self.mock_progress_tracker.update = Mock()
        self.task_dir = Path("temp_task_dir")
        self.operation = CorrelationMatrixOperation(fields=['A', 'B', 'C'])

    @patch('pamola_core.profiling.analyzers.correlation.load_data_operation')
    @patch('pamola_core.profiling.analyzers.correlation.create_correlation_matrix')
    @patch('pamola_core.profiling.analyzers.correlation.write_json')
    def test_execute_success(self, mock_write_json, mock_create_plot, mock_load_data):
        mock_load_data.return_value = self.df
        mock_create_plot.return_value = None

        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )
   
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        mock_write_json.assert_called_once()

    @patch('pamola_core.profiling.analyzers.correlation.load_data_operation')
    def test_missing_fields(self, mock_load_data):
        mock_load_data.return_value = pd.DataFrame({'X': [1, 2, 3]})

        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter
        )

        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Fields not found", result.error_message)

    @patch('pamola_core.profiling.analyzers.correlation.load_data_operation')
    def test_none_dataframe(self, mock_load_data):
        mock_load_data.return_value = None

        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter
        )

        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("No valid DataFrame", result.error_message)

    @patch('pamola_core.profiling.analyzers.correlation.load_data_operation')
    @patch('pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze_matrix')
    def test_analysis_error(self, mock_analyze, mock_load_data):
        mock_load_data.return_value = self.df
        mock_analyze.return_value = {"error": "Computation failed"}

        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter
        )

        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Computation failed", result.error_message)

    @patch("pamola_core.profiling.analyzers.correlation.create_correlation_matrix")
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze_matrix")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_visualization_success(self, mock_load_data, mock_analyze_matrix, mock_create_correlation_matrix):
        # Mock DataFrame
        df = pd.DataFrame({
            "A": [1, 2, 3, 4],
            "B": [4, 3, 2, 1],
            "C": [1, 1, 2, 2]
        })
        mock_load_data.return_value = df

        # Mock analysis result to include correlation_matrix
        mock_analyze_matrix.return_value = {
            "correlation_matrix": {
                "A": {"A": 1.0, "B": -1.0, "C": 0.5},
                "B": {"A": -1.0, "B": 1.0, "C": -0.5},
                "C": {"A": 0.5, "B": -0.5, "C": 1.0}
            },
            "significant_correlations": [("A", "B")],
        }

        # Mock create_correlation_matrix to simulate successful plot creation
        mock_create_correlation_matrix.return_value = "success"

        # Mock reporter
        reporter = MagicMock()

        # Instantiate and execute operation
        op = CorrelationMatrixOperation(fields=["A", "B", "C"])
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter
        )

        # Check that the plot function was called
        mock_create_correlation_matrix.assert_called_once()
        # Success result
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        # There is a png artifact in result
        self.assertTrue(any(str(getattr(artifact, "path", "" )).endswith(".png") for artifact in result.artifacts))
        # Reporter also receives png artifact
        reporter.add_artifact.assert_any_call("png", unittest.mock.ANY, "Correlation matrix visualization")

    @patch("pamola_core.profiling.analyzers.correlation.CorrelationAnalyzer.analyze_matrix")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_exception_branch(self, mock_load_data, mock_analyze_matrix):
        # Mock DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_load_data.return_value = df

        # Mock analyze_matrix to raise an exception
        mock_analyze_matrix.side_effect = RuntimeError("Simulated matrix error")

        # Mock reporter and progress_tracker
        reporter = MagicMock()
        progress_tracker = MagicMock()

        # Instantiate and execute operation
        op = CorrelationMatrixOperation(fields=["A", "B"])
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Check progress_tracker.update was called with error message
        progress_tracker.update.assert_any_call(0, {"step": "Error", "error": "Simulated matrix error"})
        # Check reporter.add_operation was called with status="error"
        reporter.add_operation.assert_any_call(
            "Error creating correlation matrix",
            status="error",
            details={"error": "Simulated matrix error"}
        )
        # Result is error
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Error creating correlation matrix: Simulated matrix error", result.error_message)


class TestAnalyzeCorrelations(unittest.TestCase):
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationOperation.execute")
    def test_analyze_correlations_success_and_missing_fields(self, mock_execute, mock_load_data):
        # Mock DataFrame with columns A, B, C
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        mock_load_data.return_value = df

        # Mock execute to always return a successful OperationResult
        mock_execute.return_value = MagicMock(status=OperationStatus.SUCCESS, error_message=None)

        # Mock reporter
        reporter = MagicMock()

        # Test with one valid pair and one invalid pair
        pairs = [("A", "B"), ("A", "D")]  # "D" does not exist

        results = analyze_correlations(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter,
            pairs=pairs
        )

        # "A_B" succeeds, "A_D" fails due to missing field D
        self.assertIn("A_B", results)
        self.assertIn("A_D", results)
        self.assertEqual(results["A_B"].status, OperationStatus.SUCCESS)
        self.assertEqual(results["A_D"].status, OperationStatus.ERROR)
        self.assertIn("Fields not found", results["A_D"].error_message)

        # Ensure reporter.add_operation is called for the error
        reporter.add_operation.assert_any_call(
            "Correlation Analysis: A vs D",
            status="error",
            details={"error": "Fields not found: D"}
        )

    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_analyze_correlations_no_dataframe(self, mock_load_data):
        # Mock load_data_operation returns None
        mock_load_data.return_value = None

        reporter = MagicMock()
        pairs = [("A", "B")]

        results = analyze_correlations(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter,
            pairs=pairs
        )

        # No DataFrame, should return empty dict
        self.assertEqual(results, {})
        reporter.add_operation.assert_any_call(
            "Correlation analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"}
        )

    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_missing_field1(self, mock_load_data):
        # DataFrame only has column 'B'
        df = pd.DataFrame({"B": [1, 2, 3]})
        mock_load_data.return_value = df

        reporter = MagicMock()
        pairs = [("A", "B")]  # 'A' does not exist

        results = analyze_correlations(
            data_source=MagicMock(),
            task_dir=Path("."),
            reporter=reporter,
            pairs=pairs
        )

        # Result should have key "A_B" and status is ERROR
        self.assertIn("A_B", results)
        self.assertEqual(results["A_B"].status, OperationStatus.ERROR)
        self.assertIn("Fields not found: A", results["A_B"].error_message)

        # Ensure reporter.add_operation is called with status="error"
        reporter.add_operation.assert_any_call(
            "Correlation Analysis: A vs B",
            status="error",
            details={"error": "Fields not found: A"}
        )
    @patch("pamola_core.profiling.analyzers.correlation.CorrelationOperation.execute")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_overall_tracker_update_on_error(self, mock_load_data, mock_execute):
        # DataFrame has enough columns to not enter missing_fields branch
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_load_data.return_value = df

        # Mock OperationResult returns error
        error_result = OperationResult(
            status=OperationStatus.ERROR,
            error_message="Simulated correlation error"
        )
        mock_execute.return_value = error_result

        # Mock reporter
        reporter = MagicMock()

        # Patch ProgressTracker to check update
        with patch("pamola_core.profiling.analyzers.correlation.ProgressTracker") as MockProgressTracker:
            mock_tracker = MockProgressTracker.return_value

            # Call analyze_correlations with track_progress=True (default)
            pairs = [("A", "B")]
            analyze_correlations(
                data_source=MagicMock(),
                task_dir=Path("."),
                reporter=reporter,
                pairs=pairs
            )

            # Check overall_tracker.update is called with status="error" and correct error message
            mock_tracker.update.assert_any_call(
                1,
                {"pair": "A_B", "status": "error", "error": "Simulated correlation error"}
            )

    @patch("pamola_core.profiling.analyzers.correlation.CorrelationOperation.execute")
    @patch("pamola_core.profiling.analyzers.correlation.load_data_operation")
    def test_execute_raises_exception(self, mock_load_data, mock_execute):
        # DataFrame has enough columns
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_load_data.return_value = df

        # Mock execute to raise exception
        mock_execute.side_effect = RuntimeError("Simulated execution error")

        # Mock reporter
        reporter = MagicMock()

        # Patch ProgressTracker to check update
        with patch("pamola_core.profiling.analyzers.correlation.ProgressTracker") as MockProgressTracker:
            mock_tracker = MockProgressTracker.return_value

            pairs = [("A", "B")]
            results = analyze_correlations(
                data_source=MagicMock(),
                task_dir=Path("."),
                reporter=reporter,
                pairs=pairs
            )

            # Result should be error
            self.assertIn("A_B", results)
            self.assertEqual(results["A_B"].status, OperationStatus.ERROR)
            self.assertIn("Simulated execution error", results["A_B"].error_message)

            # Reporter.add_operation was called with status="error"
            reporter.add_operation.assert_any_call(
                "Analyzing correlation between A and B",
                status="error",
                details={"error": "Simulated execution error"}
            )
            # overall_tracker.update was called with status="error"
            mock_tracker.update.assert_any_call(
                1, {"pair": "A_B", "status": "error"}
            )


class TestCorrelationAnalyzer(unittest.TestCase):
    @patch("pamola_core.profiling.analyzers.correlation.analyze_correlation")
    def test_analyze_calls_analyze_correlation(self, mock_analyze_correlation):
        # Prepare data and mock
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_analyze_correlation.return_value = {"result": "ok"}

        # Call function
        result = CorrelationAnalyzer.analyze(df, "A", "B", method="pearson", extra_param=123)

        # Ensure the original function is called with correct parameters
        mock_analyze_correlation.assert_called_once_with(
            df=df, field1="A", field2="B", method="pearson", extra_param=123
        )
        self.assertEqual(result, {"result": "ok"})

    @patch("pamola_core.profiling.analyzers.correlation.analyze_correlation_matrix")
    def test_analyze_matrix_calls_analyze_correlation_matrix(self, mock_analyze_matrix):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_analyze_matrix.return_value = {"matrix": "ok"}

        result = CorrelationAnalyzer.analyze_matrix(df, ["A", "B"], min_threshold=0.5)
        mock_analyze_matrix.assert_called_once_with(
            df=df, fields=["A", "B"], min_threshold=0.5
        )
        self.assertEqual(result, {"matrix": "ok"})

    @patch("pamola_core.profiling.analyzers.correlation.estimate_resources")
    def test_estimate_resources_calls_estimate_resources(self, mock_estimate_resources):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        mock_estimate_resources.return_value = {"cpu": 1, "mem": 2}

        result = CorrelationAnalyzer.estimate_resources(df, "A", "B")
        mock_estimate_resources.assert_called_once_with(df, "A", "B")
        self.assertEqual(result, {"cpu": 1, "mem": 2})

if __name__ == "__main__":
    unittest.main()