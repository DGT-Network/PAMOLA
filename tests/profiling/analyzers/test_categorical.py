"""
Tests for the categorical module in the pamola_core/profiling/analyzers package.
These tests ensure that the CategoricalAnalyzer and CategoricalOperation classes properly implement
categorical field analysis, resource estimation, artifact generation, error handling, and caching.
"""
import os
import unittest
import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from unittest.mock import PropertyMock, patch, MagicMock
from pamola_core.profiling.analyzers import categorical
from pamola_core.profiling.analyzers.categorical import CategoricalAnalyzer, CategoricalOperation, analyze_categorical_fields
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

class TestCategoricalOperation(unittest.TestCase):
    def setUp(self):
        self.operation = CategoricalOperation(
            field_name="Category",
            top_n=15,
            min_frequency=1,
            profile_type="categorical",
            analyze_anomalies=True,
            description="description",
            include_timestamp=True,
            save_output=True,
            generate_visualization=True,
            use_cache=False,
            force_recalculation=False,
            visualization_backend=None,
            visualization_theme=None,
            visualization_strict=False,
            visualization_timeout=120,
            use_encryption=False,
            encryption_key=None,
            encryption_mode=None
        )
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NY', 'LA', 'SF']
        })
        self.mock_data_source = DummyDataSource(df=df)
        task_dir = Path("test_task_dir/unittest/profiling/analyzers/categorical")
        os.makedirs(task_dir, exist_ok=True)
        self.mock_task_dir = task_dir
        self.mock_reporter = Reporter()
        self.mock_progress_tracker = Progress()

    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
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
        self.assertIn("validate input parameters failed", result.error_message)


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

    @patch("pamola_core.profiling.analyzers.categorical.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_with_value_dictionary(self, mock_load_data_operation, mock_analyze, mock_get_timestamped_filename):
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
        description = "Category value dictionary"
        any_item = any(artifact.description in description for artifact in result.artifacts)
        self.assertTrue(any_item)
        self.assertTrue(any(artifact.artifact_type == "csv" for artifact in result.artifacts))

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_with_empty_value_dictionary(self, mock_load_data_operation, mock_analyze):
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

        # Assertions
        description = "Category statistical analysis"
        any_item = any(artifact.description in description for artifact in result.artifacts)
        self.assertTrue(any_item)
        contain_category_description = any(
            "Category value dictionary" == val
            for artifact in self.mock_reporter.artifacts
            for item in artifact
            for val in item
        )
        self.assertFalse(contain_category_description)

    @patch("pamola_core.profiling.analyzers.categorical.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_visualization_error(self, mock_load_data_operation, mock_create_visualization):
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
        # Ensure no visualization artifact is added
        contain_category_description = any(
            f"{self.operation.field_name} distribution visualization" == val
            for artifact in self.mock_reporter.artifacts
            for item in artifact
            for val in item
        )
        self.assertFalse(contain_category_description)
        
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

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        mock_create_visualization.assert_called_once()
        description = "Category distribution visualization"
        any_item = any(artifact.description in description for artifact in result.artifacts)
        self.assertTrue(any_item)
    
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    def test_execute_exception_handling(self, mock_load_data_operation, mock_load_settings_operation):
        # Mock load_data_operation to raise an exception
        error = "Test exception: load_data_operation"
        mock_load_settings_operation.return_value = {}
        mock_load_data_operation.side_effect = Exception(error)

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        # Ensure reporter adds the error operation
        error_operations = [
            kwargs
            for args, kwargs in self.mock_reporter.operations
            if kwargs.get('status') == 'error'
        ]
        error_operation = error_operations[0]
        status = error_operation["status"]
        error_details = error_operation["details"]["error"]
        # Ensure the result is an error with the correct message
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(status, OperationStatus.ERROR.value)
        self.assertIn(error, error_details)

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_execute_no_anomalies(self, mock_load_settings_operation, mock_load_data_operation, mock_analyze):
        # Mock DataFrame with the target field
        mock_load_settings_operation.return_value = {}
        mock_df = pd.DataFrame({
            self.operation.field_name: ["A", "B", "A", "C", "B", "A"]
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
        # Ensure the result is not an error
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        # Ensure no anomalies artifact is added
        description = f"{self.operation.field_name} anomalies"
        not_all = not all(artifact.description in description for artifact in result.artifacts)
        self.assertTrue(not_all)

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_execute_with_numeric_like_strings(self, mock_load_settings_operation, mock_load_data_operation, mock_analyze):
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            self.operation.field_name: ["123", "456", "123", "789", "456", "123"]
        })
        mock_load_data_operation.return_value = mock_df
        mock_load_settings_operation.return_value = {}

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
        # Ensure the result is not an error
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        # Ensure anomalies artifact is added
        description = f"{self.operation.field_name} anomalies"
        any_item = any(artifact.description in description for artifact in result.artifacts)
        self.assertTrue(any_item)

        # Verify that the anomalies are correctly added to the result
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

    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_execute_dataframe_missing_columns_attribute(self, mock_load_settings_operation, mock_load_data_operation):
        # Mock object without 'columns' attribute
        mock_df = pd.DataFrame({
            "name_test": ["A", "B", "A"]
        })
        mock_load_data_operation.return_value = mock_df
        mock_load_settings_operation.return_value = {}

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn(
            "validate input parameters failed",
            result.error_message
        )

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_execute_analyzer_exception(self,
                                        mock_load_settings_operation,
                                        mock_load_data_operation,
                                        mock_analyze):
        mock_logger = MagicMock()
        # Mock DataFrame with the target field
        mock_df = pd.DataFrame({
            self.operation.field_name: ["A", "B", "A"]
        })
        mock_load_settings_operation.return_value = {}
        mock_load_data_operation.return_value = mock_df

        # Mock analyzer to raise an exception
        error = "Analyzer failed"
        mock_analyze.side_effect = Exception(error)

        # Execute the operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker,
            logger=mock_logger
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn(
            error,
            result.error_message
        )

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_execute_analysis_results_with_error(self,
                                                 mock_load_settings_operation,
                                                 mock_load_data_operation,
                                                 mock_analyze):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            self.operation.field_name: ['Alice', 'Bob', 'Charlie']
        })
        mock_analyze.return_value = {
            "error": "Analysis failed due to invalid data",
            "status": "error"
        }
        mock_load_settings_operation.return_value = {}
        mock_load_data_operation.return_value = df

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
        mock_result = MagicMock()
        dict_dir = self.mock_task_dir / "dictionaries"
        # Case when anomalies is None
        self.operation._save_anomalies_to_csv(
            analysis_results={},  # No anomalies key
            dict_dir=dict_dir,
            include_timestamp=True,
            result=mock_result,
            reporter=self.mock_reporter
        )
        # No artifact or other function should be called
        mock_result.add_artifact.assert_not_called()
        assert len(self.mock_reporter.artifacts) == 0

        # Case when anomalies is {}
        self.operation._save_anomalies_to_csv(
            analysis_results={"anomalies": {}},
            dict_dir=dict_dir,
            include_timestamp=True,
            result=mock_result,
            reporter=self.mock_reporter
        )
        mock_result.add_artifact.assert_not_called()
        assert len(self.mock_reporter.artifacts) == 0
        
    @patch("pamola_core.profiling.analyzers.categorical.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.categorical.pd.DataFrame.to_csv")
    def test_save_anomalies_to_csv_typo_and_numeric(self, mock_to_csv, mock_get_filename):
        mock_result = MagicMock()
        dict_dir = self.mock_task_dir / "dictionaries"
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
        self.operation._save_anomalies_to_csv(
            analysis_results, dict_dir, include_timestamp=True,
            result=mock_result, reporter=self.mock_reporter
        )
        # Check artifact added
        mock_result.add_artifact.assert_called_with(
            "csv", dict_dir / "Category_anomalies.csv", "Category anomalies", category="dictionary"
        )
        # Ensure anomalies artifact is added
        description = "Category anomalies"
        
        # Flatten all nested items and check for the description efficiently
        contain_category_description = any(
            description == val
            for artifact in self.mock_reporter.artifacts
            for item in artifact
            for val in item
        )
        
        self.assertTrue(contain_category_description)

    def test_save_anomalies_to_csv_handles_exception(self):
        mock_result = MagicMock()
        dict_dir = self.mock_task_dir / "dictionaries"
        # Patch pd.DataFrame to raise exception
        with patch("pamola_core.profiling.analyzers.categorical.pd.DataFrame", side_effect=Exception("df error")):
            self.operation._save_anomalies_to_csv(
                {"anomalies": {"numeric_like_strings": {"123": 1}}},
                dict_dir, True, mock_result, self.mock_reporter
            )
            
        error_operations = [
            kwargs
            for args, kwargs in self.mock_reporter.operations
            if kwargs.get('status') == 'warning'
        ]
        error_operation = error_operations[0]
        status = error_operation["status"]
        error_details = error_operation["details"]["warning"]
        
        self.assertEqual(status, "warning")
        self.assertIsNotNone(error_details)

    @patch("pamola_core.profiling.analyzers.categorical.ensure_directory")
    def test_prepare_directories_creates_and_returns_paths(self, mock_ensure_directory):
        op = CategoricalOperation(field_name="Category")
        base_dir = self.mock_task_dir
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
        self.mock_reporter = Reporter()
        self.mock_task_dir = Path("test_task_dir/unittest/profiling/analyzers/categorical")
        self.df = pd.DataFrame({
            "cat1": ["A", "B", "A"],
            "cat2": ["X", "Y", "Z"],
            "num": [1, 2, 3]
        })
        self.mock_data_source = DummyDataSource(df=self.df)

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

    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_analyze_fields_no_dataframe(self, mock_load_settings_operation, mock_load_data_operation):
        mock_load_settings_operation.return_value = {}
        mock_load_data_operation.return_value = None
        results = analyze_categorical_fields(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter
        )

        is_error = any(
            results[item].error_message == "Load data and validate input parameters failed"
            for item in results
        )
        self.assertTrue(is_error)

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    @patch("pamola_core.profiling.analyzers.categorical.load_data_operation")
    @patch("pamola_core.profiling.analyzers.categorical.load_settings_operation")
    def test_analyze_fields_operation_exception(self, mock_load_settings_operation, mock_load_data_operation, mock_execute):
        # Simulate exception in operation.execute
        df = pd.DataFrame({
            "num": [1, 2, 3]
        })
        mock_load_settings_operation.return_value = {}
        mock_load_data_operation.return_value = df
        mock_execute.side_effect = Exception("Operation failed")
        results = analyze_categorical_fields(
            data_source=self.mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            cat_fields=["cat1"]
        )
        self.assertNotIn("cat1", [k for k, v in results.items() if v.status == OperationStatus.SUCCESS])

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

   
    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_auto_detect_column_dtype_exception(self, mock_execute):
        # Normal DataFrame
        df = pd.DataFrame({"bad_col": [1, 2, 3]})
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)

        # Patch Series.dtype to raise exception when accessed
        with patch("pandas.Series.dtype", new_callable=PropertyMock) as mock_dtype:
            mock_dtype.side_effect = ValueError("broken dtype")
            results = analyze_categorical_fields(
                data_source=self.mock_data_source,
                task_dir=self.mock_task_dir,
                reporter=self.mock_reporter
            )

        self.assertEqual(results, {})

    def test_auto_detect_no_columns_attribute(self):
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

    @patch("pamola_core.profiling.analyzers.categorical.CategoricalOperation.execute")
    def test_overall_tracker_update_on_error(self, mock_execute):
        # Mock tracker instance
        mock_tracker = MagicMock()

        # Mock operation result with ERROR status
        error_message = "Some error"
        mock_execute.return_value = OperationResult(status=OperationStatus.ERROR, error_message=error_message)

        # DataFrame with 1 categorical column
        df = pd.DataFrame({"cat1": ["A", "B", "A"]})
        mock_data_source = MagicMock()
        mock_data_source.get_dataframe.return_value = df

        result = analyze_categorical_fields(
            data_source=mock_data_source,
            task_dir=self.mock_task_dir,
            reporter=self.mock_reporter,
            track_progress=mock_tracker,
            cat_fields=["cat1"]
        )
        self.assertIn(error_message, result["cat1"].error_message)
                  
# Add pytest-based tests for full coverage
class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
        self.encryption_keys = {}
        self.encryption_modes = {}
    def get_dataframe(self, dataset_name, **kwargs):
        if self.df is not None:
            return self.df, None
        return None, {"message": self.error or "No data"}

def dummy_load_settings_operation(data_source, dataset_name, **kwargs):
    return {}

def dummy_load_data_operation(data_source, dataset_name, **settings):
    return data_source.get_dataframe(dataset_name)

@pytest.fixture(autouse=True)
def patch_io(monkeypatch):
    monkeypatch.setattr(categorical, 'load_settings_operation', dummy_load_settings_operation)
    monkeypatch.setattr(categorical, 'load_data_operation', dummy_load_data_operation)
    monkeypatch.setattr(categorical, 'write_json', lambda *a, **k: None)
    monkeypatch.setattr(categorical, 'write_dataframe_to_csv', lambda *a, **k: None)
    monkeypatch.setattr(categorical, 'ensure_directory', lambda *a, **k: None)
    monkeypatch.setattr(categorical, 'get_timestamped_filename', lambda prefix, ext, ts: f"{prefix}.{ext}")
    monkeypatch.setattr(categorical, 'get_encryption_mode', lambda *a, **k: None)
    yield

@pytest.fixture
def dummy_dirs(tmp_path):
    return {
        'output': tmp_path / 'output',
        'visualizations': tmp_path / 'visualizations',
        'dictionaries': tmp_path / 'dictionaries'
    }

class Reporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []

    def add_operation(self, *args, **kwargs):
        self.operations.append((args, kwargs))

    def add_artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))
        
@pytest.fixture
def dummy_reporter():
    return Reporter()

class Progress:
    def __init__(self):
        self.updates = []
        self.total = 0

    def update(self, step, info):
        self.updates.append((step, info))

    def create_subtask(self, total, description, unit):
        return Progress()

    def close(self):
        pass
@pytest.fixture
def dummy_progress():
    return Progress()

@pytest.fixture
def dummy_task_dir(tmp_path):
    (tmp_path / 'output').mkdir()
    (tmp_path / 'visualizations').mkdir()
    (tmp_path / 'dictionaries').mkdir()
    return tmp_path

@pytest.fixture
def dummy_categorical_result():
    return {
        'total_records': 7,
        'null_values': 1,
        'null_percent': 1/7*100,
        'unique_values': 3,
        'entropy': 1.5,
        'cardinality_ratio': 3/7,
        'distribution_type': 'nominal',
        'top_values': {'a': 3, 'b': 2, 'c': 1},
        'value_dictionary': {'dictionary_data': [{'value': 'a', 'count': 3}, {'value': 'b', 'count': 2}, {'value': 'c', 'count': 1}]},
        'anomalies': {'potential_typos': {}, 'single_char_values': {}, 'numeric_like_strings': {}}
    }

class DummyOperationCache:
    def __init__(self):
        self.saved = None
        self.loaded = None
    def generate_cache_key(self, **kwargs):
        return 'dummy_key'
    def save_cache(self, **kwargs):
        self.saved = kwargs
        return True
    def get_cache(self, **kwargs):
        return self.loaded

@pytest.fixture(autouse=True)
def patch_operation_cache(monkeypatch):
    dummy_cache = DummyOperationCache()
    monkeypatch.setattr(categorical, 'operation_cache', dummy_cache)
    yield dummy_cache

@pytest.fixture(autouse=True)
def patch_plot(monkeypatch):
    monkeypatch.setattr(categorical, 'plot_value_distribution', lambda **kwargs: 'dummy_plot.png')
    yield

@pytest.fixture
def valid_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'categorical': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['NY', 'LA', 'SF']
    })
    
def test_analyze_valid_case():
    df = pd.DataFrame({'cat': ['a', 'b', 'a', 'c', 'b', 'a', None]})
    result = categorical.CategoricalAnalyzer.analyze(df, 'cat', top_n=2, min_frequency=1)
    assert isinstance(result, dict)

def test_estimate_resources():
    df = pd.DataFrame({'cat': ['a', 'b']})
    result = categorical.CategoricalAnalyzer.estimate_resources(df, 'cat')
    assert isinstance(result, dict)

def test_categorical_operation_execute_valid(dummy_task_dir, dummy_reporter, dummy_progress, dummy_categorical_result, valid_df):
    # Patch analyze to return dummy result
    with mock.patch('pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze', return_value=dummy_categorical_result),\
        patch("pamola_core.profiling.analyzers.categorical.load_settings_operation", return_value={}), \
        patch("pamola_core.profiling.analyzers.categorical.load_data_operation", return_value=valid_df):
        op = categorical.CategoricalOperation(field_name='categorical', top_n=2, min_frequency=1)
        result = op.execute(DummyDataSource(df=valid_df), dummy_task_dir, dummy_reporter, dummy_progress)
        assert result.status == OperationStatus.SUCCESS
        assert 'total_records' in result.metrics
        assert 'unique_values' in result.metrics

def test_categorical_operation_execute_invalid_field(dummy_task_dir, dummy_reporter, dummy_progress):
    with mock.patch('pamola_core.profiling.analyzers.categorical.CategoricalAnalyzer.analyze', return_value={'error': 'Field not found'}):
        op = categorical.CategoricalOperation(field_name='not_a_field', top_n=2, min_frequency=1)
        result = op.execute(DummyDataSource(), dummy_task_dir, dummy_reporter, dummy_progress)
        assert result.status == OperationStatus.ERROR
        assert 'error' in result.error_message or result.error_message

def test_categorical_operation_execute_empty_df(dummy_task_dir, dummy_reporter, dummy_progress):
    class EmptyDataSource:
        def get_dataframe(self, name):
            return pd.DataFrame()
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    result = op.execute(EmptyDataSource(), dummy_task_dir, dummy_reporter, dummy_progress)
    assert result.status == OperationStatus.ERROR

def test_save_output_handles_empty_dictionary(dummy_task_dir, dummy_reporter, dummy_categorical_result):
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    dummy_categorical_result['value_dictionary']['dictionary_data'] = []
    result = mock.Mock()
    op._save_output(dummy_categorical_result, dummy_task_dir / 'output', dummy_task_dir / 'dictionaries', result, dummy_reporter)
    # Should not raise

def test_save_output_handles_anomalies(dummy_task_dir, dummy_reporter, dummy_categorical_result):
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    dummy_categorical_result['anomalies'] = {
        'potential_typos': {'typo': {'count': 1, 'similar_to': 'a', 'similar_count': 3}},
        'single_char_values': {'x': 2},
        'numeric_like_strings': {'123': 1}
    }
    result = mock.Mock()
    op._save_output(dummy_categorical_result, dummy_task_dir / 'output', dummy_task_dir / 'dictionaries', result, dummy_reporter)
    # Should not raise

def test_generate_visualizations_handles_missing_top_values(dummy_task_dir, dummy_categorical_result):
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    dummy_categorical_result.pop('top_values', None)
    result = mock.Mock()
    out = op._generate_visualizations(dummy_categorical_result, dummy_task_dir / 'visualizations', result)
    assert out.startswith('Error:')

def test_get_cache_and_save_cache(dummy_task_dir, dummy_categorical_result, patch_operation_cache):
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    # Save cache
    op._original_df = pd.DataFrame({'cat': ['a', 'b']})
    result = mock.Mock()
    result.status = OperationStatus.SUCCESS
    result.metrics = {'foo': 1}
    result.error_message = None
    result.execution_time = 1.0
    result.error_trace = None
    result.artifacts = []
    op._save_cache(dummy_task_dir, result)
    # Get cache
    patch_operation_cache.loaded = {
        'result': {
            'status': 'SUCCESS',
            'metrics': {'foo': 1},
            'error_message': None,
            'execution_time': 1.0,
            'error_trace': None,
            'artifacts': []
        }
    }
    out = op._get_cache(pd.DataFrame({'cat': ['a', 'b']}))
    assert out.status == OperationStatus.SUCCESS

def test_set_input_parameters_sets_all():
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    op._set_input_parameters(field_name='cat2', top_n=3, min_frequency=2, profile_type='other', analyze_anomalies=False, generate_visualization=False, save_output=False, output_format='json', include_timestamp=False, use_cache=False, force_recalculation=True, visualization_backend='mpl', visualization_theme='dark', visualization_strict=True, visualization_timeout=10, use_encryption=True, encryption_key='key')
    assert op.field_name == 'cat2'
    assert op.top_n == 3
    assert op.min_frequency == 2
    assert op.profile_type == 'other'
    assert op.analyze_anomalies is False
    assert op.generate_visualization is False
    assert op.save_output is False
    assert op.output_format == 'json'
    assert op.include_timestamp is False
    assert op.use_cache is False
    assert op.force_recalculation is True
    assert op.visualization_backend == 'mpl'
    assert op.visualization_theme == 'dark'
    assert op.visualization_strict is True
    assert op.visualization_timeout == 10
    assert op.use_encryption is True
    assert op.encryption_key == 'key'

def test_validate_input_parameters():
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    df = pd.DataFrame({'cat': [1, 2, 3]})
    assert op._validate_input_parameters(df) is True
    op2 = categorical.CategoricalOperation(field_name='not_a_field', top_n=2, min_frequency=1)
    assert op2._validate_input_parameters(df) is False

def test_generate_data_hash():
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    df = pd.DataFrame({'cat': [1, 2, 3]})
    h = op._generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 32

def test_compute_total_steps():
    op = categorical.CategoricalOperation(field_name='cat', top_n=2, min_frequency=1)
    steps = op._compute_total_steps(use_cache=True, force_recalculation=False, save_output=True, generate_visualization=True)
    assert steps >= 7

if __name__ == "__main__":
    unittest.main()