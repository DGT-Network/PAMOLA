from datetime import datetime, timedelta
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pamola_core.profiling.analyzers.date import DateAnalyzer, DateOperation, analyze_date_fields
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
        
class TestDateOperation(unittest.TestCase):

    def setUp(self):
        self.mock_data_source = MagicMock()
        self.mock_reporter = MagicMock()
        self.mock_progress_tracker = MagicMock()
        self.task_dir = Path("temp_task_dir")
        self.field_name = "birth_date"
        self.operation = DateOperation(
            field_name=self.field_name,
            min_year=1940,
            max_year=2005,
            generate_plots=True
        )

    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    @patch("pamola_core.profiling.analyzers.date.write_json")
    @patch("pamola_core.profiling.analyzers.date.get_timestamped_filename")
    def test_execute_success(self, mock_get_timestamped_filename, mock_write_json, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "birth_date": ["2000-01-01", "1990-05-15", "1985-07-20"],
            "create_at": ["2000-01-01", "1990-05-15", "1985-07-20"],
            "other_field": [1, 2, 3]
        })
        mock_load_data_operation.return_value = mock_df
        mock_get_timestamped_filename.return_value = "birth_date_stats.json"

        # Mock analysis results
        analysis_results = {
            "valid_count": 3,
            "invalid_count": 0,
            "min_date": "1985-07-20",
            "max_date": "2000-01-01",
            "anomalies": {},
            "age_statistics": {
                "min_age": 20,
                "max_age": 35,
                "mean_age": 27.5,
                "median_age": 27.5
            }
        }

        with patch("pamola_core.profiling.analyzers.date.DateAnalyzer.analyze", return_value=analysis_results):
            # Execute operation
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                progress_tracker=self.mock_progress_tracker
            )

            # Debugging: Print result status and artifacts
            print("Result status:", result.status)
            print("Result artifacts:", result.artifacts)

            # Assertions
            self.assertEqual(result.status, OperationStatus.SUCCESS)
            mock_write_json.assert_called_once()
            self.mock_reporter.add_artifact.assert_called_once_with(
                "json",
                str(self.task_dir / "birth_date_stats.json"),
                "birth_date statistical analysis"
            )

    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_field_not_found(self, mock_load_data_operation):
        # Mock DataFrame without the target field
        mock_df = pd.DataFrame({
            "other_field": [1, 2, 3]
        })
        mock_load_data_operation.return_value = mock_df

        # Execute operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Field birth_date not found", result.error_message)

    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_none_dataframe(self, mock_load_data_operation):
        # Mock None DataFrame
        mock_load_data_operation.return_value = None

        # Execute operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("No valid DataFrame found", result.error_message)
    
    
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    @patch("pamola_core.profiling.analyzers.date.DateAnalyzer.analyze")
    def test_analysis_error(self, mock_analyze, mock_load_data_operation):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "birth_date": ["2000-01-01", "1990-05-15", "1985-07-20"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis error
        mock_analyze.return_value = {"error": "Analysis failed"}

        # Execute operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Analysis failed", result.error_message)

    @patch("pamola_core.profiling.analyzers.date.DateOperation._save_anomalies_to_csv")
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    @patch("pamola_core.profiling.analyzers.date.DateAnalyzer.analyze")
    def test_save_anomalies_to_csv(self, mock_analyze, mock_load_data_operation, mock_save_anomalies_to_csv):
        # Mock DataFrame
        mock_df = pd.DataFrame({
            "date_field": ["2023-01-01", "2023-01-02", "invalid_date"]
        })
        mock_load_data_operation.return_value = mock_df

        # Mock analysis results with anomalies
        analysis_results = {
            "valid_count": 2,
            "invalid_count": 1,
            "anomalies": {"invalid_dates": 1},
            "invalid_dates_examples": [("index_2", "invalid_date")]
        }
        mock_analyze.return_value = analysis_results

        # Mock progress tracker
        mock_progress_tracker = MagicMock()

        # Create DateOperation instance
        operation = DateOperation(field_name="date_field")

        # Execute operation
        result = operation.execute(
            data_source=MagicMock(),
            task_dir=Path("/tmp/test_task_dir"),
            reporter=MagicMock(),
            progress_tracker=mock_progress_tracker
        )

        # Assertions
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        mock_save_anomalies_to_csv.assert_called_once_with(
            analysis_results,
            Path("/tmp/test_task_dir/dictionaries"),
            True,  # include_timestamp
            result,
            unittest.mock.ANY  # reporter
        )
        mock_progress_tracker.update.assert_any_call(1, {"step": "Saved anomalies data"})

    @patch('pamola_core.profiling.analyzers.date.plot_date_distribution')
    @patch('pamola_core.profiling.analyzers.date.plot_value_distribution')
    @patch('pamola_core.profiling.analyzers.date.get_timestamped_filename')
    def test_generate_visualizations(self, mock_get_filename, mock_plot_value, mock_plot_date):
        # Setup
        analysis_results = {
            'year_distribution': {'2023': 10, '2022': 5},
            'month_distribution': {'01': 3, '02': 4, '03': 8},
            'day_of_week_distribution': {'Monday': 3, 'Tuesday': 7},
            'age_distribution': {'20-24': 5, '25-29': 10}
        }
        
        vis_dir = Path('/tmp/test/visualizations')
        include_timestamp = True
        is_birth_date = True
        
        # Mock filename generation
        mock_get_filename.side_effect = [
            'field_year_distribution.png',
            'field_month_distribution.png',
            'field_dow_distribution.png',
            'field_age_distribution.png'
        ]
        
        # Mock successful plot creation
        mock_plot_date.return_value = 'Success'
        mock_plot_value.return_value = 'Success'
        
        # Create operation result and reporter mocks
        result = MagicMock(spec=OperationResult)
        reporter = MagicMock()
        
        # Execute
        operation = DateOperation(field_name='test_field')
        operation._generate_visualizations(
            analysis_results,
            vis_dir,
            include_timestamp,
            is_birth_date,
            result,
            reporter
        )
        
        # Assert
        # Check year distribution plot
        mock_plot_date.assert_called_once_with(
            {'year_distribution': analysis_results['year_distribution']},
            str(vis_dir / 'field_year_distribution.png'),
            title='Birth Year Distribution'
        )
        
        # Check other distribution plots
        expected_plot_value_calls = [
            unittest.mock.call(
                analysis_results['month_distribution'],
                str(vis_dir / 'field_month_distribution.png'),
                title='Birth Month Distribution'
            ),
            unittest.mock.call(
                analysis_results['day_of_week_distribution'],
                str(vis_dir / 'field_dow_distribution.png'),
                title='Birth Day of Week Distribution'
            ),
            unittest.mock.call(
                analysis_results['age_distribution'],
                str(vis_dir / 'field_age_distribution.png'),
                title='Age Distribution',
                x_label='Age Group',
                y_label='Count'
            )
        ]
        
        mock_plot_value.assert_has_calls(expected_plot_value_calls)
        
        expected_artifact_calls = [
            unittest.mock.call('png', vis_dir / 'field_year_distribution.png', 'test_field year distribution'),
            unittest.mock.call('png', vis_dir / 'field_month_distribution.png', 'test_field month distribution'),
            unittest.mock.call('png', vis_dir / 'field_dow_distribution.png', 'test_field day of week distribution'),
            unittest.mock.call('png', vis_dir / 'field_age_distribution.png', 'Age distribution')
        ]
        
        result.add_artifact.assert_has_calls(expected_artifact_calls)
        
        # Fix for the reporter.add_artifact assertions
        expected_reporter_calls = [
            unittest.mock.call('png', str(vis_dir / 'field_year_distribution.png'), 'test_field year distribution'),
            unittest.mock.call('png', str(vis_dir / 'field_month_distribution.png'), 'test_field month distribution'),
            unittest.mock.call('png', str(vis_dir / 'field_dow_distribution.png'), 'test_field day of week distribution'),
            unittest.mock.call('png', str(vis_dir / 'field_age_distribution.png'), 'Age distribution')
        ]
        reporter.add_artifact.assert_has_calls(expected_reporter_calls)

    def test_save_anomalies_to_csv_creates_file_and_artifacts(self):
        result = OperationResult()
        reporter = MagicMock()
        tmp_dir = Path("test_tmp_dir")
        tmp_dir.mkdir(exist_ok=True)

        # Create fake anomalies data
        analysis_results = {
            'anomalies': {'too_old': 2},
            'too_old_examples': [
                (0, '1900-01-01', 1900),
                (1, '1899-12-31', 1899)
            ]
        }
        include_timestamp = False

        operation = DateOperation(field_name='birth_date')
        operation._save_anomalies_to_csv(
            analysis_results=analysis_results,
            dict_dir=tmp_dir,
            include_timestamp=include_timestamp,
            result=result,
            reporter=reporter
        )

        # Check that the CSV file was created
        csv_files = list(tmp_dir.glob("birth_date_anomalies*.csv"))
        self.assertTrue(len(csv_files) == 1)

        # Check the content of the file
        df = pd.read_csv(csv_files[0])
        self.assertEqual(len(df), 2)
        self.assertIn('anomaly_type', df.columns)
        self.assertEqual(df['anomaly_type'].iloc[0], 'too_old')

        # Check that artifact was added to result and reporter
        self.assertTrue(any(str(getattr(artifact, "path", "" )).endswith(".csv") for artifact in result.artifacts))
        reporter.add_artifact.assert_called()

        # Clean up temporary files
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()

    
    @patch('pandas.DataFrame.to_csv')
    def test_save_anomalies_to_csv_handles_empty_data(self, mock_to_csv):
        """Test handling of empty anomalies data"""
        # Setup
        analysis_results = {
            'anomalies': {
                'future_dates': 0,
                'too_old': 0
            }
        }
        
        dict_dir = Path('/tmp/test/dictionaries')
        include_timestamp = False
        result = MagicMock(spec=OperationResult)
        reporter = MagicMock()
        
        # Create DateOperation instance
        operation = DateOperation(field_name='test_date')
        
        # Execute
        operation._save_anomalies_to_csv(
            analysis_results,
            dict_dir,
            include_timestamp,
            result,
            reporter
        )
        
        # Assert
        mock_to_csv.assert_not_called()
        result.add_artifact.assert_not_called()
        reporter.add_artifact.assert_not_called()

    @patch('pandas.DataFrame.to_csv')
    def test_save_anomalies_to_csv_handles_exception(self, mock_to_csv):
        """Test handling of exceptions during CSV saving"""
        # Setup
        mock_to_csv.side_effect = Exception("CSV write error")
        
        analysis_results = {
            'anomalies': {
                'future_dates': 1
            },
            'future_dates_examples': [
                (0, '2025-01-01', 2025)
            ]
        }
        
        dict_dir = Path('/tmp/test/dictionaries')
        include_timestamp = False
        result = MagicMock(spec=OperationResult)
        reporter = MagicMock()
        
        # Create DateOperation instance
        operation = DateOperation(field_name='test_date')
        
        # Execute
        operation._save_anomalies_to_csv(
            analysis_results,
            dict_dir,
            include_timestamp,
            result,
            reporter
        )
        
        # Assert
        reporter.add_operation.assert_called_with(
            "Saving anomalies for test_date",
            status="warning",
            details={"warning": "CSV write error"}
        )
    def test_initialization_with_defaults(self):
        operation = DateOperation(field_name='created_at')
        self.assertEqual(operation.field_name, 'created_at')
        self.assertEqual(operation.min_year, 1940)
        self.assertEqual(operation.max_year, 2005)
        self.assertTrue(operation.generate_plots)
        self.assertTrue(operation.include_timestamp)
        self.assertEqual(operation.profile_type, 'date')
        self.assertFalse(operation.is_birth_date)

    def test_birth_date_detection_by_field_name(self):
        for field in ['birth_date', 'birthdate', 'birth_day', 'dob']:
            operation = DateOperation(field_name=field)
            self.assertTrue(operation.is_birth_date)

    def test_explicit_birth_date_override_true(self):
        operation = DateOperation(field_name='random_field', is_birth_date=True)
        self.assertTrue(operation.is_birth_date)

    def test_explicit_birth_date_override_false(self):
        operation = DateOperation(field_name='birth_date', is_birth_date=False)
        self.assertFalse(operation.is_birth_date)

    def test_custom_parameters(self):
        operation = DateOperation(
            field_name='custom_date',
            min_year=1900,
            max_year=2100,
            id_column='group_id',
            uid_column='user_id',
            description='Custom test',
            generate_plots=False,
            include_timestamp=False,
            profile_type='custom_type'
        )
        self.assertEqual(operation.min_year, 1900)
        self.assertEqual(operation.max_year, 2100)
        self.assertEqual(operation.id_column, 'group_id')
        self.assertEqual(operation.uid_column, 'user_id')
        self.assertEqual(operation.description, 'Custom test')
        self.assertFalse(operation.generate_plots)
        self.assertFalse(operation.include_timestamp)
        self.assertEqual(operation.profile_type, 'custom_type')

    def test_analyzer_instance_created(self):
        operation = DateOperation(field_name='some_date')
        self.assertIsInstance(operation.analyzer, DateAnalyzer)


class TestDateAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = DateAnalyzer()

    @patch("pamola_core.profiling.analyzers.date.analyze_date_field")
    def test_analyze_normal_case(self, mock_analyze_date_field):
        mock_analyze_date_field.return_value = {"summary": "ok"}

        df = pd.DataFrame({
            'created_at': ["2020-01-01", "2019-05-20", "2000-12-31"]
        })

        result = self.analyzer.analyze(df, "created_at", min_year=1990, max_year=2022)
        self.assertEqual(result, {"summary": "ok"})
        mock_analyze_date_field.assert_called_once()

    @patch("pamola_core.profiling.analyzers.date.analyze_date_field")
    def test_analyze_with_birth_date_flag(self, mock_analyze_date_field):
        today = datetime.now().date()
        birth_year = today.year - 30
        df = pd.DataFrame({
            'birth_date': [f"{birth_year}-01-01", f"{birth_year-1}-06-01", None]
        })

        mock_analyze_date_field.return_value = {"summary": "ok", "error": False}

        result = self.analyzer.analyze(df, "birth_date", is_birth_date=True)

        self.assertIn("age_distribution", result)
        self.assertIn("age_statistics", result)

    def test_calculate_age_distribution_valid(self):
        today = datetime.now().date()
        df = pd.DataFrame({
            "birth_date": [
                (today - timedelta(days=365 * 30)).isoformat(),
                (today - timedelta(days=365 * 45)).isoformat(),
                (today - timedelta(days=365 * 50)).isoformat(),
                None
            ]
        })

        result = self.analyzer._calculate_age_distribution(df, "birth_date")

        self.assertIn("age_distribution", result)
        self.assertIn("25-29", result["age_distribution"])
        self.assertIn("40-44", result["age_distribution"])
        self.assertIn("45-49", result["age_distribution"])

        stats = result["age_statistics"]
        self.assertIsNotNone(stats["min_age"])
        self.assertIsNotNone(stats["max_age"])
        self.assertIsNotNone(stats["mean_age"])
        self.assertIsNotNone(stats["median_age"])

    def test_calculate_age_distribution_all_invalid(self):
        df = pd.DataFrame({
            "birth_date": [None, "not-a-date", ""]
        })

        result = self.analyzer._calculate_age_distribution(df, "birth_date")

        self.assertEqual(result["age_distribution"], {})
        self.assertTrue(all(v is None for v in result["age_statistics"].values()))
        

    @patch("pamola_core.profiling.analyzers.date.estimate_resources")
    def test_estimate_resources(self, mock_estimate):
        mock_estimate.return_value = {"memory": "low", "time": "fast"}

        df = pd.DataFrame({"date": ["2020-01-01", "2021-01-01"]})
        result = self.analyzer.estimate_resources(df, "date")

        self.assertEqual(result, {"memory": "low", "time": "fast"})
        mock_estimate.assert_called_once()

    def test_calculate_age_distribution_with_attribute_error(self):
        df = pd.DataFrame({
            "birth_date": ["invalid-date"]
        })

        mock_date = MagicMock()
        del mock_date.date

        df["birth_date"] = [mock_date]

        result = self.analyzer._calculate_age_distribution(df, "birth_date")

        self.assertEqual(result["age_distribution"], {})
        self.assertTrue(all(v is None for v in result["age_statistics"].values()))

    def test_calculate_age_distribution_with_no_valid_dates(self):
        df = pd.DataFrame({
            "birth_date": ["not a date", "still not a date", None]
        })

        result = self.analyzer._calculate_age_distribution(df, "birth_date")

        self.assertEqual(result["age_distribution"], {})
        self.assertDictEqual(
            result["age_statistics"],
            {
                "min_age": None,
                "max_age": None,
                "mean_age": None,
                "median_age": None
            }
        )

    def test_calculate_age_distribution_with_future_dates(self):
        future_date = (datetime.now() + timedelta(days=365 * 10)).strftime("%Y-%m-%d")
        df = pd.DataFrame({
            "birth_date": [future_date, future_date]
        })

        result = self.analyzer._calculate_age_distribution(df, "birth_date")

        self.assertEqual(result["age_distribution"], {})
        self.assertDictEqual(
            result["age_statistics"],
            {
                "min_age": None,
                "max_age": None,
                "mean_age": None,
                "median_age": None
            }
        )
   
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_execute_handles_exception_and_returns_error(self, mock_load_data):
        # Arrange
        mock_load_data.side_effect = Exception("Simulated failure")
        field_name = "birth_date"
        operation = DateOperation(field_name=field_name)
        mock_reporter = MagicMock()
        mock_progress = MagicMock()

        # Act
        result = operation.execute(
            data_source=MagicMock(), 
            task_dir=Path("/tmp"), 
            reporter=mock_reporter, 
            progress_tracker=mock_progress
        )

        # Assert
        assert isinstance(result, OperationResult)
        assert result.status == OperationStatus.ERROR
        assert "Simulated failure" in result.error_message

        # Check that progress tracker was updated with error
        mock_progress.update.assert_called_with(0, {
            "step": "Error",
            "error": "Simulated failure"
        })

        # Check that reporter recorded the error
        mock_reporter.add_operation.assert_called_with(
            f"Error analyzing {field_name}",
            status="error",
            details={"error": "Simulated failure"}
        )

        
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_execute_handles_general_exception(self, mock_load_data):
        """Test that execute method properly handles any general exceptions"""
        # Setup
        field_name = "test_date"
        operation = DateOperation(field_name=field_name)
        mock_reporter = MagicMock()
        mock_progress_tracker = MagicMock()
        
        # Create a mock analyzer that raises an exception
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = Exception("Unexpected analysis error")
        operation.analyzer = mock_analyzer

        # Mock the DataFrame with valid data
        mock_df = pd.DataFrame({
            field_name: ["2023-01-01", "2023-02-01"]
        })

        # Execute with mocked data loading
        with patch("pamola_core.profiling.analyzers.date.load_data_operation", return_value=mock_df):
            result = operation.execute(
                data_source=MagicMock(),
                task_dir=Path("/tmp"),
                reporter=mock_reporter,
                progress_tracker=mock_progress_tracker
            )

            # Verify error result
            self.assertEqual(result.status, OperationStatus.ERROR)
            self.assertEqual(
                result.error_message,
                f"Error analyzing date field {field_name}: Unexpected analysis error"
            )

            # Verify progress tracker was updated
            mock_progress_tracker.update.assert_called_with(
                0, 
                {"step": "Error", "error": "Unexpected analysis error"}
            )

            # Verify reporter recorded the error
            mock_reporter.add_operation.assert_called_with(
                f"Error analyzing {field_name}",
                status="error",
                details={"error": "Unexpected analysis error"}
            )

class TestAnalyzeDateFields(unittest.TestCase):
    """Test cases for analyze_date_fields function"""
    
    def setUp(self):
        self.data_source = MagicMock()
        self.reporter = MagicMock()
        self.task_dir = Path("test_tmp_dir")
        self.task_dir.mkdir(exist_ok=True)

    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_analyze_date_fields_with_formatted_dates(self, mock_load_data):
        """Test with properly formatted dates to avoid parsing warnings"""
        # Setup test data with consistent date format
        mock_df = pd.DataFrame({
            'birth_date': pd.to_datetime(['1990-01-01', '1995-02-02']),
            'created_at': pd.to_datetime(['2023-01-01', '2023-02-02'])
        })
        mock_load_data.return_value = mock_df
        
        # Execute
        results = analyze_date_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            date_fields=['birth_date', 'created_at']
        )
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertIn('birth_date', results)
        self.assertIn('created_at', results)
        
        # Verify reporter calls
        expected_calls = [
            # Initial analysis call
            unittest.mock.call(
                "Date fields analysis",
                details={
                    'fields_count': 2,
                    'fields': ['birth_date', 'created_at'],
                    'id_column': None,
                    'uid_column': None,
                    'parameters': {}
                }
            ),
            # Birth date field analysis
            unittest.mock.call(
                "Analyzing date field: birth_date",
                details={
                    'field_name': 'birth_date',
                    'min_year': 1940,
                    'max_year': 2005,
                    'id_column': None,
                    'uid_column': None,
                    'is_birth_date': True,
                    'operation_type': 'date_analysis'
                }
            ),
            # Created at field analysis
            unittest.mock.call(
                "Analyzing date field: created_at",
                details={
                    'field_name': 'created_at',
                    'min_year': 1940,
                    'max_year': 2005,
                    'id_column': None,
                    'uid_column': None,
                    'is_birth_date': False,
                    'operation_type': 'date_analysis'
                }
            ),
            # Final completion call
            unittest.mock.call(
                "Date fields analysis completed",
                details={
                    'fields_analyzed': 2,
                    'successful': 0,
                    'failed': 2
                }
            )
        ]
        
        self.reporter.add_operation.assert_has_calls(expected_calls, any_order=True)

    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_analyze_date_fields_with_none_dataframe(self, mock_load_data):
        """Test analyze_date_fields when load_data_operation returns None"""
        # Setup
        mock_load_data.return_value = None
        
        # Execute
        results = analyze_date_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            date_fields=['birth_date', 'created_at']
        )
        
        # Assert
        # Verify empty results returned
        self.assertEqual(results, {})
        
        # Verify error reported
        self.reporter.add_operation.assert_called_once_with(
            "Date fields analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"}
        )

    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_analyze_date_fields_auto_detection(self, mock_load_data):
        """Test automatic detection of date fields when date_fields is None"""
        # Setup test data with various field names
        mock_df = pd.DataFrame({
            'birth_date': pd.to_datetime(['1990-01-01', '1995-02-02']),
            'create_date': pd.to_datetime(['2023-01-01', '2023-02-02']),
            'update_time': pd.to_datetime(['2023-03-01', '2023-04-02']),
            'normal_field': ['value1', 'value2'],
            'other_day': pd.to_datetime(['2023-05-01', '2023-06-02'])
        })
        mock_load_data.return_value = mock_df
        
        # Execute without specifying date_fields
        results = analyze_date_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            date_fields=None
        )
        
        # Assert
        expected_fields = {'update_time', 'create_date', 'other_day', 'birth_date'}
        self.assertEqual(set(results.keys()), expected_fields)
    
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_analyze_date_fields_with_error_and_tracker(self, mock_load_data):
        """Test error handling during field analysis with overall tracker"""
        # Setup
        mock_df = pd.DataFrame({
            'birth_date': pd.to_datetime(['1990-01-01', '1995-02-02']),
            'invalid_date': ['invalid', 'data']
        })
        mock_load_data.return_value = mock_df
        
        # Create mock overall tracker
        mock_overall_tracker = MagicMock()
        
        # Execute with overall tracker
        results = analyze_date_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            date_fields=['birth_date', 'invalid_date'],
            overall_tracker=mock_overall_tracker
        )
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertIn('birth_date', results)
        self.assertIn('invalid_date', results)

    @patch("pamola_core.profiling.analyzers.date.DateOperation.execute")
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    def test_overall_tracker_update_on_success(self, mock_load_data, mock_execute):
        # Setup DataFrame with a date field
        import pandas as pd
        df = pd.DataFrame({
            "birth_date": ["2000-01-01", "1990-05-15"]
        })
        mock_load_data.return_value = df

        # Mock OperationResult with SUCCESS
        mock_result = OperationResult(status=OperationStatus.SUCCESS)
        mock_execute.return_value = mock_result

        # Patch ProgressTracker to monitor update calls
        with patch("pamola_core.profiling.analyzers.date.ProgressTracker") as MockTracker:
            tracker_instance = MockTracker.return_value

            from pamola_core.profiling.analyzers.date import analyze_date_fields

            analyze_date_fields(
                data_source=self.data_source,
                task_dir=self.task_dir,
                reporter=self.reporter,
                date_fields=["birth_date"]
            )

            # Check that overall_tracker.update was called with status "completed"
            tracker_instance.update.assert_any_call(1, {"field": "birth_date", "status": "completed"})

    @patch("pamola_core.profiling.analyzers.date.DateOperation.execute")
    @patch("pamola_core.profiling.analyzers.date.ProgressTracker")
    @patch("pamola_core.profiling.analyzers.date.load_data_operation")
    @patch("pamola_core.profiling.analyzers.date.logger")
    def test_analyze_date_fields_exception_handling(self, mock_logger, mock_load_data, MockProgressTracker, mock_execute):
        # Setup DataFrame with a date field
        df = pd.DataFrame({"birth_date": ["2000-01-01", "1990-05-15"]})
        mock_load_data.return_value = df

        # Mock execute to raise exception
        mock_execute.side_effect = Exception("Test exception")
        tracker_instance = MockProgressTracker.return_value

        from pamola_core.profiling.analyzers.date import analyze_date_fields

        analyze_date_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            date_fields=["birth_date"]
        )

        # Check that logger.error was called
        mock_logger.error.assert_called()
        # Check that reporter.add_operation was called with status="error"
        self.reporter.add_operation.assert_any_call(
            "Analyzing birth_date field",
            status="error",
            details={"error": "Test exception"}
        )
        # Check that overall_tracker.update was called with status="error"
        tracker_instance.update.assert_any_call(1, {"field": "birth_date", "status": "error"})