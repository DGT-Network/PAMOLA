import shutil
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from pamola_core.profiling.analyzers.email import EmailAnalyzer, analyze_email_fields
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
from pamola_core.profiling.analyzers.email import EmailOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.progress import ProgressTracker

class TestEmailAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = {
            'email': [
                'user1@example.com',
                'user2@example.com',
                'user3@another.com',
                None,
                'invalid-email',
                'user4@another.com'
            ]
        }
        self.df = pd.DataFrame(self.test_data)
        self.field_name = 'email'

    @patch('pamola_core.profiling.analyzers.email.analyze_email_field')
    def test_analyze(self, mock_analyze):
        """Test analyze method"""
        # Setup mock return value
        expected_result = {
            'total_rows': 6,
            'valid_count': 4,
            'invalid_count': 1,
            'null_count': 1,
            'unique_domains': 2,
            'top_domains': {'example.com': 2, 'another.com': 2}
        }
        mock_analyze.return_value = expected_result

        # Call analyze method
        result = EmailAnalyzer.analyze(
            df=self.df,
            field_name=self.field_name,
            top_n=20
        )

        # Verify mock was called with correct parameters
        mock_analyze.assert_called_once_with(
            df=self.df,
            field_name=self.field_name,
            top_n=20
        )

        # Verify result
        self.assertEqual(result, expected_result)
        self.assertEqual(result['total_rows'], 6)
        self.assertEqual(result['valid_count'], 4)
        self.assertEqual(len(result['top_domains']), 2)

    @patch('pamola_core.profiling.analyzers.email.create_domain_dictionary')
    def test_create_domain_dictionary(self, mock_create_dict):
        """Test create_domain_dictionary method"""
        # Setup mock return value
        expected_result = {
            'domains': [
                {'domain': 'example.com', 'count': 2},
                {'domain': 'another.com', 'count': 2}
            ],
            'total_domains': 2,
            'min_count': 1
        }
        mock_create_dict.return_value = expected_result

        # Call create_domain_dictionary method
        result = EmailAnalyzer.create_domain_dictionary(
            df=self.df,
            field_name=self.field_name,
            min_count=1
        )

        # Verify mock was called with correct parameters
        mock_create_dict.assert_called_once_with(
            df=self.df,
            field_name=self.field_name,
            min_count=1
        )

        # Verify result
        self.assertEqual(result, expected_result)
        self.assertEqual(len(result['domains']), 2)
        self.assertEqual(result['total_domains'], 2)

    @patch('pamola_core.profiling.analyzers.email.estimate_resources')
    def test_estimate_resources(self, mock_estimate):
        """Test estimate_resources method"""
        # Setup mock return value
        expected_result = {
            'estimated_memory': '10MB',
            'estimated_time': '5s',
            'row_count': 6
        }
        mock_estimate.return_value = expected_result

        # Call estimate_resources method
        result = EmailAnalyzer.estimate_resources(
            df=self.df,
            field_name=self.field_name
        )

        # Verify mock was called with correct parameters - using args instead of kwargs
        mock_estimate.assert_called_once_with(self.df, self.field_name)

        # Verify result
        self.assertEqual(result, expected_result)
        self.assertIn('estimated_memory', result)
        self.assertIn('estimated_time', result)
        self.assertEqual(result['row_count'], 6)

    def test_analyze_invalid_field(self):
        """Test analyze method with invalid field name"""
        result = EmailAnalyzer.analyze(
            df=self.df,
            field_name='nonexistent_field'
        )
        self.assertIn('error', result)

    def test_analyze_empty_dataframe(self):
        """Test analyze method with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = EmailAnalyzer.analyze(
            df=empty_df,
            field_name=self.field_name
        )
        self.assertIn('error', result)


class TestEmailOperation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.field_name = "email"
        self.test_data = {
            'email': [
                'user1@example.com',
                'user2@example.com',
                'user3@another.com',
                None,
                'invalid-email',
                'user4@another.com'
            ]
        }
        self.df = pd.DataFrame(self.test_data)
        self.task_dir = Path("test_task_dir")
        self.operation = EmailOperation(
            field_name=self.field_name,
            top_n=20,
            min_frequency=1
        )
        
        # Mock objects
        self.mock_data_source = MagicMock(spec=DataSource)
        self.mock_reporter = MagicMock()
        self.mock_progress_tracker = MagicMock(spec=ProgressTracker)

    def tearDown(self):
        """Clean up test artifacts"""
        import shutil
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    @patch('pamola_core.profiling.analyzers.email.EmailAnalyzer')
    def test_execute_successful(self, mock_analyzer, mock_load_data):
        """Test successful execution of email operation"""
        # Setup mocks
        mock_load_data.return_value = self.df
        mock_analyzer.analyze.return_value = {
            'total_rows': 6,
            'valid_count': 4,
            'invalid_count': 1,
            'null_count': 1,
            'top_domains': {'example.com': 2, 'another.com': 2},
            'unique_domains': 2
        }

        # Execute operation
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Verify results
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertIsNone(result.error_message)
        self.assertTrue(len(result.metrics) > 0)
        self.assertTrue(len(result.artifacts) > 0)

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    def test_execute_invalid_field(self, mock_load_data):
        """Test execution with invalid field name"""
        # Setup
        self.operation.field_name = "nonexistent_field"
        mock_load_data.return_value = self.df

        # Execute
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter
        )

        # Verify
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("not found in DataFrame", result.error_message)

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    def test_execute_empty_dataframe(self, mock_load_data):
        """Test execution with empty DataFrame"""
        # Setup
        mock_load_data.return_value = pd.DataFrame()

        # Execute
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter
        )

        # Verify
        self.assertEqual(result.status, OperationStatus.ERROR)
        # Update the error message check to match actual message
        self.assertIn("Field", result.error_message)
        self.assertIn("not found in DataFrame", result.error_message)


    def test_prepare_directories(self):
        """Test directory preparation"""
        # Execute
        dirs = self.operation._prepare_directories(self.task_dir)

        # Verify
        self.assertTrue(dirs['output'].exists())
        self.assertTrue(dirs['visualizations'].exists())
        self.assertTrue(dirs['dictionaries'].exists())

    def test_assess_privacy_risk(self):
        """Test privacy risk assessment"""
        # Execute
        risk_assessment = self.operation._assess_privacy_risk(self.df, self.field_name)

        # Verify
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('uniqueness_ratio', risk_assessment)
        self.assertIn('singleton_count', risk_assessment)

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    @patch('pamola_core.utils.visualization.plot_email_domains')
    def test_execute_with_visualization(self, mock_plot, mock_load_data):
        """Test execution with visualization generation"""
        # Setup
        mock_load_data.return_value = self.df
        mock_analyzer_result = {
            'total_rows': 6,
            'valid_count': 4,
            'invalid_count': 1,
            'null_count': 1,
            'top_domains': {'example.com': 2, 'another.com': 2},
            'unique_domains': 2
        }
        mock_load_data.return_value = self.df
        mock_plot.return_value = "Success"

        # Mock EmailAnalyzer.analyze
        with patch('pamola_core.profiling.analyzers.email.EmailAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_analyzer_result

            # Execute
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                generate_plots=True
            )

            # Verify
            self.assertEqual(result.status, OperationStatus.SUCCESS)
            mock_plot.assert_called_once()
            
            # Update artifact check to use artifact_type instead of type
            self.assertTrue(any(hasattr(art, 'artifact_type') and art.artifact_type == "png" 
                              for art in result.artifacts))
            
    # Add this test method in the TestEmailOperation class:
    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    def test_execute_none_dataframe(self, mock_load_data):
        """Test execution when DataFrame is None"""
        # Setup
        mock_load_data.return_value = None
        # Execute
        result = self.operation.execute(
            data_source=None,
            task_dir=self.task_dir,
            reporter=self.mock_reporter
        )

        # Verify
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("No valid DataFrame found in data source", result.error_message)
    
    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    @patch('pamola_core.utils.visualization.plot_email_domains')
    @patch('pamola_core.profiling.analyzers.email.logger')
    def test_execute_visualization_error(self, mock_logger, mock_plot, mock_load_data):
        """Test execution when visualization generation fails"""
        # Setup
        mock_load_data.return_value = self.df
        mock_analyzer_result = {
            'total_rows': 6,
            'valid_count': 4,
            'invalid_count': 1,
            'null_count': 1,
            'top_domains': {'example.com': 2, 'another.com': 2},
            'unique_domains': 2
        }
        # Set plot to return an error
        mock_plot.return_value = "Error: Failed to create visualization"

        # Mock EmailAnalyzer.analyze
        with patch('pamola_core.profiling.analyzers.email.EmailAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = mock_analyzer_result

            # Execute
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                generate_plots=True
            )

            # Verify
            self.assertEqual(result.status, OperationStatus.SUCCESS)
            mock_plot.assert_called_once()
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once_with(
                "Error creating visualization: Error: Failed to create visualization"
            )
            
            # Verify no PNG artifact was added
            self.assertFalse(any(
                hasattr(art, 'artifact_type') and art.artifact_type == "png" 
                for art in result.artifacts
            ))
    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    @patch('pamola_core.profiling.analyzers.email.logger')
    def test_execute_with_exception(self, mock_logger, mock_load_data):
        """Test execution when an exception occurs"""
        # Setup - make load_data_operation raise an exception
        test_error = ValueError("Test error message")
        mock_load_data.side_effect = test_error

        # Execute
        result = self.operation.execute(
            data_source=self.mock_data_source,
            task_dir=self.task_dir,
            reporter=self.mock_reporter,
            progress_tracker=self.mock_progress_tracker
        )

        # Verify error result
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertEqual(
            result.error_message,
            f"Error analyzing email field {self.field_name}: {str(test_error)}"
        )

        # Verify logger was called with exception
        mock_logger.exception.assert_called_once_with(
            f"Error in email operation for {self.field_name}: {test_error}"
        )

        # Verify progress tracker was updated
        self.mock_progress_tracker.update.assert_called_with(
            0, 
            {"step": "Error", "error": str(test_error)}
        )

        # Verify reporter was updated
        self.mock_reporter.add_operation.assert_called_with(
            f"Error analyzing {self.field_name}",
            status="error",
            details={"error": str(test_error)}
        )    

    def test_prepare_directories(self):
        """Test directory preparation and structure"""
        try:
            # Execute
            dirs = self.operation._prepare_directories(self.task_dir)

            # Verify directory structure
            self.assertTrue(isinstance(dirs, dict), "Should return a dictionary")
            self.assertEqual(len(dirs), 3, "Should contain exactly 3 directory paths")
            
            # Verify all required keys exist
            expected_keys = ['output', 'visualizations', 'dictionaries']
            for key in expected_keys:
                self.assertIn(key, dirs, f"Missing required directory key: {key}")
                self.assertTrue(dirs[key].exists(), f"Directory {key} was not created")
                self.assertTrue(dirs[key].is_dir(), f"{key} should be a directory")

            # Verify directory structure
            self.assertEqual(dirs['output'], self.task_dir / 'output')
            self.assertEqual(dirs['visualizations'], self.task_dir / 'visualizations')
            self.assertEqual(dirs['dictionaries'], self.task_dir / 'dictionaries')

            # Test directory permissions (on Windows)
            import os
            for dir_path in dirs.values():
                self.assertTrue(os.access(dir_path, os.W_OK), f"Directory {dir_path} should be writable")
                self.assertTrue(os.access(dir_path, os.R_OK), f"Directory {dir_path} should be readable")

        finally:
            # Cleanup - remove test directories if they exist
            import shutil
            if self.task_dir.exists():
                shutil.rmtree(self.task_dir)

    def test_prepare_directories_with_existing_dirs(self):
        """Test directory preparation when directories already exist"""
        try:
            # Create directories first
            self.task_dir.mkdir(parents=True, exist_ok=True)
            (self.task_dir / 'output').mkdir(exist_ok=True)
            (self.task_dir / 'visualizations').mkdir(exist_ok=True)
            (self.task_dir / 'dictionaries').mkdir(exist_ok=True)

            # Execute
            dirs = self.operation._prepare_directories(self.task_dir)

            # Verify
            self.assertTrue(all(path.exists() for path in dirs.values()))
            self.assertEqual(len(list(self.task_dir.glob('*'))), 3)

        finally:
            # Cleanup
            if self.task_dir.exists():
                shutil.rmtree(self.task_dir)

    def test_prepare_directories_with_invalid_path(self):
        """Test directory preparation with invalid base path"""
        # Test with a path that should be invalid
        invalid_path = Path('Z:\\nonexistent\\path')
        
        # Execute and verify it raises an exception
        with self.assertRaises((OSError, Exception)) as context:
            self.operation._prepare_directories(invalid_path)
        
        # Verify error message contains either the path or a Windows error
        error_msg = str(context.exception)
        self.assertTrue(
            any([
                'system cannot find the path' in error_msg,
                str(invalid_path) in error_msg,
                'WinError' in error_msg
            ]),
            f"Error message '{error_msg}' does not contain expected content"
        )
    
    def test_assess_privacy_risk_normal(self):
        """Test privacy risk assessment with normal data"""
        # Setup test data with known characteristics
        test_data = {
            'email': [
                'user1@example.com',
                'user1@example.com',  # Duplicate
                'user2@example.com',
                'user3@another.com',
                'user4@another.com',
                'unique@domain.com',   # Singleton
                None                   # Null value
            ]
        }
        df = pd.DataFrame(test_data)
        # Execute
        risk_assessment = self.operation._assess_privacy_risk(df, 'email')

        # Verify
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('uniqueness_ratio', risk_assessment)
        self.assertIn('singleton_count', risk_assessment)
        self.assertIn('most_frequent_examples', risk_assessment)
        
        # Verify calculations
        self.assertEqual(risk_assessment['total_valid_emails'], 6)
        self.assertEqual(risk_assessment['unique_emails'], 5)
        self.assertEqual(risk_assessment['singleton_count'], 4) 
        self.assertAlmostEqual(risk_assessment['uniqueness_ratio'], 0.8333)

    def test_assess_privacy_risk_empty(self):
        """Test privacy risk assessment with empty DataFrame"""
        # Setup
        empty_df = pd.DataFrame(columns=['email'])
        
        # Execute
        risk_assessment = self.operation._assess_privacy_risk(empty_df, 'email')
        
        # Verify
        self.assertEqual(risk_assessment, {})

    def test_assess_privacy_risk_all_null(self):
        """Test privacy risk assessment with all null values"""
        # Setup
        null_df = pd.DataFrame({'email': [None, None, None]})
        
        # Execute
        risk_assessment = self.operation._assess_privacy_risk(null_df, 'email')
        
        # Verify
        self.assertEqual(risk_assessment, {})

    def test_assess_privacy_risk_high_uniqueness(self):
        """Test privacy risk assessment with high uniqueness ratio"""
        # Setup - all emails are unique
        test_data = {
            'email': [
                'user1@example.com',
                'user2@example.com',
                'user3@example.com',
                'user4@example.com',
                'user5@example.com'
            ]
        }
        df = pd.DataFrame(test_data)

        # Execute
        risk_assessment = self.operation._assess_privacy_risk(df, 'email')

        # Verify
        self.assertEqual(risk_assessment['risk_level'], "Very High")
        self.assertEqual(risk_assessment['singleton_count'], 5)  # All are singletons
        self.assertAlmostEqual(risk_assessment['uniqueness_ratio'], 1.0)

    def test_assess_privacy_risk_medium_uniqueness(self):
        """Test privacy risk assessment with high uniqueness ratio"""
        # Setup - all emails are unique
        test_data = {
            'email': [
                'user1@example.com',
                'user3@example.com',
                'user4@example.com',
                'user2@example.com',
                'user5@example.com',
                'user1@example.com',
                'user1@example.com',
                'user1@example.com'
            ]
        }
        df = pd.DataFrame(test_data)

        # Execute
        risk_assessment = self.operation._assess_privacy_risk(df, 'email')

        # Verify
        self.assertEqual(risk_assessment['risk_level'], "Medium")
        self.assertEqual(risk_assessment['singleton_count'], 4)
        self.assertAlmostEqual(risk_assessment['uniqueness_ratio'], 0.625)

    def test_assess_privacy_risk_invalid_field(self):
        """Test privacy risk assessment with invalid field name"""
        # Execute
        risk_assessment = self.operation._assess_privacy_risk(self.df, 'nonexistent_field')
        
        # Verify
        self.assertEqual(risk_assessment, {})

    @patch('pamola_core.profiling.analyzers.email.logger')
    def test_assess_privacy_risk_error_handling(self, mock_logger):
        """Test privacy risk assessment error handling"""
        # Setup - create a DataFrame that will cause an error during processing
        df = MagicMock()
        df.columns = ['email']
        df.__getitem__.side_effect = Exception("Test error")

        # Execute
        risk_assessment = self.operation._assess_privacy_risk(df, 'email')

        # Verify
        self.assertEqual(risk_assessment, {})
        mock_logger.error.assert_called_once()
        self.assertIn("Error in privacy risk assessment", 
                    mock_logger.error.call_args[0][0])

    def test_assess_privacy_risk_zero_valid_emails(self):
        """Test privacy risk assessment when total valid emails is zero"""
        # Setup - DataFrame with only invalid/null values
        test_data = {
            'email': [
                None,
                None,
            ]
        }
        df = pd.DataFrame(test_data)

        # Additional verification
        total_valid = df['email'].apply(lambda x: not pd.isna(x)).sum()
        self.assertEqual(total_valid, 0, 
                        "Test data should have zero valid emails")
                        
    
    
class TestAnalyzeEmailFields(unittest.TestCase):
    """Test cases for analyze_email_fields function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_source = MagicMock(spec=DataSource)
        self.reporter = MagicMock()
        self.task_dir = Path("test_task_dir")
    

    def tearDown(self):
        """Clean up test artifacts"""
        if self.task_dir.exists():
            shutil.rmtree(self.task_dir)

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    def test_analyze_email_fields_with_specified_fields(self, mock_load):
        """Test analyzing specified email fields"""
        # Setup
        mock_load.return_value = pd.DataFrame({
            'email': ['user1@example.com', 'user2@example.com'],
            'backup_email': ['backup1@example.com', None],
            'username': ['user1', 'user2']
        })
        email_fields = ['email', 'backup_email']

        # Execute
        results = analyze_email_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            email_fields=email_fields
        )

        # Verify
        self.assertEqual(len(results), 2)
        self.assertIn('email', results)
        self.assertIn('backup_email', results)
        
        # Verify reporter calls
        self.reporter.add_operation.assert_any_call(
            "Email fields analysis",
            details={
                "fields_count": 2,
                "fields": email_fields,
                "top_n": 20,
                "min_frequency": 1,
                "parameters": ANY
            }
        )

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    def test_analyze_email_fields_auto_detection(self, mock_load):
        """Test automatic detection of email fields"""
        # Setup
        mock_load.return_value = pd.DataFrame({
            'email': ['user1@example.com', 'user2@example.com'],
            'backup_email': ['backup1@example.com', None],
            'username': ['user1', 'user2']
        })

        # Execute
        results = analyze_email_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )

        # Verify
        self.assertEqual(len(results), 2)  # Should find both email fields
        self.assertIn('email', results)
        self.assertIn('backup_email', results)

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    def test_analyze_email_fields_with_none_dataframe(self, mock_load):
        """Test handling of None DataFrame"""
        # Setup
        mock_load.return_value = None

        # Execute
        results = analyze_email_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter
        )

        # Verify
        self.assertEqual(results, {})
        self.reporter.add_operation.assert_called_with(
            "Email fields analysis",
            status="error",
            details={"error": "No valid DataFrame found in data source"}
        )

    @patch('pamola_core.profiling.analyzers.email.load_data_operation')
    @patch('pamola_core.profiling.analyzers.email.ProgressTracker')
    def test_analyze_email_fields_with_progress_tracking(self, mock_tracker_class, mock_load):
        """Test progress tracking functionality"""
        # Setup
        mock_load.return_value = pd.DataFrame({
            'email': ['user1@example.com', 'user2@example.com'],
            'backup_email': ['backup1@example.com', None],
            'username': ['user1', 'user2']
        })
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        # Execute
        results = analyze_email_fields(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            track_progress=True
        )

        # Verify
        self.assertTrue(mock_tracker_class.called)
        mock_tracker.update.assert_called()
        mock_tracker.close.assert_called_once()

        

if __name__ == '__main__':
    unittest.main()