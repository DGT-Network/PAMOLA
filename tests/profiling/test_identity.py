import unittest
import pandas as pd
from typing import Dict, Any
from pamola_core.profiling.analyzers.identity import IdentityAnalyzer, analyze_identities


from unittest.mock import MagicMock, Mock, patch
from pathlib import Path
from typing import Dict, List, Optional

from pamola_core.profiling.analyzers.identity import IdentityAnalysisOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.progress import ProgressTracker

class TestIdentityAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample test data
        self.test_data = {
            'uid': ['001', '001', '002', '003', '003', '004'],
            'resume_id': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
            'first_name': ['John', 'John', 'Alice', 'Bob', 'Bob', 'Charlie'],
            'last_name': ['Doe', 'Doe', 'Smith', 'Johnson', 'Johnson', 'Brown'],
            'email': ['john@test.com', 'john@test.com', 'alice@test.com', 
                     'bob@test.com', 'bob@test.com', 'charlie@test.com']
        }
        self.df = pd.DataFrame(self.test_data)

    def test_analyze_identifier_distribution(self):
        """Test analyze_identifier_distribution method"""
        result = IdentityAnalyzer.analyze_identifier_distribution(
            df=self.df,
            id_field='uid',
            entity_field='resume_id',
            top_n=5
        )

        # Assert the result is a dictionary
        self.assertIsInstance(result, Dict)
        
        # Check if distribution data exists
        self.assertIn('distribution', result)
        
        # Verify counts are correct
        distribution = result['distribution']
        self.assertEqual(distribution.get('001', 0), 0)
        self.assertEqual(distribution.get('003', 0), 0)
        
        # Verify statistics
        self.assertIn('max_count', result)
        self.assertIn('avg_count', result)
        self.assertEqual(result['max_count'], 2)
        self.assertAlmostEqual(result['avg_count'], 1.5, places=2)

    def test_analyze_identifier_consistency(self):
        """Test analyze_identifier_consistency method"""
        result = IdentityAnalyzer.analyze_identifier_consistency(
            df=self.df,
            id_field='uid',
            reference_fields=['first_name', 'last_name', 'email']
        )

        # Assert result structure
        self.assertIsInstance(result, Dict)
        self.assertIn('match_percentage', result)
        self.assertIn('mismatch_count', result)
        self.assertIn('total_records', result)
        
        # Verify consistency metrics
        self.assertEqual(result['total_records'], len(self.df))
        self.assertTrue(0 <= result['match_percentage'] <= 100)
        self.assertGreaterEqual(result['mismatch_count'], 0)

    def test_find_cross_matches(self):
        """Test find_cross_matches method"""
        result = IdentityAnalyzer.find_cross_matches(
            df=self.df,
            id_field='uid',
            reference_fields=['first_name', 'last_name'],
            min_similarity=0.8,
            fuzzy_matching=False
        )

        # Assert result structure
        self.assertIsInstance(result, Dict)
        self.assertIn('total_cross_matches', result)
        self.assertIn('cross_match_examples', result)
        
        # Verify cross-match detection
        self.assertGreaterEqual(result['total_cross_matches'], 0)
        if result['total_cross_matches'] > 0:
            self.assertGreater(len(result['cross_match_examples']), 0)

    def test_compute_identifier_stats(self):
        """Test compute_identifier_stats method"""
        result = IdentityAnalyzer.compute_identifier_stats(
            df=self.df,
            id_field='uid',
            entity_field='resume_id'
        )

        # Assert result structure
        self.assertIsInstance(result, Dict)
        self.assertIn('total_records', result)
        self.assertIn('unique_identifiers', result)
        self.assertIn('coverage_percentage', result)
        
        # Verify statistics
        self.assertEqual(result['total_records'], len(self.df))
        self.assertEqual(result['unique_identifiers'], len(self.df['uid'].unique()))
        self.assertTrue(0 <= result['coverage_percentage'] <= 100)

    def test_null_values(self):
        """Test handling of null values"""
        # Create DataFrame with null values
        df_with_nulls = self.df.copy()
        df_with_nulls.loc[0, 'uid'] = None
        df_with_nulls.loc[1, 'first_name'] = None

        # Test identifier stats with nulls
        result = IdentityAnalyzer.compute_identifier_stats(
            df_with_nulls, 'uid'
        )
        self.assertIn('null_identifiers', result)
        self.assertGreater(result['null_identifiers'], 0)

        # Test consistency analysis with nulls
        result = IdentityAnalyzer.analyze_identifier_consistency(
            df_with_nulls, 'uid', ['first_name', 'last_name']
        )

class TestIdentityAnalysisOperation(unittest.TestCase):
    def setUp(self):
        """Set up test data and mocks"""
        # Sample test data
        self.test_data = {
            'uid': ['001', '001', '002', '003', '003', '004'],
            'resume_id': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
            'first_name': ['John', 'John', 'Alice', 'Bob', 'Bob', 'Charlie'],
            'last_name': ['Doe', 'Doe', 'Smith', 'Johnson', 'Johnson', 'Brown']
        }
        self.df = pd.DataFrame(self.test_data)
        
        # Create mock objects
        self.mock_data_source = Mock(spec=DataSource)
        self.mock_data_source.return_value = self.df
        
        self.mock_reporter = Mock()
        self.mock_progress_tracker = Mock(spec=ProgressTracker)
        
        # Create temporary task directory
        self.task_dir = Path('test_task_dir')
        
        # Initialize operation
        self.operation = IdentityAnalysisOperation(
            uid_field='uid',
            reference_fields=['first_name', 'last_name'],
            id_field='resume_id'
        )

    def test_initialization(self):
        """Test operation initialization"""
        operation = IdentityAnalysisOperation(
            uid_field='test_uid',
            reference_fields=['field1', 'field2'],
            id_field='test_id',
            top_n=10,
            description='Test description'
        )
        
        self.assertEqual(operation.field_name, 'test_uid')
        self.assertEqual(operation.reference_fields, ['field1', 'field2'])
        self.assertEqual(operation.id_field, 'test_id')
        self.assertEqual(operation.top_n, 10)
        self.assertEqual(operation.description, 'Test description')

    def test_execute_success(self):
        """Test successful execution"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            mock_load.return_value = self.df
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                progress_tracker=self.mock_progress_tracker
            )
            
            self.assertEqual(result.status, OperationStatus.SUCCESS)
            self.assertIsNotNone(result.metrics)
            self.assertTrue(len(result.artifacts) > 0)

    def test_execute_missing_field(self):
        """Test execution with missing required field"""
        df_missing = self.df.drop('uid', axis=1)
        
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            mock_load.return_value = df_missing
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )
            
            self.assertEqual(result.status, OperationStatus.ERROR)
            self.assertIn('not found', result.error_message)

    def test_execute_missing_reference_fields(self):
        """Test execution with missing reference fields"""
        df_missing = self.df.drop(['first_name', 'last_name'], axis=1)
        
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            mock_load.return_value = df_missing
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )
            
            self.assertEqual(result.status, OperationStatus.ERROR)
            self.assertIn('reference fields', result.error_message)

    def test_execute_with_progress_tracking(self):
        """Test execution with progress tracking"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            mock_load.return_value = self.df
            
            self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                progress_tracker=self.mock_progress_tracker
            )
            
            # Verify progress tracker was updated
            self.mock_progress_tracker.update.assert_called()
            self.mock_progress_tracker.total = 4

    def test_prepare_directories(self):
        """Test directory preparation"""
        dirs = self.operation._prepare_directories(self.task_dir)
        
        self.assertIn('output', dirs)
        self.assertIn('visualizations', dirs)
        self.assertIn('dictionaries', dirs)
        
        self.assertTrue(dirs['output'].exists())
        self.assertTrue(dirs['visualizations'].exists())
        self.assertTrue(dirs['dictionaries'].exists())

    def test_execute_with_custom_parameters(self):
        """Test execution with custom parameters"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            mock_load.return_value = self.df
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                top_n=20,
                check_cross_matches=True,
                fuzzy_matching=True,
                min_similarity=0.9
            )
            
            self.assertEqual(result.status, OperationStatus.SUCCESS)

    def test_execute_none_dataframe(self):
        """Test execution when DataFrame is None"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            # Mock load_data_operation to return None
            mock_load.return_value = None
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )
            
            # Verify error status and message
            self.assertEqual(result.status, OperationStatus.ERROR)

    def test_execute_partial_missing_reference_fields(self):
        """Test execution with some missing reference fields"""
        # Create DataFrame with only one reference field
        df_partial = self.df.drop('last_name', axis=1)
        
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load, \
            patch('pamola_core.profiling.analyzers.identity.logger') as mock_logger:
            mock_load.return_value = df_partial
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )
            
            # Verify warnings were logged
            mock_logger.warning.assert_called_with(
                "Some reference fields are missing: {'last_name'}"
            )
            
            # Verify reporter was called with warning
            self.mock_reporter.add_operation.assert_any_call(
                f"Missing reference fields for {self.operation.field_name}",
                status="warning",
                details={"missing_fields": ['last_name']}
            )
            
            # Verify operation still succeeds with remaining field
            self.assertEqual(result.status, OperationStatus.SUCCESS)
    
    def test_execute_missing_id_field(self):
        """Test execution when ID field is missing"""
        
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load, \
            patch('pamola_core.profiling.analyzers.identity.logger') as mock_logger:
            mock_load.return_value = self.df

            operation_temp = IdentityAnalysisOperation(
                uid_field='uid',
                reference_fields=['first_name', 'last_name'],
                id_field='resu5me_id'
            )
            
            result = operation_temp.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )
            
            # Verify reporter was called with warning
            self.mock_reporter.add_operation.assert_any_call(
                f"Missing ID field {operation_temp.id_field}",
                status="warning",
                details={"missing_field": operation_temp.id_field}
            )
            
            # Verify operation still succeeds without ID field
            self.assertEqual(result.status, OperationStatus.SUCCESS)

    def test_execute_visualization_error(self):
        """Test handling of visualization creation error"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load, \
            patch('pamola_core.profiling.analyzers.identity.plot_value_distribution') as mock_plot, \
            patch('pamola_core.profiling.analyzers.identity.logger') as mock_logger:
            
            # Setup mock return values
            mock_load.return_value = self.df
            mock_plot.return_value = "Error: Failed to create visualization"
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )
            
            # Verify warning was logged
            mock_logger.warning.assert_called_with(
                "Error creating distribution visualization: Error: Failed to create visualization"
            )
            
            # Verify no visualization artifact was added
            for artifact in result.artifacts:
                self.assertNotIn('distribution visualization', artifact.description)
            
            # Verify operation still succeeds despite visualization error
            self.assertEqual(result.status, OperationStatus.SUCCESS)
            
            # Verify other analysis results are still present
            self.assertIsNotNone(result.metrics)
            self.assertTrue(any('distribution' in str(artifact.path) for artifact in result.artifacts))
    def test_execute_progress_tracking_with_cross_matches(self):
        """Test progress tracking during cross-match analysis"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load:
            mock_load.return_value = self.df
            
            # Execute with progress tracker and cross matches enabled
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                progress_tracker=self.mock_progress_tracker,
                check_cross_matches=True
            )
            
            # Verify progress tracker updates for cross-match step
            self.mock_progress_tracker.update.assert_any_call(
                1, {"step": "Cross-match analysis complete"}
            )
            
            # Verify total steps includes cross-match analysis
            self.assertEqual(self.mock_progress_tracker.total, 5)  # 4 default steps + 1 for cross-match
            
            # Verify operation completed successfully
            self.assertEqual(result.status, OperationStatus.SUCCESS)
    def test_execute_exception_handling(self):
        """Test exception handling in execute method"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load, \
             patch('pamola_core.profiling.analyzers.identity.logger') as mock_logger:
            mock_load.side_effect = Exception("Simulated error")
            
            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter,
                progress_tracker=self.mock_progress_tracker
            )
            
            # Logger should log the exception
            mock_logger.exception.assert_called_with(
                f"Error in identity analysis operation for {self.operation.field_name}: Simulated error"
            )
            # Progress tracker should be updated with error
            self.mock_progress_tracker.update.assert_called_with(
                0, {"step": "Error", "error": "Simulated error"}
            )
            # Reporter should record the error
            self.mock_reporter.add_operation.assert_called_with(
                f"Error analyzing {self.operation.field_name}",
                status="error",
                details={"error": "Simulated error"}
            )
            # Result should be error status and contain error message
            self.assertEqual(result.status, OperationStatus.ERROR)
            self.assertIn("Error analyzing identity field", result.error_message)
            
    def test_execute_with_mismatch_examples_artifact(self):
        """Test artifact creation when mismatch_examples exist in consistency_analysis"""
        with patch('pamola_core.profiling.analyzers.identity.load_data_operation') as mock_load, \
             patch('pamola_core.profiling.analyzers.identity.IdentityAnalyzer.analyze_identifier_consistency') as mock_consistency, \
             patch('pamola_core.profiling.analyzers.identity.get_timestamped_filename') as mock_get_filename, \
             patch('pamola_core.profiling.analyzers.identity.write_json') as mock_write_json:
            
            mock_load.return_value = self.df
            mock_consistency.return_value = {
                'match_percentage': 80,
                'mismatch_count': 2,
                'total_records': 6,
                'mismatch_examples': [{'row': 1, 'reason': 'test'}]
            }
            mock_get_filename.return_value = "uid_mismatch_examples.json"

            result = self.operation.execute(
                data_source=self.mock_data_source,
                task_dir=self.task_dir,
                reporter=self.mock_reporter
            )

            found = False
            for call_args in mock_write_json.call_args_list:
                args, kwargs = call_args
                if (
                    isinstance(args[0], dict)
                    and 'mismatch_examples' in args[0]
                    and args[0].get('mismatch_count', None) == 2
                    and args[0].get('total_records', None) == 6
                    and str(args[1]).endswith("uid_mismatch_examples.json")
                ):
                    found = True
                    break
            self.assertTrue(found, "write_json was not called with expected mismatch_examples artifact")

            self.assertTrue(any("mismatch examples" in artifact.description for artifact in result.artifacts))
            self.mock_reporter.add_artifact.assert_any_call(
                "json",
                str(self.task_dir / "output" / "uid_mismatch_examples.json"),
                "uid mismatch examples"
            ) 
            
    def tearDown(self):
        """Clean up after tests"""
        # Remove test directories if they exist
        if self.task_dir.exists():
            import shutil
            shutil.rmtree(self.task_dir)

class DummyOperationResult:
    def __init__(self, status="success", error_message=None):
        self.status = status
        self.error_message = error_message

class TestAnalyzeIdentities(unittest.TestCase):

    @patch("pamola_core.profiling.analyzers.identity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.identity.IdentityAnalysisOperation")
    @patch("pamola_core.profiling.analyzers.identity.ProgressTracker")
    def test_analyze_identities_success(self, mock_tracker_cls, mock_operation_cls, mock_load_data):
        # Mock DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "user_id": [1, 2],
            "name": ["Alice", "Bob"],
            "birth_date": ["1990-01-01", "1992-02-02"]
        })
        mock_load_data.return_value = df

        # Mock operation result
        mock_result = DummyOperationResult(status="success")
        mock_operation = MagicMock()
        mock_operation.execute.return_value = mock_result
        mock_operation_cls.return_value = mock_operation

        # Mock reporter
        reporter = MagicMock()

        # Identity fields
        identity_fields = {
            "user_id": {
                "reference_fields": ["name", "birth_date"],
                "id_field": None
            }
        }

        # Call function
        from pamola_core.profiling.analyzers.identity import analyze_identities
        result = analyze_identities(
            data_source=MagicMock(),
            task_dir=Path("/tmp"),
            reporter=reporter,
            identity_fields=identity_fields
        )

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertIn("user_id", result)
        self.assertEqual(result["user_id"].status, "success")
        mock_operation_cls.assert_called_once_with(
            "user_id",
            reference_fields=["name", "birth_date"],
            id_field=None
        )
        mock_operation.execute.assert_called_once()

    @patch("pamola_core.profiling.analyzers.identity.load_data_operation")
    def test_analyze_identities_no_dataframe(self, mock_load_data):
        # Mock no DataFrame
        mock_load_data.return_value = None
        reporter = MagicMock()

        from pamola_core.profiling.analyzers.identity import analyze_identities
        result = analyze_identities(
            data_source=MagicMock(),
            task_dir=Path("/tmp"),
            reporter=reporter,
            identity_fields={
                "user_id": {"reference_fields": ["name"]}
            }
        )

        self.assertEqual(result, {})
        reporter.add_operation.assert_called_with(
            "Identity fields analysis", status="error",
            details={"error": "No valid DataFrame found in data source"}
        )

    @patch("pamola_core.profiling.analyzers.identity.load_data_operation")
    def test_analyze_identities_autodetect_fields(self, mock_load_data):
        # Mock DataFrame with id and reference fields
        import pandas as pd
        df = pd.DataFrame({
            "user_id": [1, 2],
            "uuid": ["a", "b"],
            "name": ["Alice", "Bob"],
            "birth_date": ["1990-01-01", "1992-02-02"]
        })
        mock_load_data.return_value = df
        reporter = MagicMock()

        # Mock IdentityAnalysisOperation
        with patch("pamola_core.profiling.analyzers.identity.IdentityAnalysisOperation") as mock_operation_cls:
            mock_result = DummyOperationResult(status="success")
            mock_operation = MagicMock()
            mock_operation.execute.return_value = mock_result
            mock_operation_cls.return_value = mock_operation

            from pamola_core.profiling.analyzers.identity import analyze_identities
            result = analyze_identities(
                data_source=MagicMock(),
                task_dir=Path("/tmp"),
                reporter=reporter,
                identity_fields=None
            )

            self.assertTrue(len(result) >= 1)
            self.assertTrue(all(isinstance(r, DummyOperationResult) for r in result.values()))
    
    @patch("pamola_core.profiling.analyzers.identity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.identity.IdentityAnalysisOperation")
    @patch("pamola_core.profiling.analyzers.identity.ProgressTracker")
    def test_overall_tracker_success_update(self, mock_tracker_cls, mock_operation_cls, mock_load_data):
        # Prepare DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "person_id": [1, 2],
            "name": ["Alice", "Bob"],
            "birth_date": ["1990-01-01", "1992-02-02"]
        })
        mock_load_data.return_value = df

        mock_result = MagicMock()
        mock_result.status = OperationStatus.SUCCESS
        mock_result.error_message = None

        mock_operation = MagicMock()
        mock_operation.execute.return_value = mock_result
        mock_operation_cls.return_value = mock_operation

        # Setup mock ProgressTracker
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        reporter = MagicMock()
        identity_fields = {
            "person_id": {
                "reference_fields": ["name", "birth_date"],
                "id_field": None
            }
        }

        from pamola_core.profiling.analyzers.identity import analyze_identities
        result = analyze_identities(
            data_source=MagicMock(),
            task_dir=Path("/tmp"),
            reporter=reporter,
            identity_fields=identity_fields
        )

        mock_tracker.update.assert_any_call(1, {"field": "person_id", "status": "completed"})

    @patch("pamola_core.profiling.analyzers.identity.load_data_operation")
    @patch("pamola_core.profiling.analyzers.identity.IdentityAnalysisOperation")
    @patch("pamola_core.profiling.analyzers.identity.ProgressTracker")
    def test_exception_during_execute(self, mock_tracker_cls, mock_operation_cls, mock_load_data):
        # 1) Setup DataFrame to return
        import pandas as pd
        df = pd.DataFrame({
            "cust_id": [1],
            "name": ["Alice"],
        })
        mock_load_data.return_value = df

        # 2) Mock IdentityAnalysisOperation.execute() to raise exception
        mock_operation = MagicMock()
        mock_operation.execute.side_effect = ValueError("test-failure")
        mock_operation_cls.return_value = mock_operation

        # 3) Mock ProgressTracker
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        # 4) Reporter spy
        reporter = MagicMock()

        identity_fields = {
            "cust_id": {
                "reference_fields": ["name"],
                "id_field": None
            }
        }

        try:
            result = analyze_identities(
                data_source=MagicMock(),
                task_dir=Path("/tmp"),
                reporter=reporter,
                identity_fields=identity_fields
            )
        except Exception:
            result = {}

        # 6) The result should not contain cust_id because it failed
        self.assertNotIn("cust_id", result)

        # 7) reporter.add_operation should be called with the correct error info
        reporter.add_operation.assert_any_call(
            "Analyzing cust_id field",
            status="error",
            details={"error": "test-failure"}
        )

        # 8) overall_tracker.update should be called to mark the error
        mock_tracker.update.assert_any_call(1, {"field": "cust_id", "status": "error"})


if __name__ == "__main__":
    unittest.main()