import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
from pathlib import Path

from pamola_core.profiling.analyzers import mvf
from pamola_core.utils.ops.op_result import OperationResult

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
    
class TestMVFAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'mvf_field': ["['A','B']", "['B','C']", "['A']", None, "[]"]
        })
        self.data_source = DummyDataSource(df=self.df)
        

    @patch('pamola_core.profiling.analyzers.mvf.analyze_mvf_fields')
    def test_analyze(self, mock_analyze):
        mock_analyze.return_value = {'result': 1}
        logger = Mock()
        result = mvf.MVFAnalyzer.analyze(self.df, 'mvf_field', task_logger=logger)
        self.assertIsNotNone(result)

    @patch('pamola_core.profiling.analyzers.mvf.parse_mvf')
    def test_parse_field(self, mock_parse):
        mock_parse.side_effect = lambda x, **kwargs: ['A', 'B'] if x else []
        df2 = mvf.MVFAnalyzer.parse_field(self.df, 'mvf_field', format_type='array')
        self.assertIn('parsed_mvf_field', df2.columns)
        self.assertTrue(all(isinstance(x, list) for x in df2['parsed_mvf_field']))

    @patch('pamola_core.profiling.analyzers.mvf.create_value_dictionary')
    def test_create_value_dictionary(self, mock_dict):
        mock_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        result = mvf.MVFAnalyzer.create_value_dictionary(self.df, 'mvf_field')
        self.assertIn('value', result.columns)
        mock_dict.assert_called_once()

    @patch('pamola_core.profiling.analyzers.mvf.create_combinations_dictionary')
    def test_create_combinations_dictionary(self, mock_dict):
        mock_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        result = mvf.MVFAnalyzer.create_combinations_dictionary(self.df, 'mvf_field')
        self.assertIn('combination', result.columns)
        mock_dict.assert_called_once()

    @patch('pamola_core.profiling.analyzers.mvf.analyze_value_count_distribution')
    def test_analyze_value_counts(self, mock_dist):
        mock_dist.return_value = {'1': 2, '2': 1}
        result = mvf.MVFAnalyzer.analyze_value_counts(self.df, 'mvf_field')
        self.assertEqual(result, {'1': 2, '2': 1})
        mock_dist.assert_called_once()

    @patch('pamola_core.profiling.analyzers.mvf.estimate_resources')
    def test_estimate_resources(self, mock_est):
        mock_est.return_value = {'memory': 100}
        result = mvf.MVFAnalyzer.estimate_resources(self.df, 'mvf_field')
        self.assertEqual(result, {'memory': 100})
        mock_est.assert_called_once()

    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_parse_field_field_not_found(self, mock_logger):
        df = pd.DataFrame({'other_field': [1, 2]})
        result = mvf.MVFAnalyzer.parse_field(df, 'mvf_field')
        self.assertTrue(result.equals(df))
        mock_logger.error.assert_called_once_with('Field mvf_field not found in DataFrame')

class TestMVFOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        self.data_source = DummyDataSource(df=self.df)
        self.data_source.__class__.__name__ = 'DataSource'
        self.task_dir = Path('test_task_dir')
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    def test_execute_success(self, mock_combo_dict, mock_value_dict, mock_analyze, mock_load):
        mock_load.return_value = self.df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, mvf.OperationStatus.SUCCESS)
        self.assertTrue(result.artifacts)
        self.assertIn('total_records', result.metrics)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_execute_no_df(self, mock_load):
        mock_load.return_value = None
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn('No valid DataFrame', result.error_message)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_execute_field_not_found(self, mock_load):
        mock_load.return_value = pd.DataFrame({'other_field': [1, 2]})
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn('not found in DataFrame', result.error_message)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    def test_execute_analysis_error(self, mock_analyze, mock_load):
        mock_load.return_value = self.df
        mock_analyze.return_value = {'error': 'fail'}
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn('fail', result.error_message)
        
    @patch.object(mvf.MVFOperation, '_handle_visualizations')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    def test_execute_with_visualization(self, mock_combo_dict, mock_value_dict, mock_analyze, mock_load, mock_handle_viz):
        # Setup DataFrame and mocks
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        mock_handle_viz.return_value = {'main': 'vis_path.png'}
        op = mvf.MVFOperation('mvf_field')
        op.generate_visualization = True
        op.visualization_backend = 'plotly'
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, mvf.OperationStatus.SUCCESS)
        mock_handle_viz.assert_called_once()
        
    @patch.object(mvf.MVFOperation, '_check_cache')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_execute_with_cache(self, mock_load, mock_cache):
        # Setup DataFrame and mocks
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_cache.return_value = OperationResult(
                            status=mvf.OperationStatus.SUCCESS
                        )
        op = mvf.MVFOperation('mvf_field')
        op.use_cache = True 
        op.generate_visualization = True
        op.visualization_backend = 'plotly'
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, mvf.OperationStatus.SUCCESS)
        mock_cache.assert_called_once()
        
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_execute_with_df_none(self, mock_load):
        # Setup DataFrame and mocks
        mock_load.return_value = None
        op = mvf.MVFOperation('mvf_field')
        op.use_cache = False
        op.generate_visualization = True
        op.visualization_backend = 'plotly'
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn("No valid DataFrame found in data source", result.error_message)
        
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_execute_with_exception_load_data(self, mock_load):
        # Setup DataFrame and mocks
        mock_load.side_effect = Exception("Error loading data")
        op = mvf.MVFOperation('mvf_field')
        op.use_cache = False
        op.generate_visualization = True
        op.visualization_backend = 'plotly'
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn("Error loading data", result.error_message)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_execute_values_viz_error(self, mock_logger, mock_combo_dict, mock_value_dict, mock_analyze, mock_load):
        # Setup DataFrame and mocks
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {},
            'value_counts_distribution': {},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        mock_value_dict.side_effect = Exception('Exception')
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_execute_combinations_viz_error(self, mock_logger, mock_combo_dict, mock_value_dict, mock_analyze, mock_load):
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.side_effect = Exception('Exception')
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    def test_execute_value_counts_non_int_key(self, mock_combo_dict, mock_value_dict, mock_analyze, mock_load):
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 2, 'two': 3},  # 'two' will cause ValueError
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.SUCCESS)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_execute_value_counts_viz_error(self, mock_logger, mock_combo_dict, mock_value_dict, mock_analyze, mock_load):
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_analyze.side_effect = Exception('Exception')
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    def test_execute_with_error_count(self, mock_combo_dict, mock_value_dict, mock_analyze, mock_load):
        mock_load.return_value = self.df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0,
            'error_count': 5,
            'error_percentage': 12.5
        }
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertIn('error_count', result.metrics)
        self.assertIn('error_percentage', result.metrics)
        self.assertEqual(result.metrics['error_count'], 5)
        self.assertEqual(result.metrics['error_percentage'], 12.5)

    @patch('pamola_core.profiling.analyzers.mvf.logger')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze', side_effect=Exception('Simulated error'))
    def test_execute_exception_handling(self, mock_analyze, mock_load, mock_logger):
        mock_load.return_value = self.df
        op = mvf.MVFOperation('mvf_field')
        reporter = MagicMock()
        progress_tracker = MagicMock()
        result = op.execute(self.data_source, self.task_dir, reporter, progress_tracker)
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn('Simulated error', result.error_message)

    def test_prepare_directories(self):
        op = mvf.MVFOperation('mvf_field')
        base_dir = Path('test_task_dir')
        dirs = op._prepare_directories(base_dir)
        self.assertIn('root', dirs)
        self.assertIn('output', dirs)
        self.assertIn('dictionaries', dirs)
        self.assertIn('logs', dirs)
        self.assertIn('cache', dirs)
        self.assertEqual(str(dirs['root']), str(Path('test_task_dir')))
        self.assertTrue(str(dirs['output']).endswith('output'))
        self.assertTrue(str(dirs['dictionaries']).endswith('dictionaries'))
        self.assertTrue(str(dirs['logs']).endswith('logs'))
        self.assertTrue(str(dirs['cache']).endswith('cache'))
        # ensure_directory should be called for each directory
        self.assertEqual(len(dirs), 5)

class TestAnalyzeMVFFields(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute')
    def test_analyze_mvf_fields(self, mock_execute, mock_load):
        df = pd.DataFrame({'f1': ["['A']"], 'f2': ["['B']"]})
        data_source = DummyDataSource(df=df)
        mock_load.return_value = df
        mock_execute.return_value = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
        reporter = MagicMock()
        result = mvf.analyze_mvf_fields(data_source, Path('task'), reporter, ['f1', 'f2'])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r.status == mvf.OperationStatus.SUCCESS for r in result.values()))

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_analyze_mvf_fields_no_df(self, mock_load):
        mock_load.return_value = None
        reporter = MagicMock()
        result = mvf.analyze_mvf_fields(MagicMock(), Path('task'), reporter, ['f1'])
        self.assertEqual(result, {})

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    def test_analyze_mvf_fields_missing_field(self, mock_load):
        df = pd.DataFrame({'f1': ["['A']"]})
        data_source = DummyDataSource(df=df)
        mock_load.return_value = df
        reporter = MagicMock()
        # f2 is missing, so only f1 is processed
        with patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute') as mock_exec:
            mock_exec.return_value = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
            result = mvf.analyze_mvf_fields(data_source, Path('task'), reporter, ['f1', 'f2'])
            self.assertIn('f1', result)
            self.assertNotIn('f2', result)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute', side_effect=Exception('Simulated error'))
    def test_analyze_mvf_fields_operation_exception(self, mock_execute, mock_load):
        df = pd.DataFrame({'f1': ["['A']"]})
        data_source = DummyDataSource(df=df)
        mock_load.return_value = df
        reporter = MagicMock()
        result = mvf.analyze_mvf_fields(data_source, Path('task'), reporter, ['f1'])
        # The result will be {} because of the exception, 'f1' will not be present
        self.assertEqual(result, {})
        # Check that reporter recorded the error
        reporter.add_operation.assert_any_call('Analyzing f1 field', status='error', details={'error': 'Simulated error'})

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute')
    def test_analyze_mvf_fields_progress(self, mock_execute, mock_load):
        df = pd.DataFrame({'f1': ["['A']"], 'f2': ["['B']"]})
        data_source = DummyDataSource(df=df)
        mock_load.return_value = df
        mock_execute.return_value = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
        reporter = MagicMock()
        # Test with track_progress=False
        result = mvf.analyze_mvf_fields(data_source, Path('task'), reporter, ['f1', 'f2'], track_progress=False)
        self.assertEqual(len(result), 2)
        # Ensure that execute was called with track_progress=False
        _, kwargs = mock_execute.call_args
        self.assertFalse(kwargs['track_progress'])


    @patch("pamola_core.utils.progress.ProgressTracker")
    @patch("pamola_core.profiling.analyzers.mvf.load_data_operation")
    @patch("pamola_core.profiling.analyzers.mvf.MVFOperation.execute")
    def test_analyze_mvf_fields_overall_tracker_error_update(self, mock_execute, mock_load, mock_tracker_cls):
        # Setup DataFrame and mocks
        df = pd.DataFrame({'f1': ["['A']"]})
        mock_load.return_value = df
        # Simulate an error result
        error_result = mvf.OperationResult(status=mvf.OperationStatus.ERROR, error_message="Simulated error")
        mock_execute.return_value = error_result
        mock_tracker = mock_tracker_cls.return_value

        reporter = MagicMock()
        # Call with track_progress=True (default)
        mvf.analyze_mvf_fields(MagicMock(), Path('task'), reporter, ['f1'])

        # Check that overall_tracker.update was called with status="error" (allow extra keys)
        calls = mock_tracker.update.call_args_list
        found = any(
            call_args[0][0] == 1 and
            call_args[0][1].get("field") == "f1" and
            call_args[0][1].get("status") == "error"
            for call_args in calls
        )
        self.assertTrue(found, f"Expected update(1, ...) with field='f1' and status='error', got: {calls}")

class TestMVFHandleVisualizations(unittest.TestCase):
    def setUp(self):
        self.analysis_results = {'some': 'result'}
        self.task_dir = Path('test_vis_dir')
        self.result = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()
        self.op = mvf.MVFOperation('mvf_field')
        self.op.logger = MagicMock()

    @patch.object(mvf.MVFOperation, '_generate_visualizations')
    def test_handle_visualizations_success(self, mock_generate):
        mock_generate.return_value = {'main': 'vis_path.png'}
        out = self.op._handle_visualizations(
            self.analysis_results, self.task_dir, self.result, self.reporter, self.progress_tracker,
            vis_theme='theme', vis_backend='plotly', vis_strict=False, vis_timeout=2, operation_timestamp='20240624')
        self.assertIn('main', out)
        self.assertTrue(any('vis_path.png' in str(a.path) for a in self.result.artifacts))
        self.reporter.add_operation.assert_any_call(
            'mvf_field main visualization', details={'artifact_type': 'png', 'path': 'vis_path.png'})

    @patch.object(mvf.MVFOperation, '_generate_visualizations', side_effect=Exception('viz error'))
    def test_handle_visualizations_visualization_error(self, mock_generate):
        out = self.op._handle_visualizations(
            self.analysis_results, self.task_dir, self.result, self.reporter, self.progress_tracker,
            vis_theme='theme', vis_backend='plotly', vis_strict=False, vis_timeout=2, operation_timestamp='20240624')
        self.assertEqual(out, {})
        self.op.logger.error.assert_any_call(
            unittest.mock.ANY
        )

    @patch('threading.Thread')
    @patch.object(mvf.MVFOperation, '_generate_visualizations')
    def test_handle_visualizations_timeout(self, mock_generate, mock_thread):
        # Simulate thread that never finishes
        class DummyThread:
            def __init__(self): self._alive = True
            def start(self): pass
            def join(self, timeout=None): pass
            def is_alive(self): return True
            @property
            def daemon(self): return False
        mock_thread.return_value = DummyThread()
        out = self.op._handle_visualizations(
            self.analysis_results, self.task_dir, self.result, self.reporter, self.progress_tracker,
            vis_theme='theme', vis_backend='plotly', vis_strict=False, vis_timeout=0, operation_timestamp='20240624')
        self.assertEqual(out, {})
        self.op.logger.error.assert_any_call(
            unittest.mock.ANY
        )

    def test_handle_visualizations_no_backend(self):
        out = self.op._handle_visualizations(
            self.analysis_results, self.task_dir, self.result, self.reporter, self.progress_tracker,
            vis_theme='theme', vis_backend=None, vis_strict=False, vis_timeout=2, operation_timestamp='20240624')
        self.assertEqual(out, {})
        self.op.logger.info.assert_any_call(
            'Generating visualizations with backend: None, timeout: 2s'
        )

class TestMVFAnalyzerDaskVectorization(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.mvf.analyze_mvf_field_with_dask')
    @patch('pamola_core.profiling.analyzers.mvf.detect_mvf_format', return_value='array')
    def test_analyze_use_dask_vectorization(self, mock_detect, mock_dask):
        # Simulate a DataFrame and dask analysis result
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        # Simulate dask result
        mock_dask.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        # Call analyze with use_dask=True, use_vectorization=True, and no flag_processed
        result = mvf.MVFAnalyzer.analyze(
            df, 'mvf_field', use_dask=True, use_vectorization=True
        )
        self.assertIn('values_analysis', result)
        self.assertEqual(result['total_records'], 3)
        mock_dask.assert_called_once()
        
    @patch('pamola_core.profiling.analyzers.mvf.analyze_mvf_field_with_parallel')
    @patch('pamola_core.profiling.analyzers.mvf.detect_mvf_format', return_value='array')
    def test_analyze_use_vectorization(self, mock_detect, mock_parallel):
        # Simulate a DataFrame and dask analysis result
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        # Simulate dask result
        mock_parallel.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        # Call analyze with use_dask=True, use_vectorization=True, and no flag_processed
        result = mvf.MVFAnalyzer.analyze(
            df, 'mvf_field', use_dask=False, use_vectorization=True
        )
        self.assertIn('values_analysis', result)
        self.assertEqual(result['total_records'], 3)
        mock_parallel.assert_called_once()
        
    @patch('pamola_core.profiling.analyzers.mvf.analyze_mvf_in_chunks')
    @patch('pamola_core.profiling.analyzers.mvf.detect_mvf_format', return_value='array')
    def test_analyze_use_chunk(self, mock_detect, mock_chunk):
        # Simulate a DataFrame and dask analysis result
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        # Simulate dask result
        mock_chunk.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        # Call analyze with use_dask=True, use_vectorization=True, and no flag_processed
        result = mvf.MVFAnalyzer.analyze(
            df, 'mvf_field', use_dask=False, use_vectorization=False
        )
        self.assertIn('values_analysis', result)
        self.assertEqual(result['total_records'], 3)
        mock_chunk.assert_called_once()
    
    @patch('pamola_core.profiling.analyzers.mvf.aggregate_mvf_analysis')    
    @patch('pamola_core.profiling.analyzers.mvf.process_mvf_partition')
    @patch('pamola_core.profiling.analyzers.mvf.detect_mvf_format', return_value='array')
    def test_analyze_use_partition(self, mock_detect, mock_partition, mock_analysispartition):
        # Simulate a DataFrame and dask analysis result
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        # Simulate dask result
        mock_partition.return_value = df
        mock_analysispartition.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 1, '2': 2},
            'total_records': 3,
            'null_count': 0,
            'null_percentage': 0.0,
            'empty_arrays_count': 0,
            'unique_values': 2,
            'unique_combinations': 1,
            'avg_values_per_record': 2.0
        }
        # Call analyze with use_dask=True, use_vectorization=True, and no flag_processed
        result = mvf.MVFAnalyzer.analyze(
            df, 'mvf_field', use_dask=False, use_vectorization=False, chunk_size=1
        )
        self.assertIn('values_analysis', result)
        self.assertEqual(result['total_records'], 3)
        mock_partition.assert_called_once()
        mock_analysispartition.assert_called_once()

class TestMVFOperationRestoreCachedArtifacts(unittest.TestCase):
    def setUp(self):
        self.op = mvf.MVFOperation('mvf_field')
        self.op.logger = MagicMock()
        self.result = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
        self.reporter = MagicMock()

    def test_restore_main_output_and_metrics(self):
        # Simulate files exist
        with patch('pathlib.Path.exists', return_value=True):
            cached = {
                'values_str_path': 'foo.csv',
                'combinations_str_path': 'bar.csv',
                'visualizations': {}
            }
            restored = self.op._restore_cached_artifacts(self.result, cached, self.reporter)
            self.assertEqual(restored, 2)
            self.reporter.add_operation.assert_any_call(
                'mvf_field generalized data (cached)', details={'artifact_type': 'csv', 'path': 'foo.csv'})
            self.reporter.add_operation.assert_any_call(
                'mvf_field generalized data (cached)', details={'artifact_type': 'csv', 'path': 'bar.csv'})

    def test_restore_visualizations(self):
        # Simulate files exist
        with patch('pathlib.Path.exists', return_value=True):
            cached = {
                'visualizations': {'main': 'foo.png', 'other': 'bar.png'}
            }
            restored = self.op._restore_cached_artifacts(self.result, cached, self.reporter)
            self.assertEqual(restored, 2)
            self.reporter.add_operation.assert_any_call(
                'mvf_field main visualization (cached)', details={'artifact_type': 'png', 'path': 'foo.png'})
            self.reporter.add_operation.assert_any_call(
                'mvf_field other visualization (cached)', details={'artifact_type': 'png', 'path': 'bar.png'})

    def test_restore_file_missing(self):
        # Simulate file does not exist
        with patch('pathlib.Path.exists', return_value=False):
            cached = {
                'values_str_path': 'foo.csv',
                'visualizations': {'main': 'foo.png'}
            }
            restored = self.op._restore_cached_artifacts(self.result, cached, self.reporter)
            self.assertEqual(restored, 0)
            self.op.logger.warning.assert_any_call('Cached file not found: foo.csv')
            self.op.logger.warning.assert_any_call('Cached file not found: foo.png')

    def test_restore_empty_cache(self):
        # No files, nothing to restore
        cached = {}
        restored = self.op._restore_cached_artifacts(self.result, cached, self.reporter)
        self.assertEqual(restored, 0)

if __name__ == '__main__':
    unittest.main()
