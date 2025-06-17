import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from pathlib import Path

from pamola_core.profiling.analyzers import mvf

class TestMVFAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'mvf_field': ["['A','B']", "['B','C']", "['A']", None, "[]"]
        })

    @patch('pamola_core.profiling.analyzers.mvf.analyze_mvf_field')
    def test_analyze(self, mock_analyze):
        mock_analyze.return_value = {'result': 1}
        result = mvf.MVFAnalyzer.analyze(self.df, 'mvf_field')
        self.assertEqual(result, {'result': 1})
        mock_analyze.assert_called_once()

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
        self.data_source = MagicMock()
        self.data_source.__class__.__name__ = 'DataSource'
        self.task_dir = Path('test_task_dir')
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.get_timestamped_filename', side_effect=lambda *a, **k: 'file.json')
    @patch('pamola_core.profiling.analyzers.mvf.write_json')
    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.plot_value_distribution', return_value='ok')
    @patch('pamola_core.profiling.analyzers.mvf.create_bar_plot', return_value='ok')
    def test_execute_success(self, mock_bar, mock_plot, mock_combo_dict, mock_value_dict, mock_analyze, mock_ensure, mock_write, mock_filename, mock_load):
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

    @patch('pamola_core.profiling.analyzers.mvf.plot_value_distribution', return_value='Error: failed to plot')
    @patch('pamola_core.profiling.analyzers.mvf.get_timestamped_filename', side_effect=lambda *a, **k: 'file.png')
    @patch('pamola_core.profiling.analyzers.mvf.write_json')
    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_execute_values_viz_error(self, mock_logger, mock_combo_dict, mock_value_dict, mock_analyze, mock_load, mock_ensure, mock_write, mock_filename, mock_plot):
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
        mock_value_dict.return_value = pd.DataFrame({'value': ['A'], 'count': [2]})
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        op.execute(self.data_source, self.task_dir, self.reporter)
        mock_logger.warning.assert_any_call('Error creating values visualization: Error: failed to plot')

    @patch('pamola_core.profiling.analyzers.mvf.plot_value_distribution', side_effect=[ 'ok', 'Error: failed to plot' ])
    @patch('pamola_core.profiling.analyzers.mvf.get_timestamped_filename', side_effect=lambda *a, **k: 'file.png')
    @patch('pamola_core.profiling.analyzers.mvf.write_json')
    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_execute_combinations_viz_error(self, mock_logger, mock_combo_dict, mock_value_dict, mock_analyze, mock_load, mock_ensure, mock_write, mock_filename, mock_plot):
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
        mock_combo_dict.return_value = pd.DataFrame({'combination': ['A,B'], 'count': [1]})
        op = mvf.MVFOperation('mvf_field')
        op.execute(self.data_source, self.task_dir, self.reporter)
        mock_logger.warning.assert_any_call('Error creating combinations visualization: Error: failed to plot')

    @patch('pamola_core.profiling.analyzers.mvf.create_bar_plot', return_value='ok')
    @patch('pamola_core.profiling.analyzers.mvf.get_timestamped_filename', side_effect=lambda *a, **k: 'file.png')
    @patch('pamola_core.profiling.analyzers.mvf.write_json')
    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    def test_execute_value_counts_non_int_key(self, mock_combo_dict, mock_value_dict, mock_analyze, mock_load, mock_ensure, mock_write, mock_filename, mock_bar):
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
        # Check that create_bar_plot was called with a dict containing the non-int key 'two'
        called_args = mock_bar.call_args[1]['data']
        self.assertIn('two', called_args)

    @patch('pamola_core.profiling.analyzers.mvf.create_bar_plot', return_value='Error: failed to plot')
    @patch('pamola_core.profiling.analyzers.mvf.get_timestamped_filename', side_effect=lambda *a, **k: 'file.png')
    @patch('pamola_core.profiling.analyzers.mvf.write_json')
    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.logger')
    def test_execute_value_counts_viz_error(self, mock_logger, mock_combo_dict, mock_value_dict, mock_analyze, mock_load, mock_ensure, mock_write, mock_filename, mock_bar):
        df = pd.DataFrame({'mvf_field': ["['A','B']", "['B','C']", "['A']"]})
        mock_load.return_value = df
        mock_analyze.return_value = {
            'values_analysis': {'A': 2, 'B': 2},
            'combinations_analysis': {'A,B': 1},
            'value_counts_distribution': {'1': 2, '2': 1},
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
        op.execute(self.data_source, self.task_dir, self.reporter)
        mock_logger.warning.assert_any_call('Error creating value counts visualization: Error: failed to plot')

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.get_timestamped_filename', side_effect=lambda *a, **k: 'file.json')
    @patch('pamola_core.profiling.analyzers.mvf.write_json')
    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_value_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.MVFAnalyzer.create_combinations_dictionary')
    @patch('pamola_core.profiling.analyzers.mvf.plot_value_distribution', return_value='ok')
    @patch('pamola_core.profiling.analyzers.mvf.create_bar_plot', return_value='ok')
    def test_execute_with_error_count(self, mock_bar, mock_plot, mock_combo_dict, mock_value_dict, mock_analyze, mock_ensure, mock_write, mock_filename, mock_load):
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
        mock_logger.exception.assert_called()
        progress_tracker.update.assert_any_call(0, {'step': 'Error', 'error': 'Simulated error'})
        reporter.add_operation.assert_any_call('Error analyzing mvf_field', status='error', details={'error': 'Simulated error'})
        self.assertEqual(result.status, mvf.OperationStatus.ERROR)
        self.assertIn('Error analyzing MVF field mvf_field: Simulated error', result.error_message)

    @patch('pamola_core.profiling.analyzers.mvf.ensure_directory')
    def test_prepare_directories(self, mock_ensure):
        op = mvf.MVFOperation('mvf_field')
        base_dir = Path('test_task_dir')
        dirs = op._prepare_directories(base_dir)
        self.assertIn('output', dirs)
        self.assertIn('visualizations', dirs)
        self.assertIn('dictionaries', dirs)
        self.assertTrue(str(dirs['output']).endswith('output'))
        self.assertTrue(str(dirs['visualizations']).endswith('visualizations'))
        self.assertTrue(str(dirs['dictionaries']).endswith('dictionaries'))
        # ensure_directory should be called for each directory
        self.assertEqual(mock_ensure.call_count, 3)

class TestAnalyzeMVFFields(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute')
    def test_analyze_mvf_fields(self, mock_execute, mock_load):
        df = pd.DataFrame({'f1': ["['A']"], 'f2': ["['B']"]})
        mock_load.return_value = df
        mock_execute.return_value = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
        reporter = MagicMock()
        result = mvf.analyze_mvf_fields(MagicMock(), Path('task'), reporter, ['f1', 'f2'])
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
        mock_load.return_value = df
        reporter = MagicMock()
        # f2 is missing, so only f1 is processed
        with patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute') as mock_exec:
            mock_exec.return_value = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
            result = mvf.analyze_mvf_fields(MagicMock(), Path('task'), reporter, ['f1', 'f2'])
            self.assertIn('f1', result)
            self.assertNotIn('f2', result)

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute', side_effect=Exception('Simulated error'))
    def test_analyze_mvf_fields_operation_exception(self, mock_execute, mock_load):
        df = pd.DataFrame({'f1': ["['A']"]})
        mock_load.return_value = df
        reporter = MagicMock()
        result = mvf.analyze_mvf_fields(MagicMock(), Path('task'), reporter, ['f1'])
        # The result will be {} because of the exception, 'f1' will not be present
        self.assertEqual(result, {})
        # Check that reporter recorded the error
        reporter.add_operation.assert_any_call('Analyzing f1 field', status='error', details={'error': 'Simulated error'})

    @patch('pamola_core.profiling.analyzers.mvf.load_data_operation')
    @patch('pamola_core.profiling.analyzers.mvf.MVFOperation.execute')
    def test_analyze_mvf_fields_progress(self, mock_execute, mock_load):
        df = pd.DataFrame({'f1': ["['A']"], 'f2': ["['B']"]})
        mock_load.return_value = df
        mock_execute.return_value = mvf.OperationResult(status=mvf.OperationStatus.SUCCESS)
        reporter = MagicMock()
        # Test with track_progress=False
        result = mvf.analyze_mvf_fields(MagicMock(), Path('task'), reporter, ['f1', 'f2'], track_progress=False)
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

if __name__ == '__main__':
    unittest.main()
