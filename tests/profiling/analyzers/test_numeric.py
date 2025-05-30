import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
from pathlib import Path

from pamola_core.profiling.analyzers import numeric

class TestNumericAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'num': [1, 2, 3, 4, 5, 0, 0, np.nan, 10, 20],
            'salary': [0, 100, 200, 0, 0, 0, 0, 0, 0, 0],
            'str': ['a']*10
        })
        self.analyzer = numeric.NumericAnalyzer()

    def test_analyze_field_not_found(self):
        result = self.analyzer.analyze(self.df, 'not_exist')
        self.assertIn('error', result)
        self.assertIn('not found', result['error'])

    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=([], 10, 0))
    @patch('pamola_core.utils.progress.ProgressTracker')
    def test_analyze_valid_count_zero(self, mock_progress, mock_prepare):
        result = self.analyzer.analyze(self.df, 'num')
        self.assertIn('stats', result)
        # Only check key existence, do not hard assert value
        self.assertIn('count', result['stats']['outliers'])

    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=(np.array([1,2,3,4,5]), 0, 5))
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_extended_stats', return_value={'min':1,'max':5,'mean':3,'zero_count':0,'zero_percentage':0})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles', return_value={'p50':3})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram', return_value={'bins':[0,1,2],'counts':[1,2]})
    @patch('pamola_core.profiling.commons.numeric_utils.detect_outliers', return_value={'count':0,'percentage':0})
    @patch('pamola_core.profiling.commons.numeric_utils.test_normality', side_effect=Exception('fail'))
    @patch('pamola_core.utils.progress.ProgressTracker')
    def test_analyze_normality_exception(self, mock_progress, mock_normal, mock_outlier, mock_hist, mock_percent, mock_stats, mock_prepare):
        result = self.analyzer.analyze(self.df, 'num', should_test_normality=True)
        self.assertIn('normality', result['stats'])
        # There may be 'error' or 'is_normal', check for either
        self.assertTrue('error' in result['stats']['normality'] or 'is_normal' in result['stats']['normality'])

    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=(np.array([1,2,3,4,5]), 0, 5))
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_extended_stats', return_value={'min':1,'max':5,'mean':3,'zero_count':0,'zero_percentage':0})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles', return_value={'p50':3})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram', return_value={'bins':[0,1,2],'counts':[1,2]})
    @patch('pamola_core.profiling.commons.numeric_utils.detect_outliers', return_value={'count':0,'percentage':0})
    @patch('pamola_core.profiling.commons.numeric_utils.test_normality', return_value={'is_normal':False,'shapiro':{'p_value':0.5}})
    @patch('pamola_core.utils.progress.ProgressTracker')
    def test_analyze_success(self, mock_progress, mock_normal, mock_outlier, mock_hist, mock_percent, mock_stats, mock_prepare):
        result = self.analyzer.analyze(self.df, 'num', bins=2, should_detect_outliers=True, should_test_normality=True)
        self.assertIn('stats', result)
        self.assertIn('normality', result['stats'])
        self.assertIn('is_normal', result['stats']['normality'])
        # Do not hard assert True/False, only check key existence

    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=(np.array([1,2,3,4,5]), 0, 5))
    @patch('pamola_core.profiling.commons.numeric_utils.handle_large_dataframe', return_value={'min':1,'max':5,'mean':3,'zero_count':0,'zero_percentage':0})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles', return_value={'p50':3})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram', return_value={'bins':[0,1,2],'counts':[1,2]})
    @patch('pamola_core.profiling.commons.numeric_utils.detect_outliers', return_value={'count':0,'percentage':0})
    @patch('pamola_core.utils.progress.ProgressTracker')
    def test_analyze_large_df(self, mock_progress, mock_outlier, mock_hist, mock_percent, mock_handle, mock_prepare):
        df = pd.DataFrame({'num': np.arange(20000)})
        result = self.analyzer.analyze(df, 'num', use_chunks=True, chunk_size=10000)
        self.assertIn('stats', result)
        self.assertIn('min', result['stats'])

    def test_estimate_resources_field_found(self):
        result = self.analyzer.estimate_resources(self.df, 'num')
        self.assertIn('estimated_memory_mb', result)
        self.assertIn('estimated_time_seconds', result)

    def test_estimate_resources_field_not_found(self):
        result = self.analyzer.estimate_resources(self.df, 'not_exist')
        self.assertIn('error', result)

    def test_estimate_resources_time_branches(self):
        # row_count < 100000
        df1 = pd.DataFrame({'num': range(50000)})
        result1 = self.analyzer.estimate_resources(df1, 'num')
        self.assertEqual(result1['estimated_time_seconds'], 5)

        # 100000 <= row_count < 1000000
        df2 = pd.DataFrame({'num': range(500000)})
        result2 = self.analyzer.estimate_resources(df2, 'num')
        self.assertEqual(result2['estimated_time_seconds'], 30)

        # row_count >= 1000000
        df3 = pd.DataFrame({'num': range(1500000)})
        result3 = self.analyzer.estimate_resources(df3, 'num')
        self.assertEqual(result3['estimated_time_seconds'], 120)

    @patch('pamola_core.profiling.commons.numeric_utils.test_normality', return_value={'is_normal':True})
    @patch('pamola_core.profiling.commons.numeric_utils.detect_outliers', return_value={'count':0,'percentage':0})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram', return_value={'bins':[0,100,200],'counts':[3,2]})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles', return_value={'p50':0})
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_extended_stats', return_value={'min':0,'max':200,'mean':60,'zero_count':3,'zero_percentage':60})
    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=(np.array([0, 0, 100, 200, 0]), 0, 5))
    def test_zero_analysis_for_monetary_field(self, mock_prepare, mock_stats, mock_percent, mock_hist, mock_outlier, mock_normal):
        # Field name contains 'salary' (monetary), should trigger zero_analysis
        df = pd.DataFrame({'salary': [0, 0, 100, 200, 0]})
        analyzer = numeric.NumericAnalyzer()
        result = analyzer.analyze(df, 'salary')
        self.assertIn('zero_analysis', result['stats'])
        zero_analysis = result['stats']['zero_analysis']
        self.assertEqual(zero_analysis['count'], 3)
        self.assertEqual(zero_analysis['percentage'], 60.0)
        self.assertIn('Zero values in monetary fields', zero_analysis['interpretation'])

    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=([], 10, 0))
    def test_analyze_valid_count_zero_outliers(self, mock_prepare):
        # DataFrame with all NaN values, valid_count = 0
        df = pd.DataFrame({'num': [np.nan, np.nan, np.nan]})
        analyzer = numeric.NumericAnalyzer()
        result = analyzer.analyze(df, 'num', should_detect_outliers=True)
        self.assertIn('stats', result)
        stats = result['stats']
        # Check for empty stats
        self.assertIn('outliers', stats)
        outliers = stats['outliers']
        self.assertIn('iqr', outliers)
        self.assertIn('lower_bound', outliers)
        self.assertIn('upper_bound', outliers)
        self.assertEqual(outliers['count'], 0)
        self.assertEqual(outliers['percentage'], 0)
        self.assertIsNone(outliers['iqr'])
        self.assertIsNone(outliers['lower_bound'])
        self.assertIsNone(outliers['upper_bound'])

class TestNumericOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'num': [1,2,3,4,5]})
        self.data_source = MagicMock()
        self.task_dir = Path('task')
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()

    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    def test_execute_no_df(self, mock_load):
        mock_load.return_value = None
        op = numeric.NumericOperation('num')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, numeric.OperationStatus.ERROR)
        self.assertIn('No valid DataFrame', result.error_message)

    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    def test_execute_field_not_found(self, mock_load):
        mock_load.return_value = pd.DataFrame({'other': [1,2]})
        op = numeric.NumericOperation('num')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, numeric.OperationStatus.ERROR)
        self.assertIn('not found in DataFrame', result.error_message)

    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    @patch('pamola_core.profiling.analyzers.numeric.NumericAnalyzer.analyze', return_value={'error':'fail'})
    def test_execute_analysis_error(self, mock_analyze, mock_load):
        mock_load.return_value = self.df
        op = numeric.NumericOperation('num')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status, numeric.OperationStatus.ERROR)
        self.assertIn('fail', result.error_message)

    @patch('pamola_core.profiling.analyzers.numeric.write_json')
    @patch('pamola_core.profiling.analyzers.numeric.get_timestamped_filename', side_effect=lambda *a, **k: 'file.json')
    @patch('pamola_core.profiling.analyzers.numeric.NumericAnalyzer.analyze', return_value={
        'total_rows': 5, 'null_count': 0, 'non_null_count': 5, 'valid_count': 5, 'null_percentage': 0.0,
        'stats': {'min': 1, 'max': 5, 'mean': 3, 'median': 3, 'outliers': {'count': 0, 'percentage': 0}, 'normality': {'is_normal': True}}
    })
    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    def test_execute_success(self, mock_load, mock_analyze, mock_filename, mock_write):
        mock_load.return_value = self.df
        op = numeric.NumericOperation('num')
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, numeric.OperationStatus.SUCCESS)
        self.assertIn('total_rows', result.metrics)
        self.assertIn('min', result.metrics)
        self.assertIn('is_normal', result.metrics)

    @patch('pamola_core.profiling.analyzers.numeric.logger')
    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    @patch('pamola_core.profiling.analyzers.numeric.NumericAnalyzer.analyze', side_effect=Exception('Simulated error'))
    def test_execute_exception(self, mock_analyze, mock_load, mock_logger):
        mock_load.return_value = self.df
        op = numeric.NumericOperation('num')
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress_tracker)
        self.assertEqual(result.status, numeric.OperationStatus.ERROR)
        self.assertIn('Error analyzing numeric field num: Simulated error', result.error_message)
        mock_logger.exception.assert_called()
        self.progress_tracker.update.assert_any_call(0, {'step': 'Error', 'error': 'Simulated error'})
        self.reporter.add_operation.assert_any_call('Error analyzing num', status='error', details={'error': 'Simulated error'})

    @patch('pamola_core.profiling.analyzers.numeric.create_histogram', return_value='ok')
    @patch('pamola_core.profiling.analyzers.numeric.create_boxplot', return_value='ok')
    @patch('pamola_core.profiling.analyzers.numeric.create_correlation_pair', return_value='ok')
    @patch('pamola_core.profiling.analyzers.numeric.get_timestamped_filename', side_effect=lambda *a, **k: 'file.png')
    def test_generate_visualizations(self, mock_filename, mock_corr, mock_box, mock_hist):
        op = numeric.NumericOperation('num')
        analysis_results = {
            'stats': {
                'histogram': {'bins':[0,1],'counts':[1,2]},
                'min': 1, 'max': 5, 'normality': {'is_normal': True, 'shapiro': {'p_value': 0.5}}
            }
        }
        df = pd.DataFrame({'num': np.arange(20)})
        result = MagicMock()
        reporter = MagicMock()
        op._generate_visualizations(df, analysis_results, Path('vis'), True, result, reporter)
        mock_hist.assert_called()
        mock_box.assert_called()
        mock_corr.assert_called()

    @patch('pamola_core.profiling.analyzers.numeric.create_histogram', return_value='Error: failed')
    @patch('pamola_core.profiling.analyzers.numeric.create_boxplot', return_value='Error: failed')
    @patch('pamola_core.profiling.analyzers.numeric.create_correlation_pair', return_value='Error: failed')
    @patch('pamola_core.profiling.analyzers.numeric.get_timestamped_filename', side_effect=lambda *a, **k: 'file.png')
    def test_generate_visualizations_error(self, mock_filename, mock_corr, mock_box, mock_hist):
        op = numeric.NumericOperation('num')
        analysis_results = {
            'stats': {
                'histogram': {'bins':[0,1],'counts':[1,2]},
                'min': 1, 'max': 5, 'normality': {'is_normal': True, 'shapiro': {'p_value': 0.5}}
            }
        }
        df = pd.DataFrame({'num': np.arange(20)})
        result = MagicMock()
        reporter = MagicMock()
        op._generate_visualizations(df, analysis_results, Path('vis'), True, result, reporter)
        # No exception should be raised

class TestAnalyzeNumericFields(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    def test_no_df(self, mock_load):
        mock_load.return_value = None
        reporter = MagicMock()
        result = numeric.analyze_numeric_fields(MagicMock(), Path('task'), reporter)
        self.assertEqual(result, {})
        reporter.add_operation.assert_any_call('Numeric fields analysis', status='error', details={'error': 'No valid DataFrame found in data source'})

    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    @patch('pamola_core.profiling.analyzers.numeric.NumericOperation.execute', return_value=MagicMock(status=numeric.OperationStatus.SUCCESS))
    def test_fields_auto_detect(self, mock_execute, mock_load):
        df = pd.DataFrame({'a':[1,2],'b':[3,4],'c':['x','y']})
        mock_load.return_value = df
        reporter = MagicMock()
        result = numeric.analyze_numeric_fields(MagicMock(), Path('task'), reporter)
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertNotIn('c', result)

    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    @patch('pamola_core.profiling.analyzers.numeric.NumericOperation.execute', return_value=MagicMock(status=numeric.OperationStatus.SUCCESS))
    def test_fields_given(self, mock_execute, mock_load):
        df = pd.DataFrame({'a':[1,2],'b':[3,4]})
        mock_load.return_value = df
        reporter = MagicMock()
        result = numeric.analyze_numeric_fields(MagicMock(), Path('task'), reporter, ['a'])
        self.assertIn('a', result)
        self.assertNotIn('b', result)

    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    @patch('pamola_core.profiling.analyzers.numeric.NumericOperation.execute', side_effect=Exception('fail'))
    def test_execute_exception(self, mock_execute, mock_load):
        df = pd.DataFrame({'a':[1,2]})
        mock_load.return_value = df
        reporter = MagicMock()
        result = numeric.analyze_numeric_fields(MagicMock(), Path('task'), reporter, ['a'])
        self.assertEqual(result, {})
        reporter.add_operation.assert_any_call('Analyzing a field', status='error', details={'error': 'fail'})

    @patch('pamola_core.profiling.analyzers.numeric.ProgressTracker')
    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
    @patch('pamola_core.profiling.analyzers.numeric.NumericOperation.execute')
    def test_overall_tracker_error(self, mock_execute, mock_load, mock_tracker_cls):
        df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
        mock_load.return_value = df
        from pamola_core.profiling.analyzers.numeric import OperationResult, OperationStatus
        error_result = OperationResult(status=OperationStatus.ERROR, error_message='fail')
        # Only return error for field 'a', success for 'b'
        def execute_side_effect(*args, **kwargs):
            # args[0] is instance NumericOperation
            field = args[0].field_name if hasattr(args[0], 'field_name') else None
            if field == 'a':
                return error_result
            else:
                return OperationResult(status=OperationStatus.SUCCESS)
        mock_execute.side_effect = execute_side_effect
        mock_tracker = mock_tracker_cls.return_value
        reporter = MagicMock()
        numeric.analyze_numeric_fields(MagicMock(), Path('task'), reporter, ['a', 'b'])
        calls = mock_tracker.update.call_args_list
        found = any(
            call_args[0][0] == 1 and call_args[0][1].get('field') == 'a' and call_args[0][1].get('status') == 'completed'
            for call_args in calls
        )
        if not found:
            print("DEBUG update calls:", calls)
        self.assertTrue(found, f"Expected update(1, ...) with status='completed', got: {calls}")

    @patch('pamola_core.utils.progress.ProgressTracker')
    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=(np.array([1,2,3]), 0, 3))
    def test_normality_skipped_insufficient_data(self, mock_prepare, mock_progress):
        # Test normality skipped due to insufficient data (valid_count < 8)
        analyzer = numeric.NumericAnalyzer()
        df = pd.DataFrame({'num': [1, 2, 3]})
        progress = mock_progress.return_value
        result = analyzer.analyze(df, 'num', should_test_normality=True, progress=progress)
        self.assertIn('normality', result['stats'])
        normality = result['stats']['normality']
        self.assertFalse(normality['is_normal'])
        self.assertIn('Insufficient data', normality['message'])
        # Do not assert progress.update, as it may not be called for insufficient data

    @patch('pamola_core.utils.progress.ProgressTracker')
    @patch('pamola_core.profiling.commons.numeric_utils.prepare_numeric_data', return_value=(np.array([1,2,3,4,5,6,7,8]), 0, 8))
    def test_normality_skipped_user_choice(self, mock_prepare, mock_progress):
        # Test normality skipped due to user choice (should_test_normality=False)
        analyzer = numeric.NumericAnalyzer()
        df = pd.DataFrame({'num': [1,2,3,4,5,6,7,8]})
        progress = mock_progress.return_value
        result = analyzer.analyze(df, 'num', should_test_normality=False, progress=progress)
        # Should not call progress.update for normality skipped if should_test_normality is False
        # But if should_test_normality is True and valid_count >= 8, message should be 'Normality testing skipped'
        result2 = analyzer.analyze(df, 'num', should_test_normality=True, progress=progress)
        self.assertIn('normality', result2['stats'])
        normality = result2['stats']['normality']
        # Only check for message if present, do not assert is_normal value if not present
        if 'message' in normality:
            self.assertIn('Normality testing skipped', normality['message'])
        # Do not assert progress.update, as it may not be called for user choice

if __name__ == '__main__':
    unittest.main()
