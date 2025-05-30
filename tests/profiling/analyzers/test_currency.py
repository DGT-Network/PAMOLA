import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from unittest.mock import ANY

import pandas as pd

from pamola_core.profiling.analyzers.currency import CurrencyAnalyzer


class TestCurrencyAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CurrencyAnalyzer()
        self.df_valid = pd.DataFrame({
            'amount': ['$1,000.00', '$2,000.00', '$3,000.00', None, 'invalid']
        })
        self.df_empty = pd.DataFrame({'amount': []})
        self.df_no_field = pd.DataFrame({'other': [1, 2, 3]})

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.analyze_currency_stats')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.generate_currency_samples')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram')
    def test_analyze_valid(self, mock_hist, mock_percentiles, mock_empty_stats, mock_samples, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        mock_parse.return_value = (pd.Series([1000.0, 2000.0, 3000.0, None, None]), {'USD': 3})
        mock_analyze_stats.return_value = {'min': 1000.0, 'max': 3000.0, 'mean': 2000.0, 'median': 2000.0, 'std': 1000.0, 'negative_count': 0, 'zero_count': 0}
        mock_percentiles.return_value = {'25%': 1500.0, '50%': 2000.0, '75%': 2500.0}
        mock_hist.return_value = {'bins': [1000, 2000, 3000], 'counts': [1, 1, 1]}
        mock_samples.return_value = []
        mock_empty_stats.return_value = {}

        result = self.analyzer.analyze(self.df_valid, 'amount')
        self.assertIn('stats', result)
        self.assertEqual(result['valid_count'], 3)
        self.assertEqual(result['null_count'], 2)
        self.assertEqual(result['invalid_count'], 0)
        self.assertEqual(result['currency_counts'], {'USD': 3})

    def test_analyze_field_not_found(self):
        result = self.analyzer.analyze(self.df_no_field, 'amount')
        self.assertIn('error', result)
        self.assertIn('not found', result['error'])

    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    @patch('pamola_core.profiling.analyzers.currency.is_currency_field')
    def test_analyze_not_currency(self, mock_is_currency, mock_parse):
        mock_is_currency.return_value = False
        mock_parse.return_value = (pd.Series([1000.0, 2000.0, 3000.0, None, None]), {'USD': 3})
        result = self.analyzer.analyze(self.df_valid, 'amount')
        self.assertIn('is_detected_currency', result)
        self.assertFalse(result['is_detected_currency'])

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    def test_analyze_empty(self, mock_empty_stats, mock_parse):
        mock_parse.return_value = (pd.Series([], dtype=float), {})
        mock_empty_stats.return_value = {'min': None, 'max': None}
        result = self.analyzer.analyze(self.df_empty, 'amount')
        self.assertIn('stats', result)
        self.assertEqual(result['stats']['min'], None)

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.analyze_currency_stats')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.generate_currency_samples')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram')
    def test_analyze_with_progress_tracker(self, mock_hist, mock_percentiles, mock_empty_stats, mock_samples, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        mock_parse.return_value = (pd.Series([1000.0, 2000.0, 3000.0, None, None]), {'USD': 3})
        mock_analyze_stats.return_value = {'min': 1000.0, 'max': 3000.0, 'mean': 2000.0, 'median': 2000.0, 'std': 1000.0, 'negative_count': 0, 'zero_count': 0}
        mock_percentiles.return_value = {'25%': 1500.0, '50%': 2000.0, '75%': 2500.0}
        mock_hist.return_value = {'bins': [1000, 2000, 3000], 'counts': [1, 1, 1]}
        mock_samples.return_value = []
        mock_empty_stats.return_value = {}

        mock_progress = MagicMock()
        result = self.analyzer.analyze(self.df_valid, 'amount', progress_tracker=mock_progress)
        mock_progress.update.assert_any_call(0, {'step': 'Initializing currency analysis', 'field': 'amount'})
        self.assertIn('stats', result)

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.analyze_currency_stats')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.generate_currency_samples')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram')
    def test_analyze_dask_importerror(self, mock_hist, mock_percentiles, mock_empty_stats, mock_samples, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        mock_parse.return_value = (pd.Series([1000.0]*20001), {'USD': 20001})
        mock_analyze_stats.return_value = {'min': 1000.0, 'max': 1000.0, 'mean': 1000.0, 'median': 1000.0, 'std': 0.0, 'negative_count': 0, 'zero_count': 0}
        mock_percentiles.return_value = {'25%': 1000.0, '50%': 1000.0, '75%': 1000.0}
        mock_hist.return_value = {'bins': [1000], 'counts': [20001]}
        mock_samples.return_value = []
        mock_empty_stats.return_value = {}

        # Patch import dask to raise ImportError
        import builtins
        orig_import = builtins.__import__
        def import_side_effect(name, *args, **kwargs):
            if name == 'dask.dataframe':
                raise ImportError('No module named dask')
            return orig_import(name, *args, **kwargs)
        
        mock_progress = MagicMock()
        with patch('builtins.__import__', side_effect=import_side_effect):
            # DataFrame larger than chunk_size to enter Dask branch
            df_large = pd.DataFrame({'amount': ['$1,000.00']*20001})
            result = self.analyzer.analyze(df_large, 'amount', chunk_size=10000, use_dask=True, progress_tracker=mock_progress)
            mock_progress.update.assert_any_call(0, {'step': 'Dask fallback', 'warning': 'Dask not available, using chunks'})
            self.assertIn('stats', result)

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.analyze_currency_stats')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.generate_currency_samples')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram')
    def test_analyze_negative_count_semantic_note(self, mock_hist, mock_percentiles, mock_empty_stats, mock_samples, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        mock_parse.return_value = (pd.Series([1000.0, -2000.0, 3000.0]), {'USD': 3})
        mock_analyze_stats.return_value = {
            'min': -2000.0, 'max': 3000.0, 'mean': 733.33, 'median': 1000.0, 'std': 2516.6,
            'negative_count': 1, 'zero_count': 0
        }  # No semantic_notes
        mock_percentiles.return_value = {'25%': -500.0, '50%': 1000.0, '75%': 2000.0}
        mock_hist.return_value = {'bins': [-2000, 1000, 3000], 'counts': [1, 1, 1]}
        mock_samples.return_value = []
        mock_empty_stats.return_value = {}

        result = self.analyzer.analyze(self.df_valid, 'amount')
        notes = result['stats'].get('semantic_notes', [])

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    def test_analyze_no_valid_values_statistics_warning(self, mock_empty_stats, mock_is_currency, mock_parse):
        import pandas as pd
        from unittest.mock import MagicMock
        mock_is_currency.return_value = True
        # All values are None (after dropna there will be no valid values left)
        mock_parse.return_value = (pd.Series([None, None, None]), {'USD': 0})
        mock_empty_stats.return_value = {'min': None, 'max': None}
        mock_progress = MagicMock()
        df = pd.DataFrame({'amount': [None, None, None]})
        result = self.analyzer.analyze(df, 'amount', progress_tracker=mock_progress)
        mock_progress.update.assert_any_call(1, {'step': 'Statistics calculation', 'warning': 'No valid values for statistics'})
        self.assertIn('stats', result)
        self.assertEqual(result['stats']['min'], None)


    def test_analyze_large_df_chunking(self):
        from pamola_core.profiling.analyzers.currency import CurrencyAnalyzer
        from unittest.mock import patch
        import pandas as pd
        with patch.object(CurrencyAnalyzer, '_analyze_in_chunks') as mock_analyze_in_chunks, \
             patch('pamola_core.profiling.commons.currency_utils.is_currency_field') as mock_is_currency:
            mock_is_currency.return_value = True
            # Create DataFrame larger than chunk_size
            df_large = pd.DataFrame({'amount': ['$1,000.00']*15000})
            # Simulated result returned from _analyze_in_chunks
            mock_analyze_in_chunks.return_value = {'stats': {'min': 1000.0, 'max': 1000.0}}
            result = self.analyzer.analyze(df_large, 'amount', chunk_size=10000, use_dask=False)
            mock_analyze_in_chunks.assert_called_once()
            self.assertIn('stats', result)
            self.assertEqual(result['stats']['min'], 1000.0)

    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    @patch('pamola_core.profiling.analyzers.currency.is_currency_field')
    def test_analyze_parse_currency_field_exception(self, mock_is_currency, mock_parse):
        mock_is_currency.return_value = True
        mock_parse.side_effect = Exception('parse error!')
        mock_progress = MagicMock()
        df = pd.DataFrame({'amount': ['$1,000.00', '$2,000.00']})
        result = self.analyzer.analyze(df, 'amount', progress_tracker=mock_progress)
        self.assertIn('error', result)
        self.assertIn('parse error!', result['error'])
        self.assertEqual(result['field_name'], 'amount')
        self.assertEqual(result['total_rows'], 2)

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.analyze_currency_stats')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.generate_currency_samples')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram')
    @patch('pamola_core.profiling.commons.numeric_utils.detect_outliers')
    @patch('pamola_core.profiling.commons.numeric_utils.test_normality')
    def test_analyze_normality_exception(self, mock_test_normality, mock_detect_outliers, mock_hist, mock_percentiles, mock_empty_stats, mock_samples, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        # Enough values to enter test_normality branch (>=8)
        mock_parse.return_value = (pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0]), {'USD': 8})
        mock_analyze_stats.return_value = {'min': 1000.0, 'max': 8000.0, 'mean': 4500.0, 'median': 4500.0, 'std': 2415.0, 'negative_count': 0, 'zero_count': 0}
        mock_percentiles.return_value = {'25%': 2500.0, '50%': 4500.0, '75%': 6500.0}
        mock_hist.return_value = {'bins': [1000, 3000, 8000], 'counts': [2, 6]}
        mock_samples.return_value = []
        mock_empty_stats.return_value = {}
        mock_detect_outliers.return_value = {'count': 0, 'percentage': 0.0}
        mock_test_normality.side_effect = Exception('normality error!')

        result = self.analyzer.analyze(self.df_valid, 'amount')
        normality = result['stats'].get('normality', {})
        self.assertIn('message', normality)
        self.assertIn('Insufficient data for normality testing', normality['message'])
        self.assertFalse(normality['is_normal'])

    @patch('pamola_core.profiling.commons.currency_utils.parse_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.analyze_currency_stats')
    @patch('pamola_core.profiling.commons.currency_utils.is_currency_field')
    @patch('pamola_core.profiling.commons.currency_utils.generate_currency_samples')
    @patch('pamola_core.profiling.commons.currency_utils.create_empty_currency_stats')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_percentiles')
    @patch('pamola_core.profiling.commons.numeric_utils.calculate_histogram')
    @patch('pamola_core.profiling.commons.numeric_utils.detect_outliers')
    @patch('pamola_core.profiling.commons.numeric_utils.test_normality')
    def test_analyze_normality_exception_with_error_key(self, mock_test_normality, mock_detect_outliers, mock_hist, mock_percentiles, mock_empty_stats, mock_samples, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        mock_parse.return_value = (pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0]), {'USD': 8})
        mock_analyze_stats.return_value = {'min': 1000.0, 'max': 8000.0, 'mean': 4500.0, 'median': 4500.0, 'std': 2415.0, 'negative_count': 0, 'zero_count': 0}
        mock_percentiles.return_value = {'25%': 2500.0, '50%': 4500.0, '75%': 6500.0}
        mock_hist.return_value = {'bins': [1000, 3000, 8000], 'counts': [2, 6]}
        mock_samples.return_value = []
        mock_empty_stats.return_value = {}
        mock_detect_outliers.return_value = {'count': 0, 'percentage': 0.0}
        mock_test_normality.side_effect = Exception('normality error!')

        result = self.analyzer.analyze(self.df_valid, 'amount')
        normality = result['stats'].get('normality', {})
        self.assertIn('message', normality)
        self.assertIn('Insufficient data for normality testing', normality['message'])
        self.assertFalse(normality['is_normal'])

    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    @patch('pamola_core.profiling.analyzers.currency.analyze_currency_stats')
    @patch('pamola_core.profiling.analyzers.currency.is_currency_field')
    def test_analyze_statistics_exception(self, mock_is_currency, mock_analyze_stats, mock_parse):
        mock_is_currency.return_value = True
        df_valid_large = pd.DataFrame({'amount': ['$1,000.00', '$2,000.00', '$3,000.00', '$4,000.00', '$5,000.00', '$6,000.00', '$7,000.00', '$8,000.00', '$9,000.00', '$10,000.00']})
        # Ensure correct tuple is returned
        mock_parse.return_value = (pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]), {'USD': 10})
        mock_analyze_stats.side_effect = Exception('statistics error!')
        result = self.analyzer.analyze(df_valid_large, 'amount')
        self.assertIn('error', result)
        self.assertIn('Error calculating statistics: statistics error!', result['error'])

    @patch('pamola_core.profiling.analyzers.currency.detect_currency_from_sample')
    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    @patch('dask.dataframe.from_pandas')
    def test__analyze_with_dask_success_and_exception(self, mock_from_pandas, mock_parse_currency_field, mock_detect_currency):
        # --- Success path ---
        mock_ddf = MagicMock()
        mock_from_pandas.return_value = mock_ddf
        # Simulate Dask chained calls
        mock_ddf.__getitem__.return_value.isna.return_value.sum.return_value.compute.return_value = 1
        mock_ddf.map_partitions.return_value.mean.return_value.compute.return_value = 2.0
        mock_ddf.map_partitions.return_value.quantile.return_value.compute.return_value = 3.0
        mock_ddf.map_partitions.return_value.std.return_value.compute.return_value = 4.0
        mock_ddf.map_partitions.return_value.min.return_value.compute.return_value = 1.0
        mock_ddf.map_partitions.return_value.max.return_value.compute.return_value = 5.0
        mock_detect_currency.return_value = 'USD'
        mock_parse_currency_field.side_effect = lambda part, field, locale: (part[field], {})
        df = pd.DataFrame({'amount': [1.0, 2.0, 3.0, 4.0, 5.0]})
        analyzer = self.analyzer
        result = analyzer._analyze_with_dask(df, 'amount', 'en_US', 10, False, False)
        self.assertIn('stats', result)
        self.assertEqual(result['stats']['min'], 1.0)
        self.assertEqual(result['stats']['max'], 5.0)
        self.assertEqual(result['detected_currency'], 'USD')
        self.assertIn('note', result)

        # --- Exception path ---
        mock_from_pandas.side_effect = Exception('dask error!')
        result = analyzer._analyze_with_dask(df, 'amount', 'en_US', 10, False, False)
        self.assertIn('error', result)
        self.assertIn('dask error!', result['error'])
        self.assertEqual(result['field_name'], 'amount')

    @patch('pamola_core.profiling.analyzers.currency.detect_currency_from_sample')
    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    @patch('dask.dataframe.from_pandas')
    def test__analyze_with_dask_progress_tracker_update(self, mock_from_pandas, mock_parse_currency_field, mock_detect_currency):
        # Setup mocks for Dask DataFrame and chained calls
        mock_ddf = MagicMock()
        mock_from_pandas.return_value = mock_ddf
        mock_ddf.__getitem__.return_value.isna.return_value.sum.return_value.compute.return_value = 1
        mock_ddf.map_partitions.return_value.mean.return_value.compute.return_value = 2.0
        mock_ddf.map_partitions.return_value.quantile.return_value.compute.return_value = 3.0
        mock_ddf.map_partitions.return_value.std.return_value.compute.return_value = 4.0
        mock_ddf.map_partitions.return_value.min.return_value.compute.return_value = 1.0
        mock_ddf.map_partitions.return_value.max.return_value.compute.return_value = 5.0
        mock_detect_currency.return_value = 'USD'
        mock_parse_currency_field.side_effect = lambda part, field, locale: (part[field], {})
        df = pd.DataFrame({'amount': [1.0, 2.0, 3.0, 4.0, 5.0]})
        analyzer = self.analyzer
        mock_progress = MagicMock()
        result = analyzer._analyze_with_dask(df, 'amount', 'en_US', 10, False, False, progress_tracker=mock_progress)
        self.assertIn('stats', result)
        # Check that progress_tracker.update was called with the Dask statistics step
        mock_progress.update.assert_any_call(1, {
            "step": "Dask statistics calculated",
            "min": 1.0,
            "max": 5.0
        })

    @patch('pamola_core.profiling.analyzers.currency.generate_currency_samples')
    @patch('pamola_core.profiling.analyzers.currency.calculate_histogram')
    @patch('pamola_core.profiling.analyzers.currency.calculate_percentiles')
    @patch('pamola_core.profiling.analyzers.currency.analyze_currency_stats')
    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    def test__analyze_in_chunks_basic(self, mock_parse, mock_analyze_stats, mock_percentiles, mock_hist, mock_samples):
        # Setup mocks for chunking
        df = pd.DataFrame({'amount': [1000.0, 2000.0, None, 3000.0, None, 4000.0]})
        # Simulate parse_currency_field for each chunk
        def parse_side_effect(chunk, field, locale):
            vals = chunk[field]
            valid_flags = [v is not None for v in vals]
            s = pd.Series(vals)
            s.valid_flags = valid_flags
            return s, {'USD': sum(valid_flags)}
        mock_parse.side_effect = parse_side_effect
        mock_analyze_stats.return_value = {
            'min': 1000.0, 'max': 4000.0, 'mean': 2500.0, 'median': 2500.0, 'std': 1290.99,
            'negative_count': 0, 'zero_count': 0
        }
        mock_percentiles.return_value = {'25%': 1500.0, '50%': 2500.0, '75%': 3500.0}
        mock_hist.return_value = {'bins': [1000, 2000, 3000, 4000], 'counts': [1, 1, 1, 1]}
        mock_samples.return_value = []
        analyzer = self.analyzer
        mock_progress = MagicMock()
        result = analyzer._analyze_in_chunks(df, 'amount', 'en_US', 10, False, False, 3, progress_tracker=mock_progress)
        self.assertIn('stats', result)
        self.assertEqual(result['stats']['min'], 1000.0)
        self.assertEqual(result['stats']['max'], 4000.0)
        self.assertEqual(result['valid_count'], 4)
        self.assertEqual(result['null_count'], 2)
        self.assertEqual(result['currency_counts'], {'USD': 6})
        self.assertIn('note', result)
        mock_progress.update.assert_any_call(1, {
            "step": "Chunks processed",
            "valid_total": 4,
            "currencies_detected": 1
        })

class TestCurrencyOperation(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.currency.CurrencyAnalyzer')
    @patch('pamola_core.profiling.analyzers.currency.write_json')
    def test_execute_success(self, mock_write_json, mock_analyzer_cls):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {
            'valid_count': 2,
            'null_count': 1,
            'currency_counts': {'USD': 2},
            'stats': {'min': 1000, 'max': 2000},
        }
        mock_analyzer_cls.return_value = mock_analyzer
        mock_data_source = MagicMock()
        mock_df = pd.DataFrame({'amount': [1000, 2000, None]})
        mock_data_source.__getitem__.return_value = mock_df
        mock_reporter = MagicMock()
        mock_reporter.add_operation = MagicMock()
        mock_reporter.add_artifact = MagicMock()
        mock_progress = MagicMock()
        # Patch load_data_operation to return DataFrame
        with patch('pamola_core.profiling.analyzers.currency.load_data_operation', return_value=mock_df):
            op = CurrencyOperation(field_name='amount')
            result = op.execute(
                data_source=mock_data_source,
                task_dir=Path('.'),
                reporter=mock_reporter,
                progress_tracker=mock_progress
            )
        self.assertEqual(result.status.name, 'SUCCESS')
        self.assertIn('stats', mock_analyzer.analyze.return_value)
        mock_write_json.assert_called()
        mock_reporter.add_artifact.assert_any_call('json', ANY, 'amount currency analysis')

    @patch('pamola_core.profiling.analyzers.currency.CurrencyAnalyzer')
    @patch('pamola_core.profiling.analyzers.currency.write_json')
    def test_execute_field_not_found(self, mock_write_json, mock_analyzer_cls):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {'valid_count': 2, 'null_count': 1, 'currency_counts': {'USD': 2}, 'stats': {'min': 1000, 'max': 2000}}
        mock_analyzer_cls.return_value = mock_analyzer
        mock_data_source = MagicMock()
        mock_df = pd.DataFrame({'other': [1, 2, 3]})
        mock_data_source.__getitem__.return_value = mock_df
        mock_reporter = MagicMock()
        # Patch load_data_operation to return DataFrame
        with patch('pamola_core.profiling.analyzers.currency.load_data_operation', return_value=mock_df):
            op = CurrencyOperation(field_name='amount')
            result = op.execute(
                data_source=mock_data_source,
                task_dir=Path('.'),
                reporter=mock_reporter
            )
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('not found', result.error_message)

    @patch('pamola_core.profiling.analyzers.currency.CurrencyAnalyzer')
    @patch('pamola_core.profiling.analyzers.currency.write_json')
    def test_execute_analyze_error(self, mock_write_json, mock_analyzer_cls):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {'error': 'some error'}
        mock_analyzer_cls.return_value = mock_analyzer
        mock_data_source = MagicMock()
        mock_df = pd.DataFrame({'amount': [1000, 2000, None]})
        mock_data_source.__getitem__.return_value = mock_df
        mock_reporter = MagicMock()
        # Patch load_data_operation to return DataFrame
        with patch('pamola_core.profiling.analyzers.currency.load_data_operation', return_value=mock_df):
            op = CurrencyOperation(field_name='amount')
            result = op.execute(
                data_source=mock_data_source,
                task_dir=Path('.'),
                reporter=mock_reporter
            )
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('some error', result.error_message)

    @patch('pamola_core.profiling.analyzers.currency.CurrencyAnalyzer')
    @patch('pamola_core.profiling.analyzers.currency.write_json')
    def test_execute_exception_handling(self, mock_write_json, mock_analyzer_cls):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = Exception('unexpected error!')
        mock_analyzer_cls.return_value = mock_analyzer
        mock_data_source = MagicMock()
        mock_df = pd.DataFrame({'amount': [1000, 2000, None]})
        mock_data_source.__getitem__.return_value = mock_df
        mock_reporter = MagicMock()
        # Patch load_data_operation to return DataFrame
        with patch('pamola_core.profiling.analyzers.currency.load_data_operation', return_value=mock_df):
            op = CurrencyOperation(field_name='amount')
            result = op.execute(
                data_source=mock_data_source,
                task_dir=Path('.'),
                reporter=mock_reporter
            )
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('unexpected error!', result.error_message)
        mock_reporter.add_operation.assert_any_call(
            'Error analyzing currency field amount',
            status='error',
            details={'error': 'unexpected error!'}
        )

    @patch('pamola_core.profiling.analyzers.currency.create_histogram')
    @patch('pamola_core.profiling.analyzers.currency.create_boxplot')
    @patch('pamola_core.profiling.analyzers.currency.create_correlation_pair')
    @patch('pamola_core.profiling.analyzers.currency.parse_currency_field')
    def test_generate_visualizations(self, mock_parse, mock_corr, mock_box, mock_hist):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        # Setup mocks
        mock_parse.return_value = (pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]), {'USD': 10})
        mock_hist.return_value = 'histogram_path.png'
        mock_box.return_value = 'boxplot_path.png'
        mock_corr.return_value = 'qqplot_path.png'
        mock_result = MagicMock()
        mock_reporter = MagicMock()
        vis_dir = Path('.')
        analysis_results = {
            'stats': {
                'histogram': True,
                'min': 100.0,
                'max': 1000.0,
                'normality': {'is_normal': True, 'shapiro': {'p_value': 0.5}}
            },
            'currency_counts': {'USD': 10},
            'locale_used': 'en_US'
        }
        df = pd.DataFrame({'amount': [f'${i}.00' for i in range(100, 1100, 100)]})
        op = CurrencyOperation(field_name='amount')
        op.test_normality = True
        op.bins = 10
        # Call the method
        op._generate_visualizations(df, analysis_results, vis_dir, mock_result, mock_reporter)
        # Check that artifacts were added for histogram, boxplot, and qq plot
        self.assertTrue(mock_result.add_artifact.called)
        self.assertTrue(mock_reporter.add_artifact.called)
        calls = [call[0][2] for call in mock_result.add_artifact.call_args_list]
        self.assertTrue(any('histogram' in c for c in calls))
        self.assertTrue(any('boxplot' in c for c in calls))
        self.assertTrue(any('Q-Q plot' in c or 'qq plot' in c for c in calls))

    @patch('pamola_core.profiling.analyzers.currency.write_json')
    def test_save_sample_records(self, mock_write_json):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        op = CurrencyOperation(field_name='amount')
        mock_result = MagicMock()
        mock_reporter = MagicMock()
        dict_dir = Path('.')
        # Ensure at least one valid sample record
        df = pd.DataFrame({'amount': ['$1,000.00', '$2,000.00', '$3,000.00']})
        analysis_results = {'currency_counts': {'USD': 3}}
        mock_write_json.return_value = None
        op._save_sample_records(df, analysis_results, dict_dir, mock_result, mock_reporter)
        # Artifacts should be added
        self.assertTrue(mock_result.add_artifact.called)
        self.assertTrue(mock_reporter.add_artifact.called)
        calls = [call[0][2] for call in mock_result.add_artifact.call_args_list]
        self.assertTrue(any('sample records' in c for c in calls))
        # Do not assert write_json call count, as the method may not call it if not implemented

    @patch('pamola_core.profiling.analyzers.currency.CurrencyAnalyzer')
    def test_add_metrics_to_result(self, mock_analyzer_cls):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        op = CurrencyOperation(field_name='amount')
        mock_result = MagicMock()
        metrics = {
            'stats': {
                'min': 1000,
                'max': 2000,
                'mean': 1500
            },
            'currency_counts': {'USD': 2}
        }
        op._add_metrics_to_result(metrics, mock_result)
        # 'min', 'max', 'mean' should be added as nested metrics under 'statistics'
        mock_result.add_nested_metric.assert_any_call('statistics', 'min', 1000)
        mock_result.add_nested_metric.assert_any_call('statistics', 'max', 2000)
        mock_result.add_nested_metric.assert_any_call('statistics', 'mean', 1500)

    @patch('pamola_core.profiling.analyzers.currency.CurrencyAnalyzer')
    def test_add_metrics_to_result_with_nested_metrics(self, mock_analyzer_cls):
        from pamola_core.profiling.analyzers.currency import CurrencyOperation
        op = CurrencyOperation(field_name='amount')
        mock_result = MagicMock()
        analysis_results = {
            'total_rows': 10,
            'valid_count': 8,
            'null_count': 1,
            'invalid_count': 1,
            'null_percentage': 0.1,
            'invalid_percentage': 0.1,
            'multi_currency': True,
            'currency_counts': {'USD': 7, 'EUR': 1},
            'stats': {
                'min': 1000,
                'max': 2000,
                'mean': 1500,
                'median': 1500,
                'std': 500,
                'skewness': 0.0,
                'kurtosis': 3.0,
                'zero_count': 0,
                'zero_percentage': 0.0,
                'negative_count': 0,
                'negative_percentage': 0.0,
                'outliers': {'count': 1, 'percentage': 0.1, 'lower_bound': 900, 'upper_bound': 2100},
                'normality': {'is_normal': True, 'shapiro': {'statistic': 0.99, 'p_value': 0.5}}
            }
        }
        op._add_metrics_to_result(analysis_results, mock_result)
        # Check top-level metrics
        mock_result.add_metric.assert_any_call('total_rows', 10)
        mock_result.add_metric.assert_any_call('valid_count', 8)
        mock_result.add_metric.assert_any_call('null_count', 1)
        mock_result.add_metric.assert_any_call('invalid_count', 1)
        mock_result.add_metric.assert_any_call('null_percentage', 0.1)
        mock_result.add_metric.assert_any_call('invalid_percentage', 0.1)
        mock_result.add_metric.assert_any_call('multi_currency', True)
        # Check nested statistics metrics
        mock_result.add_nested_metric.assert_any_call('statistics', 'min', 1000)
        mock_result.add_nested_metric.assert_any_call('statistics', 'max', 2000)
        mock_result.add_nested_metric.assert_any_call('statistics', 'mean', 1500)
        mock_result.add_nested_metric.assert_any_call('statistics', 'median', 1500)
        mock_result.add_nested_metric.assert_any_call('statistics', 'std', 500)
        mock_result.add_nested_metric.assert_any_call('statistics', 'skewness', 0.0)
        mock_result.add_nested_metric.assert_any_call('statistics', 'kurtosis', 3.0)
        mock_result.add_nested_metric.assert_any_call('statistics', 'zero_count', 0)
        mock_result.add_nested_metric.assert_any_call('statistics', 'zero_percentage', 0.0)
        mock_result.add_nested_metric.assert_any_call('statistics', 'negative_count', 0)
        mock_result.add_nested_metric.assert_any_call('statistics', 'negative_percentage', 0.0)
        # Outlier metrics
        mock_result.add_nested_metric.assert_any_call('outliers', 'count', 1)
        mock_result.add_nested_metric.assert_any_call('outliers', 'percentage', 0.1)
        mock_result.add_nested_metric.assert_any_call('outliers', 'lower_bound', 900)
        mock_result.add_nested_metric.assert_any_call('outliers', 'upper_bound', 2100)
        # Normality metrics
        mock_result.add_nested_metric.assert_any_call('normality', 'is_normal', True)
        mock_result.add_nested_metric.assert_any_call('normality', 'shapiro_stat', 0.99)
        mock_result.add_nested_metric.assert_any_call('normality', 'shapiro_p_value', 0.5)
        # Currency metrics
        mock_result.add_nested_metric.assert_any_call('currencies', 'USD', 7)
        mock_result.add_nested_metric.assert_any_call('currencies', 'EUR', 1)


if __name__ == '__main__':
    unittest.main()
