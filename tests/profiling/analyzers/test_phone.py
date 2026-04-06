import unittest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
from pathlib import Path
import numpy as np

from pamola_core.profiling.analyzers import phone
from pamola_core.utils.ops.op_result import OperationStatus, OperationResult

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

    def apply_data_types(self, df, dataset_name=None, **kwargs):
        return df
    
class TestPhoneAnalyzer(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.phone.analyze_phone_field')
    def test_analyze(self, mock_analyze):
        mock_analyze.return_value = {'result': 1}
        df = pd.DataFrame({'phone': ['123']})
        result = phone.PhoneAnalyzer.analyze(df, 'phone')
        self.assertEqual(result, {'result': 1})
        mock_analyze.assert_called_once()

    @patch('pamola_core.profiling.analyzers.phone.create_country_code_dictionary')
    def test_create_country_code_dictionary(self, mock_func):
        mock_func.return_value = {'country_codes': [{'code': '84', 'count': 2}]}
        df = pd.DataFrame({'phone': ['+84123']})
        result = phone.PhoneAnalyzer.create_country_code_dictionary(df, 'phone')
        self.assertIn('country_codes', result)

    @patch('pamola_core.profiling.analyzers.phone.create_operator_code_dictionary')
    def test_create_operator_code_dictionary(self, mock_func):
        mock_func.return_value = {'operator_codes': [{'code': '123', 'count': 1}]}
        df = pd.DataFrame({'phone': ['+84123']})
        result = phone.PhoneAnalyzer.create_operator_code_dictionary(df, 'phone', country_code='84')
        self.assertIn('operator_codes', result)

    @patch('pamola_core.profiling.analyzers.phone.create_messenger_dictionary')
    def test_create_messenger_dictionary(self, mock_func):
        mock_func.return_value = {'messengers': [{'name': 'Zalo', 'count': 1}]}
        df = pd.DataFrame({'phone': ['+84123']})
        result = phone.PhoneAnalyzer.create_messenger_dictionary(df, 'phone')
        self.assertIn('messengers', result)

    @patch('pamola_core.profiling.analyzers.phone.estimate_resources')
    def test_estimate_resources(self, mock_func):
        mock_func.return_value = {'memory': 100}
        df = pd.DataFrame({'phone': ['+84123']})
        result = phone.PhoneAnalyzer.estimate_resources(df, 'phone')
        self.assertIn('memory', result)
        
    @patch('pamola_core.profiling.analyzers.phone.analyze_phone_field_with_dask', return_value={'min':1,'max':5,'mean':3,'zero_count':0,'zero_percentage':0})
    @patch('pamola_core.utils.progress.HierarchicalProgressTracker')
    def test_analyze_large_df_use_dask(self, mock_tracker, mock_handle):
        df = pd.DataFrame({'phone': np.arange(20000)})
        result = phone.PhoneAnalyzer.analyze(df, 'phone', use_dask=True)
        self.assertIn('max', result)
        self.assertIn('min', result)
        self.assertIn('mean', result)
        self.assertIn('zero_count', result)
        self.assertIn('zero_percentage', result)
        mock_handle.assert_called_once()
        
    @patch('pamola_core.profiling.analyzers.phone.analyze_phone_field_with_joblib', return_value={'min':1,'max':5,'mean':3,'zero_count':0,'zero_percentage':0})
    @patch('pamola_core.utils.progress.HierarchicalProgressTracker')
    def test_analyze_large_df_use_vectorization(self, mock_tracker, mock_handle):
        df = pd.DataFrame({'phone': np.arange(20000)})
        result = phone.PhoneAnalyzer.analyze(df, 'phone', use_vectorization=True)
        self.assertIn('max', result)
        self.assertIn('min', result)
        self.assertIn('mean', result)
        self.assertIn('zero_count', result)
        self.assertIn('zero_percentage', result)
        mock_handle.assert_called_once()
        
    @patch('pamola_core.profiling.analyzers.phone.analyze_phone_field_with_chunk', return_value={'min':1,'max':5,'mean':3,'zero_count':0,'zero_percentage':0})
    @patch('pamola_core.utils.progress.HierarchicalProgressTracker')
    def test_analyze_large_df_chunk(self, mock_tracker, mock_handle):
        df = pd.DataFrame({'phone': np.arange(20000)})
        result = phone.PhoneAnalyzer.analyze(df, 'phone', chunk_size=10000)
        self.assertIn('max', result)
        self.assertIn('min', result)
        self.assertIn('mean', result)
        self.assertIn('zero_count', result)
        self.assertIn('zero_percentage', result)
        mock_handle.assert_called_once()

class TestPhoneOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'phone': ['+84123', None]})
        self.data_source = DummyDataSource(df=self.df)
        self.data_source.__class__.__name__ = 'DataSource'
        self.task_dir = Path('test_task_dir')
        self.reporter = MagicMock()
        self.progress = MagicMock()
        self.progress.total = 0

    @patch('pamola_core.profiling.commons.helpers.load_data_operation')
    def test_execute_no_dataframe(self, mock_load):
        mock_load.return_value = None
        op = phone.PhoneOperation('phone', use_cache=False)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertTrue(len(result.error_message) > 0)

    @patch('pamola_core.profiling.commons.helpers.load_data_operation')
    def test_execute_field_not_found(self, mock_load):
        mock_load.return_value = pd.DataFrame({'other': [1]})
        op = phone.PhoneOperation('phone', use_cache=False)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('not found', result.error_message)

    @patch('pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze')
    @patch('pamola_core.profiling.commons.helpers.load_data_operation')
    def test_execute_analysis_error(self, mock_load, mock_analyze):
        mock_load.return_value = self.df
        mock_analyze.return_value = {'error': 'fail'}
        op = phone.PhoneOperation('phone', use_cache=False)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('fail', result.error_message)

    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.utils.ops.op_base.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    def test_execute_success_return_result(
        self, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json
    ):
        # Setup
        field_name = "phone"
        df = pd.DataFrame({field_name: ["0123456789", "0987654321"]})
        mock_load_data.return_value = df
        mock_analyze.return_value = {
            "total_rows": 2,
            "null_count": 0,
            "null_percentage": 0.0,
            "valid_count": 2,
            "valid_percentage": 100.0,
            "format_error_count": 0,
            "has_comment_count": 0,
            "country_codes": [{"code": "84", "count": 2}],
            "operator_codes": [{"operator": "Viettel", "count": 1}, {"operator": "Vinaphone", "count": 1}],
            "messenger_mentions": {"Zalo": 1, "Telegram": 0}
        }
        mock_country_dict.return_value = {"country_codes": [{"code": "84", "count": 2}]}
        mock_operator_dict.return_value = {"operator_codes": [{"operator": "Viettel", "count": 1}]}
        mock_messenger_dict.return_value = {"messengers": [{"messenger": "Zalo", "count": 1}]}
        mock_plot.return_value = "ok"
        mock_to_csv.return_value = None

        # Reporter and progress tracker mocks
        reporter = MagicMock()
        progress_tracker = MagicMock()

        # Run
        op = phone.PhoneOperation(field_name=field_name, use_cache=False)
        result = op.execute(
            data_source=self.data_source,
            task_dir=Path("test_task_dir"),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Assert
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertGreaterEqual(len(result.artifacts), 1)
        self.assertIn("total_records", result.metrics)
        reporter.add_operation.assert_any_call(ANY, details=ANY)
        reporter.add_artifact.assert_any_call("json", ANY, ANY)
        progress_tracker.update.assert_any_call(1, ANY)

    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.utils.ops.op_base.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    def test_execute_with_normalization_metrics(
        self, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json
    ):
        # Setup
        field_name = "phone"
        df = pd.DataFrame({field_name: ["0123456789", "0987654321"]})
        mock_load_data.return_value = df
        mock_analyze.return_value = {
            "total_rows": 2,
            "null_count": 0,
            "null_percentage": 0.0,
            "valid_count": 2,
            "valid_percentage": 100.0,
            "format_error_count": 0,
            "has_comment_count": 0,
            "country_codes": [{"code": "84", "count": 2}],
            "operator_codes": [{"operator": "Viettel", "count": 1}],
            "messenger_mentions": {"Zalo": 1},
            "normalization_success_count": 2,
            "normalization_success_percentage": 100.0
        }
        mock_country_dict.return_value = {"country_codes": [{"code": "84", "count": 2}]}
        mock_operator_dict.return_value = {"operator_codes": [{"operator": "Viettel", "count": 1}]}
        mock_messenger_dict.return_value = {"messengers": [{"messenger": "Zalo", "count": 1}]}
        mock_plot.return_value = "ok"
        mock_to_csv.return_value = None

        reporter = MagicMock()
        progress_tracker = MagicMock()

        # Configure data source mock
        mock_data_source = MagicMock()
        mock_data_source.apply_data_types.side_effect = lambda df, *args, **kwargs: df

        op = phone.PhoneOperation(field_name=field_name, use_cache=False)
        result = op.execute(
            data_source=mock_data_source,
            task_dir=Path("test_task_dir"),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Assert result status is SUCCESS
        if result.status != OperationStatus.SUCCESS:
            self.fail(f"Expected SUCCESS status, got {result.status}. Error: {result.error_message}")

        # Assert normalization metrics are present
        self.assertIn("normalization_success_count", result.metrics)
        self.assertIn("normalization_success_percentage", result.metrics)
        self.assertEqual(result.metrics["normalization_success_count"], 2)
        self.assertEqual(result.metrics["normalization_success_percentage"], 100.0)

    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.utils.ops.op_base.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_execute_country_code_plot_error(
        self, mock_logger, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json
    ):
        # Setup
        field_name = "phone"
        df = pd.DataFrame({field_name: ["0123456789", "0987654321"]})
        mock_load_data.return_value = df
        mock_analyze.return_value = {
            "total_rows": 2,
            "null_count": 0,
            "null_percentage": 0.0,
            "valid_count": 2,
            "valid_percentage": 100.0,
            "format_error_count": 0,
            "has_comment_count": 0,
            "country_codes": [{"code": "84", "count": 2}],
            "operator_codes": [],
            "messenger_mentions": {}
        }
        mock_country_dict.return_value = {"country_codes": [{"code": "84", "count": 2}]}
        mock_operator_dict.return_value = {"operator_codes": []}
        mock_messenger_dict.return_value = {"messengers": []}
        mock_plot.return_value = "Error: failed to plot"
        mock_to_csv.return_value = None

        reporter = MagicMock()
        progress_tracker = MagicMock()

        op = phone.PhoneOperation(field_name=field_name, use_cache=False)
        op.generate_visualization = True
        op.visualization_backend = 'plotly'
        result = op.execute(
            data_source=self.data_source,
            task_dir=Path("test_task_dir"),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Assert logger.warning was called with the error message
        mock_logger.warning.assert_any_call("Error creating country code visualization: Error: failed to plot")

    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.utils.ops.op_base.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_execute_exception_handling(
        self, mock_logger, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json
    ):
        # Setup
        field_name = "phone"
        df = pd.DataFrame({field_name: ["0123456789", "0987654321"]})
        mock_load_data.return_value = df
        mock_analyze.return_value = {
            "total_rows": 2,
            "null_count": 0,
            "null_percentage": 0.0,
            "valid_count": 2,
            "valid_percentage": 100.0,
            "format_error_count": 0,
            "has_comment_count": 0,
            "country_codes": [{"code": "84", "count": 2}],
            "operator_codes": [],
            "messenger_mentions": {}
        }
        mock_country_dict.return_value = {"country_codes": [{"code": "84", "count": 2}]}
        mock_operator_dict.return_value = {"operator_codes": []}
        mock_messenger_dict.return_value = {"messengers": []}
        mock_plot.return_value = "ok"
        mock_to_csv.return_value = None

        # Cause an error when writing file
        mock_write_json.side_effect = Exception("Disk write error")

        reporter = MagicMock()
        progress_tracker = MagicMock()

        op = phone.PhoneOperation(field_name=field_name, use_cache=False)
        result = op.execute(
            data_source=self.data_source,
            task_dir=Path("test_task_dir"),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Assert logger.exception is called
        mock_logger.exception.assert_called()
        # Assert the result is ERROR and error_message is correct
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Disk write error", result.error_message)
        
    @patch.object(phone.PhoneOperation, '_check_cache')    
    @patch('pamola_core.profiling.commons.helpers.load_data_operation')
    def test_execute_use_cache(self, mock_load, mock_cache):
        field_name = "phone"
        df = pd.DataFrame({field_name: ["0123456789", "0987654321"]})
        mock_load.return_value = df
        mock_cache.return_value = OperationResult(
                            status=phone.OperationStatus.SUCCESS
                        )
        op = phone.PhoneOperation(field_name=field_name, use_cache=True)
        result = op.execute(self.data_source, self.task_dir, self.reporter, self.progress)
        self.assertEqual(result.status, phone.OperationStatus.SUCCESS)

class TestHandleVisualizations(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.task_dir = Path('test_vis_dir')
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()
        self.analysis_results = {
            'stats': {
                'histogram': {'bins': [0, 1], 'counts': [1, 2]},
                'min': 1, 'max': 10, 'normality': {'is_normal': True, 'shapiro': {'p_value': 0.5}}
            },
            'country_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'operator_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'messenger_mentions':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        }
        self.result = phone.OperationResult(status=phone.OperationStatus.SUCCESS)
        self.op = phone.PhoneOperation(field_name='phone', use_cache=False)
        self.op.metrics = {"country_codes": [{"code": "84", "count": 2}]}
        self.op.field_name = "phone"
        
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_success(self, mock_logger, mock_plot):
        mock_plot.return_value = "success"
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_plot.assert_called()

    @patch("pamola_core.utils.visualization.plot_value_distribution", return_value="Error: failed to plot")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_plot_error(self, mock_logger, mock_plot):
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_logger.warning.assert_any_call("Error creating country code visualization: Error: failed to plot")

    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_no_country_codes(self, mock_logger, mock_plot):
        mock_plot.return_value = 'Error'
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        analysis_results = {
            'operator_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'messenger_mentions':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        }
        result = self.op._handle_visualizations(analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_plot.assert_called()
        assert ('Error creating country code visualization',) not in [call.args for call in mock_logger.warning.call_args_list]
    
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_no_operator_codes(self, mock_logger, mock_plot):
        mock_plot.return_value = 'Error'
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        analysis_results = {
            'country_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'operator_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'messenger_mentions':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        }
        result = self.op._handle_visualizations(analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_plot.assert_called()
        assert ('Error creating operator code visualization',) not in [call.args for call in mock_logger.warning.call_args_list]
    
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_no_messenger_mentions(self, mock_logger, mock_plot):
        mock_plot.return_value = 'Error'
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        analysis_results = {
            'country_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            'operator_codes':{
                'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        }
        result = self.op._handle_visualizations(analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_plot.assert_called()
        assert ('Error creating messenger mentions visualization',) not in [call.args for call in mock_logger.warning.call_args_list]

    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_backend_matplotlib(self, mock_logger, mock_plot):
        mock_plot.return_value = "success"
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='matplotlib',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_plot.assert_called()
        mock_logger.warning.assert_not_called()

    @patch('threading.Thread')
    @patch('pamola_core.utils.visualization.plot_value_distribution')
    def test_handle_visualizations_timeout(self, mock_generate, mock_thread):
        class DummyThread:
            def __init__(self): self._alive = True
            def start(self): pass
            def join(self, timeout=None): pass
            def is_alive(self): return True
            @property
            def daemon(self): return False
        mock_thread.return_value = DummyThread()
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='matplotlib',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        # When timeout occurs, visualization_paths is reset to empty list
        self.assertEqual(result, [])

class TestCheckCache(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'phone': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.task_dir = Path('test_vis_dir')
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()
        self.op = phone.PhoneOperation('phone', use_cache=True)
        
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_no_cache(self, mock_cache_key, mock_operation_cache):
        # _check_cache(df) — one param only
        mock_cache_key.return_value = 'cache_key'
        out = self.op._check_cache(self.df)
        self.assertEqual(out, None)

    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_cache(self, mock_cache_key):
        # _check_cache(df) — one param; set operation_cache directly on instance
        mock_cache_key.return_value = 'cache_key'
        mock_op_cache = MagicMock()
        mock_op_cache.get_cache.return_value = {
            'status': 'SUCCESS',
            'metrics': {},
            'error_message': None,
            'execution_time': 1.0,
            'error_trace': None,
            'artifacts': []
        }
        self.op.operation_cache = mock_op_cache
        out = self.op._check_cache(self.df)
        self.assertEqual(out.status, phone.OperationStatus.SUCCESS)

    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_cache_normalization_success_count(self, mock_cache_key):
        # _check_cache(df) — one param; metrics from cached result
        mock_cache_key.return_value = 'cache_key'
        mock_op_cache = MagicMock()
        mock_op_cache.get_cache.return_value = {
            'status': 'SUCCESS',
            'metrics': {'normalization_success_count': 1},
            'error_message': None,
            'execution_time': 1.0,
            'error_trace': None,
            'artifacts': []
        }
        self.op.operation_cache = mock_op_cache
        out = self.op._check_cache(self.df)
        self.assertEqual(out.status, phone.OperationStatus.SUCCESS)
        self.assertEqual(out.metrics['normalization_success_count'], 1)

    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_cache_exception(self, mock_cache_key):
        # _check_cache(df) — one param
        mock_cache_key.return_value = 'cache_key'
        mock_op_cache = MagicMock()
        mock_op_cache.get_cache.side_effect = Exception("Cache Exception")
        self.op.operation_cache = mock_op_cache
        out = self.op._check_cache(self.df)
        self.assertEqual(out, None)

class TestSaveToCache(unittest.TestCase):
    def setUp(self):
        from pamola_core.profiling.analyzers.phone import PhoneOperation
        self.op = PhoneOperation(field_name='phone', use_cache=True)
        self.df = pd.DataFrame({'phone': [1, 2, 3]})
        self.task_dir = Path('test_vis_dir')
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()
        self.analysis_results = {'country_codes': [{'code': '84', 'count': 2}]}
        self.artifacts = {'main': 'vis_path.png'}
        self.metrics = {'total': 3}

    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_success(self, mock_cache_key):
        # _save_to_cache(df, analysis_results, result, task_dir) — result is OperationResult
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = True
        self.op.operation_cache = mock_op_cache
        result = self.op._save_to_cache(self.df, self.analysis_results, MagicMock(), self.task_dir)
        self.assertTrue(result)
        mock_cache_key.assert_called()

    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_false(self, mock_cache_key):
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = False
        self.op.operation_cache = mock_op_cache
        result = self.op._save_to_cache(self.df, self.analysis_results, MagicMock(), self.task_dir)
        self.assertFalse(result)
        mock_op_cache.save_cache.assert_called()

    @patch.object(phone.PhoneOperation, '_generate_cache_key', side_effect=Exception('Cache write error'))
    def test_save_to_cache_exception(self, mock_cache_key):
        result = self.op._save_to_cache(self.df, self.analysis_results, MagicMock(), self.task_dir)
        self.assertFalse(result)

    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_partial_data(self, mock_cache_key):
        # result param as MagicMock — should still attempt to save
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = True
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        self.op._save_to_cache(self.df, self.analysis_results, MagicMock(), self.task_dir)
        mock_cache_key.assert_called()

    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_empty_analysis_results(self, mock_cache_key):
        # Empty analysis_results
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = True
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        self.op._save_to_cache(self.df, {}, MagicMock(), self.task_dir)
        mock_cache_key.assert_called()
        
if __name__ == "__main__":
    unittest.main()