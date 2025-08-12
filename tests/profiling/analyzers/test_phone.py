import unittest
from unittest.mock import MagicMock, patch, call, ANY
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
    @patch('pamola_core.utils.progress.ProgressTracker')
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
    @patch('pamola_core.utils.progress.ProgressTracker')
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
    @patch('pamola_core.utils.progress.ProgressTracker')
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

    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_execute_no_dataframe(self, mock_load):
        mock_load.return_value = None
        op = phone.PhoneOperation('phone', use_cache=False)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('No valid DataFrame', result.error_message)

    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_execute_field_not_found(self, mock_load):
        mock_load.return_value = pd.DataFrame({'other': [1]})
        op = phone.PhoneOperation('phone', use_cache=False)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('not found', result.error_message)

    @patch('pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_execute_analysis_error(self, mock_load, mock_analyze):
        mock_load.return_value = self.df
        mock_analyze.return_value = {'error': 'fail'}
        op = phone.PhoneOperation('phone', use_cache=False)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('fail', result.error_message)

    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.analyzers.phone.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.profiling.analyzers.phone.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    def test_execute_success_return_result(
        self, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json, mock_get_filename
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
        mock_get_filename.side_effect = lambda *a, **k: (
            f"{k.get('base_name', a[0] if a else 'file')}.{k.get('extension', a[1] if len(a) > 1 else 'json')}"
        )
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

    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.analyzers.phone.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.profiling.analyzers.phone.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    def test_execute_with_normalization_metrics(
        self, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json, mock_get_filename
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
        mock_get_filename.side_effect = lambda *a, **k: (
            f"{k.get('base_name', a[0] if a else 'file')}.{k.get('extension', a[1] if len(a) > 1 else 'json')}"
        )
        mock_plot.return_value = "ok"
        mock_to_csv.return_value = None

        reporter = MagicMock()
        progress_tracker = MagicMock()

        op = phone.PhoneOperation(field_name=field_name, use_cache=False)
        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("test_task_dir"),
            reporter=reporter,
            progress_tracker=progress_tracker
        )

        # Assert normalization metrics are present
        self.assertIn("normalization_success_count", result.metrics)
        self.assertIn("normalization_success_percentage", result.metrics)
        self.assertEqual(result.metrics["normalization_success_count"], 2)
        self.assertEqual(result.metrics["normalization_success_percentage"], 100.0)

    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.analyzers.phone.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.profiling.analyzers.phone.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_execute_country_code_plot_error(
        self, mock_logger, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json, mock_get_filename
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
        mock_get_filename.side_effect = lambda *a, **k: (
            f"{k.get('base_name', a[0] if a else 'file')}.{k.get('extension', a[1] if len(a) > 1 else 'json')}"
        )
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

    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.write_json")
    @patch("pamola_core.profiling.analyzers.phone.load_data_operation")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_country_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_operator_code_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.PhoneAnalyzer.create_messenger_dictionary")
    @patch("pamola_core.profiling.analyzers.phone.pd.DataFrame.to_csv")
    @patch("pamola_core.profiling.analyzers.phone.ensure_directory")
    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_execute_exception_handling(
        self, mock_logger, mock_plot, mock_ensure_dir, mock_to_csv, mock_messenger_dict,
        mock_operator_dict, mock_country_dict, mock_analyze, mock_load_data,
        mock_write_json, mock_get_filename
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
        mock_get_filename.side_effect = lambda *a, **k: (
            f"{k.get('base_name', a[0] if a else 'file')}."
            f"{k.get('extension', a[1] if len(a) > 1 else 'json')}"
        )

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
        mock_logger.exception.assert_any_call(
            f"Error in phone operation for {field_name}: Disk write error"
        )
        # Assert progress_tracker.update is called with error
        progress_tracker.update.assert_any_call(0, {"step": "Error", "error": "Disk write error"})
        # Assert reporter.add_operation is called with status error
        reporter.add_operation.assert_any_call(
            f"Error analyzing {field_name}",
            status="error",
            details={"error": "Disk write error"}
        )
        # Assert the result is ERROR and error_message is correct
        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Disk write error", result.error_message)
        
    @patch.object(phone.PhoneOperation, '_check_cache')    
    @patch('pamola_core.profiling.analyzers.numeric.load_data_operation')
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

class TestAnalyzePhoneFields(unittest.TestCase):
    @patch('pamola_core.profiling.analyzers.phone.PhoneOperation.execute')
    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_analyze_phone_fields_success(self, mock_load_data, mock_execute):
        # Setup DataFrame with multiple phone fields
        df = pd.DataFrame({
            'phone1': ['0123456789', '0987654321'],
            'phone2': ['0123456788', '0987654320'],
            'other': [1, 2]
        })
        mock_load_data.return_value = df
        # Mock OperationResult
        from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)
        # Reporter mock
        reporter = MagicMock()
        # Import function under test
        from pamola_core.profiling.analyzers.phone import analyze_phone_fields
        # Run
        results = analyze_phone_fields(
            data_source=MagicMock(),
            task_dir=Path('test_task_dir'),
            reporter=reporter,
            phone_fields=['phone1', 'phone2']
        )
        # Assert
        self.assertIn('phone1', results)
        self.assertIn('phone2', results)
        self.assertEqual(results['phone1'].status.name, 'SUCCESS')
        self.assertEqual(results['phone2'].status.name, 'SUCCESS')
        self.assertTrue(reporter.add_operation.called)

    @patch('pamola_core.profiling.analyzers.phone.PhoneOperation.execute')
    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_analyze_phone_fields_no_dataframe(self, mock_load_data, mock_execute):
        mock_load_data.return_value = None
        reporter = MagicMock()
        from pamola_core.profiling.analyzers.phone import analyze_phone_fields
        results = analyze_phone_fields(
            data_source=MagicMock(),
            task_dir=Path('test_task_dir'),
            reporter=reporter,
            phone_fields=['phone1']
        )
        self.assertEqual(results, {})
        reporter.add_operation.assert_any_call('Phone fields analysis', status='error', details={'error': 'No valid DataFrame found in data source'})

    @patch('pamola_core.profiling.analyzers.phone.PhoneOperation.execute')
    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_analyze_phone_fields_auto_detect(self, mock_load_data, mock_execute):
        # DataFrame with phone-like columns
        df = pd.DataFrame({
            'home_phone': ['1'],
            'work_phone': ['2'],
            'cell_phone': ['3'],
            'other': [4]
        })
        mock_load_data.return_value = df
        from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
        mock_execute.return_value = OperationResult(status=OperationStatus.SUCCESS)
        reporter = MagicMock()
        from pamola_core.profiling.analyzers.phone import analyze_phone_fields
        results = analyze_phone_fields(
            data_source=MagicMock(),
            task_dir=Path('test_task_dir'),
            reporter=reporter,
            phone_fields=None
        )
        self.assertIn('home_phone', results)
        self.assertIn('work_phone', results)
        self.assertIn('cell_phone', results)
        self.assertEqual(results['home_phone'].status.name, 'SUCCESS')
        self.assertEqual(results['work_phone'].status.name, 'SUCCESS')
        self.assertEqual(results['cell_phone'].status.name, 'SUCCESS')

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
    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename", return_value="vis.png")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_success(self, mock_logger, mock_get_filename, mock_plot):
        mock_plot.return_value = "success"
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       True,
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_plot.assert_called()
        mock_get_filename.assert_called()

    @patch("pamola_core.utils.visualization.plot_value_distribution", return_value="Error: failed to plot")
    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename", return_value="vis.png")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_plot_error(self, mock_logger, mock_get_filename, mock_plot):
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       True,
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='plotly',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        mock_logger.warning.assert_any_call("Error creating country code visualization: Error: failed to plot")

    @patch("pamola_core.utils.visualization.plot_value_distribution")
    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_no_country_codes(self, mock_logger, mock_get_filename, mock_plot):
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
                                       True,
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
    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_no_operator_codes(self, mock_logger, mock_get_filename, mock_plot):
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
                                       True,
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
    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_no_messenger_mentions(self, mock_logger, mock_get_filename, mock_plot):
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
                                       True,
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
    @patch("pamola_core.profiling.analyzers.phone.get_timestamped_filename")
    @patch("pamola_core.profiling.analyzers.phone.logger")
    def test_handle_visualizations_backend_matplotlib(self, mock_logger, mock_get_filename, mock_plot):
        mock_plot.return_value = "success"
        self.op.generate_visualization = True
        self.op.visualization_backend = "plotly"
        result = self.op._handle_visualizations(self.analysis_results,
                                       self.task_dir,
                                       True,
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
                                       True,
                                       self.result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='matplotlib',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress_tracker)
        self.assertEqual(result, {})

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
        mock_cache_key.return_value = 'cache_key'
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out, None)
        
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_cache(self, mock_cache_key, mock_operation_cache, mock_get_cache):
        mock_cache_key.return_value = 'cache_key'
        mock_get_cache.return_value = {
            'artifacts': [],
            'analysis_results': {}
        }
        mock_operation_cache.return_value = {'main': 'vis_path.png'}
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out.status, phone.OperationStatus.SUCCESS)
    
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_cache_normalization_success_count(self, mock_cache_key, mock_operation_cache, mock_get_cache):
        mock_cache_key.return_value = 'cache_key'
        mock_get_cache.return_value = {
            'analysis_results': {
                'normalization_success_count': 1
            }
        }
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out.status, phone.OperationStatus.SUCCESS)
        self.assertEqual(out.metrics['normalization_success_count'], 1)
                        
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key')
    def test_cache_exception(self, mock_cache_key, mock_operation_cache, mock_get_cache):
        mock_cache_key.return_value = 'cache_key'
        mock_get_cache.side_effect = Exception("Cache Exception")
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
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

    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    @patch('pamola_core.utils.ops.op_cache.OperationCache.save_cache')
    def test_save_to_cache_success(self, mock_save_cache, mock_cache_key, mock_operation_cache):
        result = self.op._save_to_cache(self.df, self.analysis_results, self.artifacts, self.task_dir)
        self.assertTrue(result)
        mock_cache_key.assert_called
        
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    @patch('pamola_core.utils.ops.op_cache.OperationCache.save_cache')
    def test_save_to_cache_false(self, mock_save_cache, mock_cache_key, mock_operation_cache):
        mock_save_cache.return_value = False
        result = self.op._save_to_cache(self.df, self.analysis_results, self.artifacts, self.task_dir)
        self.assertFalse(result)
        mock_save_cache.assert_called

    @patch('pamola_core.utils.ops.op_cache.operation_cache', side_effect=Exception('Cache write error'))
    @patch.object(phone.PhoneOperation, '_generate_cache_key', side_effect=Exception('Cache write error'))
    @patch('pamola_core.utils.ops.op_cache.OperationCache.save_cache')
    @patch('pamola_core.profiling.analyzers.phone.logger')
    def test_save_to_cache_exception(self, mock_logger, mock_save_cache, mock_cache_key, mock_operation_cache):
        result = self.op._save_to_cache(self.df, self.analysis_results, self.artifacts, self.task_dir)
        self.assertFalse(result)

    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_partial_data(self, mock_cache_key, mock_operation_cache):
        # Missing artifacts and metrics
        result = self.op._save_to_cache(self.df, self.analysis_results, None, self.task_dir)
        mock_cache_key.assert_called()

    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(phone.PhoneOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_empty_analysis_results(self, mock_cache_key, mock_operation_cache):
        # Empty analysis_results
        self.op._save_to_cache(self.df, {}, self.artifacts, self.task_dir)
        mock_cache_key.assert_called()
        
if __name__ == "__main__":
    unittest.main()