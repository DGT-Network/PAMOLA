import unittest
from unittest.mock import MagicMock, patch, call, ANY
import pandas as pd
from pathlib import Path

from pamola_core.profiling.analyzers import phone
from pamola_core.utils.ops.op_result import OperationStatus

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

class TestPhoneOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'phone': ['+84123', None]})
        self.data_source = MagicMock()
        self.data_source.__class__.__name__ = 'DataSource'
        self.task_dir = Path('test_task_dir')
        self.reporter = MagicMock()
        self.progress = MagicMock()
        self.progress.total = 0

    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_execute_no_dataframe(self, mock_load):
        mock_load.return_value = None
        op = phone.PhoneOperation('phone')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('No valid DataFrame', result.error_message)

    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_execute_field_not_found(self, mock_load):
        mock_load.return_value = pd.DataFrame({'other': [1]})
        op = phone.PhoneOperation('phone')
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        self.assertEqual(result.status.name, 'ERROR')
        self.assertIn('not found', result.error_message)

    @patch('pamola_core.profiling.analyzers.phone.PhoneAnalyzer.analyze')
    @patch('pamola_core.profiling.analyzers.phone.load_data_operation')
    def test_execute_analysis_error(self, mock_load, mock_analyze):
        mock_load.return_value = self.df
        mock_analyze.return_value = {'error': 'fail'}
        op = phone.PhoneOperation('phone')
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
        op = phone.PhoneOperation(field_name=field_name)
        result = op.execute(
            data_source=MagicMock(),
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

        op = phone.PhoneOperation(field_name=field_name)
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

        op = phone.PhoneOperation(field_name=field_name)
        result = op.execute(
            data_source=MagicMock(),
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

        op = phone.PhoneOperation(field_name=field_name)
        result = op.execute(
            data_source=MagicMock(),
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

