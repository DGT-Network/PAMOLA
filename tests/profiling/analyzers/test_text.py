import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation
from pamola_core.profiling.commons.text_utils import extract_text_and_ids, find_dictionary_file, analyze_language
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
    
class TestTextSemanticCategorizerOperation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'test_field': ['+84123', None]})
        self.data_source = DummyDataSource(df=self.df)
        self.data_source.__class__.__name__ = 'DataSource'
        self.task_dir = Path('test_task_dir')
        self.reporter = MagicMock()
        self.progress = MagicMock()
        self.op = TextSemanticCategorizerOperation(field_name="test_field", entity_type="job", use_cache=False)

    def test_init_sets_attributes(self):
        self.assertEqual(self.op.field_name, "test_field")
        self.assertEqual(self.op.entity_type, "job")
        self.assertTrue(self.op.perform_categorization)

    def test_extract_text_and_ids_with_id_field(self):
        # extract_text_and_ids is a module-level function in text_utils
        df = pd.DataFrame({"test_field": ["a", "b"], "id": [1, 2]})
        texts, ids = extract_text_and_ids(df, "test_field", "id")
        self.assertEqual(texts, ["a", "b"])
        self.assertEqual(ids, ["1", "2"])

    def test_extract_text_and_ids_without_id_field(self):
        # extract_text_and_ids is a module-level function in text_utils
        df = pd.DataFrame({"test_field": ["a", "b"]})
        texts, ids = extract_text_and_ids(df, "test_field", None)
        self.assertEqual(texts, ["a", "b"])
        self.assertEqual(ids, ["0", "1"])

    @patch("pamola_core.utils.ops.op_base.ensure_directory")
    def test_prepare_directories(self, mock_ensure):
        # ensure_directory is called in op_base._prepare_directories (inherited)
        task_dir = Path("/tmp/task")
        dirs = self.op._prepare_directories(task_dir)
        self.assertIn("output", dirs)
        self.assertIn("dictionaries", dirs)
        self.assertIn("visualizations", dirs)
        self.assertIn("cache", dirs)
        mock_ensure.assert_called()

    def test_initialize_categorization_results(self):
        text_values = ["a", "b", ""]
        result = self.op._initialize_categorization_results(text_values)
        self.assertEqual(result["summary"]["total_texts"], 2)
        self.assertEqual(result["summary"]["num_matched"], 0)
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)

    def test_load_cache_uses_check_cache(self):
        # _load_cache does not exist; _check_cache(df) is the actual method
        # Verify _check_cache returns None when use_cache=False (default)
        out = self.op._check_cache(self.df)
        self.assertIsNone(out)

    def test_save_cache_uses_save_to_cache(self):
        # _save_cache does not exist; _save_to_cache(df, result, task_dir) is the actual method
        # When use_cache=False (default), should return False
        result = MagicMock()
        res = self.op._save_to_cache(self.df, result, Path("/tmp/task"))
        self.assertFalse(res)

    def test_get_cache_parameters(self):
        # _prepare_execution_parameters does not exist; verify _get_cache_parameters works
        params = self.op._get_cache_parameters()
        self.assertIn("field_name", params)
        self.assertIn("entity_type", params)

    def test_generate_cache_key(self):
        # _get_cache_key does not exist; _generate_cache_key(df) is the actual method
        # Patch the underlying cache key generation
        with patch.object(self.op, 'operation_cache') as mock_cache:
            mock_cache.generate_cache_key.return_value = "text_semantic_test_field_job_abc123"
            key = self.op._generate_cache_key(self.df)
            self.assertIsInstance(key, str)

    def test_analyze_language_module_function(self):
        # analyze_language is a module-level function in text_utils, not an instance method
        # Patch it at its source location
        with patch("pamola_core.profiling.commons.text_utils.analyze_language",
                   return_value={"predominant_language": "en", "language_distribution": {"en": 2, "fr": 1}}) as mock_detect:
            res = analyze_language(["a", "b", "c"])
            self.assertEqual(res["predominant_language"], "en")

    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch("pamola_core.profiling.analyzers.text.analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field")
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)

    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch("pamola_core.profiling.analyzers.text.analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis_use_dask(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field", use_dask=True)
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)

    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch("pamola_core.profiling.analyzers.text.analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis_use_vectorization(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field", use_vectorization=True)
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)

    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch("pamola_core.profiling.analyzers.text.analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis_chunk_size(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field", chunk_size=1)
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)

    @patch("pamola_core.profiling.commons.text_utils.logger")
    @patch("pamola_core.profiling.commons.text_utils.Path.exists", return_value=True)
    def test_find_dictionary_file_explicit(self, mock_exists, mock_logger):
        # find_dictionary_file is a module-level function in text_utils
        # signature: (dictionary_path, entity_type, dictionaries_dir, task_logger=None)
        path = find_dictionary_file("/tmp/dict.json", "job", Path("/tmp/task/dictionaries"))
        self.assertEqual(path, Path("/tmp/dict.json"))

    @patch("pamola_core.profiling.commons.text_utils.logger")
    @patch("pamola_core.profiling.commons.text_utils.Path.exists", return_value=True)
    def test_find_dictionary_file_task_dir(self, mock_exists, mock_logger):
        # find_dictionary_file is a module-level function; None dictionary_path falls back to task dir
        path = find_dictionary_file(None, "job", Path("/tmp/task/dictionaries"))
        self.assertIsNotNone(path)
        self.assertEqual(path, Path("/tmp/task/dictionaries/job.json"))

    @patch("pamola_core.profiling.commons.text_utils.logger")
    @patch("pamola_core.profiling.commons.text_utils.Path.exists", side_effect=[False, False, True])
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"data_repository": "/repo"}')
    def test_find_dictionary_file_global(self, mock_open, mock_exists, mock_logger):
        with patch("json.load", return_value={"data_repository": "/repo"}):
            path = find_dictionary_file(None, "job", Path("/tmp/task/dictionaries"))
            self.assertIsNone(path)  # repo_dict_path.exists() returns False for the 3rd side_effect

    def test_compile_analysis_results(self):
        basic = {"null_empty_analysis": {"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}},
                 "length_stats": {"mean": 1, "max": 2},
                 "language_analysis": {"predominant_language": "en"}}
        cat = {"summary": {"num_matched": 1, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 1, "percentage_matched": 50},
               "categorization": [], "category_distribution": {}, "aliases_distribution": {}, "hierarchy_analysis": {}, "unresolved": []}
        res = self.op._compile_analysis_results(basic, cat, "test_field")
        self.assertIn("metrics", res)
        self.assertIn("field_name", res)

    def test_add_metrics_to_result(self):
        class DummyResult:
            def __init__(self):
                self.metrics = {}
            def add_metric(self, k, v):
                self.metrics[k] = v
        analysis_results = {
            "metrics": {"a": 1, "b": 2},
            "null_empty_analysis": {"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}},
            "length_stats": {"mean": 1, "max": 2},
            "language_analysis": {"predominant_language": "en"},
            "match_summary": {"num_matched": 1, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 1, "percentage_matched": 50}
        }
        dummy = DummyResult()
        self.op._add_metrics_to_result(analysis_results, dummy)
        self.assertEqual(dummy.metrics["a"], 1)
        self.assertEqual(dummy.metrics["b"], 2)

    def test_add_metrics_to_result_no_metrics(self):
        class DummyResult:
            def __init__(self):
                self.metrics = {}
            def add_metric(self, k, v):
                self.metrics[k] = v
        analysis_results = {
            "null_empty_analysis": {"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}},
            "length_stats": {"mean": 1, "max": 2},
            "language_analysis": {"predominant_language": "en"},
            "match_summary": {"num_matched": 1, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 1, "percentage_matched": 50}
        }
        dummy = DummyResult()
        self.op._add_metrics_to_result(analysis_results, dummy)
        self.assertEqual(dummy.metrics["total_records"], 2)
        self.assertEqual(dummy.metrics["num_matched"], 1)

    def test_create_and_register_artifact(self):
        class DummyResult:
            def __init__(self):
                self.artifacts = []
            def add_artifact(self, **kwargs):
                self.artifacts.append(kwargs)
        class DummyReporter:
            def __init__(self):
                self.artifacts = []
            def add_artifact(self, *args):
                self.artifacts.append(args)
        dummy_result = DummyResult()
        dummy_reporter = DummyReporter()
        self.op._create_and_register_artifact("json", Path("/tmp/file.json"), "desc", dummy_result, dummy_reporter, category="cat")
        self.assertEqual(dummy_result.artifacts[0]["artifact_type"], "json")
        self.assertEqual(dummy_reporter.artifacts[0][0], "json")

    @patch("pamola_core.profiling.analyzers.text.create_pie_chart", return_value=True)
    def test_create_visualization(self, mock_pie):
        mock_pie.__name__ = "create_pie_chart"
        # _create_visualization takes operation_timestamp (not include_timestamp)
        class DummyResult:
            def add_artifact(self, **kwargs):
                self.kwargs = kwargs
        class DummyReporter:
            def add_artifact(self, *args):
                self.args = args
        dummy_result = DummyResult()
        dummy_reporter = DummyReporter()
        self.op._create_visualization(
            data={"a": 1},
            vis_func=mock_pie,
            filename="file",
            title="title",
            output_dir=Path("/tmp"),
            operation_timestamp="20240101_000000",
            result=dummy_result,
            reporter=dummy_reporter,
            description="desc",
            vis_theme=None,
            vis_backend="matplotlib",
            vis_strict=False,
            additional_params={"show_percentages": True}
        )
        self.assertEqual(dummy_result.kwargs["artifact_type"], "png")
        self.assertEqual(dummy_reporter.args[0], "png")

    @patch("pamola_core.profiling.analyzers.text.create_pie_chart", return_value="Error: fail")
    def test_create_visualization_error(self, mock_pie):
        mock_pie.__name__ = "create_pie_chart"
        # _create_visualization takes operation_timestamp (not include_timestamp)
        class DummyResult:
            def add_artifact(self, **kwargs):
                self.kwargs = kwargs
        class DummyReporter:
            def add_artifact(self, *args):
                self.args = args
        dummy_result = DummyResult()
        dummy_reporter = DummyReporter()
        self.op._create_visualization(
            data={"a": 1},
            vis_func=mock_pie,
            filename="file",
            title="title",
            output_dir=Path("/tmp"),
            operation_timestamp="20240101_000000",
            result=dummy_result,
            reporter=dummy_reporter,
            description="desc",
            vis_theme=None,
            vis_backend="matplotlib",
            vis_strict=False,
        )
        self.assertFalse(hasattr(dummy_result, "kwargs"))

    @patch("pamola_core.profiling.analyzers.text.create_pie_chart", return_value=True)
    @patch("pamola_core.profiling.analyzers.text.create_bar_plot", return_value=True)
    @patch("pamola_core.profiling.analyzers.text.plot_text_length_distribution", return_value=True)
    def test_generate_visualizations(self, mock_plot, mock_bar, mock_pie):
        mock_plot.__name__ = "plot_text_length_distribution"
        mock_bar.__name__ = "create_bar_plot"
        mock_pie.__name__ = "create_pie_chart"
        # _generate_visualizations takes operation_timestamp (not include_timestamp)
        class DummyResult:
            def add_artifact(self, **kwargs):
                self.kwargs = kwargs
        class DummyReporter:
            def add_artifact(self, *args):
                self.args = args
        dummy_result = DummyResult()
        dummy_reporter = DummyReporter()
        analysis_results = {
            "category_distribution": {"a": 1},
            "aliases_distribution": {"b": 2},
            "length_stats": {"length_distribution": [1, 2]}
        }
        self.op._generate_visualizations(
            analysis_results,
            Path("/tmp/vis"),
            "20240101_000000",
            dummy_result,
            dummy_reporter,
            vis_theme=None,
            vis_backend="matplotlib",
            vis_strict=False,
        )
        self.assertEqual(dummy_result.kwargs["artifact_type"], "png")

    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    def test_execute_progress_and_none_df(self, mock_load):
        # Setup
        mock_load.return_value = None
        data_source = MagicMock()
        task_dir = Path("/tmp/task")
        reporter = MagicMock()
        progress_tracker = MagicMock()
        # Call execute
        res = self.op.execute(
            data_source=data_source,
            task_dir=task_dir,
            reporter=reporter,
            progress_tracker=progress_tracker,
            dataset_name="main"
        )
        # Check progress_tracker.update was called
        progress_tracker.update.assert_called_with(1, {"step": "Initialization", "field": self.op.field_name})
        # Check OperationResult returned for None df (error result)
        self.assertTrue(len(res.error_message) > 0)

    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    def test_execute_field_not_in_df(self, mock_load):
        # DataFrame does not have the required field
        mock_load.return_value = pd.DataFrame({"other_field": [1, 2]})
        data_source = MagicMock()
        task_dir = Path("/tmp/task")
        reporter = MagicMock()
        progress_tracker = MagicMock()
        # Call execute
        res = self.op.execute(
            data_source=data_source,
            task_dir=task_dir,
            reporter=reporter,
            progress_tracker=progress_tracker,
            dataset_name="main"
        )
        # Check OperationResult returned for missing field is an error
        self.assertEqual(res.status, OperationStatus.ERROR)
        # reporter.add_operation should not be called (field check fails before it)
        reporter.add_operation.assert_not_called()

    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    def test_execute_progress_tracker_update(self, mock_load):
        # DataFrame with correct field
        mock_load.return_value = pd.DataFrame({"test_field": [1, 2]})
        data_source = MagicMock()
        task_dir = Path("/tmp/task")
        reporter = MagicMock()
        progress_tracker = MagicMock()
        # Call execute
        self.op.execute(
            data_source=data_source,
            task_dir=task_dir,
            reporter=reporter,
            progress_tracker=progress_tracker,
            dataset_name="main"
        )
        # Check progress_tracker.update was called for Initialization
        progress_tracker.update.assert_any_call(1, {"step": "Initialization", "field": self.op.field_name})

    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch.object(TextSemanticCategorizerOperation, "_generate_visualizations")
    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    def test_execute_uses_cache(
        self, mock_load_data, mock_generate_vis, mock_logger
    ):
        # _load_cache does not exist; _check_cache(df) is the actual API
        df = pd.DataFrame({"test_field": ["a", "b"]})
        mock_load_data.return_value = df
        with patch.object(self.op, "_check_cache", return_value=OperationResult(status=OperationStatus.SUCCESS)), \
             patch.object(self.op, "_prepare_directories", return_value={
                "output": Path("/tmp/task/output"),
                "dictionaries": Path("/tmp/task/dictionaries"),
                "visualizations": Path("/tmp/task/visualizations"),
                "cache": Path("/tmp/task/cache")
             }):
            self.op.use_cache = True
            result = self.op.execute(
                data_source=self.data_source,
                task_dir=self.task_dir,
                reporter=self.reporter,
                progress_tracker=self.progress
            )
            self.assertEqual(result.status, OperationStatus.SUCCESS)

    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch.object(TextSemanticCategorizerOperation, "_generate_visualizations")
    @patch("pamola_core.profiling.commons.helpers.load_data_operation")
    def test_execute_visualization(
        self, mock_load_data, mock_generate_vis, mock_logger
    ):
        # _load_cache does not exist; _check_cache(df) is the actual API
        df = pd.DataFrame({"test_field": ["a", "b"]})
        mock_load_data.return_value = df
        with patch.object(self.op, "_check_cache", return_value=OperationResult(status=OperationStatus.SUCCESS)), \
             patch.object(self.op, "_prepare_directories", return_value={
                "output": Path("/tmp/task/output"),
                "dictionaries": Path("/tmp/task/dictionaries"),
                "visualizations": Path("/tmp/task/visualizations"),
                "cache": Path("/tmp/task/cache")
             }):
            self.op.generate_visualization = True
            self.op.visualization_backend = 'plotly'
            result = self.op.execute(
                data_source=self.data_source,
                task_dir=self.task_dir,
                reporter=self.reporter,
                progress_tracker=self.progress
            )
            self.assertEqual(result.status, OperationStatus.SUCCESS)
    
    @patch("pamola_core.profiling.analyzers.text.create_entity_extractor")
    def test_perform_semantic_categorization_basic(self, mock_create_extractor):
        mock_extractor = MagicMock()
        mock_extractor.extract_entities.return_value = {
            "entities": [
                {"text": "engineer", "record_id": "0", "category": "STEM", "match_method": "dictionary", "confidence": 1.0},
                {"text": "doctor", "record_id": "1", "category": "Healthcare", "match_method": "dictionary", "confidence": 1.0},
            ],
            "unresolved": [{"text": "", "record_id": "2"}, {"text": None, "record_id": "3"}]
        }
        mock_create_extractor.return_value = mock_extractor
        text_values = ["engineer", "doctor", "", None]
        record_ids = ["0", "1", "2", "3"]
        result = self.op._perform_semantic_categorization(
            text_values, record_ids, Path("/tmp/dict.json"),
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertIn("summary", result)
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)
        self.assertEqual(len(result["categorization"]), 2)

    @patch("pamola_core.profiling.analyzers.text.read_json", create=True)
    @patch("pamola_core.profiling.analyzers.text.logger")
    def test_perform_semantic_categorization_empty(self, mock_logger, mock_read_json):
        text_values = []
        record_ids = []
        mock_read_json.return_value = {}
        dictionary_path = Path("/tmp/dict.json")
        result = self.op._perform_semantic_categorization(
            text_values, record_ids, dictionary_path,
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertEqual(result["summary"]["total_texts"], 0)
        self.assertEqual(result["summary"]["num_matched"], 0)
        self.assertEqual(result["categorization"], [])
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["category_distribution"], {})

    @patch("pamola_core.profiling.analyzers.text.write_json")
    @patch("pamola_core.utils.ops.op_base.ensure_directory")
    @patch("pamola_core.profiling.analyzers.text.write_dataframe_to_csv")
    def test_save_categorization_artifacts(self, mock_write_csv, mock_ensure_dir, mock_write_json):
        categorization_results = {
            "categorization": [{"record_id": "1", "matched_alias": "eng", "matched_category": "A", "matched_domain": "D", "method": "exact"}],
            "unresolved": [{"record_id": "2", "text": "unknown"}],
            "summary": {"total_texts": 2, "num_matched": 1},
            "category_distribution": {"A": 1},
            "aliases_distribution": {},
            "hierarchy_analysis": {}
        }
        record_ids = ["1", "2"]
        text_values = ["engineer", "unknown"]
        dirs = {"dictionaries": Path("/tmp/dictionaries")}
        result = MagicMock()
        reporter = MagicMock()
        self.op._save_categorization_artifacts(
            categorization_results, record_ids, text_values, "20240101_000000", dirs, result, reporter
        )
        self.assertTrue(mock_write_json.called)
        self.assertTrue(mock_write_csv.called)
        self.assertTrue(result.add_artifact.called)
        self.assertTrue(reporter.add_artifact.called)

    def test_save_main_artifacts(self):
        # _save_main_artifacts uses datetime.now().strftime directly (no get_timestamped_filename)
        with patch("pamola_core.profiling.analyzers.text.write_json") as mock_write_json, \
             patch.object(TextSemanticCategorizerOperation, "_create_and_register_artifact") as mock_create_artifact:
            analysis_results = {"a": 1}
            dirs = {"output": Path("/tmp/output")}
            result = MagicMock()
            reporter = MagicMock()
            self.op._save_main_artifacts(analysis_results, dirs, "20240101_000000", result, reporter)
            # Check that the json file is written correctly
            mock_write_json.assert_called_once()
            args, kwargs = mock_write_json.call_args
            self.assertEqual(args[0], analysis_results)
            self.assertIn("output", str(args[1]))
            # Check that the artifact is registered correctly
            mock_create_artifact.assert_called_once()
            artifact_args, artifact_kwargs = mock_create_artifact.call_args
            self.assertEqual(artifact_kwargs["artifact_type"], "json")
            self.assertIn("Semantic analysis", artifact_kwargs["description"])
            self.assertEqual(artifact_kwargs["result"], result)
            self.assertEqual(artifact_kwargs["reporter"], reporter)

    def test_categorize_texts_in_chunks_basic(self):
        text_values = ["engineer", "doctor", "unknown", ""]
        record_ids = ["0", "1", "2", "3"]
        mock_result = {
            "categorization": [], "unresolved": [],
            "summary": {"total_texts": 4, "num_matched": 0, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 0},
            "category_distribution": {}, "aliases_distribution": {}, "hierarchy_analysis": {},
        }
        with patch.object(self.op, '_perform_semantic_categorization', return_value=mock_result):
            result = self.op._categorize_texts_in_chunks(
                text_values, record_ids, Path("/tmp/dict.json"),
                language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8
            )
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)
        self.assertIn("summary", result)
        self.assertIn("category_distribution", result)
        self.assertEqual(len(result["unresolved"]), 0)
        
    def test_categorize_texts_in_chunks_use_dask(self):
        mock_result = {
            "categorization": [], "unresolved": [],
            "summary": {"total_texts": 4, "num_matched": 0, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 0},
            "category_distribution": {}, "aliases_distribution": {}, "hierarchy_analysis": {},
        }
        with patch.object(self.op, '_perform_semantic_categorization', return_value=mock_result):
            result = self.op._categorize_texts_in_chunks(
                ["engineer", "doctor", "unknown", ""], ["0", "1", "2", "3"],
                Path("/tmp/dict.json"), language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8,
                use_dask=True
            )
        self.assertIn("unresolved", result)
        self.assertEqual(len(result["unresolved"]), 0)
            
    def test_categorize_texts_in_chunks_use_vectorization(self):
        mock_result = {
            "categorization": [], "unresolved": [],
            "summary": {"total_texts": 4, "num_matched": 0, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 0},
            "category_distribution": {}, "aliases_distribution": {}, "hierarchy_analysis": {},
        }
        with patch.object(self.op, '_perform_semantic_categorization', return_value=mock_result):
            result = self.op._categorize_texts_in_chunks(
                ["engineer", "doctor", "unknown", ""], ["0", "1", "2", "3"],
                Path("/tmp/dict.json"), language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8,
                use_vectorization=True
            )
        self.assertIn("unresolved", result)
        self.assertEqual(len(result["unresolved"]), 0)
            
    def test_categorize_texts_in_chunks_chunk_size(self):
        mock_result = {
            "categorization": [], "unresolved": [],
            "summary": {"total_texts": 2, "num_matched": 0, "num_ner_matched": 0, "num_auto_clustered": 0, "num_unresolved": 0},
            "category_distribution": {}, "aliases_distribution": {}, "hierarchy_analysis": {},
        }
        with patch.object(self.op, '_perform_semantic_categorization', return_value=mock_result):
            result = self.op._categorize_texts_in_chunks(
                ["engineer", "doctor", "unknown", ""], ["0", "1", "2", "3"],
                Path("/tmp/dict.json"), language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8,
                chunk_size=2
            )
        self.assertIn("unresolved", result)
        self.assertEqual(len(result["unresolved"]), 0)

    @patch("pamola_core.profiling.analyzers.text.create_entity_extractor")
    def test_perform_semantic_categorization_full(self, mock_create_extractor):
        # Setup mock extractor that returns matched entities
        mock_extractor = MagicMock()
        mock_extractor.extract_entities.return_value = {
            "entities": [
                {"text": "engineer", "record_id": "0", "category": "STEM", "match_method": "dictionary", "confidence": 1.0},
                {"text": "doctor", "record_id": "1", "category": "Healthcare", "match_method": "dictionary", "confidence": 1.0},
            ],
            "unresolved": [
                {"text": "unknown", "record_id": "2"},
            ]
        }
        mock_create_extractor.return_value = mock_extractor

        text_values = ["engineer", "doctor", "unknown", ""]
        record_ids = ["0", "1", "2", "3"]
        result = self.op._perform_semantic_categorization(
            text_values, record_ids, Path("/tmp/dict.json"),
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertIn("summary", result)
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)
        self.assertEqual(len(result["categorization"]), 2)

        # Case with empty input
        mock_extractor.extract_entities.return_value = {"entities": [], "unresolved": []}
        result = self.op._perform_semantic_categorization(
            [], [], Path("/tmp/dict.json"),
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertEqual(result["summary"]["total_texts"], 0)
        self.assertEqual(result["categorization"], [])
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["category_distribution"], {})

    @patch("pamola_core.profiling.analyzers.text.read_json", create=True)
    def test_perform_semantic_categorization_with_matches(self, mock_read_json):
        # Patch create_entity_extractor to return a mock extractor
        with patch("pamola_core.profiling.analyzers.text.create_entity_extractor") as mock_factory:
            mock_extractor = MagicMock()
            # Simulate extraction_results with matches
            extraction_results = {
                "entities": [
                    {"match_method": "dictionary", "record_id": "1", "matched_category": "A", "matched_alias": "a_alias"},
                    {"match_method": "ner", "record_id": "2", "matched_category": "B", "matched_alias": "b_alias"},
                    {"match_method": "other", "record_id": "3", "matched_category": "C", "matched_alias": "c_alias"}
                ],
                "unresolved": []
            }
            mock_extractor.extract_entities.return_value = extraction_results
            mock_factory.return_value = mock_extractor

            text_values = ["foo", "bar", "baz"]
            record_ids = ["1", "2", "3"]
            dictionary_path = None
            result = self.op._perform_semantic_categorization(
                text_values, record_ids, dictionary_path,
                language="en", match_strategy="exact", use_ner=True, perform_clustering=False, clustering_threshold=0.8
            )
            # Check that categorization contains all matches
            self.assertEqual(len(result["categorization"]), 3)
            self.assertEqual(result["summary"]["num_matched"], 1)  # Only method==dictionary
            self.assertEqual(result["summary"]["num_ner_matched"], 1)  # Only method==ner
            self.assertEqual(result["summary"]["num_auto_clustered"], 0)
            self.assertEqual(result["summary"]["num_unresolved"], 0)
            self.assertEqual(result["category_distribution"], {"A": 1, "B": 1, "C": 1})
            self.assertEqual(result["aliases_distribution"], {"a_alias": 1, "b_alias": 1, "c_alias": 1})
            self.assertListEqual(result["unresolved"], [])

            self.assertGreaterEqual(result["summary"]["num_ner_matched"], 0)

    @patch("pamola_core.profiling.commons.helpers.load_data_operation", return_value="not_a_dataframe")
    def test_execute_with_invalid_input(self, mock_load):
        # Test with invalid input (not a DataFrame) — causes AttributeError on .columns access
        data_source = MagicMock()
        task_dir = Path("/tmp/task")
        reporter = MagicMock()
        progress_tracker = MagicMock()
        res = self.op.execute(
            data_source=data_source,
            task_dir=task_dir,
            reporter=reporter,
            progress_tracker=progress_tracker,
            dataset_name="main"
        )
        # Should return an error OperationResult
        self.assertEqual(res.status, OperationStatus.ERROR)
        self.assertTrue(len(res.error_message) > 0)
        progress_tracker.update.assert_called()
            
    @patch('threading.Thread')
    def test_handle_visualizations_timeout(self, mock_thread):
        class DummyThread:
            def __init__(self): self._alive = True
            def start(self): pass
            def join(self, timeout=None): pass
            def is_alive(self): return True
            @property
            def daemon(self): return False
        mock_thread.return_value = DummyThread()
        analysis_results = {
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
        result = OperationResult(status=OperationStatus.SUCCESS)
        result = self.op._handle_visualizations(analysis_results,
                                       self.task_dir,
                                       "20240101_000000",
                                       result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='matplotlib',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress)
        self.assertEqual(result, [])
        
        
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_no_cache(self, mock_cache_key, mock_operation_cache):
        # _check_cache(df) — only one param (not reporter, task_dir)
        mock_cache_key.return_value = 'cache_key'
        out = self.op._check_cache(self.df)
        self.assertEqual(out, None)
        
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_cache(self, mock_cache_key):
        # _check_cache(df) — only one param; set operation_cache directly
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
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        out = self.op._check_cache(self.df)
        self.assertEqual(out.status, OperationStatus.SUCCESS)

    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_cache_match_summary(self, mock_cache_key):
        # _check_cache(df) — only one param; set operation_cache directly
        mock_cache_key.return_value = 'cache_key'
        mock_op_cache = MagicMock()
        mock_op_cache.get_cache.return_value = {
            'status': 'SUCCESS',
            'metrics': {'num_matched': 1, 'num_ner_matched': 2, 'num_auto_clustered': 1},
            'error_message': None,
            'execution_time': 1.0,
            'error_trace': None,
            'artifacts': []
        }
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        out = self.op._check_cache(self.df)
        self.assertEqual(out.status, OperationStatus.SUCCESS)

    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_cache_exception(self, mock_cache_key):
        # _check_cache(df) — only one param
        mock_cache_key.return_value = 'cache_key'
        mock_op_cache = MagicMock()
        mock_op_cache.get_cache.side_effect = Exception("Cache Exception")
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        out = self.op._check_cache(self.df)
        self.assertEqual(out, None)

    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_success(self, mock_cache_key):
        # _save_to_cache(df, result, task_dir) — 3 params (not 6)
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = True
        result_obj = MagicMock()
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        result = self.op._save_to_cache(self.df, result_obj, self.task_dir)
        self.assertTrue(result)
        mock_cache_key.assert_called()

    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_false(self, mock_cache_key):
        # _save_to_cache returns False when save_cache returns False
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = False
        result_obj = MagicMock()
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        result = self.op._save_to_cache(self.df, result_obj, self.task_dir)
        self.assertFalse(result)
        mock_op_cache.save_cache.assert_called()

    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', side_effect=Exception('Cache write error'))
    def test_save_to_cache_exception(self, mock_cache_key):
        # _save_to_cache returns False on exception
        result_obj = MagicMock()
        self.op.use_cache = True
        result = self.op._save_to_cache(self.df, result_obj, self.task_dir)
        self.assertFalse(result)

    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_empty_analysis_results(self, mock_cache_key):
        # _save_to_cache with empty result — should attempt to save
        mock_op_cache = MagicMock()
        mock_op_cache.save_cache.return_value = True
        self.op.use_cache = True
        self.op.operation_cache = mock_op_cache
        result_obj = MagicMock()
        result = self.op._save_to_cache(self.df, result_obj, self.task_dir)
        mock_cache_key.assert_called()

if __name__ == "__main__":
    unittest.main()