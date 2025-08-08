import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.text import TextSemanticCategorizerOperation
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
        df = pd.DataFrame({"test_field": ["a", "b"], "id": [1, 2]})
        texts, ids = self.op._extract_text_and_ids(df, "test_field", "id")
        self.assertEqual(texts, ["a", "b"])
        self.assertEqual(ids, ["1", "2"])

    def test_extract_text_and_ids_without_id_field(self):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        texts, ids = self.op._extract_text_and_ids(df, "test_field", None)
        self.assertEqual(texts, ["a", "b"])
        self.assertEqual(ids, ["0", "1"])

    @patch("pamola_core.profiling.analyzers.text.ensure_directory")
    def test_prepare_directories(self, mock_ensure):
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

    @patch("pamola_core.profiling.analyzers.text.read_json")
    @patch("pamola_core.profiling.analyzers.text.Path.exists", return_value=True)
    def test_load_cache_success(self, mock_exists, mock_read_json):
        mock_read_json.return_value = {"result": 1}
        cache_key = "abc"
        cache_dir = Path("/tmp/cache")
        result = self.op._load_cache(cache_key, cache_dir)
        self.assertEqual(result, {"result": 1})

    @patch("pamola_core.profiling.analyzers.text.Path.exists", return_value=False)
    def test_load_cache_file_not_exists(self, mock_exists):
        cache_key = "abc"
        cache_dir = Path("/tmp/cache")
        result = self.op._load_cache(cache_key, cache_dir)
        self.assertIsNone(result)

    @patch("pamola_core.profiling.analyzers.text.write_json")
    @patch("pamola_core.profiling.analyzers.text.ensure_directory")
    def test_save_cache_success(self, mock_ensure, mock_write_json):
        cache_key = "abc"
        cache_dir = Path("/tmp/cache")
        res = self.op._save_cache({"a": 1}, cache_key, cache_dir)
        self.assertTrue(res)
        mock_write_json.assert_called()

    def test_save_cache_no_cache_dir(self):
        res = self.op._save_cache({"a": 1}, "abc", None)
        self.assertFalse(res)

    def test_prepare_execution_parameters(self):
        task_dir = Path("/tmp/task")
        dirs = {"cache": Path("/tmp/task/cache")}
        kwargs = {"min_word_length": 5, "use_cache": True}
        params = self.op._prepare_execution_parameters(kwargs, task_dir, dirs)
        self.assertEqual(params["min_word_length"], 5)
        self.assertTrue(params["use_cache"])
        self.assertEqual(params["cache_dir"], Path("/tmp/task/cache"))

    def test_get_cache_key(self):
        class DummyDS:
            def get_identifier(self):
                return "id123"
        ds = DummyDS()
        key = self.op._get_cache_key(ds, Path("/tmp/task"))
        self.assertIn("test_field", key)
        self.assertIn("job", key)
        self.assertIn("text_semantic_", key)

    @patch("pamola_core.profiling.analyzers.text.detect_languages", return_value={"en": 2, "fr": 1})
    def test_analyze_language(self, mock_detect):
        res = self.op._analyze_language(["a", "b", "c"])
        self.assertEqual(res["predominant_language"], "en")
        self.assertEqual(res["language_distribution"], {"en": 2, "fr": 1})

    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch.object(TextSemanticCategorizerOperation, "_analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field")
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)
        
    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch.object(TextSemanticCategorizerOperation, "_analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis_use_dask(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field", use_dask=True)
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)
        
    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch.object(TextSemanticCategorizerOperation, "_analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis_use_vectorization(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field", use_vectorization=True)
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)
        
    @patch("pamola_core.profiling.analyzers.text.analyze_null_and_empty", return_value={"total_records": 2, "null_values": {"percentage": 0}, "empty_strings": {"percentage": 0}})
    @patch("pamola_core.profiling.analyzers.text.calculate_length_stats", return_value={"mean": 1, "max": 2, "length_distribution": [1,2]})
    @patch.object(TextSemanticCategorizerOperation, "_analyze_language", return_value={"predominant_language": "en", "language_distribution": {"en": 2}})
    def test_perform_basic_analysis_chunk_size(self, mock_lang, mock_len, mock_null):
        df = pd.DataFrame({"test_field": ["a", "b"]})
        res = self.op._perform_basic_analysis(df, "test_field", chunk_size=1)
        self.assertIn("null_empty_analysis", res)
        self.assertIn("language_analysis", res)
        self.assertIn("length_stats", res)

    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch("pamola_core.profiling.analyzers.text.Path.exists", return_value=True)
    def test_find_dictionary_file_explicit(self, mock_exists, mock_logger):
        self.op.dictionary_path = "/tmp/dict.json"
        path = self.op._find_dictionary_file("job", Path("/tmp/task"), Path("/tmp/task/dictionaries"))
        self.assertEqual(path, Path("/tmp/dict.json"))

    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch("pamola_core.profiling.analyzers.text.Path.exists", return_value=True)
    def test_find_dictionary_file_task_dir(self, mock_exists, mock_logger):
        self.op.dictionary_path = None
        path = self.op._find_dictionary_file("job", Path("/tmp/task"), Path("/tmp/task/dictionaries"))
        self.assertIsNotNone(path)
        self.assertEqual(path, Path("/tmp/task/dictionaries/job.json"))

    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch("pamola_core.profiling.analyzers.text.Path.exists", side_effect=[False, False, True])
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"data_repository": "/repo"}')
    def test_find_dictionary_file_global(self, mock_open, mock_exists, mock_logger):
        self.op.dictionary_path = None
        with patch("json.load", return_value={"data_repository": "/repo"}):
            path = self.op._find_dictionary_file("job", Path("/tmp/task"), Path("/tmp/task/dictionaries"))
            self.assertIsNone(path)  # Because repo_dict_path.exists() is not checked further

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

    @patch("pamola_core.profiling.analyzers.text.get_timestamped_filename", return_value="file.png")
    @patch("pamola_core.profiling.analyzers.text.create_pie_chart", return_value=True)
    def test_create_visualization(self, mock_pie, mock_get):
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
            include_timestamp=True,
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
    @patch("pamola_core.profiling.analyzers.text.get_timestamped_filename", return_value="file.png")
    def test_create_visualization_error(self, mock_get, mock_pie):
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
            include_timestamp=True,
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
    @patch("pamola_core.profiling.analyzers.text.get_timestamped_filename", return_value="file.png")
    def test_generate_visualizations(self, mock_get, mock_plot, mock_bar, mock_pie):
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
            True,
            dummy_result,
            dummy_reporter,
            vis_theme=None,
            vis_backend="matplotlib",
            vis_strict=False,
        )
        self.assertEqual(dummy_result.kwargs["artifact_type"], "png")

    @patch("pamola_core.profiling.analyzers.text.load_data_operation")
    @patch("pamola_core.profiling.analyzers.text.OperationResult")
    @patch("pamola_core.profiling.analyzers.text.OperationStatus")
    def test_execute_progress_and_none_df(self, mock_status, mock_result, mock_load):
        # Setup
        mock_status.ERROR = "ERROR"
        mock_result.return_value = "error_result"
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
        # Check OperationResult returned for None df
        self.assertEqual(res, "error_result")

    @patch("pamola_core.profiling.analyzers.text.load_data_operation")
    @patch("pamola_core.profiling.analyzers.text.OperationResult")
    @patch("pamola_core.profiling.analyzers.text.OperationStatus")
    def test_execute_field_not_in_df(self, mock_status, mock_result, mock_load):
        # Setup
        mock_status.ERROR = "ERROR"
        mock_result.return_value = "error_result"
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
        # Check OperationResult returned for missing field
        self.assertEqual(res, "error_result")
        # reporter.add_operation should not be called
        reporter.add_operation.assert_not_called()

    @patch("pamola_core.profiling.analyzers.text.load_data_operation")
    @patch("pamola_core.profiling.analyzers.text.OperationResult")
    @patch("pamola_core.profiling.analyzers.text.OperationStatus")
    def test_execute_progress_tracker_update(self, mock_status, mock_result, mock_load):
        # Setup
        mock_status.ERROR = "ERROR"
        mock_result.return_value = "error_result"
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
    @patch.object(TextSemanticCategorizerOperation, "_load_cache")
    @patch("pamola_core.profiling.analyzers.text.load_data_operation")
    def test_execute_uses_cache(
        self, mock_load_data, mock_load_cache, mock_generate_vis, mock_logger
    ):
        # Setup
        df = pd.DataFrame({"test_field": ["a", "b"]})
        mock_load_data.return_value = df
        cached_result = {"metrics": {"m1": 1, "m2": 2}}
        mock_load_cache.return_value = cached_result
        # Patch _get_cache_key to avoid hashing
        with patch.object(self.op, "_check_cache", return_value=OperationResult(status=OperationStatus.SUCCESS)), \
             patch.object(self.op, "_get_cache_key", return_value="cache_key"), \
             patch.object(self.op, "cache_dir", Path("/tmp/task/cache")), \
             patch.object(self.op, "_prepare_directories", return_value={
                "output": Path("/tmp/task/output"),
                "dictionaries": Path("/tmp/task/dictionaries"),
                "visualizations": Path("/tmp/task/visualizations"),
                "cache": Path("/tmp/task/cache")
             }), \
            patch.object(self.op, "_prepare_execution_parameters", return_value={"include_timestamp": True, "chunk_size": 10000}):
            self.op.use_cache = True
            result = self.op.execute(
                data_source=self.data_source,
                task_dir=self.task_dir,
                reporter=self.reporter,
                progress_tracker=self.progress
            )
            self.assertEqual(result.status, OperationStatus.SUCCESS)
        # logger.info is called
        mock_logger.info.assert_any_call(f"Using cached results for {self.op.field_name}")
        # progress_tracker.update is called with step Loaded from cache
        self.progress.update.assert_any_call(5, {"step": "Loaded from cache", "field": self.op.field_name})

    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch.object(TextSemanticCategorizerOperation, "_generate_visualizations")
    @patch.object(TextSemanticCategorizerOperation, "_load_cache")
    @patch("pamola_core.profiling.analyzers.text.load_data_operation")
    def test_execute_visualization(
        self, mock_load_data, mock_load_cache, mock_generate_vis, mock_logger
    ):
        # Setup
        df = pd.DataFrame({"test_field": ["a", "b"]})
        mock_load_data.return_value = df
        cached_result = {"metrics": {"m1": 1, "m2": 2}}
        # Patch _get_cache_key to avoid hashing
        with patch.object(self.op, "_check_cache", return_value=OperationResult(status=OperationStatus.SUCCESS)), \
             patch.object(self.op, "_get_cache_key", return_value="cache_key"), \
             patch.object(self.op, "cache_dir", Path("/tmp/task/cache")), \
             patch.object(self.op, "_prepare_directories", return_value={
                "output": Path("/tmp/task/output"),
                "dictionaries": Path("/tmp/task/dictionaries"),
                "visualizations": Path("/tmp/task/visualizations"),
                "cache": Path("/tmp/task/cache")
             }), \
            patch.object(self.op, "_prepare_execution_parameters",
                         return_value={"include_timestamp": True, "chunk_size": 10000, "perform_categorization": None}):
            self.op.generate_visualization = True
            self.op.visualization_backend = 'plotly'
            result = self.op.execute(
                data_source=self.data_source,
                task_dir=self.task_dir,
                reporter=self.reporter,
                progress_tracker=self.progress
            )
            self.assertEqual(result.status, OperationStatus.SUCCESS)
    
    @patch("pamola_core.profiling.analyzers.text.logger")
    @patch("pamola_core.profiling.analyzers.text.read_json")
    def test_perform_semantic_categorization_basic(self, mock_read_json, mock_logger):
        # Prepare text values and record ids
        text_values = ["engineer", "doctor", "", None]
        record_ids = ["0", "1", "2", "3"]
        # Mock dictionary file and loading
        dictionary = {"engineer": "STEM", "doctor": "Healthcare"}
        mock_read_json.return_value = dictionary
        dictionary_path = Path("/tmp/dict.json")
        # Call the function
        result = self.op._perform_semantic_categorization(
            text_values, record_ids, dictionary_path,
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertIn("summary", result)
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)
        self.assertEqual(result["summary"]["num_matched"], 0)
        self.assertEqual(len(result["categorization"]), 0)
        self.assertEqual(len(result["unresolved"]), 0)

    @patch("pamola_core.profiling.analyzers.text.read_json")
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
    @patch("pamola_core.profiling.analyzers.text.ensure_directory")
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
            categorization_results, record_ids, text_values, dirs, result, reporter
        )
        self.assertTrue(mock_write_json.called)
        self.assertTrue(mock_write_csv.called)
        self.assertTrue(result.add_artifact.called)
        self.assertTrue(reporter.add_artifact.called)

    def test_save_main_artifacts(self):
        with patch("pamola_core.profiling.analyzers.text.write_json") as mock_write_json, \
             patch("pamola_core.profiling.analyzers.text.get_timestamped_filename", return_value="test_file.json") as mock_get_filename, \
             patch.object(TextSemanticCategorizerOperation, "_create_and_register_artifact") as mock_create_artifact:
            analysis_results = {"a": 1}
            dirs = {"output": Path("/tmp/output")}
            include_timestamp = True
            result = MagicMock()
            reporter = MagicMock()
            self.op._save_main_artifacts(analysis_results, dirs, include_timestamp, result, reporter)
            # Check that the json file is written correctly
            mock_write_json.assert_called_once()
            args, kwargs = mock_write_json.call_args
            self.assertEqual(args[0], analysis_results)
            self.assertTrue(str(args[1]).endswith("test_file.json"))
            # Check that the artifact is registered correctly
            mock_create_artifact.assert_called_once()
            artifact_args, artifact_kwargs = mock_create_artifact.call_args
            self.assertEqual(artifact_kwargs["artifact_type"], "json")
            self.assertTrue(str(artifact_kwargs["path"]).endswith("test_file.json"))
            self.assertIn("Semantic analysis", artifact_kwargs["description"])
            self.assertEqual(artifact_kwargs["result"], result)
            self.assertEqual(artifact_kwargs["reporter"], reporter)

    @patch("pamola_core.profiling.analyzers.text.read_json")
    def test_categorize_texts_in_chunks_basic(self, mock_read_json):
        text_values = ["engineer", "doctor", "unknown", ""]
        record_ids = ["0", "1", "2", "3"]
        dictionary = {"engineer": "STEM", "doctor": "Healthcare"}
        mock_read_json.return_value = dictionary
        dictionary_path = Path("/tmp/dict.json")
        result = self.op._categorize_texts_in_chunks(
            text_values, record_ids, dictionary_path,
            language="en", match_strategy="exact",
            use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)
        self.assertIn("summary", result)
        self.assertIn("category_distribution", result)
        self.assertEqual(len(result["categorization"]), 0)
        self.assertEqual(len(result["unresolved"]), 0)
        
    def test_categorize_texts_in_chunks_use_dask(self):
        with patch("pamola_core.profiling.analyzers.text.read_json") as mock_read_json:
            text_values = ["engineer", "doctor", "unknown", ""]
            record_ids = ["0", "1", "2", "3"]
            dictionary = {"engineer": "STEM", "doctor": "Healthcare"}
            mock_read_json.return_value = dictionary
            dictionary_path = Path("/tmp/dict.json")
            result = self.op._categorize_texts_in_chunks(
                text_values, record_ids, dictionary_path,
                language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8,
                use_dask=True
            )
            self.assertIn("categorization", result)
            self.assertIn("unresolved", result)
            self.assertIn("summary", result)
            self.assertIn("category_distribution", result)
            self.assertEqual(len(result["categorization"]), 0)
            self.assertEqual(len(result["unresolved"]), 0)
            
    def test_categorize_texts_in_chunks_use_vectorization(self):
        with patch("pamola_core.profiling.analyzers.text.read_json") as mock_read_json:
            text_values = ["engineer", "doctor", "unknown", ""]
            record_ids = ["0", "1", "2", "3"]
            dictionary = {"engineer": "STEM", "doctor": "Healthcare"}
            mock_read_json.return_value = dictionary
            dictionary_path = Path("/tmp/dict.json")
            result = self.op._categorize_texts_in_chunks(
                text_values, record_ids, dictionary_path,
                language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8,
                use_vectorization=True
            )
            self.assertIn("categorization", result)
            self.assertIn("unresolved", result)
            self.assertIn("summary", result)
            self.assertIn("category_distribution", result)
            self.assertEqual(len(result["categorization"]), 0)
            self.assertEqual(len(result["unresolved"]), 0)
            
    def test_categorize_texts_in_chunks_chunk_size(self):
        with patch("pamola_core.profiling.analyzers.text.read_json") as mock_read_json:
            text_values = ["engineer", "doctor", "unknown", ""]
            record_ids = ["0", "1", "2", "3"]
            dictionary = {"engineer": "STEM", "doctor": "Healthcare"}
            mock_read_json.return_value = dictionary
            dictionary_path = Path("/tmp/dict.json")
            result = self.op._categorize_texts_in_chunks(
                text_values, record_ids, dictionary_path,
                language="en", match_strategy="exact",
                use_ner=False, perform_clustering=False, clustering_threshold=0.8,
                chunk_size=2
            )
            self.assertIn("categorization", result)
            self.assertIn("unresolved", result)
            self.assertIn("summary", result)
            self.assertIn("category_distribution", result)
            self.assertEqual(len(result["categorization"]), 0)
            self.assertEqual(len(result["unresolved"]), 0)

    @patch("pamola_core.profiling.analyzers.text.read_json")
    @patch("pamola_core.profiling.analyzers.text.logger")
    def test_perform_semantic_categorization_full(self, mock_logger, mock_read_json):
        # Normal categorization case
        text_values = ["engineer", "doctor", "unknown", ""]
        record_ids = ["0", "1", "2", "3"]
        dictionary = {"engineer": "STEM", "doctor": "Healthcare"}
        mock_read_json.return_value = dictionary
        dictionary_path = Path("/tmp/dict.json")
        result = self.op._perform_semantic_categorization(
            text_values, record_ids, dictionary_path,
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertIn("summary", result)
        self.assertIn("categorization", result)
        self.assertIn("unresolved", result)
        self.assertEqual(result["summary"]["num_matched"], 0)
        self.assertEqual(len(result["categorization"]), 0)
        self.assertEqual(len(result["unresolved"]), 0)

        # Case with no dictionary
        mock_read_json.return_value = {}
        result = self.op._perform_semantic_categorization(
            text_values, record_ids, dictionary_path,
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertEqual(result["summary"]["num_matched"], 0)
        self.assertEqual(len(result["categorization"]), 0)
        self.assertEqual(len(result["unresolved"]), 0)
        self.assertEqual(result["category_distribution"], {})

        # Case with empty input
        result = self.op._perform_semantic_categorization(
            [], [], dictionary_path,
            language="en", match_strategy="exact", use_ner=False, perform_clustering=False, clustering_threshold=0.8
        )
        self.assertEqual(result["summary"]["total_texts"], 0)
        self.assertEqual(result["summary"]["num_matched"], 0)
        self.assertEqual(result["categorization"], [])
        self.assertEqual(result["unresolved"], [])
        self.assertEqual(result["category_distribution"], {})

    @patch("pamola_core.profiling.analyzers.text.read_json")
    def test_perform_semantic_categorization_with_matches(self, mock_read_json):
        # Patch create_entity_extractor to return a mock extractor
        with patch("pamola_core.profiling.analyzers.text.create_entity_extractor") as mock_factory:
            mock_extractor = MagicMock()
            # Simulate extraction_results with matches
            extraction_results = {
                "matches": [
                    {"method": "dictionary", "record_id": "1", "category": "A", "alias": "a_alias"},
                    {"method": "ner", "record_id": "2", "category": "B", "alias": "b_alias"},
                    {"method": "other", "record_id": "3", "category": "C", "alias": "c_alias"}
                ],
                "unmatched": []
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

    @patch("pamola_core.profiling.analyzers.text.cluster_by_similarity")
    def test_perform_semantic_categorization_clustering(self, mock_cluster):
        # Simulate cluster_by_similarity returning 2 clusters
        mock_cluster.return_value = {
            "A": [0, 2],  # text 0 and 2 are in cluster A
            "B": [1]      # text 1 is in cluster B
        }
        # Patch create_entity_extractor to return unmatched
        with patch("pamola_core.profiling.analyzers.text.create_entity_extractor") as mock_factory:
            mock_extractor = MagicMock()
            extraction_results = {
                "matches": [],
                "unmatched": [
                    {"record_id": "1", "text": "foo"},
                    {"record_id": "2", "text": "bar"},
                    {"record_id": "3", "text": "baz"}
                ]
            }
            mock_extractor.extract_entities.return_value = extraction_results
            mock_factory.return_value = mock_extractor

            text_values = ["foo", "bar", "baz"]
            record_ids = ["1", "2", "3"]
            dictionary_path = None
            result = self.op._perform_semantic_categorization(
                text_values, record_ids, dictionary_path,
                language="en", match_strategy="exact", use_ner=True, perform_clustering=True, clustering_threshold=0.8
            )
            # Check that categorization contains all record_ids
            categorization = result["categorization"]
            self.assertEqual(len(categorization), 3)
            record_ids_in_cat = {item["record_id"] for item in categorization}
            self.assertSetEqual(record_ids_in_cat, {"1", "2", "3"})
            # Check cluster_id, matched_category, matched_alias
            cluster_ids = {item["cluster_id"] for item in categorization}
            self.assertSetEqual(cluster_ids, {"A", "B"})
            categories = {item["matched_category"] for item in categorization}
            self.assertSetEqual(categories, {"CLUSTER_A", "CLUSTER_B"})
            aliases = {item["matched_alias"] for item in categorization}
            self.assertSetEqual(aliases, {"cluster_a", "cluster_b"})
            # Check category_counts, alias_counts
            self.assertEqual(result["category_distribution"], {"CLUSTER_A": 2, "CLUSTER_B": 1})
            self.assertEqual(result["aliases_distribution"], {"cluster_a": 2, "cluster_b": 1})
            # No unresolved left because all are clustered
            self.assertListEqual(result["unresolved"], [])
            # Check summary
            self.assertEqual(result["summary"]["num_auto_clustered"], 3)
            self.assertEqual(result["summary"]["num_unresolved"], 0)
            self.assertEqual(result["summary"]["num_matched"], 0)
            self.assertEqual(result["summary"]["num_ner_matched"], 0)

    @patch("pamola_core.profiling.analyzers.text.load_data_operation")
    @patch("pamola_core.profiling.analyzers.text.OperationResult")
    @patch("pamola_core.profiling.analyzers.text.OperationStatus")
    def test_execute_with_invalid_input(self, mock_status, mock_result, mock_load):
        # Test with invalid input (not a DataFrame)
        data_source = MagicMock()
        task_dir = Path("/tmp/task")
        reporter = MagicMock()
        progress_tracker = MagicMock()
        # Patch load_data_operation to return an invalid type
        with patch("pamola_core.profiling.analyzers.text.load_data_operation", return_value="not_a_dataframe"), \
             patch("pamola_core.profiling.analyzers.text.OperationResult") as mock_result, \
             patch("pamola_core.profiling.analyzers.text.OperationStatus") as mock_status:
            mock_status.ERROR = "ERROR"
            mock_result.return_value = "error_result"
            res = self.op.execute(
                data_source=data_source,
                task_dir=task_dir,
                reporter=reporter,
                progress_tracker=progress_tracker,
                dataset_name="main"
            )
            # Should return an error OperationResult or raise
            self.assertEqual(res, "error_result")
            reporter.add_operation.assert_called()
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
                                       True,
                                       result,
                                       self.reporter,
                                       vis_theme='theme',
                                       vis_backend='matplotlib',
                                       vis_strict=False,
                                       vis_timeout=2,
                                       progress_tracker=self.progress)
        self.assertEqual(result, {})
        
        
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_no_cache(self, mock_cache_key, mock_operation_cache):
        mock_cache_key.return_value = 'cache_key'
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out, None)
        
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_cache(self, mock_cache_key, mock_operation_cache, mock_get_cache):
        mock_cache_key.return_value = 'cache_key'
        mock_get_cache.return_value = {
            'analysis_results': {
                'null_empty_analysis':{
                    'total_records': 2,
                    'null_values':{
                        'percentage': 100
                    },
                    'empty_strings':{
                        'percentage': 100
                    }
                },
                'length_stats':{
                    'mean': 1,
                    'max': 2
                },
                'language_analysis':{
                    'predominant_language': 'en'
                }
            }
        }
        mock_operation_cache.return_value = {'main': 'vis_path.png'}
        self.op.use_cache = True
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out.status, OperationStatus.SUCCESS)
        
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_cache_match_summary(self, mock_cache_key, mock_operation_cache, mock_get_cache):
        mock_cache_key.return_value = 'cache_key'
        mock_get_cache.return_value = {
            'analysis_results': {
                'null_empty_analysis':{
                    'total_records': 2,
                    'null_values':{
                        'percentage': 100
                    },
                    'empty_strings':{
                        'percentage': 100
                    }
                },
                'length_stats':{
                    'mean': 1,
                    'max': 2
                },
                'language_analysis':{
                    'predominant_language': 'en'
                },
                'match_summary':{
                    'num_matched': 1,
                    'num_ner_matched': 2,
                    'num_auto_clustered': 1,
                    'num_unresolved': 0,
                    'percentage_matched': 100
                }
            }
        }
        mock_operation_cache.return_value = {'main': 'vis_path.png'}
        self.op.use_cache = True
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out.status, OperationStatus.SUCCESS)
                 
    @patch('pamola_core.utils.ops.op_cache.OperationCache.get_cache')
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key')
    def test_cache_exception(self, mock_cache_key, mock_operation_cache, mock_get_cache):
        mock_cache_key.return_value = 'cache_key'
        mock_get_cache.side_effect = Exception("Cache Exception")
        self.op.use_cache = True
        out = self.op._check_cache(
            self.df, self.reporter, self.task_dir
        )
        self.assertEqual(out, None)
        
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', return_value='cache_key')
    @patch('pamola_core.utils.ops.op_cache.OperationCache.save_cache')
    def test_save_to_cache_success(self, mock_save_cache, mock_cache_key, mock_operation_cache):
        analysis_results = {
            'analysis_results': {
                'null_empty_analysis':{
                    'total_records': 2,
                    'null_values':{
                        'percentage': 100
                    },
                    'empty_strings':{
                        'percentage': 100
                    }
                },
                'length_stats':{
                    'mean': 1,
                    'max': 2
                },
                'language_analysis':{
                    'predominant_language': 'en'
                }
            }
        }
        analysis_result_path= {}
        categorization_result_paths = {}
        visualization_paths = []
        self.op.use_cache = True
        result = self.op._save_to_cache(self.df, analysis_results, analysis_result_path, categorization_result_paths, visualization_paths, self.task_dir)
        self.assertTrue(result)
        mock_cache_key.assert_called
        
    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', return_value='cache_key')
    @patch('pamola_core.utils.ops.op_cache.OperationCache.save_cache')
    def test_save_to_cache_false(self, mock_save_cache, mock_cache_key, mock_operation_cache):
        mock_save_cache.return_value = False
        self.op.use_cache = True
        analysis_results = {
            'analysis_results': {
                'null_empty_analysis':{
                    'total_records': 2,
                    'null_values':{
                        'percentage': 100
                    },
                    'empty_strings':{
                        'percentage': 100
                    }
                },
                'length_stats':{
                    'mean': 1,
                    'max': 2
                },
                'language_analysis':{
                    'predominant_language': 'en'
                }
            }
        }
        analysis_result_path= {}
        categorization_result_paths = {}
        visualization_paths = []
        self.op.use_cache = True
        result = self.op._save_to_cache(self.df, analysis_results, analysis_result_path, categorization_result_paths, visualization_paths, self.task_dir)
        self.assertFalse(result)
        mock_save_cache.assert_called

    @patch('pamola_core.utils.ops.op_cache.operation_cache', side_effect=Exception('Cache write error'))
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', side_effect=Exception('Cache write error'))
    @patch('pamola_core.utils.ops.op_cache.OperationCache.save_cache')
    @patch('pamola_core.profiling.analyzers.phone.logger')
    def test_save_to_cache_exception(self, mock_logger, mock_save_cache, mock_cache_key, mock_operation_cache):
        self.op.use_cache = True
        analysis_results = {
            'analysis_results': {
                'null_empty_analysis':{
                    'total_records': 2,
                    'null_values':{
                        'percentage': 100
                    },
                    'empty_strings':{
                        'percentage': 100
                    }
                },
                'length_stats':{
                    'mean': 1,
                    'max': 2
                },
                'language_analysis':{
                    'predominant_language': 'en'
                }
            }
        }
        analysis_result_path= {}
        categorization_result_paths = {}
        visualization_paths = []
        self.op.use_cache = True
        result = self.op._save_to_cache(self.df, analysis_results, analysis_result_path, categorization_result_paths, visualization_paths, self.task_dir)
        self.assertFalse(result)

    @patch('pamola_core.utils.ops.op_cache.operation_cache')
    @patch.object(TextSemanticCategorizerOperation, '_generate_cache_key', return_value='cache_key')
    def test_save_to_cache_empty_analysis_results(self, mock_cache_key, mock_operation_cache):
        # Empty analysis_results
        self.op.use_cache = True
        analysis_results = {}
        analysis_result_path= {}
        categorization_result_paths = {}
        visualization_paths = []
        self.op.use_cache = True
        result = self.op._save_to_cache(self.df, analysis_results, analysis_result_path, categorization_result_paths, visualization_paths, self.task_dir)
        mock_cache_key.assert_called()

if __name__ == "__main__":
    unittest.main()