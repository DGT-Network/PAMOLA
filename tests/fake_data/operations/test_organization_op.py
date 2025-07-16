import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pamola_core.fake_data import FakeOrganizationOperation
from pamola_core.fake_data.commons.base import NullStrategy
from pamola_core.fake_data.generators.organization import OrganizationGenerator
from pamola_core.utils.ops.op_registry import unregister_operation, get_operation_class
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
import pytest

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

class TestFakeOrganizationOperationInit(unittest.TestCase):

    def test_initialization_with_defaults(self):
        # Initialize with only the required field_name
        op = FakeOrganizationOperation(field_name="organization")

        # Check default attributes
        self.assertEqual(op.field_name, "organization")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "prgn")
        self.assertFalse(op.save_mapping)
        self.assertIsNone(op.mapping_store_path)
        self.assertEqual(op.error_logging_level, "WARNING")
        self.assertEqual(op.max_retries, 3)
        self.assertIsNone(op.id_field)
        self.assertIsNone(op.key)
        self.assertIsNone(op.context_salt)
        self.assertEqual(op.column_prefix, "_")
        self.assertEqual(op.organization_type, "general")
        self.assertEqual(op.region, "en")
        self.assertTrue(op.preserve_type)

        self.assertTrue(op.collect_type_distribution)
        self.assertIsNone(op.type_field)
        self.assertIsNone(op.region_field)
        self.assertFalse(op.detailed_metrics)

        # Check initial counters
        self.assertEqual(op.process_count, 0)
        self.assertEqual(op.retry_count, 0)
        self.assertEqual(op.error_count, 0)

        # Verify the generator is a valid instance
        self.assertIsInstance(op.generator, OrganizationGenerator)

        # Check default values inside the OrganizationGenerator config
        config = op.generator.config
        self.assertEqual(config["organization_type"], "general")
        self.assertEqual(config["dictionaries"], {})
        self.assertEqual(config["prefixes"], {})
        self.assertEqual(config["suffixes"], {})
        self.assertEqual(config["add_prefix_probability"], 0.3)
        self.assertEqual(config["add_suffix_probability"], 0.5)
        self.assertEqual(config["region"], "en")
        self.assertTrue(config["preserve_type"])
        self.assertIsNone(config["industry"])
        self.assertIsNone(config["key"])
        self.assertIsNone(config["context_salt"])

        # When detailed_metrics = False, detailed stats should not exist
        self.assertFalse(hasattr(op, "_type_stats"))
        self.assertFalse(hasattr(op, "_region_stats"))
        self.assertFalse(hasattr(op, "_prefix_suffix_stats"))
        self.assertFalse(hasattr(op, "_generation_times"))

    def test_initialization_with_parameters(self):
        kwargs = {
            "field_name": "organization_name",
            "mode": "ENRICH",
            "output_field_name": "organization_enriched",
            "organization_type": "general",
            "dictionaries": {"general": ["Corp", "Inc"]},
            "prefixes": {"general": ["Global", "National"]},
            "suffixes": {"general": ["Ltd", "LLC"]},
            "add_prefix_probability": 0.4,
            "add_suffix_probability": 0.6,
            "region": "en",
            "preserve_type": True,
            "industry": "technology",
            "chunk_size": 10000,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "abcd",
            "mapping_store_path": "C:/fake_data/operation/mappings.json",
            "id_field": "id",
            "key": None,
            "context_salt": "context-123",
            "save_mapping": True,
            "column_prefix": "_",
            "collect_type_distribution": True,
            "type_field": "type",
            "region_field": "region",
            "detailed_metrics": True,
            "error_logging_level": "WARNING",
            "max_retries": 3
        }

        op = FakeOrganizationOperation(**kwargs)

        # Top-level attribute checks
        self.assertEqual(op.field_name, "organization_name")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "organization_enriched")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "abcd")
        self.assertEqual(op.mapping_store_path, "C:/fake_data/operation/mappings.json")
        self.assertEqual(op.id_field, "id")
        self.assertIsNone(op.key)
        self.assertEqual(op.context_salt, "context-123")
        self.assertTrue(op.save_mapping)
        self.assertEqual(op.column_prefix, "_")
        self.assertEqual(op.organization_type, "general")
        self.assertEqual(op.region, "en")
        self.assertTrue(op.preserve_type)
        self.assertEqual(op.error_logging_level, "WARNING")
        self.assertEqual(op.max_retries, 3)

        # Metrics-related attributes
        self.assertEqual(op.process_count, 0)
        self.assertEqual(op.error_count, 0)
        self.assertEqual(op.retry_count, 0)

        # Detailed metrics
        self.assertTrue(op.detailed_metrics)
        self.assertTrue(hasattr(op, "_type_stats"))
        self.assertTrue(hasattr(op, "_region_stats"))
        self.assertTrue(hasattr(op, "_prefix_suffix_stats"))
        self.assertTrue(hasattr(op, "_generation_times"))

        self.assertIsInstance(op._type_stats, Counter)
        self.assertIsInstance(op._region_stats, Counter)
        self.assertIsInstance(op._prefix_suffix_stats, dict)
        self.assertIsInstance(op._generation_times, list)

        # Generator and config validation
        self.assertIsInstance(op.generator, OrganizationGenerator)
        config = op.generator.config
        self.assertEqual(config["organization_type"], "general")
        self.assertEqual(config["dictionaries"], {"general": ["Corp", "Inc"]})
        self.assertEqual(config["prefixes"], {"general": ["Global", "National"]})
        self.assertEqual(config["suffixes"], {"general": ["Ltd", "LLC"]})
        self.assertEqual(config["add_prefix_probability"], 0.4)
        self.assertEqual(config["add_suffix_probability"], 0.6)
        self.assertEqual(config["region"], "en")
        self.assertTrue(config["preserve_type"])
        self.assertEqual(config["industry"], "technology")
        self.assertIsNone(config["key"])
        self.assertEqual(config["context_salt"], "context-123")


class TestProcessValue(unittest.TestCase):

    def setUp(self):
        self.mock_generator = Mock()
        self.mock_generator.generate_like.return_value = "synthetic_organization_name"
        self.mock_generator.detect_organization_type.return_value = "general"
        self.mock_generator._determine_region_from_name.return_value = "en"
        self.mock_generator._prefixes = {"general": {"en": ["synthetic"]}}
        self.mock_generator._suffixes = {"general": {"en": ["name"]}}

        self.mock_mapping_store = Mock()
        self.mock_mapping_store.get_mapping.return_value = None

        self.op = FakeOrganizationOperation(field_name="organization_name")
        self.op.generator = self.mock_generator
        self.op.mapping_store = self.mock_mapping_store
        self.op.consistency_mechanism = "mapping"
        self.op.detailed_metrics = True
        self.op.max_retries = 2
        self.op._domain_stats = Counter()
        self.op._format_stats = Counter()
        self.op._type_stats = Counter()
        self.op._region_stats = Counter()
        self.op._prefix_suffix_stats = Counter()
        self.op._generation_times = []
        self.op.retry_count = 0
        self.op.error_count = 0

    def test_returns_existing_mapping_if_found(self):
        self.mock_mapping_store.get_mapping.return_value = "Existing Co"
        self.mock_generator.detect_organization_type.return_value = "general"

        result = self.op.process_value("Original Co")

        self.assertEqual(result, "Existing Co")
        self.mock_mapping_store.get_mapping.assert_called_once_with("organization_name", "Original Co")
        self.assertEqual(self.op._type_stats["general"], 1)
        self.assertEqual(len(self.op._generation_times), 1)

    def test_generate_and_store_mapping(self):
        self.mock_mapping_store.get_mapping.return_value = None

        result = self.op.process_value("New Co")

        self.assertEqual(result, "synthetic_organization_name")
        self.mock_generator.generate_like.assert_called_once_with("New Co")
        self.mock_mapping_store.add_mapping.assert_called_once_with("organization_name", "New Co", "synthetic_organization_name")
        self.assertEqual(self.op._type_stats["general"], 1)
        self.assertEqual(self.op._region_stats["en"], 1)
        self.assertEqual(len(self.op._generation_times), 1)

    def test_retry_once_on_failure(self):
        self.mock_generator.generate_like.side_effect = [Exception("fail"), "recovered_value"]
        self.mock_generator.detect_organization_type.return_value = "general"
        self.mock_generator._determine_region_from_name.return_value = "en"

        result = self.op.process_value("Flaky Org")

        self.assertEqual(result, "recovered_value")
        self.assertEqual(self.op.retry_count, 1)
        self.assertEqual(self.op.error_count, 0)

    def test_fail_after_max_retries(self):
        self.mock_generator.generate_like.side_effect = Exception("persistent error")

        result = self.op.process_value("Broken Org")

        self.assertEqual(result, "Broken Org")
        self.assertEqual(self.op.retry_count, 3)
        self.assertEqual(self.op.error_count, 1)


class TestProcessBatch(unittest.TestCase):

    def setUp(self):
        self.op = FakeOrganizationOperation(field_name="organization_name")
        self.op.type_field = "type"
        self.op.region_field = "region"
        self.op.id_field = "id"
        self.op.output_field_name = "organization_enriched"
        self.op.column_prefix = "_"
        self.op.mode = "ENRICH"
        self.op.organization_type = "general"
        self.op.region = "en"
        self.op.context_salt = "salt123"
        self.op.null_strategy = "PRESERVE"
        self.op.process_value = Mock(side_effect=lambda v, **p: f"gen_{v}" if pd.notna(v) else np.nan)
        self.op.process_count = 0
        self.op._type_stats = Counter()
        self.op._region_stats = Counter()
        self.op._generation_times = []

    def test_process_batch_with_valid_values(self):
        df = pd.DataFrame({
            "organization_name": ["Alpha", "Beta", "Gamma"],
            "type": ["startup", "enterprise", "nonprofit"],
            "region": ["us", "eu", "apac"],
            "id": [1, 2, 3]
        })

        result = self.op.process_batch(df)

        expected = ["gen_Alpha", "gen_Beta", "gen_Gamma"]
        self.assertListEqual(result["organization_enriched"].tolist(), expected)
        self.assertEqual(self.op.process_value.call_count, 3)
        self.assertEqual(self.op.process_count, 3)

    def test_process_batch_with_null_and_preserve(self):
        self.op.null_strategy = "PRESERVE"
        df = pd.DataFrame({
            "organization_name": ["Alpha", None, "Gamma"],
            "type": [None, None, None],
            "region": [None, None, None],
            "id": [1, 2, 3]
        })

        result = self.op.process_batch(df)

        expected = ["gen_Alpha", np.nan, "gen_Gamma"]
        self.assertEqual(result["organization_enriched"].tolist(), expected)
        self.assertEqual(self.op.process_value.call_count, 2)

    def test_process_batch_with_null_and_exclude(self):
        self.op.null_strategy = "EXCLUDE"
        df = pd.DataFrame({
            "organization_name": ["Alpha", None, "Gamma"],
            "type": [None, None, None],
            "region": [None, None, None],
            "id": [1, 2, 3]
        })

        result = self.op.process_batch(df)

        expected = ["gen_Alpha", np.nan, "gen_Gamma"]
        self.assertEqual(result["organization_enriched"].tolist(), expected)
        self.assertEqual(self.op.process_value.call_count, 2)

    def test_process_batch_with_null_and_error(self):
        self.op.null_strategy = "ERROR"
        df = pd.DataFrame({
            "organization_name": ["Alpha", None],
            "type": [None, None],
            "region": [None, None],
            "id": [1, 2]
        })

        with self.assertRaises(ValueError) as cm:
            self.op.process_batch(df)
        self.assertIn("Null value found in organization_name", str(cm.exception))

    def test_process_batch_with_exception_in_process_value(self):
        def raise_for_beta(value, **params):
            if value == "Beta":
                raise RuntimeError("Generation failed")
            return f"gen_{value}"

        self.op.process_value = Mock(side_effect=raise_for_beta)
        df = pd.DataFrame({
            "organization_name": ["Alpha", "Beta", "Gamma"],
            "type": ["t1", "t2", "t3"],
            "region": ["r1", "r2", "r3"],
            "id": [101, 102, 103]
        })

        result = self.op.process_batch(df)
        expected = ["gen_Alpha", "Beta", "gen_Gamma"]
        self.assertEqual(result["organization_enriched"].tolist(), expected)


class PrepareData:

    def create_data_source(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "organization_name": ["Tech Global", "HealthCare United", "Green Future"],
            "type": ["tech", "health", "energy"],
            "region": ["us", "us", "vn"]
        })

    def create_kwargs(self):
        return {
            "field_name": "organization_name",
            "mode": "ENRICH",
            "output_field_name": "organization_enriched",
            "organization_type": "general",
            "dictionaries": None,
            "prefixes": None,
            "suffixes": None,
            "add_prefix_probability": 0.4,
            "add_suffix_probability": 0.6,
            "region": "en",
            "preserve_type": True,
            "industry": None,
            "chunk_size": 10000,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "abcd",
            "mapping_store_path": "C:/fake_data/operation/mappings.json",
            "id_field": "id",
            "key": None,
            "context_salt": "context-123",
            "save_mapping": True,
            "column_prefix": "_",
            "collect_type_distribution": True,
            "type_field": "type",
            "region_field": "region",
            "detailed_metrics": True,
            "error_logging_level": "WARNING",
            "max_retries": 3
        }

    def create_task_dir(self):
        task_dir = Path("test_task_dir/fake_data/unittest/operation/organization")
        os.makedirs(task_dir, exist_ok=True)
        return task_dir


class TestFakeOrganzationOperationExecute(unittest.TestCase):

    def setUp(self):
        prepare_data = PrepareData()
        self.task_dir = prepare_data.create_task_dir()
        self.data_source = prepare_data.create_data_source()
        self.kwargs = prepare_data.create_kwargs()

        if get_operation_class("FakeOrganizationOperation") is not None:
            unregister_operation("FakeOrganizationOperation")

    def test_execute_success_with_enrich_mode(self):
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeOrganizationOperation(**self.kwargs)
        op.use_cache = False
        with patch("pamola_core.utils.io.load_settings_operation", return_value={}), \
         patch("pamola_core.utils.io.load_data_operation", return_value=df_data_source.copy()):
            result = op.execute(
                data_source=df,
                task_dir=self.task_dir,
                reporter=None,
                progress_tracker=None,
                **self.kwargs
            )

            # Check result is an instance of OperationResult
            self.assertIsInstance(result, OperationResult)

            # Check that status is SUCCESS
            self.assertEqual(result.status, OperationStatus.SUCCESS)

            # Check error message is None
            self.assertIsNone(result.error_message)

            # Check mapping store
            mapping_path = Path(self.task_dir) / "maps" / f"{op.name}_{op.field_name}_mapping.json"
            self.assertTrue(mapping_path.is_file(), msg=f"Mapping file not found at: {mapping_path}")

            # Check artifacts
            self.assertIsInstance(result.artifacts, list)
            for artifact in result.artifacts:
                self.assertIsInstance(artifact, OperationArtifact)
                self.assertTrue(
                    artifact.path.is_file(),
                    msg=f"Artifact file does not exist: {artifact.path}"
                )
                self.assertTrue(
                    str(artifact.path).endswith((".json", ".csv", ".txt", ".png")),
                    msg=f"Unexpected artifact file type: {artifact.path}"
                )
                self.assertIsInstance(artifact.description, str)
                self.assertIn(artifact.category, ["output", "metrics", "visualization"])
                self.assertIsInstance(artifact.tags, list)
                self.assertIsInstance(artifact.creation_time, str)
                self.assertIsInstance(artifact.size, int)

            # Check metrics
            if self.kwargs.get("detailed_metrics") and hasattr(result, "metrics"):
                self.assertIsInstance(result.metrics, dict)
                self.assertEqual(result.metrics["output_field"]["name"], "organization_enriched")
                self.assertIsInstance(result.metrics["original_data"], dict)
                self.assertIsInstance(result.metrics["generated_data"], dict)
                self.assertEqual(len(result.metrics["original_data"]), len(result.metrics["generated_data"]))

            # Check that execution time is recorded
            self.assertIsInstance(result.execution_time, float)

    def test_execute_success_with_replace_mode(self):
        self.kwargs["mode"] = "REPLACE"
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeOrganizationOperation(**self.kwargs)
        op.use_cache = False
        with patch("pamola_core.utils.io.load_settings_operation", return_value={}), \
         patch("pamola_core.utils.io.load_data_operation", return_value=df_data_source.copy()):
            result = op.execute(
                data_source=df,
                task_dir=self.task_dir,
                reporter=None,
                progress_tracker=None,
                **self.kwargs
            )

            # Check result is an instance of OperationResult
            self.assertIsInstance(result, OperationResult)

            # Check that status is SUCCESS
            self.assertEqual(result.status, OperationStatus.SUCCESS)

            # Check error message is None
            self.assertIsNone(result.error_message)

            # Check mapping store
            mapping_path = Path(self.task_dir) / "maps" / f"{op.name}_{op.field_name}_mapping.json"
            self.assertTrue(mapping_path.is_file(), msg=f"Mapping file not found at: {mapping_path}")

            # Check artifacts
            self.assertIsInstance(result.artifacts, list)
            for artifact in result.artifacts:
                self.assertIsInstance(artifact, OperationArtifact)
                self.assertTrue(
                    artifact.path.is_file(),
                    msg=f"Artifact file does not exist: {artifact.path}"
                )
                self.assertTrue(
                    str(artifact.path).endswith((".json", ".csv", ".txt", ".png")),
                    msg=f"Unexpected artifact file type: {artifact.path}"
                )
                self.assertIsInstance(artifact.description, str)
                self.assertIn(artifact.category, ["output", "metrics", "visualization"])
                self.assertIsInstance(artifact.tags, list)
                self.assertIsInstance(artifact.creation_time, str)
                self.assertIsInstance(artifact.size, int)

            # Check that execution time is recorded
            self.assertIsInstance(result.execution_time, float)

    def test_execute_missing_field_name_column(self):
        df_data_source = self.data_source.drop(columns=["organization_name"])
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeOrganizationOperation(**self.kwargs)
        with patch("pamola_core.utils.io.load_settings_operation", return_value={}), \
         patch("pamola_core.utils.io.load_data_operation", return_value=df_data_source.copy()):
            result = op.execute(
                data_source=df,
                task_dir=self.task_dir,
                reporter=None,
                progress_tracker=None,
                **self.kwargs
            )

            self.assertEqual(result.status, OperationStatus.ERROR)
            self.assertIsInstance(result.error_message, str)
            self.assertIn("organization_name", result.error_message.lower())
            
    def test_perfect_match(self):
        original = pd.Series(["A", "B", "C"])
        generated = pd.Series(["A", "B", "C"])
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIn("length_similarity", metrics)
        self.assertEqual(metrics["length_similarity"], 1.0)
        self.assertIn("type_preservation_ratio", metrics)
        self.assertEqual(metrics["type_preservation_ratio"], 1.0)
        self.assertIn("type_diversity_ratio", metrics)
        self.assertEqual(metrics["type_diversity_ratio"], 1.0)
        self.assertIn("word_count_similarity", metrics)
        self.assertEqual(metrics["word_count_similarity"], 1.0)

    def test_all_nulls_replaced(self):
        original = pd.Series([None, None, None])
        generated = pd.Series(["X", "Y", "Z"])
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        assert metrics == {}

    def test_all_unique_generated(self):
        original = pd.Series(["A", "A", "B", "B"])
        generated = pd.Series(["X", "Y", "Z", "W"])
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIn("type_preservation_ratio", metrics)
        self.assertEqual(metrics["type_preservation_ratio"], 1.0)

    def test_mismatched_lengths(self):
        original = pd.Series(["A", "B"])
        generated = None
        with self.assertRaises(Exception):
            op = FakeOrganizationOperation(**self.kwargs)
            op._calculate_quality_metrics(original, generated)

    def test_partial_overlap_and_nulls(self):
        original = pd.Series(["A", None, "C", "D"])
        generated = pd.Series([1, None, 3, 4, 5])
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIn("length_similarity", metrics)
        assert metrics["length_similarity"] == 0.0

    def test_empty_input(self):
        original = pd.Series([])
        generated = pd.Series([])
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIsInstance(metrics, dict)
        # Should not error, but may have zeros or None






            
    def test_collect_metrics_basic(self):
        generated = self.data_source
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._collect_metrics(generated)
        self.assertIsInstance(metrics, dict)
        assert len(metrics) > 0

    def test_collect_metrics_with_REPLACE(self):
        self.kwargs['mode'] = 'REPLACE'
        generated = self.data_source
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._collect_metrics(generated)
        self.assertIsInstance(metrics, dict)
        self.assertNotIn("output_field", metrics)

    def test_collect_metrics_with_ENRICH(self):
        self.kwargs['mode'] = 'ENRICH'
        generated = self.data_source
        generated["organization_enriched"] = "organization_enriched"
        op = FakeOrganizationOperation(**self.kwargs)
        metrics = op._collect_metrics(generated)
        self.assertIsInstance(metrics, dict)
        self.assertIn("output_field", metrics)

    def test_collect_metrics_mismatched_lengths(self):
        generated = pd.Series(["X"])
        with self.assertRaises(Exception):
            op = FakeOrganizationOperation(**self.kwargs)
            op._collect_metrics(generated)

    def test_collect_metrics_with_types(self):
        generated = self.data_source
        generated["organization_enriched"] = "organization_enriched"
        generated["type"] = ["t1", "t2", "t1"]
        generated["region"] = ["r1", "r2", "r1"]
        op = FakeOrganizationOperation(**self.kwargs)
        op.type_field = "type"
        op.region_field = "region"
        metrics = op._collect_metrics(generated)
        self.assertIsInstance(metrics, dict)
        self.assertIn("output_field", metrics)
        self.assertIn("organization_generator", metrics)
        self.assertIsNotNone(metrics["organization_generator"]["organization_type"])
        self.assertIsNotNone(metrics["organization_generator"]["region"])

    def test_save_metrics_creates_file(self):
        metrics = {"a": 1, "b": 2}
        metrics_path = self.task_dir
        op = FakeOrganizationOperation(**self.kwargs)
        metrics_path_result = op._save_metrics(metrics, metrics_path)
        self.assertTrue(metrics_path_result.exists())

    def test_save_metrics_content(self):
        metrics = {"foo": "bar", "num": 42}
        metrics_path = self.task_dir
        op = FakeOrganizationOperation(**self.kwargs)
        metrics_path_result = op._save_metrics(metrics, metrics_path)
        import json
        with open(metrics_path_result, "r") as f:
            data = json.load(f)
        self.assertEqual(data, metrics)

    def test_save_metrics_overwrite(self):
        metrics1 = {"x": 1}
        metrics2 = {"y": 2}
        metrics_path = self.task_dir
        op = FakeOrganizationOperation(**self.kwargs)
        op._save_metrics(metrics1, metrics_path)
        metrics_path_result = op._save_metrics(metrics2, metrics_path)
        import json
        with open(metrics_path_result, "r") as f:
            data = json.load(f)
        self.assertEqual(data, metrics2)

    def test_save_metrics_invalid_path(self):
        metrics = None
        with self.assertRaises(Exception):
            op = FakeOrganizationOperation(**self.kwargs)
            metrics_path = self.task_dir
            op._save_metrics(metrics, metrics_path)

    def test_save_metrics_empty_metrics(self):
        metrics = {}
        op = FakeOrganizationOperation(**self.kwargs)
        metrics_path = self.task_dir
        metrics_path_result = op._save_metrics(metrics, metrics_path)
        import json
        with open(metrics_path_result, "r") as f:
            data = json.load(f)
        self.assertEqual(data, metrics)

if __name__ == "__main__":
    unittest.main()