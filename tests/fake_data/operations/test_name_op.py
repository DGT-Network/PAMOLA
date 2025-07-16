import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from pamola_core.fake_data import FakeNameOperation
from pamola_core.fake_data.commons.base import NullStrategy
from pamola_core.fake_data.generators.name import NameGenerator
from pamola_core.utils.ops.op_registry import unregister_operation, get_operation_class
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact

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
    
class TestFakeNameOperationInit(unittest.TestCase):

    def test_initialization_with_defaults(self):
        # Initialize FakeNameOperation with only required parameter
        op = FakeNameOperation(field_name="full_name")

        # Check basic configuration attributes
        self.assertEqual(op.field_name, "full_name")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "prgn")

        # Additional config fields
        self.assertFalse(op.save_mapping)
        self.assertIsNone(op.mapping_store_path)
        self.assertIsNone(op.id_field)
        self.assertIsNone(op.key)
        self.assertIsNone(op.context_salt)
        self.assertEqual(op.column_prefix, "_")
        self.assertEqual(op.language, "en")
        self.assertIsNone(op.gender_field)

        # Generator assertions
        self.assertIsInstance(op.generator, NameGenerator)

        # Generator config values
        gen_config = op.generator.config
        self.assertEqual(gen_config["language"], "en")
        self.assertFalse(gen_config["gender_from_name"])
        self.assertIsNone(gen_config["format"])
        self.assertEqual(gen_config["f_m_ratio"], 0.5)
        self.assertFalse(gen_config["use_faker"])
        self.assertEqual(gen_config["case"], "title")
        self.assertEqual(gen_config["dictionaries"], {})
        self.assertIsNone(gen_config["key"])
        self.assertIsNone(gen_config["context_salt"])

        # Internal counters
        self.assertEqual(op.process_count, 0)
        self.assertIsNone(op._original_df)

    def test_initialization_with_parameters(self):
        kwargs = {
            "field_name": "full_name",
            "mode": "ENRICH",
            "output_field_name": "full_name_enriched",
            "language": "en",
            "gender_field": "gender",
            "gender_from_name": False,
            "format": None,
            "f_m_ratio": 0.5,
            "use_faker": False,
            "case": "title",
            "dictionaries": None,
            "chunk_size": 10000,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "prgn",
            "mapping_store_path": "C:/fake_data/name_operation/mappings.json",
            "id_field": "id",
            "key": "my-secret-key",
            "context_salt": "context-123",
            "save_mapping": True,
            "column_prefix": "_"
        }

        op = FakeNameOperation(**kwargs)

        # Check operation attributes
        self.assertEqual(op.field_name, "full_name")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "full_name_enriched")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "prgn")
        self.assertTrue(op.save_mapping)
        self.assertEqual(op.mapping_store_path, "C:/fake_data/name_operation/mappings.json")
        self.assertEqual(op.id_field, "id")
        self.assertEqual(op.key, "my-secret-key")
        self.assertEqual(op.context_salt, "context-123")
        self.assertEqual(op.column_prefix, "_")
        self.assertEqual(op.language, "en")
        self.assertEqual(op.gender_field, "gender")

        # Ensure that the generator is an instance of NameGenerator class
        self.assertIsInstance(op.generator, NameGenerator)

        # Check generator configuration
        gen_config = op.generator.config
        self.assertEqual(gen_config["language"], "en")
        self.assertFalse(gen_config["gender_from_name"])
        self.assertIsNone(gen_config["format"])
        self.assertEqual(gen_config["f_m_ratio"], 0.5)
        self.assertFalse(gen_config["use_faker"])
        self.assertEqual(gen_config["case"], "title")
        self.assertEqual(gen_config["dictionaries"], {})
        self.assertEqual(gen_config["key"], "my-secret-key")
        self.assertEqual(gen_config["context_salt"], "context-123")

        # Check internal counters and state
        self.assertEqual(op.process_count, 0)
        self.assertIsNone(op._original_df)


class TestFakeNameOperationProcessValue(unittest.TestCase):

    def setUp(self):
        self.mock_generator = Mock()
        self.mock_generator.generate_like.return_value = "synthetic_name"

        self.mock_mapping_store = Mock()
        self.mock_mapping_store.get_mapping.return_value = None

        self.op = FakeNameOperation(field_name="full_name")
        self.op.generator = self.mock_generator
        self.op.mapping_store = self.mock_mapping_store
        self.op.consistency_mechanism = "mapping"
        self.op.max_retries = 2
        self.op._format_stats = Counter()
        self.op._generation_times = []
        self.op.retry_count = 0
        self.op.error_count = 0

    def test_process_value_with_existing_mapping(self):
        self.mock_mapping_store.get_mapping.return_value = "mapped_name"
        result = self.op.process_value("John Doe", first_name="John", last_name="Doe")
        self.assertEqual(result, "mapped_name")
        self.mock_mapping_store.get_mapping.assert_called_once()

    def test_process_value_with_new_mapping_added(self):
        self.op.mapping_store.get_mapping.return_value = None
        result = self.op.process_value("Jane Doe", first_name="Jane", last_name="Doe")
        self.assertEqual(result, "synthetic_name")
        self.mock_mapping_store.add_mapping.assert_called_once_with(
            "full_name", "Jane Doe", "synthetic_name"
        )

    @patch("pamola_core.fake_data.operations.name_op.PRNGenerator")
    def test_process_value_with_prgn_consistency(self, MockPRGN):
        self.op.consistency_mechanism = "prgn"
        self.op.generator.prgn_generator = None
        self.mock_generator.generate_like.return_value = "synthetic_name_prgn"

        result = self.op.process_value("Alice Smith", first_name="Alice", last_name="Smith")

        self.assertEqual(result, "synthetic_name_prgn")
        MockPRGN.assert_called_once()
        # Ensure generate_like is called correctly
        self.mock_generator.generate_like.assert_called_once_with("Alice Smith", first_name="Alice", last_name="Smith")


class TestFakeNameProcessBatch(unittest.TestCase):
    def setUp(self):
        self.op = FakeNameOperation(
            field_name="name",
            gender_field="gender",
            id_field="id",
            mode="REPLACE",
            null_strategy="PRESERVE",
            context_salt="test_salt",
            column_prefix="syn_",
            language="en"
        )
        self.op.process_value = Mock(
            side_effect=lambda val, **kwargs: f"generated_{val}_{kwargs.get('gender') or ''}"
        )
        self.op.process_count = 0

    def test_replace_mode_with_valid_values(self):
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "gender": ["F", "M"],
            "id": [1, 2]
        })
        result = self.op.process_batch(df)
        expected = ["generated_Alice_F", "generated_Bob_M"]
        self.assertListEqual(result["name"].tolist(), expected)
        self.assertEqual(self.op.process_count, 2)

    def test_enrich_mode_creates_new_column(self):
        self.op.mode = "ENRICH"
        df = pd.DataFrame({
            "name": ["Tom"],
            "gender": ["M"]
        })
        result = self.op.process_batch(df)
        self.assertIn("syn_name", result.columns)
        self.assertEqual(result["syn_name"].iloc[0], "generated_Tom_M")
        self.assertEqual(result["name"].iloc[0], "Tom")

    def test_preserve_null_strategy(self):
        self.op.null_strategy = "PRESERVE"
        df = pd.DataFrame({
            "name": [None],
            "gender": ["F"]
        })
        result = self.op.process_batch(df)
        self.assertTrue(pd.isna(result["name"].iloc[0]))
        self.assertEqual(self.op.process_count, 0)

    def test_exclude_null_strategy(self):
        self.op.null_strategy = "EXCLUDE"
        df = pd.DataFrame({
            "name": [None],
            "gender": ["M"]
        })
        result = self.op.process_batch(df)
        self.assertTrue(pd.isna(result["name"].iloc[0]))
        self.assertEqual(self.op.process_count, 0)

    def test_error_on_null_value(self):
        self.op.null_strategy = "ERROR"
        df = pd.DataFrame({
            "name": [None],
            "gender": ["F"]
        })
        with self.assertRaises(ValueError):
            self.op.process_batch(df)

    def test_process_value_failure_fallback(self):
        def fail(val, **kwargs):
            raise Exception("fail")

        self.op.process_value = Mock(side_effect=fail)
        df = pd.DataFrame({
            "name": ["fail_value"],
            "gender": ["F"]
        })
        result = self.op.process_batch(df)
        self.assertEqual(result["name"].iloc[0], "fail_value")  # fallback to original
        self.assertEqual(self.op.process_count, 0)

    def test_gender_conversion_variants(self):
        df = pd.DataFrame({
            "name": ["Alex", "Maria", "Ivan", "Anna"],
            "gender": ["MALE", "FEMALE", "МУЖ", "ЖЕН"]
        })
        result = self.op.process_batch(df)
        expected = [
            "generated_Alex_M",
            "generated_Maria_F",
            "generated_Ivan_M",
            "generated_Anna_F"
        ]
        self.assertListEqual(result["name"].tolist(), expected)

    def test_missing_gender_field(self):
        df = pd.DataFrame({
            "name": ["Sam", "Charlie"]
        })
        result = self.op.process_batch(df)
        expected = ["generated_Sam_", "generated_Charlie_"]
        self.assertListEqual(result["name"].tolist(), expected)
        self.assertEqual(self.op.process_count, 2)

    def test_record_id_collected(self):
        df = pd.DataFrame({
            "name": ["Diana"],
            "gender": ["F"],
            "id": [999]
        })

        called_kwargs = {}

        def capture(val, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return "generated_value"

        self.op.process_value = Mock(side_effect=capture)
        self.op.process_batch(df)
        self.assertEqual(called_kwargs["record_id"], 999)
        self.assertEqual(called_kwargs["gender"], "F")


class PrepareData:

    def create_data_source(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "full_name": ["John Doe", "Jane Smith", "Alice Nguyen"],
            "gender": ["male", "female", "female"]
        })

    def create_kwargs(self):
        return {
            "field_name": "full_name",
            "mode": "ENRICH",
            "output_field_name": "full_name_enriched",
            "language": "en",
            "gender_field": "gender",
            "gender_from_name": False,
            "format": None,
            "f_m_ratio": 0.5,
            "use_faker": False,
            "case": "title",
            "dictionaries": None,
            "chunk_size": 10000,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "prgn",
            "mapping_store_path": "C:/fake_data/name_operation/mappings.json",
            "id_field": "id",
            "key": "my-secret-key",
            "context_salt": "context-123",
            "save_mapping": True,
            "column_prefix": "_"
        }

    def create_task_dir(self):
        task_dir = Path("test_task_dir/fake_data/unittest/operation/name")
        os.makedirs(task_dir, exist_ok=True)
        return task_dir


class TestFakeNameOperationExecute(unittest.TestCase):

    def setUp(self):
        prepare_data = PrepareData()
        self.task_dir = prepare_data.create_task_dir()
        self.data_source = prepare_data.create_data_source()
        self.kwargs = prepare_data.create_kwargs()

        if get_operation_class("FakeNameOperation") is not None:
            unregister_operation("FakeNameOperation")

    def test_execute_success_with_enrich_mode(self):
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeNameOperation(**self.kwargs)
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
            if hasattr(result, "metrics"):
                self.assertIsInstance(result.metrics, dict)
                self.assertEqual(result.metrics["output_field"]["name"], "full_name_enriched")
                self.assertIsInstance(result.metrics["original_data"], dict)
                self.assertIsInstance(result.metrics["generated_data"], dict)
                self.assertEqual(len(result.metrics["original_data"]), len(result.metrics["generated_data"]))

            # Check that execution time is recorded
            self.assertIsInstance(result.execution_time, float)

    def test_execute_success_with_replace_mode(self):
        self.kwargs["mode"] = "REPLACE"
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeNameOperation(**self.kwargs)
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
        df_data_source = self.data_source.drop(columns=["full_name"])
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeNameOperation(**self.kwargs)
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
            self.assertIn("full_name", result.error_message.lower())


if __name__ == "__main__":
    unittest.main()