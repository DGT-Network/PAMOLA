import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from pamola_core.fake_data import FakePhoneOperation
from pamola_core.fake_data.commons.base import NullStrategy
from pamola_core.fake_data.generators.phone import PhoneGenerator
from pamola_core.utils.ops.op_registry import unregister_operation, get_operation_class
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus, OperationArtifact
import time

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
    
class TestFakePhoneOperationInit(unittest.TestCase):

    def test_initialization_with_defaults(self):
        # Initialize with only the required field_name
        op = FakePhoneOperation(field_name="phone_number")

        # Basic attributes
        self.assertEqual(op.field_name, "phone_number")
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

        self.assertEqual(op.default_country, "us")
        self.assertEqual(op.region, "us")
        self.assertTrue(op.international_format)
        self.assertFalse(op.local_formatting)
        self.assertIsNone(op.country_code_field)
        self.assertFalse(op.detailed_metrics)

        # Metrics initialization
        self.assertEqual(op.process_count, 0)
        self.assertEqual(op.retry_count, 0)
        self.assertEqual(op.error_count, 0)

        # Generator instance
        self.assertIsInstance(op.generator, PhoneGenerator)

        # Check config inside the generator
        config = op.generator.config
        self.assertEqual(config["default_country"], "us")
        self.assertEqual(config["validate_source"], True)
        self.assertEqual(config["handle_invalid_phone"], "generate_new")
        self.assertEqual(config["preserve_country_code"], True)
        self.assertEqual(config["preserve_operator_code"], False)
        self.assertEqual(config["region"], "us")
        self.assertIsNone(config["key"])
        self.assertIsNone(config["context_salt"])
        self.assertIsNone(config["country_codes"])
        self.assertIsNone(config["operator_codes_dict"])
        self.assertIsNone(config["format"])

        # Detailed stats should NOT be initialized by default
        self.assertFalse(hasattr(op, "_country_stats"))
        self.assertFalse(hasattr(op, "_format_stats"))
        self.assertFalse(hasattr(op, "_generation_times"))

    def test_initialization_with_parameters(self):
        kwargs = {
            "field_name": "phone_number",
            "mode": "ENRICH",
            "output_field_name": "phone_number_enriched",
            "country_codes": {"us": 1.0},
            "operator_codes_dict": None,
            "format": "+1 (XXX) XXX-XXXX",
            "validate_source": True,
            "handle_invalid_phone": "generate_new",
            "default_country": "us",
            "preserve_country_code": True,
            "preserve_operator_code": False,
            "region": "us",
            "chunk_size": 10000,
            "null_strategy": NullStrategy.PRESERVE,
            "consistency_mechanism": "prgn",
            "mapping_store_path": "C:/fake_data/phone_operation/mappings.json",
            "id_field": "id",
            "key": "my-secret-key",
            "context_salt": "context-123",
            "save_mapping": True,
            "column_prefix": "_",
            "international_format": True,
            "local_formatting": False,
            "country_code_field": "country_code",
            "detailed_metrics": True,
            "error_logging_level": "INFO",
            "max_retries": 3
        }

        op = FakePhoneOperation(**kwargs)

        # Top-level attributes
        self.assertEqual(op.field_name, "phone_number")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "phone_number_enriched")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "prgn")
        self.assertEqual(op.mapping_store_path, "C:/fake_data/phone_operation/mappings.json")
        self.assertEqual(op.id_field, "id")
        self.assertEqual(op.key, "my-secret-key")
        self.assertEqual(op.context_salt, "context-123")
        self.assertTrue(op.save_mapping)
        self.assertEqual(op.column_prefix, "_")
        self.assertEqual(op.default_country, "us")
        self.assertEqual(op.region, "us")
        self.assertEqual(op.error_logging_level, "INFO")
        self.assertEqual(op.max_retries, 3)

        self.assertTrue(op.international_format)
        self.assertFalse(op.local_formatting)
        self.assertEqual(op.country_code_field, "country_code")
        self.assertTrue(op.detailed_metrics)

        # Initial counters
        self.assertEqual(op.process_count, 0)
        self.assertEqual(op.error_count, 0)
        self.assertEqual(op.retry_count, 0)

        # Detailed metrics attributes
        self.assertTrue(hasattr(op, "_country_stats"))
        self.assertTrue(hasattr(op, "_format_stats"))
        self.assertTrue(hasattr(op, "_generation_times"))
        self.assertIsInstance(op._country_stats, Counter)
        self.assertIsInstance(op._format_stats, Counter)
        self.assertIsInstance(op._generation_times, list)

        # Generator and config validation
        self.assertIsInstance(op.generator, PhoneGenerator)
        config = op.generator.config
        self.assertEqual(config["country_codes"], {"us": 1.0})
        self.assertIsNone(config["operator_codes_dict"])
        self.assertEqual(config["format"], "+1 (XXX) XXX-XXXX")
        self.assertEqual(config["validate_source"], True)
        self.assertEqual(config["handle_invalid_phone"], "generate_new")
        self.assertEqual(config["default_country"], "us")
        self.assertEqual(config["preserve_country_code"], True)
        self.assertEqual(config["preserve_operator_code"], False)
        self.assertEqual(config["region"], "us")
        self.assertEqual(config["key"], "my-secret-key")
        self.assertEqual(config["context_salt"], "context-123")


class TestProcessValue(unittest.TestCase):

    def setUp(self):
        # Mock generator and its methods
        self.mock_generator = Mock()
        self.mock_generator.generate_like.return_value = "+1 (123) 456-7890"
        self.mock_generator.extract_country_code.return_value = "us"

        # Mock mapping store (simulate not found)
        self.mock_mapping_store = Mock()
        self.mock_mapping_store.get_mapping.return_value = None
        self.mock_mapping_store.add_mapping = Mock()

        # Initialize operation
        self.op = FakePhoneOperation(field_name="phone_number")
        self.op.generator = self.mock_generator
        self.op.mapping_store = self.mock_mapping_store
        self.op.consistency_mechanism = "mapping"
        self.op.detailed_metrics = True
        self.op.max_retries = 2

        # Metrics and counters
        self.op._generation_times = []
        self.op._country_stats = Counter()
        self.op._format_stats = Counter()
        self.op.retry_count = 0
        self.op.error_count = 0

    def test_process_value_returns_existing_mapping(self):
        # Mock the return value from the mapping store (mapped phone number)
        self.mock_mapping_store.get_mapping.return_value = "+44 7700 900124"

        # Input phone number
        input_value = "+44 7700 900123"  # realistic UK mobile number

        # Call the method to test
        result = self.op.process_value(input_value)

        # Assert that the result returned is the mapped value
        self.assertEqual(result, "+44 7700 900124")

        # Assert that get_mapping was called correctly
        self.mock_mapping_store.get_mapping.assert_called_once_with("phone_number", input_value)

        # Assert that extract_country_code was called with the mapped phone number
        self.mock_generator.extract_country_code.assert_called_once_with("+44 7700 900124")

        # Assert that country statistics were updated correctly
        self.assertEqual(self.op._country_stats["us"], 1)

        # Assert that generation times were logged
        self.assertEqual(len(self.op._generation_times), 1)

    def test_process_value_generate_new_and_store_mapping(self):
        self.mock_mapping_store.get_mapping.return_value = None

        # Use a realistic phone number as the input
        input_value = "+1 415-555-1234"  # A realistic US phone number

        result = self.op.process_value(input_value)

        # Assert that the result is a newly generated phone number
        self.assertEqual(result, "+1 (123) 456-7890")

        # Assert that generate_like was called with the input phone number
        self.mock_generator.generate_like.assert_called_once_with(input_value)

        # Assert that the mapping was added for this phone number
        self.mock_mapping_store.add_mapping.assert_called_once_with(
            "phone_number", input_value, "+1 (123) 456-7890"
        )

        # Assert that country statistics were updated
        self.assertEqual(self.op._country_stats["us"], 1)

        # Assert that the format statistics for international parentheses with dashes are updated
        self.assertEqual(self.op._format_stats["international_parentheses_dashes"], 1)

        # Assert that generation times were logged
        self.assertEqual(len(self.op._generation_times), 1)

    def test_process_value_retries_then_success(self):
        self.mock_generator.generate_like.side_effect = [Exception("fail"), "+1 999-888-7777"]

        # Use a realistic phone number for the input value
        input_value = "+1 202-555-0123"  # A realistic US phone number

        result = self.op.process_value(input_value)

        # Assert that the result is the successfully generated phone number after retries
        self.assertEqual(result, "+1 999-888-7777")

        # Assert that one retry attempt was made
        self.assertEqual(self.op.retry_count, 1)

        # Assert that there were no errors encountered
        self.assertEqual(self.op.error_count, 0)

        # Assert that country statistics for 'us' were updated
        self.assertEqual(self.op._country_stats["us"], 1)

        # Assert that the generation time was logged
        self.assertEqual(len(self.op._generation_times), 1)

    def test_process_value_all_retries_fail_returns_original(self):
        self.mock_generator.generate_like.side_effect = Exception("fail")

        result = self.op.process_value("fail_all")

        self.assertEqual(result, "fail_all")
        self.assertEqual(self.op.retry_count, 3)  # initial + 2 retries
        self.assertEqual(self.op.error_count, 1)
        self.assertEqual(len(self.op._generation_times), 0)

    def test_process_value_returns_none_for_nan(self):
        import numpy as np
        self.mock_generator.generate_like.side_effect = Exception("fail")

        result = self.op.process_value(np.nan)

        self.assertIsNone(result)
        self.assertEqual(self.op.retry_count, 3)
        self.assertEqual(self.op.error_count, 1)
        self.assertEqual(len(self.op._generation_times), 0)

    def test_process_value_format_stats_international_spaces(self):
        self.mock_generator.generate_like.return_value = "+33 1 23 45 67 89"
        self.mock_generator.extract_country_code.return_value = "fr"

        # Use a realistic French phone number input
        input_value = "+33 1 42 68 90 12"  # A realistic French phone number

        result = self.op.process_value(input_value)

        # Assert that the generated value matches the expected output
        self.assertEqual(result, "+33 1 23 45 67 89")

        # Assert that the format stats for "international_spaces" were updated
        self.assertEqual(self.op._format_stats["international_spaces"], 1)

        # Assert that the country stats for "fr" were updated
        self.assertEqual(self.op._country_stats["fr"], 1)

        # Assert that the generation time was recorded
        self.assertEqual(len(self.op._generation_times), 1)


class TestProcessBatch(unittest.TestCase):

    def setUp(self):
        self.op = FakePhoneOperation(field_name="phone_number")
        self.op.type_field = "type"
        self.op.region_field = "region"
        self.op.id_field = "id"
        self.op.output_field_name = "phone_number_enriched"
        self.op.column_prefix = "_"
        self.op.mode = "ENRICH"
        self.op.region = "en"
        self.op.context_salt = "phone-gen-salt"
        self.op.null_strategy = "PRESERVE"
        self.op.detailed_metrics = True  # Ensure detailed metrics is enabled
        self.op.process_value = Mock(side_effect=lambda v, **p: f"gen_{v}" if pd.notna(v) else np.nan)
        self.op.process_count = 0
        self.op._country_stats = Counter()
        self.op._format_stats = Counter()
        self.op._generation_times = []

    def test_process_batch_with_valid_values(self):
        # Prepare a DataFrame with valid values
        data = {
            'phone_number': ['123-456-7890', '987-654-3210'],
            'type': ['mobile', 'mobile'],
            'region': ['us', 'us'],
            'id': [1, 2]
        }
        batch = pd.DataFrame(data)

        # Process the batch
        result = self.op.process_batch(batch)

        # Assert that the generated values are as expected
        self.assertEqual(result['phone_number_enriched'][0], 'gen_123-456-7890')
        self.assertEqual(result['phone_number_enriched'][1], 'gen_987-654-3210')

        # Ensure process_value was called for each phone number
        self.op.process_value.assert_any_call('123-456-7890', country_code=None, region='en', international_format=True,
                                              local_formatting=False, context_salt='phone-gen-salt', record_id=1)
        self.op.process_value.assert_any_call('987-654-3210', country_code=None, region='en', international_format=True,
                                              local_formatting=False, context_salt='phone-gen-salt', record_id=2)

        # Check if statistics were updated
        self.assertEqual(self.op.process_count, 2)

    def test_process_batch_with_null_values(self):
        # Prepare a DataFrame with null values
        data = {
            'phone_number': ['123-456-7890', None],
            'type': ['mobile', 'mobile'],
            'region': ['us', 'us'],
            'id': [1, 2]
        }
        batch = pd.DataFrame(data)

        # Process the batch
        result = self.op.process_batch(batch)

        # Assert that the null value is handled according to the null strategy
        self.assertEqual(result['phone_number_enriched'][0], 'gen_123-456-7890')
        self.assertTrue(pd.isna(result['phone_number_enriched'][1]))  # Expect NaN due to PRESERVE strategy

        # Ensure process_value was called for the non-null value
        self.op.process_value.assert_called_once_with('123-456-7890', country_code=None, region='en', international_format=True, local_formatting=False, context_salt='phone-gen-salt', record_id=1)

        # Check if statistics were updated
        self.assertEqual(self.op.process_count, 1)


class PrepareData:

    def create_data_source(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "phone_number": ["+1-202-555-0100", "2025550123", None],
            "country_code": ["us", "us", "us"]
        })

    def create_kwargs(self):
        return {
            "field_name": "phone_number",
            "mode": "ENRICH",
            "output_field_name": "phone_number_enriched",
            "country_codes": {"us": 1.0},
            "operator_codes_dict": None,
            "format": "+1 (XXX) XXX-XXXX",
            "validate_source": True,
            "handle_invalid_phone": "generate_new",
            "default_country": "us",
            "preserve_country_code": True,
            "preserve_operator_code": False,
            "region": "us",
            "chunk_size": 10000,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "prgn",
            "mapping_store_path": "C:/fake_data/phone_operation/mappings.json",
            "id_field": "id",
            "key": "my-secret-key",
            "context_salt": "context-123",
            "save_mapping": True,
            "column_prefix": "_",
            "international_format": True,
            "local_formatting": False,
            "country_code_field": "country_code",
            "detailed_metrics": True,
            "error_logging_level": "INFO",
            "max_retries": 3
        }

    def create_task_dir(self):
        task_dir = Path("test_task_dir/fake_data/unittest/operation/phone")
        os.makedirs(task_dir, exist_ok=True)
        return task_dir


class TestFakePhoneOperationExecute(unittest.TestCase):

    def setUp(self):
        prepare_data = PrepareData()
        self.task_dir = prepare_data.create_task_dir()
        self.data_source = prepare_data.create_data_source()
        self.kwargs = prepare_data.create_kwargs()
        self.metrics_data = {
            "phone_generator": {"format": "+1 (XXX) XXX-XXXX", "default_country": "us"},
            "original_data": {"a": 1},
            "generated_data": {"a": 2}
        }

        if get_operation_class("FakePhoneOperation") is not None:
            unregister_operation("FakePhoneOperation")

    def test_execute_success_with_enrich_mode(self):
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakePhoneOperation(**self.kwargs)
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
                self.assertEqual(result.metrics["output_field"]["name"], "phone_number_enriched")
                self.assertIsInstance(result.metrics["original_data"], dict)
                self.assertIsInstance(result.metrics["generated_data"], dict)
                self.assertEqual(len(result.metrics["original_data"]), len(result.metrics["generated_data"]))

            # Check that execution time is recorded
            self.assertIsInstance(result.execution_time, float)

    def test_execute_success_with_replace_mode(self):
        self.kwargs["mode"] = "REPLACE"
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakePhoneOperation(**self.kwargs)
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
        df_data_source = self.data_source.drop(columns=["phone_number"])
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakePhoneOperation(**self.kwargs)
        
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
            self.assertIn("phone_number", result.error_message.lower())


    def test_perfect_match(self):
        original = pd.Series(["+1-202-555-0100", "2025550123"])
        generated = pd.Series(["+1-202-555-0100", "2025550123"])
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIn("length_similarity", metrics)
        self.assertEqual(metrics["length_similarity"], 1.0)

    def test_all_nulls_replaced(self):
        original = pd.Series([None, None])
        generated = pd.Series(["+1-202-555-0100", "2025550123"])
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        assert metrics == {}

    def test_all_unique_generated(self):
        original = pd.Series(["+1-202-555-0100", "+1-202-555-0100"])
        generated = pd.Series(["+1-202-555-0101", "+1-202-555-0102"])
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIn("length_similarity", metrics)
        self.assertEqual(metrics["length_similarity"], 1.0)
        self.assertIn("country_code_preservation_ratio", metrics)
        self.assertEqual(metrics["country_code_preservation_ratio"], 1.0)
        self.assertIn("country_code_diversity_ratio", metrics)
        self.assertEqual(metrics["country_code_diversity_ratio"], 1.0)
        self.assertIn("format_preservation_ratio", metrics)
        self.assertEqual(metrics["format_preservation_ratio"], 1.0)

    def test_mismatched_lengths(self):
        original = None
        generated = pd.Series(["+1-202-555-0100", "2025550123"])
        with self.assertRaises(Exception):
            op = FakePhoneOperation(**self.kwargs)
            op._calculate_quality_metrics(original, generated)

    def test_partial_overlap_and_nulls(self):
        original = pd.Series(["+1-202-555-0100", None, "2025550123"])
        generated = pd.Series(["+1-202-555-0100", "+1-202-555-0109", None])
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIn("length_similarity", metrics)
        self.assertGreater(metrics["length_similarity"], 0)
        self.assertLess(metrics["length_similarity"], 1)

    def test_empty_input(self):
        original = pd.Series([])
        generated = pd.Series([])
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(original, generated)
        self.assertIsInstance(metrics, dict)


    def test_international_parentheses_dashes(self):
        df = self.data_source
        df["phone_number_enriched"] = "+018-75(421)"
        op = FakePhoneOperation(**self.kwargs)
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        self.assertEqual(result["format_distribution"]["international_parentheses_dashes"], 1.0)

    def test_international_spaces(self):
        df = self.data_source
        df["phone_number_enriched"] = "+33 1 23 45 67 89"
        df["phone_number_enriched2"] = "+49 30 123456"
        op = FakePhoneOperation(**self.kwargs)
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        self.assertEqual(result["format_distribution"]["international_spaces"], 1.0)

    def test_local_dots(self):
        df = self.data_source
        df["phone_number_enriched"] = "123.456.7890"
        df["phone_number_enriched2"] = "555.123.4567"
        op = FakePhoneOperation(**self.kwargs)
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        self.assertEqual(result["format_distribution"]["unknown"], 1.0)

    def test_local_dashes(self):
        df = self.data_source
        df["phone_number_enriched"] = "(123)-456-7890"
        df["phone_number_enriched2"] = "(555)-123-4567"
        op = FakePhoneOperation(**self.kwargs)
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        self.assertEqual(result["format_distribution"]["local_parentheses_dashes"], 1.0)

    def test_mixed_formats(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7],
            "phone_number_enriched": [
                "+1 (123) 456-7890",
                "123-456-7890",
                "+33 1 23 45 67 89",
                "555.123.4567",
                None,
                "(123)-456-7890",
                "+1234567890"
            ]
        })
        op = FakePhoneOperation(**self.kwargs)
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        assert result["format_distribution"]["international_parentheses_dashes"] > 0.0
        assert result["format_distribution"]["international_spaces"] > 0.0
        assert result["format_distribution"]["e164"] > 0.0
        assert result["format_distribution"]["local_parentheses_dashes"] > 0.0
        assert result["format_distribution"]["unknown"] > 0.0
        
    def test_mixed_formats_REPLACE(self):
        self.kwargs["mode"] = "REPLACE"
        op = FakePhoneOperation(**self.kwargs)
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7],
            op.field_name: [
                "+1 (123) 456-7890",
                "123-456-7890",
                "+33 1 23 45 67 89",
                "555.123.4567",
                None,
                "(123)-456-7890",
                "+1234567890"
            ]
        })
        
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        assert result["format_distribution"]["international_parentheses_dashes"] > 0.0
        assert result["format_distribution"]["international_spaces"] > 0.0
        assert result["format_distribution"]["e164"] > 0.0
        assert result["format_distribution"]["local_parentheses_dashes"] > 0.0
        assert result["format_distribution"]["unknown"] > 0.0

    def test_empty_and_nulls(self):
        df = pd.DataFrame({
            "id": [1, 2, 3,],
            "phone_number_enriched": [
                None,
                "",
                np.nan
            ]
        })
        op = FakePhoneOperation(**self.kwargs)
        result = op._analyze_formats(df)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["format_distribution"]["unknown"], 1.0)
        
    def test_analyze_formats_format_stats(self):
        self.kwargs["mode"] = "REPLACE"
        op = FakePhoneOperation(**self.kwargs)
        op._format_stats = Counter({'a': 3, 'b': 4})
        df = pd.DataFrame({
            "id": [1, 2, 3,],
            op.field_name: [
                None,
                "",
                np.nan
            ]
        })
        result = op._analyze_formats(df)
        self.assertIn("format_distribution", result)
        assert len(result["format_distribution"]) == len(op._format_stats.most_common())

    def test_basic_metrics(self):
        op = FakePhoneOperation(**self.kwargs)
        df = self.data_source
        df["phone_number"] = "+1254558_(451)"
        metrics = op._collect_metrics(df)
        self.assertIn("phone_generator", metrics)
        self.assertIn("format", metrics["phone_generator"])
        self.assertIn("default_country", metrics["phone_generator"])
        self.assertIn("region", metrics["phone_generator"])
        self.assertIn("international_format", metrics["phone_generator"])
        self.assertIn("local_formatting", metrics["phone_generator"])
        self.assertIn("preserve_country_code", metrics["phone_generator"])
        self.assertIn("preserve_operator_code", metrics["phone_generator"])
        self.assertIn("validate_source", metrics["phone_generator"])
        self.assertIn("handle_invalid_phone", metrics["phone_generator"])

    def test_metrics_with_all_nulls(self):
        df = pd.DataFrame({"phone_number_enriched": [None, None, None]})
        df["phone_number"] = "+1254558_(451)"
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._collect_metrics(df)
        self.assertIn("phone_generator", metrics)

    def test_metrics_with_missing_columns(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        df["phone_number"] = "+1254558_(451)"
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._collect_metrics(df)
        self.assertIn("phone_generator", metrics)

    def test_metrics_with_extra_columns(self):
        df = self.data_source
        df["extra"] = [1, 2, 3]
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._collect_metrics(df)
        self.assertIn("phone_generator", metrics)

    def test_metrics_with_detailed_metrics(self):
        op = FakePhoneOperation(**self.kwargs)
        op.detailed_metrics = True
        op._generation_times = [0.1, 0.2, 0.3]
        op.start_time = time.time() - 1
        op.process_count = 3
        op.error_count = 1
        op.retry_count = 2
        df = self.data_source
        metrics = op._collect_metrics(df)
        self.assertIn("performance", metrics)
        self.assertIn("avg_record_generation_time", metrics["performance"])
        self.assertIn("min_record_generation_time", metrics["performance"])
        self.assertIn("max_record_generation_time", metrics["performance"])

    def test_metrics_with_original_df_quality(self):
        op = FakePhoneOperation(**self.kwargs)
        op.mode = "ENRICH"
        op._original_df = pd.DataFrame({
            "phone_number": ["+1-202-555-0100", "2025550123", None],
            "phone_number_enriched": ["+1-202-555-0100", "2025550123", None]
        })
        op.output_field_name = "phone_number_enriched"
        metrics = op._collect_metrics(op._original_df)
        self.assertIn("quality_metrics", metrics)
        self.assertIn("length_similarity", metrics["quality_metrics"])

    def test_metrics_country_format_exceptions(self):
        # Patch _analyze_country_distribution and _analyze_formats to raise
        op = FakePhoneOperation(**self.kwargs)
        op._analyze_country_distribution = Mock(side_effect=Exception("fail country"))
        op._analyze_formats = Mock(side_effect=Exception("fail format"))
        df = self.data_source
        metrics = op._collect_metrics(df)
        self.assertIn("phone_generator", metrics)
        # Should not raise, should log warning and continue

    def test_metrics_with_empty_df(self):
        df = self.data_source
        op = FakePhoneOperation(**self.kwargs)
        metrics = op._collect_metrics(df)
        self.assertIn("phone_generator", metrics)

    @patch("pamola_core.fake_data.operations.phone_op.io.write_json")
    @patch("pamola_core.fake_data.operations.phone_op.io.ensure_directory")
    @patch("pamola_core.fake_data.operations.phone_op.metrics.create_metrics_collector")
    def test_save_metrics_file_creation(self, mock_collector, mock_ensure_dir, mock_write_json):
        mock_collector.return_value.visualize_metrics.return_value = {}
        op = FakePhoneOperation(**self.kwargs)
        path = op._save_metrics(self.metrics_data, self.task_dir)
        self.assertTrue(str(path).endswith("_metrics.json"))
        mock_ensure_dir.assert_any_call(self.task_dir / "metrics")
        mock_write_json.assert_called_once()

    @patch("pamola_core.fake_data.operations.phone_op.io.write_json")
    @patch("pamola_core.fake_data.operations.phone_op.io.ensure_directory")
    @patch("pamola_core.fake_data.operations.phone_op.metrics.create_metrics_collector")
    def test_save_metrics_with_visualizations(self, mock_collector, mock_ensure_dir, mock_write_json):
        # Simulate visualizations
        mock_collector.return_value.visualize_metrics.return_value = {"viz1": Path("viz1.png")}
        op = FakePhoneOperation(**self.kwargs)
        op.mode = "ENRICH"
        op._original_df = pd.DataFrame({
            "phone_number": ["+1-202-555-0100", "2025550123"],
            "phone_number_enriched": ["+1-202-555-0100", "2025550123"]
        })
        op.output_field_name = "phone_number_enriched"
        metrics_data = {
            "phone_generator": {"format": "+1 (XXX) XXX-XXXX", "default_country": "us"},
            "original_data": {"a": 1},
            "generated_data": {"a": 2}
        }
        path = op._save_metrics(metrics_data, self.task_dir)
        self.assertIn("visualizations", metrics_data)
        self.assertIn("viz1", metrics_data["visualizations"])
        self.assertTrue(str(path).endswith("_metrics.json"))

    @patch("pamola_core.fake_data.operations.phone_op.io.write_json")
    @patch("pamola_core.fake_data.operations.phone_op.io.ensure_directory")
    @patch("pamola_core.fake_data.operations.phone_op.metrics.create_metrics_collector")
    def test_save_metrics_encryption(self, mock_collector, mock_ensure_dir, mock_write_json):
        mock_collector.return_value.visualize_metrics.return_value = {}
        op = FakePhoneOperation(**self.kwargs)
        path = op._save_metrics(self.metrics_data, self.task_dir, use_encryption=True, encryption_key="abc")
        args, kwargs = mock_write_json.call_args
        self.assertEqual(kwargs["encryption_key"], "abc")
        self.assertTrue(str(path).endswith("_metrics.json"))

    @patch("pamola_core.fake_data.operations.phone_op.io.write_json")
    @patch("pamola_core.fake_data.operations.phone_op.io.ensure_directory")
    @patch("pamola_core.fake_data.operations.phone_op.metrics.create_metrics_collector")
    def test_save_metrics_visualization_exception(self, mock_collector, mock_ensure_dir, mock_write_json):
        # Simulate exception in visualization
        mock_collector.return_value.visualize_metrics.side_effect = Exception("fail viz")
        op = FakePhoneOperation(**self.kwargs)
        op.mode = "ENRICH"
        op._original_df = pd.DataFrame({
            "phone_number": ["+1-202-555-0100", "2025550123"],
            "phone_number_enriched": ["+1-202-555-0100", "2025550123"]
        })
        op.output_field_name = "phone_number_enriched"
        metrics_data = {
            "phone_generator": {"format": "+1 (XXX) XXX-XXXX", "default_country": "us"},
            "original_data": {"a": 1},
            "generated_data": {"a": 2}
        }
        # Should not raise
        path = op._save_metrics(metrics_data, self.task_dir)
        self.assertTrue(str(path).endswith("_metrics.json"))

    @patch("pamola_core.fake_data.operations.phone_op.io.write_json")
    @patch("pamola_core.fake_data.operations.phone_op.io.ensure_directory")
    @patch("pamola_core.fake_data.operations.phone_op.metrics.create_metrics_collector")
    def test_save_metrics_empty_metrics(self, mock_collector, mock_ensure_dir, mock_write_json):
        mock_collector.return_value.visualize_metrics.return_value = {}
        op = FakePhoneOperation(**self.kwargs)
        path = op._save_metrics({}, self.task_dir)
        self.assertTrue(str(path).endswith("_metrics.json"))

if __name__ == "__main__":
    unittest.main()