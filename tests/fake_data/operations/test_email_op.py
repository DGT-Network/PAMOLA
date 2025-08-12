import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from pamola_core.fake_data import FakeEmailOperation
from pamola_core.fake_data.commons.base import NullStrategy
from pamola_core.fake_data.generators.email import EmailGenerator
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
    
class TestFakeEmailOperationInit(unittest.TestCase):

    #@patch('pamola_core.fake_data.operations.email_op.register')
    #@patch('pamola_core.fake_data.operations.email_op.EmailGenerator')
    def test_initialization_with_defaults(self):
        # Initialize FakeEmailOperation with default parameters
        op = FakeEmailOperation(field_name="email")

        # Check default attributes of FakeEmailOperation
        self.assertEqual(op.field_name, "email")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "prgn")
        self.assertFalse(op.save_mapping)
        self.assertIsNone(op.mapping_store_path)
        self.assertEqual(op.error_logging_level, "WARNING")
        self.assertEqual(op.max_retries, 3)
        self.assertIsNone(op.first_name_field)
        self.assertIsNone(op.last_name_field)
        self.assertIsNone(op.full_name_field)
        self.assertIsNone(op.name_format)
        self.assertIsNone(op.id_field)
        self.assertIsNone(op.key)
        self.assertIsNone(op.context_salt)
        self.assertEqual(op.column_prefix, "_")

        # Ensure that the generator is an instance of EmailGenerator class
        self.assertIsInstance(op.generator, EmailGenerator)

        # Ensure that the generator has default values for its configuration
        self.assertIsNone(op.generator.domains)
        self.assertIsNone(op.generator.format)
        self.assertIsNone(op.generator.format_ratio)
        self.assertEqual(op.generator.validate_source, True)
        self.assertEqual(op.generator.handle_invalid_email, "generate_new")
        self.assertEqual(op.generator.max_length, 254)
        self.assertEqual(op.generator.separator_options, ['.', '_', '-', ''])
        self.assertEqual(op.generator.number_suffix_probability, 0.4)
        self.assertEqual(op.generator.preserve_domain_ratio, 0.5)
        self.assertEqual(op.generator.business_domain_ratio, 0.2)

    #@patch('pamola_core.fake_data.operations.email_op.register')
    #@patch('pamola_core.fake_data.operations.email_op.EmailGenerator')
    def test_initialization_with_parameters(self):
        kwargs = {
            "field_name": "email",
            "mode": "ENRICH",
            "output_field_name": "email_enriched",
            "domains": ["example.com", "company.org"],
            "format": "nickname",
            "format_ratio": {"{first}.{last}@{domain}": 0.7, "{f}{last}@{domain}": 0.3},
            "first_name_field": "first_name",
            "last_name_field": "last_name",
            "full_name_field": "full_name",
            "name_format": "{first} {last}",
            "validate_source": True,
            "handle_invalid_email": "generate_new",
            "nicknames_dict": None,
            "max_length": 254,
            "chunk_size": 10000,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "abcd",
            "mapping_store_path": "C:/fake_data/email_operation/mappings.json",
            "id_field": "id",
            "key": None,
            "context_salt": "email-context-001",
            "save_mapping": True,
            "column_prefix": "_",
            "separator_options": ["_", ".", ""],
            "number_suffix_probability": 0.4,
            "preserve_domain_ratio": 0.5,
            "business_domain_ratio": 0.2,
            "detailed_metrics": True,
            "error_logging_level": "WARNING",
            "max_retries": 3
        }

        op = FakeEmailOperation(**kwargs)

        # Check attributes of FakeEmailOperation
        self.assertEqual(op.field_name, "email")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "email_enriched")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "abcd")
        self.assertTrue(op.save_mapping)
        self.assertEqual(op.mapping_store_path, "C:/fake_data/email_operation/mappings.json")
        self.assertEqual(op.error_logging_level, "WARNING")
        self.assertEqual(op.max_retries, 3)
        self.assertEqual(op.first_name_field, "first_name")
        self.assertEqual(op.last_name_field, "last_name")
        self.assertEqual(op.full_name_field, "full_name")
        self.assertEqual(op.name_format, "{first} {last}")
        self.assertEqual(op.id_field, "id")
        self.assertIsNone(op.key)
        self.assertEqual(op.context_salt, "email-context-001")
        self.assertEqual(op.column_prefix, "_")

        # Ensure that _domain_stats is an instance of Counter (to track domain statistics)
        self.assertIsInstance(op._domain_stats, Counter)

        # Ensure that _format_stats is an instance of Counter (to track format statistics)
        self.assertIsInstance(op._format_stats, Counter)

        # Ensure that the generator is an instance of EmailGenerator class
        self.assertIsInstance(op.generator, EmailGenerator)

        # Check configuration inside EmailGenerator
        self.assertEqual(op.generator.domains, ["example.com", "company.org"])
        self.assertEqual(op.generator.format, "nickname")
        self.assertEqual(op.generator.format_ratio, {"{first}.{last}@{domain}": 0.7, "{f}{last}@{domain}": 0.3})
        self.assertEqual(op.generator.validate_source, True)
        self.assertEqual(op.generator.handle_invalid_email, "generate_new")
        self.assertEqual(op.generator.max_length, 254)
        self.assertEqual(op.generator.separator_options, ["_", ".", ""])
        self.assertEqual(op.generator.number_suffix_probability, 0.4)
        self.assertEqual(op.generator.preserve_domain_ratio, 0.5)
        self.assertEqual(op.generator.business_domain_ratio, 0.2)


class TestFakeEmailOperationProcessValue(unittest.TestCase):

    def setUp(self):
        self.mock_generator = Mock()
        self.mock_generator.generate_like.return_value = "synthetic@example.com"
        self.mock_generator.parse_email_format.return_value = "{first}.{last}@{domain}"

        self.mock_mapping_store = Mock()
        self.mock_mapping_store.get_mapping.return_value = None

        self.op = FakeEmailOperation(field_name="email")
        self.op.generator = self.mock_generator
        self.op.mapping_store = self.mock_mapping_store
        self.op.consistency_mechanism = "mapping"
        self.op.detailed_metrics = True
        self.op.max_retries = 2
        self.op._domain_stats = Counter()
        self.op._format_stats = Counter()
        self.op._generation_times = []
        self.op.retry_count = 0
        self.op.error_count = 0

    def test_process_value_with_existing_mapping(self):
        self.mock_mapping_store.get_mapping.return_value = "mapped@example.com"
        result = self.op.process_value("user@example.com", first_name="John", last_name="Doe")
        self.assertEqual(result, "mapped@example.com")
        self.mock_mapping_store.get_mapping.assert_called_once()

    def test_process_value_with_new_mapping_added(self):
        self.op.mapping_store.get_mapping.return_value = None
        result = self.op.process_value("user@example.com", first_name="John", last_name="Doe")
        self.assertEqual(result, "synthetic@example.com")
        self.mock_mapping_store.add_mapping.assert_called_once_with(
            "email", "user@example.com", "synthetic@example.com"
        )

    def test_process_value_retry_one_time(self):
        self.mock_generator.generate_like.side_effect = [Exception("fail1"), "recovered@example.com"]
        result = self.op.process_value("user@example.com", first_name="Jane")
        self.assertEqual(result, "recovered@example.com")
        self.assertEqual(self.op.retry_count, 1)

    def test_process_value_retry_all_time(self):
        self.mock_generator.generate_like.side_effect = Exception("failure")
        result = self.op.process_value("user@example.com", first_name="Jane")
        self.assertEqual(result, "user@example.com")  # fallback to original
        self.assertEqual(self.op.retry_count, 3)  # max_retries + 1
        self.assertEqual(self.op.error_count, 1)

    def test_process_value_return_none_if_input_is_nan(self):
        self.mock_generator.generate_like.side_effect = Exception("error")
        result = self.op.process_value(float("nan"))
        self.assertIsNone(result)

    @patch("pamola_core.fake_data.commons.prgn.PRNGenerator")
    def test_process_value_with_prgn_consistency(self, MockPRGN):
        self.op.consistency_mechanism = "prgn"
        self.op.generator.prgn_generator = None
        self.mock_generator.generate_like.return_value = "prgn@example.com"

        result = self.op.process_value("user@example.com", first_name="PRGN")
        self.assertEqual(result, "prgn@example.com")
        MockPRGN.assert_called_once()


class TestFakeEmailOperationProcessBatch(unittest.TestCase):
    def setUp(self):
        self.op = FakeEmailOperation(field_name="email")
        self.op.process_value = Mock(side_effect=lambda v, **kwargs: f"generated_{v}")
        self.op.process_count = 0
        self.op.null_strategy = "PRESERVE"
        self.op.mode = "REPLACE"
        self.op.first_name_field = "first_name"
        self.op.last_name_field = "last_name"
        self.op.full_name_field = "full_name"
        self.op.id_field = "id"
        self.op.name_format = "{first}.{last}@example.com"
        self.op.context_salt = "test"
        self.op.column_prefix = "syn_"
        self.op.output_field_name = None

    def test_replace_mode_with_valid_values(self):
        df = pd.DataFrame({
            "email": ["a@example.com", "b@example.com"],
            "first_name": ["Alice", "Bob"],
            "last_name": ["Smith", "Jones"],
            "id": [1, 2]
        })

        result = self.op.process_batch(df)

        expected = ["generated_a@example.com", "generated_b@example.com"]
        self.assertListEqual(result["email"].tolist(), expected)
        self.assertEqual(self.op.process_count, 2)

    def test_enrich_mode_creates_new_column(self):
        self.op.mode = "ENRICH"
        df = pd.DataFrame({
            "email": ["c@example.com"],
            "first_name": ["Carol"],
            "last_name": ["White"]
        })

        result = self.op.process_batch(df)

        self.assertIn("syn_email", result.columns)
        self.assertEqual(result["syn_email"][0], "generated_c@example.com")

    def test_preserve_null_strategy(self):
        self.op.null_strategy = "PRESERVE"
        df = pd.DataFrame({"email": [None]})

        result = self.op.process_batch(df)
        self.assertTrue(pd.isna(result["email"][0]))
        self.assertEqual(self.op.process_count, 0)

    def test_exclude_null_strategy(self):
        self.op.null_strategy = "EXCLUDE"
        df = pd.DataFrame({"email": [None]})

        result = self.op.process_batch(df)
        self.assertTrue(pd.isna(result["email"][0]))
        self.assertEqual(self.op.process_count, 0)

    def test_error_on_null_value(self):
        self.op.null_strategy = "ERROR"
        df = pd.DataFrame({"email": [None]})

        with self.assertRaises(ValueError):
            self.op.process_batch(df)

    def test_fallback_to_original_on_process_value_failure(self):
        def fail_then_pass(val, **kwargs):
            raise Exception("Boom")

        self.op.process_value = Mock(side_effect=fail_then_pass)
        df = pd.DataFrame({"email": ["fail@example.com"]})

        result = self.op.process_batch(df)
        self.assertEqual(result["email"][0], "fail@example.com")

    def test_missing_optional_name_fields(self):
        self.op.first_name_field = "first_name"
        df = pd.DataFrame({"email": ["x@example.com"]})  # first_name missing

        # Should not raise, still call process_value
        result = self.op.process_batch(df)
        self.assertEqual(result["email"][0], "generated_x@example.com")

    def test_record_id_collected_if_available(self):
        df = pd.DataFrame({
            "email": ["a@example.com"],
            "id": [42]
        })
        called_kwargs = {}

        def capturing_process_value(val, **kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return "gen"

        self.op.process_value = Mock(side_effect=capturing_process_value)
        self.op.process_batch(df)
        self.assertEqual(called_kwargs["record_id"], 42)


class PrepareData:

    def create_data_source(self):
        return pd.DataFrame({
            "id": list(range(1, 11)),
            "first_name": ["John", "Jane", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
            "last_name": ["Doe", "Smith", "Nguyen", "Brown", "Wilson", "Taylor", "Clark", "Hall", "Lewis", "Young"],
            "full_name": [
                "John Doe", "Jane Smith", "Alice Nguyen", "Bob Brown", "Charlie Wilson",
                "Diana Taylor", "Eve Clark", "Frank Hall", "Grace Lewis", "Henry Young"
            ],
            "gender": ["male", "female", "female", "male", "male", "female", "female", "male", "female", "male"],
            "email": [
                "johndoe@example.com", "janesmith@example.com", "alicenguyen@example.com",
                "bobbrown@example.com", "charliewilson@example.com", "dianataylor@example.com",
                "eveclark@example.com", "frankhall@example.com", "gracelewis@example.com", "henryyoung@example.com"
            ]
        })


    def create_kwargs(self):
        return {
            "field_name": "email",
            "mode": "ENRICH",
            "output_field_name": "email_enriched",
            "domains": ["example.com", "company.org"],
            "format": "nickname",
            "format_ratio": {"{first}.{last}@{domain}": 0.7, "{f}{last}@{domain}": 0.3},
            "first_name_field": "first_name",
            "last_name_field": "last_name",
            "full_name_field": "full_name",
            "name_format": "{first} {last}",
            "validate_source": True,
            "handle_invalid_email": "generate_new",
            "nicknames_dict": None,
            "max_length": 254,
            "chunk_size": 3,
            "null_strategy": "PRESERVE",
            "consistency_mechanism": "prgn",   # "prgn", "mapping"
            "mapping_store_path": "C:/operation/fake_data/email/maps/mappings.json",
            "id_field": "id",
            "key": None,
            "context_salt": "email-context-001",
            "save_mapping": True,
            "column_prefix": "_",
            "separator_options": ["_", ".", ""],
            "number_suffix_probability": 0.4,
            "preserve_domain_ratio": 0.5,
            "business_domain_ratio": 0.2,
            "detailed_metrics": True,
            "error_logging_level": "WARNING",
            "max_retries": 3,
            "use_cache": True,
            "force_recalculation": True,
            "use_dask": False,
            "npartitions": 2,
            "use_vectorization": False,
            "parallel_processes": 2,
            "use_encryption": False,
            "encryption_key":  None,
            "visualization_backend": "plotly",
            "visualization_theme": None,
            "visualization_strict": False
        }

    def create_task_dir(self):
        task_dir = Path("test_task_dir/fake_data/unittest/operation/email")
        os.makedirs(task_dir, exist_ok=True)
        return task_dir


class TestFakeEmailOperationExecute(unittest.TestCase):

    def setUp(self):
        prepare_data = PrepareData()
        self.task_dir = prepare_data.create_task_dir()
        self.data_source = prepare_data.create_data_source()
        self.kwargs = prepare_data.create_kwargs()

        if get_operation_class("FakeEmailOperation") is not None:
            unregister_operation("FakeEmailOperation")

    def test_execute_success_with_enrich_mode(self):
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)
        op = FakeEmailOperation(**self.kwargs)
        
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
                self.assertEqual(result.metrics["output_field"]["name"], "email_enriched")
                self.assertIsInstance(result.metrics["original_data"], dict)
                self.assertIsInstance(result.metrics["generated_data"], dict)
                self.assertEqual(len(result.metrics["original_data"]), len(result.metrics["generated_data"]))

            # Check that execution time is recorded
            self.assertIsInstance(result.execution_time, float)

    def test_execute_success_with_replace_mode(self):
        self.kwargs["mode"] = "REPLACE"
        df_data_source = self.data_source
        df = DummyDataSource(df_data_source)
        op = FakeEmailOperation(**self.kwargs)
        
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
        df_data_source = self.data_source.drop(columns=["email"])
        df = DummyDataSource(df_data_source)   # Missing field_name column in data_source
        op = FakeEmailOperation(**self.kwargs)
        
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
            self.assertIn("email", result.error_message.lower())

    def test_calculate_quality_metrics_all_nan(self):
        import pandas as pd
        orig = pd.Series([None, None, None])
        gen = pd.Series([None, None, None])
        op = FakeEmailOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(orig, gen)
        assert isinstance(metrics, dict)
        assert metrics == {}

    def test_calculate_quality_metrics_partial_nan(self):
        orig = pd.Series(['a@x.com', None, 'c@z.com'])
        gen = pd.Series(['a@x.com', 'b@y.com', None])
        op = FakeEmailOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(orig, gen)
        assert 'length_similarity' in metrics
        assert 'domain_preservation_ratio' in metrics
        assert 'domain_diversity_ratio' in metrics
        assert 'local_part_length_similarity' in metrics
        assert 'separator_similarity' in metrics

    def test_calculate_quality_metrics_no_at_symbol(self):
        orig = pd.Series(['axcom', 'bycom', 'czcom'])
        gen = pd.Series(['axcom', 'bycom', 'czcom'])
        op = FakeEmailOperation(**self.kwargs)
        metrics = op._calculate_quality_metrics(orig, gen)
        # Should not raise, but metrics may be missing domain/local part keys
        assert isinstance(metrics, dict)

    def test_analyze_domain_distribution_with_none(self):
        op = FakeEmailOperation(**self.kwargs)
        op._domain_stats = None
        df = self.data_source
        result = op._analyze_domain_distribution(df)
        # Should return an empty dict or handle None gracefully
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    def test_analyze_domain_distribution_with_REPLACE_mode(self):
        self.kwargs["mode"] = "REPLACE"
        op = FakeEmailOperation(**self.kwargs)
        op._domain_stats = None
        df = self.data_source
        result = op._analyze_domain_distribution(df)
        self.assertIsInstance(result, dict)
        assert result['total_emails'] > 0

if __name__ == "__main__":
    unittest.main()