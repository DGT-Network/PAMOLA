import unittest
from pathlib import Path
from unittest.mock import Mock, patch, call, ANY
import pandas as pd
from pamola_core.common.constants import Constants
from pamola_core.fake_data.commons.base import BaseGenerator, NullStrategy, ValidationError
from pamola_core.fake_data.commons.mapping_store import MappingStore
from pamola_core.fake_data.commons.operations import GeneratorOperation, FieldOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class TestGeneratorOperationInit(unittest.TestCase):

    def test_initialization_with_defaults(self):
        mock_generator = Mock(spec=BaseGenerator)

        op = GeneratorOperation(
            field_name="test_field",
            generator=mock_generator
        )

        self.assertEqual(op.field_name, "test_field")
        self.assertEqual(op.generator, mock_generator)
        self.assertEqual(op.mode, "REPLACE")
        self.assertEqual(op.chunk_size, 10000)
        self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
        self.assertEqual(op.consistency_mechanism, "prgn")
        self.assertIsInstance(op.mapping_store, MappingStore)
        self.assertEqual(op.generator_params, {})
        self.assertTrue(op.use_cache)
        self.assertFalse(op.force_recalculation)
        self.assertFalse(op.use_dask)
        self.assertEqual(op.npartitions, 1)
        self.assertFalse(op.use_vectorization)
        self.assertEqual(op.parallel_processes, 1)
        self.assertFalse(op.use_encryption)
        self.assertIsNone(op.encryption_key)
        self.assertIsNone(op.visualization_backend)
        self.assertIsNone(op.visualization_theme)
        self.assertFalse(op.visualization_strict)
        self.assertIn("Operation for generating fake", op.description)

    def test_initialization_with_parameters(self):
        mock_generator = Mock(spec=BaseGenerator)
        custom_mapping_store = MappingStore()
        custom_params = {"locale": "en_US"}

        op = GeneratorOperation(
            field_name="email",
            generator=mock_generator,
            mode="ENRICH",
            output_field_name="synthetic_email",
            chunk_size=5000,
            null_strategy=NullStrategy.EXCLUDE,
            consistency_mechanism="mapping",
            mapping_store=custom_mapping_store,
            generator_params=custom_params,
            use_cache=False,
            force_recalculation=True,
            use_dask=True,
            npartitions=4,
            use_vectorization=True,
            parallel_processes=2,
            use_encryption=True,
            encryption_key="secret",
            visualization_backend="plotly",
            visualization_theme="dark",
            visualization_strict=True
        )

        self.assertEqual(op.field_name, "email")
        self.assertEqual(op.generator, mock_generator)
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "synthetic_email")
        self.assertEqual(op.chunk_size, 5000)
        self.assertEqual(op.null_strategy, NullStrategy.EXCLUDE)
        self.assertEqual(op.consistency_mechanism, "mapping")
        self.assertEqual(op.mapping_store, custom_mapping_store)
        self.assertEqual(op.generator_params, custom_params)
        self.assertFalse(op.use_cache)
        self.assertTrue(op.force_recalculation)
        self.assertTrue(op.use_dask)
        self.assertEqual(op.npartitions, 4)
        self.assertTrue(op.use_vectorization)
        self.assertEqual(op.parallel_processes, 2)
        self.assertTrue(op.use_encryption)
        self.assertEqual(op.encryption_key, "secret")
        self.assertEqual(op.visualization_backend, "plotly")
        self.assertEqual(op.visualization_theme, "dark")
        self.assertTrue(op.visualization_strict)


class TestGeneratorOperationProcessBatch(unittest.TestCase):

    def setUp(self):
        self.mock_generator = Mock(spec=BaseGenerator)
        self.mock_generator.generate_like.return_value = "synthetic"

    def create_operation(self, mode="REPLACE", null_strategy=NullStrategy.PRESERVE):
        op = GeneratorOperation(
            field_name="name",
            generator=self.mock_generator,
            mode=mode,
            null_strategy=null_strategy
        )
        op._process_value = Mock(side_effect=lambda x: f"synthetic_{x}" if x else "synthetic_NULL")
        return op

    def test_replace_mode_with_valid_values(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        op = self.create_operation(mode="REPLACE")
        result = op.process_batch(df)
        self.assertListEqual(result["name"].tolist(), ["synthetic_Alice", "synthetic_Bob"])

    def test_enrich_mode_with_valid_values(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        op = self.create_operation(mode="ENRICH")
        op.output_field_name = "synthetic_name"
        result = op.process_batch(df)
        self.assertListEqual(result["synthetic_name"].tolist(), ["synthetic_Alice", "synthetic_Bob"])
        self.assertListEqual(result["name"].tolist(), ["Alice", "Bob"])  # original unchanged

    def test_null_preserve(self):
        df = pd.DataFrame({"name": ["Alice", None, "Bob"]})
        op = self.create_operation(mode="REPLACE", null_strategy=NullStrategy.PRESERVE)
        result = op.process_batch(df)
        self.assertListEqual(result["name"].tolist(), ["synthetic_Alice", None, "synthetic_Bob"])

    def test_null_replace(self):
        df = pd.DataFrame({"name": [None, "Charlie"]})
        op = self.create_operation(mode="REPLACE", null_strategy=NullStrategy.REPLACE)
        result = op.process_batch(df)
        self.assertListEqual(result["name"].tolist(), ["synthetic_NULL", "synthetic_Charlie"])

    def test_all_nulls_returns_same_batch(self):
        df = pd.DataFrame({"name": [None, None]})
        op = self.create_operation(mode="ENRICH", null_strategy=NullStrategy.PRESERVE)
        op.output_field_name = "synthetic_name"
        result = op.process_batch(df)
        self.assertTrue((result["synthetic_name"].isna()).all())

    def test_output_structure_is_preserved(self):
        df = pd.DataFrame({"name": ["X"], "other": [123]})
        op = self.create_operation(mode="ENRICH")
        op.output_field_name = "synthetic_name"
        result = op.process_batch(df)
        self.assertIn("name", result.columns)
        self.assertIn("other", result.columns)
        self.assertIn("synthetic_name", result.columns)


class TestGeneratorOperationHandleNullValues(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            "name": ["Alice", None, "Bob", None]
        })
        self.mock_generator = Mock(spec=BaseGenerator)

    def create_op(self, null_strategy):
        return GeneratorOperation(
            field_name="name",
            generator=self.mock_generator,
            null_strategy=null_strategy
        )

    def test_error_strategy_raises_validation_error(self):
        op = self.create_op(NullStrategy.ERROR)
        with self.assertRaises(ValidationError) as context:
            op.handle_null_values(self.data)

        self.assertIn("NULL values", str(context.exception))

    def test_exclude_strategy_filters_nulls(self):
        op = self.create_op(NullStrategy.EXCLUDE)
        result = op.handle_null_values(self.data)
        self.assertEqual(len(result), 2)
        self.assertListEqual(result["name"].tolist(), ["Alice", "Bob"])

    def test_preserve_strategy_returns_as_is(self):
        op = self.create_op(NullStrategy.PRESERVE)
        result = op.handle_null_values(self.data)
        pd.testing.assert_frame_equal(result, self.data)

    def test_replace_strategy_returns_as_is(self):
        op = self.create_op(NullStrategy.REPLACE)
        result = op.handle_null_values(self.data)
        pd.testing.assert_frame_equal(result, self.data)


class TestGeneratorOperationExecute(unittest.TestCase):
    def setUp(self):
        self.generator = Mock(spec=BaseGenerator)
        self.mapping_store = Mock(spec=MappingStore)
        self.mapping_store.get_field_mappings.return_value = {
            "Alice": "SyntheticAlice",
            "Bob": "SyntheticBob"
        }

        self.op = GeneratorOperation(
            field_name="name",
            generator=self.generator,
            consistency_mechanism="mapping",
            mapping_store=self.mapping_store
        )

        self.task_dir = Path("/fake/task_dir")
        self.reporter = Mock()
        self.progress_tracker = Mock()

    @patch("pamola_core.fake_data.commons.operations.write_json")
    @patch("pamola_core.fake_data.commons.operations.ensure_directory")
    @patch("pamola_core.fake_data.commons.operations.FieldOperation.execute")
    def test_execute_saves_mappings_on_success(self, mock_super_execute, mock_ensure_dir, mock_write_json):
        # Mock super().execute() return value
        mock_result = Mock(spec=OperationResult)
        mock_result.status = OperationStatus.SUCCESS
        mock_result.add_artifact = Mock()
        mock_super_execute.return_value = mock_result

        # Mock map dir
        mock_ensure_dir.return_value = self.task_dir / "maps"

        result = self.op.execute(
            data_source=Mock(),
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        # Ensure write_json was called twice
        self.assertEqual(mock_write_json.call_count, 2)

        # Build expected filenames using actual `self.op.name`
        expected_serialized = [
            {"original": "Alice", "synthetic": "SyntheticAlice", "field": "name"},
            {"original": "Bob", "synthetic": "SyntheticBob", "field": "name"},
        ]
        expected_main_file = self.task_dir / f"{self.op.name}_{self.op.field_name}_mappings.json"
        expected_detail_file = self.task_dir / "maps" / f"{self.op.field_name}_mappings.json"

        mock_write_json.assert_any_call(expected_serialized, expected_main_file)
        mock_write_json.assert_any_call(expected_serialized, expected_detail_file)

        # Assert artifact was added with correct metadata
        mock_result.add_artifact.assert_called_once()
        _, kwargs = mock_result.add_artifact.call_args
        self.assertEqual(kwargs["artifact_type"], "json")
        self.assertEqual(kwargs["category"], Constants.Artifact_Category_Mapping)

        self.assertEqual(result, mock_result)

    @patch("pamola_core.fake_data.commons.operations.write_json")
    @patch("pamola_core.fake_data.commons.operations.ensure_directory")
    @patch("pamola_core.fake_data.commons.operations.FieldOperation.execute")
    def test_execute_skips_mapping_if_not_success(self, mock_super_execute, mock_ensure_dir, mock_write_json):
        # Simulate a failed operation result
        mock_result = Mock(spec=OperationResult)
        mock_result.status = OperationStatus.ERROR
        mock_result.add_artifact = Mock()
        mock_super_execute.return_value = mock_result

        # Call execute method
        result = self.op.execute(
            data_source=Mock(),
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        # Ensure no mapping file is written
        mock_write_json.assert_not_called()
        mock_result.add_artifact.assert_not_called()

        # Ensure the result is returned correctly
        self.assertEqual(result, mock_result)


class TestFieldOperationInit(unittest.TestCase):

    def test_init_with_valid_params(self):
        op = FieldOperation(
            field_name="test_field",
            mode="ENRICH",
            output_field_name="test_output",
            chunk_size=5000,
            null_strategy="replace",
            use_cache=False,
            force_recalculation=True,
            use_dask=True,
            npartitions=4,
            use_vectorization=True,
            parallel_processes=2,
            use_encryption=True,
            encryption_key="secret_key",
            visualization_backend="plotly",
            visualization_theme="dark",
            visualization_strict=True
        )

        self.assertEqual(op.field_name, "test_field")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "test_output")
        self.assertEqual(op.chunk_size, 5000)
        self.assertEqual(op.null_strategy, NullStrategy.REPLACE)
        self.assertFalse(op.use_cache)
        self.assertTrue(op.force_recalculation)
        self.assertTrue(op.use_dask)
        self.assertEqual(op.npartitions, 4)
        self.assertTrue(op.use_vectorization)
        self.assertEqual(op.parallel_processes, 2)
        self.assertTrue(op.use_encryption)
        self.assertEqual(op.encryption_key, "secret_key")
        self.assertEqual(op.visualization_backend, "plotly")
        self.assertEqual(op.visualization_theme, "dark")
        self.assertTrue(op.visualization_strict)

    def test_init_with_invalid_null_strategy_defaults_to_preserve(self):
        with patch("pamola_core.fake_data.commons.operations.logger") as mock_logger:
            op = FieldOperation(field_name="test", null_strategy="invalid_strategy")

            # Should fall back to PRESERVE
            self.assertEqual(op.null_strategy, NullStrategy.PRESERVE)
            mock_logger.warning.assert_called_once_with("Unknown NULL strategy: invalid_strategy. Using PRESERVE.")

    def test_init_with_enum_null_strategy(self):
        op = FieldOperation(field_name="test", null_strategy=NullStrategy.EXCLUDE)
        self.assertEqual(op.null_strategy, NullStrategy.EXCLUDE)

    def test_metrics_collector_initialized(self):
        op = FieldOperation(field_name="some_field")
        self.assertIsNotNone(op._metrics_collector)
        self.assertIsNone(op._original_df)


class TestFieldOperationExecute(unittest.TestCase):
    def setUp(self):
        self.op = FieldOperation(field_name="test_field")
        self.op.force_recalculation = True  # Force execution even if cache exists
        self.task_dir = Path("/fake/task_dir")
        self.reporter = Mock()
        self.progress_tracker = Mock()

        # Patch the instance method 'process_batch' with a simple lambda
        patcher = patch.object(self.op, "process_batch", side_effect=lambda batch: batch)
        self.mock_process_batch = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("pamola_core.fake_data.commons.operations.generate_metrics_report")
    @patch("pamola_core.fake_data.commons.operations.ensure_directory")
    @patch.object(FieldOperation, "_save_metrics", return_value=Path("/fake/task_dir/metrics.json"))
    @patch.object(FieldOperation, "_save_result", return_value=Path("/fake/task_dir/output.csv"))
    @patch.object(FieldOperation, "_collect_metrics", return_value={"performance": {}})
    @patch.object(FieldOperation, "handle_null_values")
    @patch.object(FieldOperation, "preprocess_data")
    @patch.object(FieldOperation, "_prepare_directories")
    @patch("pamola_core.fake_data.commons.operations.load_data_operation")
    def test_execute_success(self, mock_load_data, mock_prepare_dirs, mock_preprocess, mock_handle_null,
                             mock_collect_metrics, mock_save_result, mock_save_metrics,
                             mock_ensure_dir, mock_generate_report):
        # Set up mocks to return the input dataframe unchanged
        mock_handle_null.side_effect = lambda df: df
        mock_preprocess.side_effect = lambda df: df

        # Return fake output and visualization directory paths
        mock_prepare_dirs.return_value = {
            "output": self.task_dir / "output",
            "visualizations": self.task_dir / "visualizations"
        }

        # Create simulated input data
        df = pd.DataFrame({
            self.op.field_name: [1, 2, 3],
            "other_field": [4, 5, 6]
        })
        mock_load_data.return_value = df

        # Execute the operation
        result = self.op.execute(
            data_source=Mock(),
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        # Verify that dummy_process_batch was called
        self.assertTrue(self.mock_process_batch.called, "Expected process_batch to be called")

        # Verify result status and structure
        self.assertEqual(result.status, OperationStatus.SUCCESS)
        self.assertTrue(hasattr(result, "metrics"))
        self.assertIsInstance(result.metrics, dict)

        # Confirm all mocks were called correctly
        mock_prepare_dirs.assert_called_once_with(self.task_dir)
        mock_load_data.assert_called_once()
        mock_save_result.assert_called_once()
        mock_save_metrics.assert_called_once()
        mock_ensure_dir.assert_called_once()
        mock_generate_report.assert_called_once()

    @patch("pamola_core.fake_data.commons.operations.load_data_operation")
    @patch.object(FieldOperation, "_prepare_directories")
    def test_execute_field_missing_returns_error(self, mock_prepare_dirs, mock_load_data):
        mock_prepare_dirs.return_value = {
            "output": self.task_dir / "output",
            "visualizations": self.task_dir / "visualizations"
        }

        # DataFrame missing the required field
        df = pd.DataFrame({"other_field": [1, 2, 3]})
        mock_load_data.return_value = df

        result = self.op.execute(
            data_source=Mock(),
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=None
        )

        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("not found in the data", result.error_message)
        self.reporter.add_operation.assert_any_call(
            f"Operation {self.op.name}", status="info", details=ANY
        )

    @patch("pamola_core.fake_data.commons.operations.load_data_operation")
    @patch.object(FieldOperation, "_prepare_directories")
    @patch.object(FieldOperation, "_get_cache")
    def test_execute_uses_cache(self, mock_get_cache, mock_prepare_dirs, mock_load_data):
        mock_prepare_dirs.return_value = {
            "output": self.task_dir / "output",
            "visualizations": self.task_dir / "visualizations"
        }

        df = pd.DataFrame({self.op.field_name: [1, 2, 3]})
        mock_load_data.return_value = df

        cached_result = Mock(spec=OperationResult)
        cached_result.status = OperationStatus.SUCCESS
        mock_get_cache.return_value = cached_result

        self.op.use_cache = True
        self.op.force_recalculation = False

        result = self.op.execute(
            data_source=Mock(),
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        # It should return cached result and skip processing
        self.assertEqual(result, cached_result)
        self.reporter.add_operation.assert_any_call(
            f"Operation {self.op.name}", status="info",
            details={"message": "Result loaded from cache – execution skipped"}
        )

    def test_execute_catches_exception_and_reports_error(self):
        # Patch method to raise error
        self.op._prepare_directories = Mock(side_effect=Exception("fail prepare dirs"))

        result = self.op.execute(
            data_source=Mock(),
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=None
        )

        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("fail prepare dirs", result.error_message)
        self.reporter.add_operation.assert_called_with(
            f"Operation {self.op.name}", status="error",
            details={"message": "fail prepare dirs"}
        )


if __name__ == "__main__":
    unittest.main()