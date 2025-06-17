import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path

from pamola_core.profiling.analyzers.group import GroupAnalyzerOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

class TestGroupAnalyzerOperation(unittest.TestCase):
    def setUp(self):
        # Minimum configuration for GroupAnalyzerOperation
        self.subset_name = "test_subset"
        self.fields_config = {"field1": 1, "field2": 2}
        self.text_length_threshold = 5
        self.minhash_similarity_threshold = 0.5
        self.operation = GroupAnalyzerOperation(
            subset_name=self.subset_name,
            fields_config=self.fields_config
        )
        self.task_dir = Path("test_task_dir")
        self.reporter = MagicMock()
        self.progress_tracker = MagicMock()
        self.data_source = MagicMock()

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    @patch('pamola_core.profiling.analyzers.group.DataWriter')
    def test_execute_success(self, mock_data_writer, mock_load_data):
        # Create a simulated DataFrame
        df = pd.DataFrame({
            "resume_id": [1, 1, 2, 2],
            "field1": ["a", "b", "a", "a"],
            "field2": ["x", "x", "y", "y"]
        })
        mock_load_data.return_value = df
        mock_writer_instance = mock_data_writer.return_value
        mock_writer_instance.write_metrics.return_value = MagicMock(path="metrics.json")

        result = self.operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        self.assertEqual(result.status.value, "success")
        self.reporter.add_operation.assert_called()
        self.reporter.add_artifact.assert_any_call(
            "json", "metrics.json", f"{self.subset_name} group analysis metrics"
        )

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    def test_execute_missing_resume_id(self, mock_load_data):
        # DataFrame missing resume_id column
        df = pd.DataFrame({
            "field1": ["a", "b"],
            "field2": ["x", "y"]
        })
        mock_load_data.return_value = df

        result = self.operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        self.assertEqual(result.status.value, "error")
        self.assertIn("resume_id", result.error_message)

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    def test_execute_missing_fields(self, mock_load_data):
        # DataFrame missing a field in fields_config
        df = pd.DataFrame({
            "resume_id": [1, 2],
            "field1": ["a", "b"]
        })
        mock_load_data.return_value = df

        result = self.operation.execute(
            data_source=self.data_source,
            task_dir=self.task_dir,
            reporter=self.reporter,
            progress_tracker=self.progress_tracker
        )

        self.assertEqual(result.status.value, "error")
        self.assertIn("Fields not found", result.error_message)

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    @patch('pamola_core.profiling.analyzers.group.DataWriter')
    def test_variance_distribution_0_1_to_0_2(self, mock_data_writer, mock_load_data):
        # Create DataFrame with 2 groups, each group 2 records, field1 has 2 different values
        # => variance = (2-1)/(2-1) = 1.0, but we will adjust fields_config so that weighted_variance = 0.15
        df = pd.DataFrame({
            "resume_id": [1, 1, 2, 2],
            "field1": ["a", "b", "a", "a"],  # group 1: ["a", "b"], group 2: ["a", "a"]
        })
        mock_load_data.return_value = df
        mock_writer_instance = mock_data_writer.return_value
        mock_writer_instance.write_metrics.return_value = MagicMock(path="metrics.json")

        # fields_config so that weighted_variance of group 2 is 0 (all same), group 1 is 1.0
        # To test the <0.2 branch, we will patch _calculate_weighted_variance to return 0.15 for group 1
        op = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1}
        )

        # Patch _analyze_group to return weighted_variance = 0.15 for group 1, 0 for group 2
        def fake_analyze_group(group_df, fields):
            if group_df["field1"].nunique() == 2:
                return {
                    "weighted_variance": 0.15,
                    "max_field_variance": 0.15,
                    "total_records": len(group_df),
                    "field_variances": {"field1": 0.15},
                    "duplication_ratios": {"field1": 2.0},
                    "should_aggregate": False
                }
            else:
                return {
                    "weighted_variance": 0.0,
                    "max_field_variance": 0.0,
                    "total_records": len(group_df),
                    "field_variances": {"field1": 0.0},
                    "duplication_ratios": {"field1": 2.0},
                    "should_aggregate": True
                }
        op._analyze_group = fake_analyze_group

        # Patch the plotting functions to not create real files
        op._generate_variance_distribution = MagicMock()
        op._generate_field_heatmap = MagicMock()

        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("test_task_dir"),
            reporter=MagicMock(),
            progress_tracker=MagicMock()
        )

        # Check metrics in result
        self.assertEqual(result.status.value, "success")
        # Get metrics from result (if you save metrics to result.metrics)
        # Or directly check variance_distribution via mock
        # Here we check the _generate_variance_distribution function is called with the correct dict
        called_args = op._generate_variance_distribution.call_args[0][0]
        self.assertEqual(called_args["0.1_to_0.2"], 1)

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    @patch('pamola_core.profiling.analyzers.group.DataWriter')
    def test_variance_distribution_0_5_to_0_8(self, mock_data_writer, mock_load_data):
        # weighted_variance = 0.7 (within 0.5_to_0.8)
        df = pd.DataFrame({
            "resume_id": [1, 1],
            "field1": ["a", "b"]
        })
        mock_load_data.return_value = df
        mock_writer_instance = mock_data_writer.return_value
        mock_writer_instance.write_metrics.return_value = MagicMock(path="metrics.json")

        op = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1}
        )

        # Patch _analyze_group to return weighted_variance = 0.7
        op._analyze_group = lambda group_df, fields: {
            "weighted_variance": 0.7,
            "max_field_variance": 0.7,
            "total_records": len(group_df),
            "field_variances": {"field1": 0.7},
            "duplication_ratios": {"field1": 2.0},
            "should_aggregate": False
        }

        op._generate_variance_distribution = MagicMock()
        op._generate_field_heatmap = MagicMock()

        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("test_task_dir"),
            reporter=MagicMock(),
            progress_tracker=MagicMock()
        )

        called_args = op._generate_variance_distribution.call_args[0][0]
        self.assertEqual(called_args["0.5_to_0.8"], 1)
        self.assertEqual(result.status.value, "success")

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    @patch('pamola_core.profiling.analyzers.group.DataWriter')
    def test_variance_distribution_above_0_8(self, mock_data_writer, mock_load_data):
        # weighted_variance = 0.85 (within above_0.8)
        df = pd.DataFrame({
            "resume_id": [1, 1],
            "field1": ["a", "b"]
        })
        mock_load_data.return_value = df
        mock_writer_instance = mock_data_writer.return_value
        mock_writer_instance.write_metrics.return_value = MagicMock(path="metrics.json")

        op = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1}
        )

        # Patch _analyze_group to return weighted_variance = 0.85
        op._analyze_group = lambda group_df, fields: {
            "weighted_variance": 0.85,
            "max_field_variance": 0.85,
            "total_records": len(group_df),
            "field_variances": {"field1": 0.85},
            "duplication_ratios": {"field1": 2.0},
            "should_aggregate": False
        }

        op._generate_variance_distribution = MagicMock()
        op._generate_field_heatmap = MagicMock()

        result = op.execute(
            data_source=MagicMock(),
            task_dir=Path("test_task_dir"),
            reporter=MagicMock(),
            progress_tracker=MagicMock()
        )

        called_args = op._generate_variance_distribution.call_args[0][0]
        self.assertEqual(called_args["above_0.8"], 1)
        self.assertEqual(result.status.value, "success")

    @patch('pamola_core.profiling.analyzers.group.load_data_operation')
    def test_execute_exception_handling(self, mock_load_data):
        # Simulate load_data_operation returning a valid DataFrame
        mock_load_data.return_value = MagicMock()
        # Initialize operation with valid fields_config
        op = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1}
        )
        # Patch _analyze_group to raise exception
        op._analyze_group = MagicMock(side_effect=Exception("Test exception"))

        # Patch logger to check log
        op.logger = MagicMock()

        # Create DataFrame with enough columns to not trigger previous errors
        df = MagicMock()
        df.columns = ["resume_id", "field1"]
        df.groupby.return_value = {"g1": df}
        mock_load_data.return_value = df

        # Call execute and check result
        result = op.execute(
            data_source=MagicMock(spec=DataSource),
            task_dir=Path("test_task_dir"),
            reporter=MagicMock(),
            progress_tracker=MagicMock()
        )

        self.assertEqual(result.status, OperationStatus.ERROR)
        self.assertIn("Error executing group analysis", result.error_message)
        op.logger.error.assert_called()

    def test_calculate_field_variance_all_identical(self):
        series = pd.Series(["a", "a", "a"])
        variance, duplication = self.operation._calculate_field_variance(series)
        self.assertEqual(variance, 0.0)
        self.assertEqual(duplication, 3.0)

    def test_calculate_field_variance_all_unique(self):
        series = pd.Series(["a", "b", "c"])
        variance, duplication = self.operation._calculate_field_variance(series)
        self.assertEqual(variance, 1.0)
        self.assertEqual(duplication, 1.0)

    def test_calculate_field_variance_some_duplicates(self):
        series = pd.Series(["a", "b", "a", "c"])
        variance, duplication = self.operation._calculate_field_variance(series)
        # unique_count = 3, total_records = 4
        # variance = (3-1)/(4-1) = 2/3 ≈ 0.666...
        self.assertAlmostEqual(variance, 2/3)
        self.assertAlmostEqual(duplication, 4/3)

    def test_calculate_field_variance_with_nan(self):
        series = pd.Series(["a", None, "a", float('nan')])
        variance, duplication = self.operation._calculate_field_variance(series)
        # unique: "a", None, nan (None and nan are treated as None)
        # unique_count = 2, total_records = 4
        # variance = (2-1)/(4-1) = 1/3 ≈ 0.333...
        self.assertAlmostEqual(variance, 1/3)
        self.assertAlmostEqual(duplication, 4/2)

    def test_calculate_field_variance_single_value(self):
        series = pd.Series(["a"])
        variance, duplication = self.operation._calculate_field_variance(series)
        self.assertEqual(variance, 0.0)
        self.assertEqual(duplication, 0.0)
    def test_get_unique_values_improved_short_strings(self):
        series = pd.Series(["a", "b", "a", "c"])
        unique, counts = self.operation._get_unique_values_improved(series)
        self.assertEqual(unique, {"a", "b", "c"})
        self.assertEqual(counts, {"a": 2, "b": 1, "c": 1})

    def test_get_unique_values_improved_long_strings_md5(self):
        series = pd.Series([
            "longtext_abcdefgh1", "longtext_abcdefgh2", "longtext_abcdefgh1"
        ])

        # Use a local operation variable with special configuration
        operation = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1},
            text_length_threshold=10,
            hash_algorithm="md5"  # Must use MD5
        )
        # text_length_threshold=10 so will hash strings with length 10 or more
        unique, counts = operation._get_unique_values_improved(series)
        self.assertEqual(len(unique), 2)
        self.assertEqual(sum(counts.values()), 3)
        # Check all keys are md5 strings
        for k in unique:
            self.assertIsInstance(k, str)
            self.assertEqual(len(k), 32)
            # Check if it's a hex string (optional)
            int(k, 16)  # will raise if not hex

    def test_get_unique_values_improved_none_and_nan(self):
        series = pd.Series([None, float('nan'), "a", None])
        unique, counts = self.operation._get_unique_values_improved(series)
        self.assertIn(None, unique)
        self.assertIn("a", unique)
        self.assertEqual(counts[None], 3)
        self.assertEqual(counts["a"], 1)

    def test_get_unique_values_improved_numbers(self):
        series = pd.Series([1, 2, 1, 3])
        unique, counts = self.operation._get_unique_values_improved(series)
        self.assertEqual(unique, {1, 2, 3})
        self.assertEqual(counts, {1: 2, 2: 1, 3: 1})

    def test_get_unique_values_improved_mixed_types(self):
        series = pd.Series([1, "1", None, float('nan'), "longtext"])
        unique, counts = self.operation._get_unique_values_improved(series)
        # "longtext" will be hashed
        self.assertIn(1, unique)
        self.assertIn("1", unique)
        self.assertIn(None, unique)
        self.assertEqual(sum(counts.values()), 5)

    @patch('pamola_core.utils.nlp.minhash.compute_minhash', return_value=[1,2,3,4,5,6,7,8,9,10])
    def test_get_unique_values_improved_minhash_import(self, mock_minhash):
        # Strings longer than text_length_threshold
        series = pd.Series([
            "longtext_abcdefghij", "longtext_abcdefghik", "longtext_abcdefghij"
        ])
        operation = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1},
            text_length_threshold=10,
            hash_algorithm="minhash"
        )
        # The first call will trigger the try to import minhash
        unique, counts = operation._get_unique_values_improved(series)
        # Ensure MinHash has been imported and flag is set
        self.assertTrue(hasattr(operation, '_minhash_imported'))
        self.assertTrue(operation._minhash_imported)
        # Ensure compute_minhash is called
        mock_minhash.assert_called()
        # Check keys are list-like strings
        for k in unique:
            self.assertTrue(str([1,2,3,4,5,6,7,8]) in k or str([1,2,3,4,5,6,7,8,9,10]) in k)
    
    def test_minhash_importerror_fallback_to_md5(self):
        operation = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1},
            hash_algorithm="minhash",
            text_length_threshold=10  # Fixed >= 10 to pass schema
        )
        operation.logger = MagicMock()

        class DummyModule:
            def __getattr__(self, name):
                raise ImportError("No module named 'pamola_core.utils.nlp.minhash'")

        with patch.dict('sys.modules', {"pamola_core.utils.nlp.minhash": DummyModule()}):
            if hasattr(operation, "_minhash_imported"):
                delattr(operation, "_minhash_imported")
            # Long string to ensure minhash is triggered
            series = pd.Series(["longtext_abcdefgh1", "longtext_abcdefgh2"])
            operation._get_unique_values_improved(series)
            operation.logger.warning.assert_called_with("MinHash library not available, falling back to MD5")
            self.assertFalse(operation._minhash_imported)
            self.assertFalse(operation.use_minhash)

    @patch('pamola_core.utils.nlp.minhash.compute_minhash', side_effect=[
        [1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20], [1,2,3,4,5,6,7,8,9,10]
    ])
    def test_get_unique_values_improved_minhash_clustering(self, mock_minhash):
        # Strings longer than text_length_threshold, with 2 different minhash values
        series = pd.Series([
            "longtext_abcdefghij", "longtext_abcdefghik", "longtext_abcdefghij"
        ])
        operation = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1},
            text_length_threshold=10,
            hash_algorithm="minhash"
        )
        # Patch cluster function to check if it's called
        operation._cluster_minhash_signatures_from_keys = MagicMock(return_value={
            "cluster1": ["[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"],
            "cluster2": ["[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]"]
        })
        operation._minhash_imported = True  # Simulate successful import

        # Patch _compute_minhash to use mock_minhash
        operation._compute_minhash = mock_minhash

        unique, counts = operation._get_unique_values_improved(series)
        # Ensure cluster function is called because there is more than 1 minhash key
        operation._cluster_minhash_signatures_from_keys.assert_called()
        # Ensure unique returns the correct number of clusters
        self.assertEqual(
            set(unique),
            {"[1, 2, 3, 4, 5, 6, 7, 8]", "[11, 12, 13, 14, 15, 16, 17, 18]"}
        )

    @patch('pamola_core.utils.nlp.minhash.compute_minhash')
    def test_merge_duplicate_keys_in_get_unique_values_improved(self, mock_minhash):
        # Return 2 different signatures but similar enough to cluster
        mock_minhash.side_effect = [
            [1, 2, 3, 4, 5, 6, 7, 8],      # for text1
            [1, 2, 3, 4, 5, 6, 7, 99],     # for text2 (7/8 elements are the same)
        ]
        operation = GroupAnalyzerOperation(
            subset_name="test",
            fields_config={"field1": 1},
            text_length_threshold=10,
            hash_algorithm="minhash",
            minhash_similarity_threshold=0.7  # 7/8 = 0.875 > 0.7
        )
        # Two different long strings but signatures are similar enough to cluster
        series = pd.Series([
            "longtext_abcdefghij1",
            "longtext_abcdefghij2"
        ])
        unique, counts = operation._get_unique_values_improved(series)
        # After clustering, there should be only 1 key
        self.assertEqual(len(unique), 1)
        # Total count should be 2
        for v in counts.values():
            self.assertEqual(v, 2)


    def test_cluster_minhash_signatures(self):
        # Signature strings in the form '[1, 2, 3, 4]'
        sigs = [
            "[1, 2, 3, 4]",
            "[1, 2, 3, 5]",
            "[10, 11, 12, 13]",
            "[10, 11, 12, 14]"
        ]
        clusters = self.operation._cluster_minhash_signatures_from_keys(sigs)
        # Each cluster contains only 1 element
        self.assertEqual(len(clusters), 4)
        for sig in sigs:
            self.assertIn([sig], clusters)

    def test_single_signature(self):
        sigs = ["[1, 2, 3, 4]"]
        clusters = self.operation._cluster_minhash_signatures_from_keys(sigs)
        self.assertEqual(clusters, [sigs])

    def test_empty_list(self):
        clusters = self.operation._cluster_minhash_signatures_from_keys([])
        self.assertEqual(clusters, [[]])
    def test_cluster_minhash_signatures_skip_invalid_key(self):
        # Create a list of signatures, one key is not a valid list string
        sigs = [
            "[1, 2, 3, 4]",
            "not_a_list",  # this key will cause ValueError when eval
            "[10, 11, 12, 13]"
        ]
        clusters = self.operation._cluster_minhash_signatures_from_keys(sigs)
        # Result: only cluster valid keys, error key is skipped
        # Depending on the logic, each valid key is 1 cluster
        self.assertEqual(len(clusters), 2)
        self.assertIn(["[1, 2, 3, 4]"], clusters)
        self.assertIn(["[10, 11, 12, 13]"], clusters)

    def test_cluster_minhash_signatures_skip_processed_key(self):
        # Create a list of signatures, one has duplicate key causing key2 to be in processed
        sigs = [
            "[1, 2, 3, 4]",
            "[1, 2, 3, 4]",  # this key will be skipped because it's already in processed
            "[10, 11, 12, 13]"
        ]
        clusters = self.operation._cluster_minhash_signatures_from_keys(sigs)
        # Result: each valid key is 1 cluster, duplicate key is skipped
        self.assertEqual(len(clusters), 2)
        self.assertIn(["[1, 2, 3, 4]"], clusters)
        self.assertIn(["[10, 11, 12, 13]"], clusters)
    
    def test_calculate_simple_jaccard(self):
        op = self.operation
        # Exactly the same case
        a = [1, 2, 3]
        b = [1, 2, 3]
        self.assertEqual(op._calculate_simple_jaccard(a, b), 1.0)

        # Completely disjoint case
        a = [1, 2, 3]
        b = [4, 5, 6]
        self.assertEqual(op._calculate_simple_jaccard(a, b), 0.0)

        # Partially overlapping case
        a = [1, 2, 3]
        b = [2, 3, 4]
        # Intersection = {2,3}, Union = {1,2,3,4} => 2/4 = 0.5
        self.assertEqual(op._calculate_simple_jaccard(a, b), 0.5)

        # One empty list
        a = []
        b = [1, 2, 3]
        self.assertEqual(op._calculate_simple_jaccard(a, b), 0.0)

        # Both empty
        a = []
        b = []
        self.assertEqual(op._calculate_simple_jaccard(a, b), 0.0)

    def test_calculate_weighted_variance(self):
        op = self.operation
        # Normal case: 2 fields, each with 1 weight
        field_variances = {"field1": 0.5, "field2": 0.2}
        op.fields_config = {"field1": 1, "field2": 1}
        self.assertAlmostEqual(op._calculate_weighted_variance(field_variances), 0.35)

        # Different weights case
        field_variances = {"field1": 0.5, "field2": 0.2}
        op.fields_config = {"field1": 2, "field2": 1}
        # (0.5*2 + 0.2*1) / (2+1) = (1.0+0.2)/3 = 0.4
        self.assertAlmostEqual(op._calculate_weighted_variance(field_variances), 0.4)

        # Only 1 field case
        field_variances = {"field1": 0.7}
        op.fields_config = {"field1": 1}
        self.assertAlmostEqual(op._calculate_weighted_variance(field_variances), 0.7)

        # No fields case
        field_variances = {}
        op.fields_config = {}
        self.assertEqual(op._calculate_weighted_variance(field_variances), 0.0)

        # Field with weight 0 case
        field_variances = {"field1": 0.5, "field2": 0.2}
        op.fields_config = {"field1": 0, "field2": 0}
        self.assertEqual(op._calculate_weighted_variance(field_variances), 0.0)

        # Negative weights (should be treated as normal numbers, but negative weights are not typical)
        field_variances = {"field1": 0.5, "field2": 0.2}
        op.fields_config = {"field1": -1, "field2": 2}
        # (0.5*-1 + 0.2*2) / (2-1) = (-0.5+0.4)/1 = -0.1
        self.assertAlmostEqual(op._calculate_weighted_variance(field_variances), -0.1)

        # Variances has extra field not in config: should ignore extra
        field_variances = {"field1": 0.5, "field2": 0.2}
        op.fields_config = {"field1": 1, "field2": 2}
        self.assertAlmostEqual(op._calculate_weighted_variance(field_variances), 0.3)

        # Config has extra field not in variances: should ignore extra
        field_variances = {"field1": 0.5}
        op.fields_config = {"field1": 1, "field2": 1}
        self.assertAlmostEqual(op._calculate_weighted_variance(field_variances), 0.5)

        # All variances zero
        field_variances = {"field1": 0.0, "field2": 0.0}
        op.fields_config = {"field1": 1, "field2": 1}
        self.assertEqual(op._calculate_weighted_variance(field_variances), 0.0)

        # Non-numeric variance: should skip or treat as zero
        field_variances = {"field1": 0, "field2": 0.2}
        op.fields_config = {"field1": 1, "field2": 1}
        # Should not raise, should treat non-numeric as zero or skip
        try:
            result = op._calculate_weighted_variance(field_variances)
            self.assertIsInstance(result, float)
        except Exception as e:
            self.fail(f"Should not raise exception: {e}")

    def test_should_aggregate(self):
        op = self.operation
        # Set thresholds for testing
        op.large_group_threshold = 10
        op.large_group_variance_threshold = 0.1
        op.variance_threshold = 0.2

        # Small group, variance below threshold
        self.assertTrue(op._should_aggregate(0.15, 5))
        # Small group, variance above threshold
        self.assertFalse(op._should_aggregate(0.25, 5))
        # Small group, variance exactly at threshold
        self.assertTrue(op._should_aggregate(0.2, 5))

        # Large group, variance below large group threshold
        self.assertTrue(op._should_aggregate(0.05, 20))
        # Large group, variance above large group threshold
        self.assertFalse(op._should_aggregate(0.15, 20))
        # Large group, variance exactly at large group threshold
        self.assertTrue(op._should_aggregate(0.1, 20))

        # Edge: group size exactly at threshold (should use small group logic)
        self.assertTrue(op._should_aggregate(0.2, 10))
        self.assertFalse(op._should_aggregate(0.21, 10))

    def test_calculate_field_metrics(self):
        op = self.operation
        # Prepare group_metrics for two groups and two fields
        group_metrics = {
            "g1": {
                "field_variances": {"field1": 0.5, "field2": 0.2},
                "duplication_ratios": {"field1": 2.0, "field2": 1.0},
                "total_records": 4
            },
            "g2": {
                "field_variances": {"field1": 0.7, "field2": 0.4},
                "duplication_ratios": {"field1": 1.0, "field2": 2.0},
                "total_records": 2
            }
        }
        fields = ["field1", "field2"]
        metrics = op._calculate_field_metrics(group_metrics, fields)
        # Check avg_variance
        self.assertAlmostEqual(metrics["field1"]["avg_variance"], (0.5+0.7)/2)
        self.assertAlmostEqual(metrics["field2"]["avg_variance"], (0.2+0.4)/2)
        # Check max_variance
        self.assertEqual(metrics["field1"]["max_variance"], 0.7)
        self.assertEqual(metrics["field2"]["max_variance"], 0.4)
        # Check avg_duplication_ratio
        self.assertAlmostEqual(metrics["field1"]["avg_duplication_ratio"], (2.0+1.0)/2)
        self.assertAlmostEqual(metrics["field2"]["avg_duplication_ratio"], (1.0+2.0)/2)
        # Check unique_values_total (sum of total_records/duplication_ratio for each group)
        expected_field1 = 4/2.0 + 2/1.0  # 2 + 2 = 4
        expected_field2 = 4/1.0 + 2/2.0  # 4 + 1 = 5
        self.assertEqual(metrics["field1"]["unique_values_total"], int(expected_field1))
        self.assertEqual(metrics["field2"]["unique_values_total"], int(expected_field2))

        # Edge: field not present in one group
        group_metrics = {
            "g1": {
                "field_variances": {"field1": 0.5},
                "duplication_ratios": {"field1": 2.0},
                "total_records": 4
            },
            "g2": {
                "field_variances": {"field2": 0.4},
                "duplication_ratios": {"field2": 2.0},
                "total_records": 2
            }
        }
        fields = ["field1", "field2"]
        metrics = op._calculate_field_metrics(group_metrics, fields)
        self.assertAlmostEqual(metrics["field1"]["avg_variance"], 0.5)
        self.assertAlmostEqual(metrics["field2"]["avg_variance"], 0.4)
        self.assertEqual(metrics["field1"]["unique_values_total"], 2)
        self.assertEqual(metrics["field2"]["unique_values_total"], 1)

        # Edge: empty group_metrics
        metrics = op._calculate_field_metrics({}, fields)
        self.assertEqual(metrics["field1"]["avg_variance"], 0)
        self.assertEqual(metrics["field2"]["avg_variance"], 0)
        self.assertEqual(metrics["field1"]["unique_values_total"], 0)
        self.assertEqual(metrics["field2"]["unique_values_total"], 0)

    def test_generate_variance_distribution(self):
        op = self.operation
        variance_distribution = {
            "below_0.1": 1,
            "0.1_to_0.2": 1,
            "0.2_to_0.5": 1,
            "0.5_to_0.8": 1,
            "above_0.8": 2,
        }
        output_path = Path("dummy.png")
        # Delete file if it exists
        if output_path.exists():
            output_path.unlink()
        op._generate_variance_distribution(variance_distribution, output_path)
        # Delete file after test
        if output_path.exists():
            output_path.unlink()
    
    def test_generate_variance_distribution_fallback_matplotlib(self):
        op = self.operation
        variance_distribution = {
            "below_0.1": 1,
            "0.1_to_0.2": 1,
            "0.2_to_0.5": 1,
            "0.5_to_0.8": 1,
            "above_0.8": 2,
        }
        output_path = Path("dummy_fallback.png")
        if output_path.exists():
            output_path.unlink()
        # Patch create_histogram to raise TypeError
        with patch("pamola_core.profiling.analyzers.group.create_histogram", side_effect=TypeError("Mocked error")):
            op.logger = MagicMock()
            op._generate_variance_distribution(variance_distribution, output_path)
            # Ensure logger.info is called with fallback message
            op.logger.info.assert_any_call(f"Generated variance distribution visualization at {output_path}")
            # Check the file was created by matplotlib fallback
            self.assertTrue(output_path.exists())
            output_path.unlink()

    def test_generate_variance_distribution_fallback_on_general_exception(self):
        op = self.operation
        variance_distribution = {
            "below_0.1": 1,
            "0.1_to_0.2": 1,
            "0.2_to_0.5": 1,
            "0.5_to_0.8": 1,
            "above_0.8": 2,
        }
        output_path = Path("dummy_fallback2.png")
        if output_path.exists():
            output_path.unlink()
        # Patch create_histogram to raise RuntimeError (not TypeError/ValueError)
        with patch("pamola_core.profiling.analyzers.group.create_histogram", side_effect=RuntimeError("Mocked runtime error")):
            op.logger = MagicMock()
            op._generate_variance_distribution(variance_distribution, output_path)
            # Ensure logger.error is called with error message
            op.logger.error.assert_any_call("Error generating variance distribution visualization: Mocked runtime error")
            # Ensure logger.info is called with fallback message
            op.logger.info.assert_any_call(f"Generated fallback variance distribution visualization at {output_path}")
            # Check the file was created by matplotlib fallback
            self.assertTrue(output_path.exists())
            output_path.unlink()
            
    def test_generate_field_heatmap(self):
        op = self.operation
        group_metrics = {
            "g1": {"field_variances": {"field1": 0.5, "field2": 0.2}},
            "g2": {"field_variances": {"field1": 0.7, "field2": 0.4}},
            "g3": {"field_variances": {"field1": 0.1, "field2": 0.9}},
        }
        output_path = Path("test_heatmap.png")
        if output_path.exists():
            output_path.unlink()
        op._generate_field_heatmap(group_metrics, output_path)
        
        # Delete file after test
        if output_path.exists():
            output_path.unlink()

    def test_generate_field_heatmap_fallback_matplotlib(self):
        op = self.operation
        group_metrics = {
            "g1": {"field_variances": {"field1": 0.5, "field2": 0.2}},
            "g2": {"field_variances": {"field1": 0.7, "field2": 0.4}},
            "g3": {"field_variances": {"field1": 0.1, "field2": 0.9}},
        }
        output_path = Path("test_heatmap_fallback.png")
        if output_path.exists():
            output_path.unlink()
        # Patch create_heatmap (or the function you use to draw heatmap) to raise TypeError
        with patch("pamola_core.profiling.analyzers.group.create_heatmap", side_effect=TypeError("Mocked error")):
            op.logger = MagicMock()
            op._generate_field_heatmap(group_metrics, output_path)
        if output_path.exists():
            # You can check logger.info or logger.error if desired
            output_path.unlink()
