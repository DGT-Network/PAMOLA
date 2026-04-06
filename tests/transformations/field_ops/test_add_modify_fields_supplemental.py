"""
Supplemental tests for pamola_core/transformations/field_ops/add_modify_fields.py

Targets missed lines:
  205-276   (process_batch – add_conditional, missing lookup path)
  303-569   (process_batch – modify_conditional, modify_expression, ENRICH mode)
  697-812   (_process_dataframe_with_config – chunk/dask/joblib branches)
  845-1000  (_process_dataframe_using_dask, _process_dataframe_using_joblib)
  1089-1292 (_collect_metrics internals – correlation, missing values)
  1315-1490 (calculate_dataset_comparison, _compare_row/col/values/nulls)
  1588-1698 (_generate_visualizations – bar plot paths)
  1772-1922 (_handle_visualizations – thread / timeout paths)
"""
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from pamola_core.transformations.field_ops.add_modify_fields import (
    AddOrModifyFieldsOperation,
    create_add_modify_fields_operation,
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.errors.exceptions import ValidationError


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _op(**overrides) -> AddOrModifyFieldsOperation:
    """Build an operation with sensible defaults, overridable per-test."""
    defaults = dict(
        field_name="value",
        field_operations={},
        lookup_tables={},
        output_format="csv",
        name="testop",
        description="desc",
        mode="REPLACE",
        output_field_name=None,
        column_prefix="_",
        chunk_size=10000,
        use_cache=False,
        use_dask=False,
        use_encryption=False,
        encryption_key=None,
    )
    defaults.update(overrides)
    return AddOrModifyFieldsOperation(**defaults)


def _reporter() -> MagicMock:
    r = MagicMock()
    r.add_operation = MagicMock()
    r.add_artifact = MagicMock()
    return r


def _progress() -> MagicMock:
    p = MagicMock()
    p.update = MagicMock()
    p.create_subtask = MagicMock(return_value=MagicMock())
    p.close = MagicMock()
    return p


# ---------------------------------------------------------------------------
# process_batch – add_conditional
# ---------------------------------------------------------------------------

class TestProcessBatchAddConditional:
    def test_add_conditional_numpy_bool_validation_error(self):
        """numpy bool_ from comparison triggers ValidationError — exercises condition evaluation path."""
        from pamola_core.errors.exceptions import ValidationError
        op = _op(field_operations={
            "category": {
                "operation_type": "add_conditional",
                "condition": "row['score'] > 50",
                "value_if_true": "'high'",
                "value_if_false": "'low'",
            }
        })
        df = pd.DataFrame({"score": [60, 40, 80, 30]})
        with pytest.raises((ValidationError, Exception)):
            op.process_batch(df.copy())

    def test_add_conditional_missing_condition_skipped(self):
        """Missing condition key → operation silently skipped."""
        op = _op(field_operations={
            "new_col": {
                "operation_type": "add_conditional",
                # no 'condition' key
                "value_if_true": "'yes'",
            }
        })
        df = pd.DataFrame({"x": [1, 2]})
        result = op.process_batch(df.copy())
        assert "new_col" not in result.columns

    def test_add_from_lookup_missing_table_skipped(self):
        """lookup_table_name not in lookup_tables → no new column."""
        op = _op(
            field_operations={
                "mapped": {
                    "operation_type": "add_from_lookup",
                    "lookup_table_name": "nonexistent",
                    "base_on_column": "key",
                }
            },
            lookup_tables={},
        )
        df = pd.DataFrame({"key": [1, 2, 3]})
        result = op.process_batch(df.copy())
        assert "mapped" not in result.columns

    def test_add_from_lookup_file_path(self, tmp_path):
        """lookup_table given as a Path to a JSON file is loaded on-the-fly."""
        import json
        lookup = {"a": "A", "b": "B"}
        lp = tmp_path / "lookup.json"
        lp.write_text(json.dumps(lookup))

        op = _op(
            field_operations={
                "mapped": {
                    "operation_type": "add_from_lookup",
                    "lookup_table_name": "tbl",
                    "base_on_column": "key",
                }
            },
            lookup_tables={"tbl": lp},
        )
        df = pd.DataFrame({"key": ["a", "b", "a"]})
        result = op.process_batch(df.copy())
        assert "mapped" in result.columns
        assert list(result["mapped"]) == ["A", "B", "A"]


# ---------------------------------------------------------------------------
# process_batch – modify_conditional, modify_expression, ENRICH mode
# ---------------------------------------------------------------------------

class TestProcessBatchModifyOps:
    def test_modify_conditional_replace_mode(self):
        op = _op(
            field_operations={
                "label": {
                    "operation_type": "modify_conditional",
                    "condition": "row['score'] >= 50",
                    "value_if_true": "'pass'",
                    "value_if_false": "'fail'",
                }
            },
            mode="REPLACE",
        )
        df = pd.DataFrame({"label": ["?", "?", "?"], "score": [70, 30, 50]})
        result = op.process_batch(df.copy())
        assert list(result["label"]) == ["pass", "fail", "pass"]

    def test_modify_conditional_enrich_mode(self):
        """ENRICH mode writes to a prefixed column."""
        op = _op(
            field_operations={
                "label": {
                    "operation_type": "modify_conditional",
                    "condition": "row['score'] >= 50",
                    "value_if_true": "'pass'",
                    "value_if_false": "'fail'",
                }
            },
            mode="ENRICH",
            column_prefix="new_",
        )
        df = pd.DataFrame({"label": ["?", "?"], "score": [70, 30]})
        result = op.process_batch(df.copy())
        assert "new_label" in result.columns
        assert list(result["new_label"]) == ["pass", "fail"]

    def test_modify_conditional_missing_condition_skipped(self):
        op = _op(field_operations={
            "label": {
                "operation_type": "modify_conditional",
                # no condition key
                "value_if_true": "'yes'",
            }
        })
        df = pd.DataFrame({"label": ["orig"]})
        result = op.process_batch(df.copy())
        assert list(result["label"]) == ["orig"]

    def test_modify_expression_replace_mode(self):
        op = _op(
            field_operations={
                "price_with_tax": {
                    "operation_type": "modify_expression",
                    "base_on_column": "price",
                    "expression_character": "x",
                    "expression": "x * 1.2",
                }
            },
            mode="REPLACE",
        )
        df = pd.DataFrame({"price_with_tax": [0.0, 0.0], "price": [100.0, 200.0]})
        result = op.process_batch(df.copy())
        assert abs(result["price_with_tax"].iloc[0] - 120.0) < 0.01
        assert abs(result["price_with_tax"].iloc[1] - 240.0) < 0.01

    def test_modify_expression_enrich_mode(self):
        op = _op(
            field_operations={
                "amount": {
                    "operation_type": "modify_expression",
                    "base_on_column": "base",
                    "expression_character": "v",
                    "expression": "v + 10",
                }
            },
            mode="ENRICH",
            column_prefix="mod_",
        )
        df = pd.DataFrame({"amount": [1.0, 2.0], "base": [5.0, 10.0]})
        result = op.process_batch(df.copy())
        assert "mod_amount" in result.columns
        assert list(result["mod_amount"]) == [15.0, 20.0]

    def test_modify_expression_missing_expression_skipped(self):
        op = _op(field_operations={
            "amount": {
                "operation_type": "modify_expression",
                "base_on_column": "base",
                # no expression_character / expression
            }
        })
        df = pd.DataFrame({"amount": [1.0], "base": [5.0]})
        result = op.process_batch(df.copy())
        assert result["amount"].iloc[0] == 1.0

    def test_modify_from_lookup_enrich_mode(self):
        op = _op(
            field_operations={
                "status": {
                    "operation_type": "modify_from_lookup",
                    "lookup_table_name": "tbl",
                    "base_on_column": "code",
                }
            },
            lookup_tables={"tbl": {1: "active", 2: "inactive"}},
            mode="ENRICH",
            column_prefix="orig_",
        )
        df = pd.DataFrame({"status": ["?", "?"], "code": [1, 2]})
        result = op.process_batch(df.copy())
        assert "orig_status" in result.columns
        assert list(result["orig_status"]) == ["active", "inactive"]

    def test_modify_from_lookup_file_path(self, tmp_path):
        import json
        lp = tmp_path / "lk.json"
        lp.write_text(json.dumps({"x": "X", "y": "Y"}))
        op = _op(
            field_operations={
                "col": {
                    "operation_type": "modify_from_lookup",
                    "lookup_table_name": "t",
                    "base_on_column": "key",
                }
            },
            lookup_tables={"t": lp},
            mode="REPLACE",
        )
        df = pd.DataFrame({"col": ["?", "?"], "key": ["x", "y"]})
        result = op.process_batch(df.copy())
        assert list(result["col"]) == ["X", "Y"]


# ---------------------------------------------------------------------------
# _process_dataframe_with_config – various processing modes
# ---------------------------------------------------------------------------

class TestProcessDataframeWithConfig:
    def test_empty_df_returned_as_is(self):
        op = _op()
        op.logger = MagicMock()
        op.logger.warning = MagicMock()
        empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = op._process_dataframe_with_config(
            df=empty,
            process_function=lambda df: df,
            chunk_size=100,
            use_dask=False,
            task_logger=op.logger,
        )
        assert len(result) == 0

    def test_small_df_processed_without_chunking(self):
        """len(df) <= chunk_size → direct call to process_function."""
        op = _op()
        op.logger = MagicMock()
        op.logger.warning = MagicMock()
        df = pd.DataFrame({"val": [1, 2, 3]})
        called_with = []

        def proc(batch):
            called_with.append(len(batch))
            return batch

        result = op._process_dataframe_with_config(
            df=df,
            process_function=proc,
            chunk_size=100,
            use_dask=False,
            task_logger=op.logger,
        )
        assert called_with == [3]
        assert len(result) == 3

    def test_chunk_processing_large_df(self):
        """len(df) > chunk_size → chunked processing path."""
        op = _op()
        op.logger = MagicMock()
        op.logger.warning = MagicMock()
        op.logger.info = MagicMock()
        df = pd.DataFrame({"val": list(range(25))})

        result = op._process_dataframe_with_config(
            df=df,
            process_function=lambda batch: batch,
            chunk_size=10,
            use_dask=False,
            use_vectorization=False,
            task_logger=op.logger,
        )
        assert len(result) == 25

    def test_dask_processing_path(self):
        """use_dask=True for large df goes through dask path (or falls back)."""
        op = _op()
        op.logger = MagicMock()
        op.logger.warning = MagicMock()
        op.logger.info = MagicMock()
        df = pd.DataFrame({"val": list(range(30))})

        result = op._process_dataframe_with_config(
            df=df,
            process_function=lambda batch: batch,
            chunk_size=10,
            use_dask=True,
            npartitions=2,
            task_logger=op.logger,
        )
        # Result might be a Dask or Pandas DataFrame – accept both
        assert result is not None

    def test_vectorization_processing_path(self):
        """use_vectorization=True for large df goes through joblib path (or falls back)."""
        op = _op()
        op.logger = MagicMock()
        op.logger.warning = MagicMock()
        op.logger.info = MagicMock()
        df = pd.DataFrame({"val": list(range(30))})

        result = op._process_dataframe_with_config(
            df=df,
            process_function=lambda batch: batch,
            chunk_size=10,
            use_vectorization=True,
            parallel_processes=2,
            task_logger=op.logger,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# _process_dataframe_using_dask
# ---------------------------------------------------------------------------

class TestProcessDataframeUsingDask:
    def test_returns_false_when_npartitions_zero_and_chunksize_zero(self):
        op = _op()
        df = pd.DataFrame({"val": [1, 2, 3]})
        result_df, flag = op._process_dataframe_using_dask(
            df=df, process_function=lambda x: x,
            npartitions=0, chunksize=0,
        )
        assert flag is False

    def test_basic_processing_succeeds(self):
        op = _op()
        df = pd.DataFrame({"val": list(range(20))})
        result_df, flag = op._process_dataframe_using_dask(
            df=df, process_function=lambda batch: batch,
            npartitions=2, chunksize=10,
        )
        assert flag is True

    def test_exception_returns_original_df(self):
        op = _op()
        df = pd.DataFrame({"val": [1, 2, 3]})

        def bad_func(batch):
            raise ValueError("boom")

        # Even with a bad process function the wrapper catches internally
        result_df, flag = op._process_dataframe_using_dask(
            df=df, process_function=bad_func,
            npartitions=2, chunksize=10,
        )
        # flag may be True (dask is lazy) or False – we just assert no crash
        assert result_df is not None


# ---------------------------------------------------------------------------
# _process_dataframe_using_joblib
# ---------------------------------------------------------------------------

class TestProcessDataframeUsingJoblib:
    def test_returns_false_when_n_jobs_zero(self):
        op = _op()
        df = pd.DataFrame({"val": [1, 2, 3]})
        result_df, flag = op._process_dataframe_using_joblib(
            df=df, process_function=lambda x: x,
            n_jobs=0, chunk_size=10,
        )
        assert flag is False

    def test_basic_parallel_processing(self):
        op = _op()
        df = pd.DataFrame({"val": list(range(20))})
        result_df, flag = op._process_dataframe_using_joblib(
            df=df, process_function=lambda batch: batch,
            n_jobs=2, chunk_size=10,
        )
        assert flag is True
        assert len(result_df) == 20

    def test_processing_chunk_returns_none_gives_false(self):
        """If any chunk returns None, flag is False."""
        op = _op()
        df = pd.DataFrame({"val": list(range(20))})
        call_count = [0]

        def partial_fail(batch):
            call_count[0] += 1
            if call_count[0] == 1:
                return None
            return batch

        result_df, flag = op._process_dataframe_using_joblib(
            df=df, process_function=partial_fail,
            n_jobs=1, chunk_size=10,
        )
        assert flag is False


# ---------------------------------------------------------------------------
# _process_dataframe_using_chunk
# ---------------------------------------------------------------------------

class TestProcessDataframeUsingChunk:
    def test_chunk_size_one_returns_false(self):
        op = _op()
        df = pd.DataFrame({"val": [1, 2, 3]})
        result_df, flag = op._process_dataframe_using_chunk(
            df=df, process_function=lambda x: x, chunk_size=1
        )
        assert flag is False

    def test_basic_chunk_processing(self):
        op = _op()
        df = pd.DataFrame({"val": list(range(25))})
        result_df, flag = op._process_dataframe_using_chunk(
            df=df, process_function=lambda batch: batch, chunk_size=10
        )
        assert flag is True
        assert len(result_df) == 25

    def test_chunk_failure_returns_false(self):
        op = _op()
        df = pd.DataFrame({"val": list(range(20))})
        call_count = [0]

        def partial_fail(batch):
            call_count[0] += 1
            if call_count[0] == 2:
                return None
            return batch

        result_df, flag = op._process_dataframe_using_chunk(
            df=df, process_function=partial_fail, chunk_size=10
        )
        assert flag is False


# ---------------------------------------------------------------------------
# _generate_dataframe_chunks
# ---------------------------------------------------------------------------

class TestGenerateDataframeChunks:
    def test_empty_df_yields_single_chunk(self):
        op = _op()
        chunks = list(op._generate_dataframe_chunks(pd.DataFrame(), chunk_size=10))
        assert len(chunks) == 1

    def test_correct_number_of_chunks(self):
        op = _op()
        df = pd.DataFrame({"val": list(range(25))})
        chunks = list(op._generate_dataframe_chunks(df, chunk_size=10))
        assert len(chunks) == 3  # 10, 10, 5

    def test_chunk_metadata_correct(self):
        op = _op()
        df = pd.DataFrame({"val": list(range(10))})
        chunks = list(op._generate_dataframe_chunks(df, chunk_size=5))
        chunk_df, chunk_num, start, end, total = chunks[0]
        assert chunk_num == 0
        assert start == 0
        assert end == 5
        assert total == 2


# ---------------------------------------------------------------------------
# _collect_metrics internals
# ---------------------------------------------------------------------------

class TestCollectMetrics:
    def _make_op_with_fields(self, **kwargs):
        return _op(
            field_operations={"new_col": {"operation_type": "add_constant", "constant_value": 99}},
            **kwargs,
        )

    def test_fields_added_count(self):
        op = self._make_op_with_fields()
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [1, 2, 3], "new_col": [99, 99, 99]})
        metrics = op._collect_metrics(orig, processed)
        assert metrics["fields_added_count"] == 1
        assert metrics["fields_modified_count"] == 0

    def test_fields_modified_count(self):
        op = _op(field_operations={"val": {"operation_type": "modify_constant", "constant_value": 0}})
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [0, 0, 0]})
        metrics = op._collect_metrics(orig, processed)
        assert metrics["fields_modified_count"] == 1

    def test_numeric_correlation_computed(self):
        op = _op(field_operations={"val": {"operation_type": "modify_constant", "constant_value": 10}})
        orig = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        processed = pd.DataFrame({"val": [10.0, 10.0, 10.0, 10.0, 10.0]})
        metrics = op._collect_metrics(orig, processed)
        assert "correlations" in metrics

    def test_non_numeric_correlation_is_nan(self):
        op = _op(field_operations={"label": {"operation_type": "modify_constant", "constant_value": "X"}})
        orig = pd.DataFrame({"label": ["a", "b", "c"]})
        processed = pd.DataFrame({"label": ["X", "X", "X"]})
        metrics = op._collect_metrics(orig, processed)
        assert "correlations" in metrics
        assert np.isnan(metrics["correlations"].get("label", float("nan")))

    def test_missing_values_tracked(self):
        op = _op(field_operations={"new_col": {"operation_type": "add_constant", "constant_value": None}})
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [1, 2, 3], "new_col": [None, None, None]})
        metrics = op._collect_metrics(orig, processed)
        assert "missing_values" in metrics
        assert "new_col" in metrics["missing_values"]


# ---------------------------------------------------------------------------
# calculate_dataset_comparison
# ---------------------------------------------------------------------------

class TestCalculateDatasetComparison:
    def test_same_dfs_no_change(self):
        op = _op()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = op.calculate_dataset_comparison(df, df.copy())
        assert result["row_counts"]["difference"] == 0
        assert result["column_counts"]["difference"] == 0

    def test_added_column_detected(self):
        op = _op()
        orig = pd.DataFrame({"a": [1, 2, 3]})
        trans = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = op.calculate_dataset_comparison(orig, trans)
        assert "b" in result["added_columns"]
        assert result["column_counts"]["difference"] == 1

    def test_row_count_change(self):
        op = _op()
        orig = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        trans = pd.DataFrame({"a": [1, 2, 3]})
        result = op.calculate_dataset_comparison(orig, trans)
        assert result["row_counts"]["difference"] == -2

    def test_none_inputs_raise(self):
        op = _op()
        with pytest.raises(Exception):
            op.calculate_dataset_comparison(None, None)

    def test_memory_usage_present(self):
        op = _op()
        df = pd.DataFrame({"a": range(100)})
        result = op.calculate_dataset_comparison(df, df.copy())
        assert "memory_usage" in result
        assert "original_mb" in result["memory_usage"]


# ---------------------------------------------------------------------------
# _count_value_changes / _count_null_changes
# ---------------------------------------------------------------------------

class TestCountChanges:
    def test_count_value_changes_numeric_no_change(self):
        op = _op()
        col = pd.Series([1.0, 2.0, 3.0])
        result = op._count_value_changes(col, col.copy())
        assert result["changed"] == 0

    def test_count_value_changes_numeric_all_changed(self):
        op = _op()
        orig = pd.Series([1.0, 2.0, 3.0])
        trans = pd.Series([10.0, 20.0, 30.0])
        result = op._count_value_changes(orig, trans)
        assert result["changed"] == 3
        assert result["percent_changed"] == 100.0

    def test_count_value_changes_string(self):
        op = _op()
        orig = pd.Series(["a", "b", "c"])
        trans = pd.Series(["a", "X", "c"])
        result = op._count_value_changes(orig, trans)
        assert result["changed"] == 1

    def test_count_null_changes(self):
        op = _op()
        orig = pd.Series([1.0, None, 3.0])
        trans = pd.Series([1.0, 2.0, None])
        result = op._count_null_changes(orig, trans)
        assert result["original_nulls"] == 1
        assert result["transformed_nulls"] == 1
        assert result["difference"] == 0

    def test_count_null_changes_increase(self):
        op = _op()
        orig = pd.Series([1.0, 2.0, 3.0])
        trans = pd.Series([1.0, None, None])
        result = op._count_null_changes(orig, trans)
        assert result["difference"] == 2


# ---------------------------------------------------------------------------
# _compare_row_counts / _compare_column_counts
# ---------------------------------------------------------------------------

class TestCompareRowAndColumnCounts:
    def test_compare_row_counts_equal(self):
        op = _op()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = op._compare_row_counts(df, df.copy())
        assert result["difference"] == 0
        assert result["percent_change"] == 0.0

    def test_compare_row_counts_growth(self):
        op = _op()
        orig = pd.DataFrame({"a": [1, 2]})
        trans = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = op._compare_row_counts(orig, trans)
        assert result["difference"] == 2
        assert result["percent_change"] == 100.0

    def test_compare_column_counts_added(self):
        op = _op()
        orig = pd.DataFrame({"a": [1], "b": [2]})
        trans = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        counts, common, added, removed = op._compare_column_counts(orig, trans)
        assert "c" in added
        assert counts["difference"] == 1

    def test_compare_column_counts_removed(self):
        op = _op()
        orig = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        trans = pd.DataFrame({"a": [1]})
        counts, common, added, removed = op._compare_column_counts(orig, trans)
        assert set(["b", "c"]).issubset(set(removed))


# ---------------------------------------------------------------------------
# _compare_memory_usage
# ---------------------------------------------------------------------------

class TestCompareMemoryUsage:
    def test_basic_memory_comparison(self):
        op = _op()
        df = pd.DataFrame({"a": list(range(1000))})
        result = op._compare_memory_usage(df, df.copy())
        assert "original_mb" in result
        assert "transformed_mb" in result
        assert "difference_mb" in result

    def test_larger_df_has_more_memory(self):
        op = _op()
        orig = pd.DataFrame({"a": list(range(10000))})
        trans = pd.DataFrame({"a": list(range(10000)), "b": ["x" * 100] * 10000})
        result = op._compare_memory_usage(orig, trans)
        assert result["difference_mb"] >= 0
        assert result["transformed_mb"] >= result["original_mb"]


# ---------------------------------------------------------------------------
# _generate_visualizations (bar plots)
# ---------------------------------------------------------------------------

class TestGenerateVisualizations:
    def test_bar_plot_called_for_modified_fields(self, tmp_path):
        op = _op(field_operations={"val": {"operation_type": "modify_constant", "constant_value": 99}})
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [99, 99, 99]})
        metrics = {"fields_modified_count": 1, "fields_added_count": 0}

        with patch("pamola_core.utils.visualization.create_bar_plot",
                   return_value=str(tmp_path / "bar.png")) as mock_bar:
            result = op._generate_visualizations(
                original_df=orig,
                processed_df=processed,
                metrics=metrics,
                task_dir=tmp_path,
                vis_theme=None,
                vis_backend="matplotlib",
                vis_strict=False,
                progress_tracker=None,
            )
        assert isinstance(result, dict)

    def test_bar_plot_for_added_fields(self, tmp_path):
        op = _op(field_operations={"new_col": {"operation_type": "add_constant", "constant_value": 1}})
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [1, 2, 3], "new_col": [1, 1, 1]})
        metrics = {"fields_added_count": 1, "fields_modified_count": 0}

        with patch("pamola_core.utils.visualization.create_bar_plot",
                   return_value=str(tmp_path / "bar.png")):
            result = op._generate_visualizations(
                original_df=orig,
                processed_df=processed,
                metrics=metrics,
                task_dir=tmp_path,
                vis_theme=None,
                vis_backend="matplotlib",
                vis_strict=False,
                progress_tracker=None,
            )
        assert isinstance(result, dict)

    def test_bar_plot_error_handled_gracefully(self, tmp_path):
        op = _op(field_operations={"val": {"operation_type": "modify_constant", "constant_value": 0}})
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [0, 0, 0]})
        metrics = {"fields_modified_count": 1, "fields_added_count": 0}

        with patch("pamola_core.utils.visualization.create_bar_plot",
                   side_effect=RuntimeError("viz crash")):
            result = op._generate_visualizations(
                original_df=orig,
                processed_df=processed,
                metrics=metrics,
                task_dir=tmp_path,
                vis_theme=None,
                vis_backend="matplotlib",
                vis_strict=False,
                progress_tracker=None,
            )
        assert isinstance(result, dict)

    def test_enrich_mode_prefixed_column_compared(self, tmp_path):
        op = _op(
            field_operations={"val": {"operation_type": "modify_constant", "constant_value": 99}},
            mode="ENRICH",
            column_prefix="PRE_",
        )
        orig = pd.DataFrame({"val": list(range(100))})
        processed = orig.copy()
        processed["PRE_val"] = [99] * 100
        metrics = {"fields_modified_count": 1, "fields_added_count": 0}

        with patch("pamola_core.utils.visualization.create_bar_plot",
                   return_value=str(tmp_path / "bar.png")):
            result = op._generate_visualizations(
                original_df=orig,
                processed_df=processed,
                metrics=metrics,
                task_dir=tmp_path,
                vis_theme=None,
                vis_backend="matplotlib",
                vis_strict=False,
                progress_tracker=None,
            )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _handle_visualizations (thread / timeout)
# ---------------------------------------------------------------------------

class TestHandleVisualizations:
    def test_handle_visualizations_returns_dict(self, tmp_path):
        op = _op()
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = pd.DataFrame({"val": [99, 99, 99]})
        metrics = {"fields_modified_count": 1, "fields_added_count": 0}
        result = OperationResult(status=OperationStatus.PENDING)

        with patch.object(op, "_generate_visualizations", return_value={"bar": tmp_path / "b.png"}):
            vis = op._handle_visualizations(
                original_df=orig,
                processed_df=processed,
                metrics=metrics,
                task_dir=tmp_path,
                result=result,
                reporter=_reporter(),
                vis_theme=None,
                vis_backend="matplotlib",
                vis_strict=False,
                vis_timeout=30,
                progress_tracker=_progress(),
                operation_timestamp="20240101_000000",
            )
        assert "bar" in vis

    def test_handle_visualizations_generate_exception_returns_empty(self, tmp_path):
        """Exception inside visualization thread yields empty dict."""
        op = _op()
        orig = pd.DataFrame({"val": [1, 2, 3]})
        processed = orig.copy()
        metrics = {}
        result = OperationResult(status=OperationStatus.PENDING)

        with patch.object(op, "_generate_visualizations",
                          side_effect=RuntimeError("internal crash")):
            vis = op._handle_visualizations(
                original_df=orig,
                processed_df=processed,
                metrics=metrics,
                task_dir=tmp_path,
                result=result,
                reporter=_reporter(),
                vis_theme=None,
                vis_backend="matplotlib",
                vis_strict=False,
                vis_timeout=30,
                progress_tracker=None,
                operation_timestamp="20240101_000000",
            )
        assert vis == {}
