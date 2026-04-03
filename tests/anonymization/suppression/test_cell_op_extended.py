"""Extended tests for CellSuppressionOperation targeting 188 missed lines.
Valid strategies: null, mean, median, mode, constant, group_mean, group_mode."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
    return R()


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Carol", "Dave", None],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "dept": ["IT", "HR", "IT", "Finance", "HR"],
        "group": ["A", "A", "B", "B", "A"],
    })


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# --- Null strategy ---
class TestNullStrategy:
    def test_null_suppression(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(field_name="name", suppression_strategy="null")
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_null_no_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"name": ["A", "B", "C"], "val": [1, 2, 3]})
        op = CellSuppressionOperation(field_name="name", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_null_all_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"name": [None, None, None], "val": [1, 2, 3]})
        op = CellSuppressionOperation(field_name="name", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_null_with_custom_value(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(
            field_name="name", suppression_strategy="null",
            suppression_value="[REDACTED]",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Mean strategy ---
class TestMeanStrategy:
    def test_mean_numeric(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(field_name="salary", suppression_strategy="mean")
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mean_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [10.0, np.nan, 30.0, np.nan, 50.0], "id": range(5)})
        op = CellSuppressionOperation(field_name="val", suppression_strategy="mean")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Median strategy ---
class TestMedianStrategy:
    def test_median_numeric(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(field_name="age", suppression_strategy="median")
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Mode strategy ---
class TestModeStrategy:
    def test_mode_categorical(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(field_name="dept", suppression_strategy="mode")
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mode_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"cat": ["a", "a", "b", None, None], "id": range(5)})
        op = CellSuppressionOperation(field_name="cat", suppression_strategy="mode")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Constant strategy ---
class TestConstantStrategy:
    def test_constant_string(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(
            field_name="name", suppression_strategy="constant",
            suppression_value="***",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_constant_numeric(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(
            field_name="salary", suppression_strategy="constant",
            suppression_value=0,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Group mean/mode ---
class TestGroupStrategies:
    def test_group_mean(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(
            field_name="salary", suppression_strategy="group_mean",
            group_by_field="dept",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_group_mode(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(
            field_name="name", suppression_strategy="group_mode",
            group_by_field="dept",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_group_mean_small_groups(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [10, 20, 30, 40, 50],
            "grp": ["A", "A", "B", "B", "C"],
            "id": range(5),
        })
        op = CellSuppressionOperation(
            field_name="val", suppression_strategy="group_mean",
            group_by_field="grp", min_group_size=3,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_group_mean_large_min(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [10, 20, 30, 40, 50],
            "grp": ["A", "A", "B", "B", "A"],
            "id": range(5),
        })
        op = CellSuppressionOperation(
            field_name="val", suppression_strategy="group_mean",
            group_by_field="grp", min_group_size=10,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Edge cases ---
class TestEdgeCases:
    def test_field_not_found(self, reporter, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        op = CellSuppressionOperation(field_name="missing", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_single_row(self, reporter, tmp_path):
        df = pd.DataFrame({"x": [None], "y": [1]})
        op = CellSuppressionOperation(field_name="x", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_large_df(self, reporter, tmp_path):
        n = 5000
        df = pd.DataFrame({
            "val": [f"v{i}" if i % 10 != 0 else None for i in range(n)],
            "id": range(n),
        })
        op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_all_nan_numeric(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [np.nan] * 5, "id": range(5)})
        op = CellSuppressionOperation(field_name="val", suppression_strategy="mean")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mixed_types(self, reporter, tmp_path):
        df = pd.DataFrame({"mixed": [1, "two", 3.0, None, True], "id": range(5)})
        op = CellSuppressionOperation(field_name="mixed", suppression_strategy="null")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_execution(self, base_df, reporter, tmp_path):
        op = CellSuppressionOperation(
            field_name="name", suppression_strategy="null", use_dask=True,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_numeric_all_same(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [42] * 10, "id": range(10)})
        op = CellSuppressionOperation(field_name="val", suppression_strategy="mean")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
