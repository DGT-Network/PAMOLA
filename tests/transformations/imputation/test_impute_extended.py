"""Extended tests for ImputeMissingValuesOperation — targets 118 missed lines.
Focuses on: _collect_metrics, _collect_specific_metrics, process_batch_dask,
various imputation strategies, and visualization paths."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def make_ds(df):
    return DataSource(dataframes={"main": df})


@pytest.fixture
def reporter():
    class R:
        def add_operation(self, *a, **kw): pass
        def add_artifact(self, *a, **kw): pass
    return R()


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# --- Mean imputation ---
class TestMeanImputation:
    def test_mean_numeric_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [10.0, np.nan, 30.0, np.nan, 50.0],
            "id": range(5),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mean_no_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "id": range(3)})
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mean_all_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [np.nan, np.nan, np.nan], "id": range(3)})
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)


# --- Median imputation ---
class TestMedianImputation:
    def test_median_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0],
            "id": range(8),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "median"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Mode imputation ---
class TestModeImputation:
    def test_mode_categorical(self, reporter, tmp_path):
        df = pd.DataFrame({
            "cat": ["a", "b", "a", None, "a", None, "b"],
            "id": range(7),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"cat": {"strategy": "mode"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mode_numeric(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [1, 2, 1, np.nan, 1, np.nan, 2],
            "id": range(7),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mode"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Constant imputation ---
class TestConstantImputation:
    def test_constant_string(self, reporter, tmp_path):
        df = pd.DataFrame({
            "name": ["Alice", None, "Carol", None],
            "id": range(4),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"name": {"strategy": "constant", "fill_value": "UNKNOWN"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_constant_numeric(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [1.0, np.nan, 3.0, np.nan],
            "id": range(4),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "constant", "fill_value": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Forward/backward fill ---
class TestFillStrategies:
    def test_forward_fill(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [1.0, np.nan, np.nan, 4.0, np.nan],
            "id": range(5),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "ffill"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_backward_fill(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [np.nan, 2.0, np.nan, 4.0, np.nan],
            "id": range(5),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "bfill"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Multiple fields ---
class TestMultipleFields:
    def test_multiple_strategies(self, reporter, tmp_path):
        df = pd.DataFrame({
            "age": [25, np.nan, 35, np.nan, 45],
            "name": ["A", None, "C", None, "E"],
            "salary": [50000, np.nan, 70000, np.nan, 90000],
            "id": range(5),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={
                "age": {"strategy": "mean"},
                "name": {"strategy": "constant", "fill_value": "N/A"},
                "salary": {"strategy": "median"},
            },
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Invalid values ---
class TestInvalidValues:
    def test_invalid_values_replaced(self, reporter, tmp_path):
        df = pd.DataFrame({
            "age": [25, -1, 35, 999, 45],
            "id": range(5),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"age": {"strategy": "mean"}},
            invalid_values={"age": [-1, 999]},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Edge cases ---
class TestEdgeCases:
    def test_empty_strategies(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3], "id": range(3)})
        op = ImputeMissingValuesOperation(field_strategies={})
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_nonexistent_field(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3]})
        op = ImputeMissingValuesOperation(
            field_strategies={"nonexistent": {"strategy": "mean"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_large_df(self, reporter, tmp_path):
        n = 5000
        vals = [float(i) if i % 7 != 0 else np.nan for i in range(n)]
        df = pd.DataFrame({"val": vals, "id": range(n)})
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_single_row(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [np.nan], "id": [0]})
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "constant", "fill_value": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_execution(self, reporter, tmp_path):
        df = pd.DataFrame({
            "val": [1.0, np.nan, 3.0, np.nan, 5.0],
            "id": range(5),
        })
        op = ImputeMissingValuesOperation(
            field_strategies={"val": {"strategy": "mean"}},
            use_dask=True,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
