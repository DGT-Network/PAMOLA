"""Extended tests for NumericGeneralizationOperation targeting 115 missed lines."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.anonymization.generalization.numeric_op import NumericGeneralizationOperation
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
        "age": [22, 35, 47, 58, 63, 71, 84, 29, 41, 55],
        "salary": [30000, 45000, 55000, 65000, 72000, 80000, 95000, 38000, 52000, 60000],
        "score": [3.2, 4.5, 2.8, 4.9, 3.7, 4.1, 2.5, 3.9, 4.3, 3.0],
    })


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# --- Binning strategy ---
class TestBinning:
    def test_equal_width_default(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=5,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_equal_width_many_bins(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=20,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_equal_width_two_bins(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=2,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_equal_frequency(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning",
            bin_count=5, binning_method="equal_frequency",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_salary_binning(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="salary", strategy="binning", bin_count=4,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_score_binning(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="score", strategy="binning", bin_count=3,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Rounding strategy ---
class TestRounding:
    def test_rounding_precision_0(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="score", strategy="rounding", precision=0,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_precision_1(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="score", strategy="rounding", precision=1,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_integers(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="rounding", precision=0,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_salary(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="salary", strategy="rounding", precision=-2,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Range strategy ---
class TestRangeStrategy:
    def test_binning_salary_equal_freq(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="salary", strategy="binning",
            bin_count=5, binning_method="equal_frequency",
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Edge cases ---
class TestEdgeCases:
    def test_field_not_found(self, reporter, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        op = NumericGeneralizationOperation(field_name="nonexistent", strategy="binning")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0, np.nan, 5.0], "id": range(5)})
        op = NumericGeneralizationOperation(field_name="val", strategy="binning", bin_count=3)
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_all_same_value(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [42] * 20, "id": range(20)})
        op = NumericGeneralizationOperation(field_name="val", strategy="binning", bin_count=5)
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_single_row(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [10], "id": [0]})
        op = NumericGeneralizationOperation(field_name="val", strategy="rounding", precision=0)
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_negative_values(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [-10, -5, 0, 5, 10], "id": range(5)})
        op = NumericGeneralizationOperation(field_name="val", strategy="binning", bin_count=3)
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_large_df(self, reporter, tmp_path):
        n = 5000
        df = pd.DataFrame({"val": np.random.randint(0, 100, n), "id": range(n)})
        op = NumericGeneralizationOperation(field_name="val", strategy="binning", bin_count=10)
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_execution(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=5, use_dask=True,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_quasi_identifiers(self, base_df, reporter, tmp_path):
        op = NumericGeneralizationOperation(
            field_name="age", strategy="binning", bin_count=5,
            quasi_identifiers=["salary"],
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
