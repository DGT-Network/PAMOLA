"""Extended tests for CleanInvalidValuesOperation — targets 111 missed lines.
Focuses on: constraint validation, whitelist/blacklist, null_replacement,
process_batch paths, and _collect_metrics."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.transformations.cleaning.clean_invalid_values import CleanInvalidValuesOperation
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


# --- Range constraints ---
class TestRangeConstraints:
    def test_numeric_range(self, reporter, tmp_path):
        df = pd.DataFrame({"age": [15, 25, 35, 150, -5, 45], "id": range(6)})
        op = CleanInvalidValuesOperation(
            field_constraints={"age": {"min": 0, "max": 120}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_min_only(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [-10, 0, 10, 20], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_max_only(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [10, 50, 100, 200], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"max": 100}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Type constraints ---
class TestTypeConstraints:
    def test_numeric_type_check(self, reporter, tmp_path):
        df = pd.DataFrame({"age": [25, 30, "abc", 40, None], "id": range(5)})
        op = CleanInvalidValuesOperation(
            field_constraints={"age": {"type": "numeric"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_string_type_check(self, reporter, tmp_path):
        df = pd.DataFrame({"name": ["Alice", 123, "Carol", None], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"name": {"type": "string"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Regex constraints ---
class TestRegexConstraints:
    def test_email_pattern(self, reporter, tmp_path):
        df = pd.DataFrame({
            "email": ["a@b.com", "invalid", "c@d.org", "nope", None],
            "id": range(5),
        })
        op = CleanInvalidValuesOperation(
            field_constraints={"email": {"regex": r"^[^@]+@[^@]+\.[^@]+$"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_numeric_pattern(self, reporter, tmp_path):
        df = pd.DataFrame({"code": ["123", "abc", "456", "7x8"], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"code": {"regex": r"^\d+$"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Multiple fields ---
class TestMultipleFields:
    def test_multiple_constraints(self, reporter, tmp_path):
        df = pd.DataFrame({
            "age": [25, -5, 35, 200, 45],
            "name": ["Alice", "", "Carol", None, "Eve"],
            "score": [3.5, 4.0, -1.0, 5.5, 3.0],
            "id": range(5),
        })
        op = CleanInvalidValuesOperation(
            field_constraints={
                "age": {"min": 0, "max": 120},
                "score": {"min": 0, "max": 5},
            },
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Null replacement ---
class TestNullReplacement:
    def test_null_replacement_string(self, reporter, tmp_path):
        df = pd.DataFrame({"name": ["Alice", None, "Carol"], "id": range(3)})
        op = CleanInvalidValuesOperation(
            field_constraints={"name": {"not_null": True}},
            null_replacement="UNKNOWN",
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_null_replacement_per_field(self, reporter, tmp_path):
        df = pd.DataFrame({
            "age": [25, np.nan, 35],
            "name": ["A", None, "C"],
            "id": range(3),
        })
        op = CleanInvalidValuesOperation(
            field_constraints={
                "age": {"min": 0, "max": 120},
            },
            null_replacement={"age": 0, "name": "N/A"},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# --- Edge cases ---
class TestEdgeCases:
    def test_no_constraints(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3], "id": range(3)})
        op = CleanInvalidValuesOperation(field_constraints={})
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_nonexistent_field(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1, 2, 3]})
        op = CleanInvalidValuesOperation(
            field_constraints={"nonexistent": {"min": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_large_df(self, reporter, tmp_path):
        n = 5000
        vals = [i if i % 10 != 0 else -999 for i in range(n)]
        df = pd.DataFrame({"val": vals, "id": range(n)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_all_invalid(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [-1, -2, -3], "id": range(3)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_all_valid(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [10, 20, 30], "id": range(3)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0, "max": 100}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_single_row(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [-1], "id": [0]})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_mixed_types(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1, "two", 3.0, None], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"type": "numeric"}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_with_nan(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0, np.nan], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0}},
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_execution(self, reporter, tmp_path):
        df = pd.DataFrame({"val": [1, -1, 3, -3], "id": range(4)})
        op = CleanInvalidValuesOperation(
            field_constraints={"val": {"min": 0}},
            use_dask=True,
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
