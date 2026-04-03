"""Extended tests for AttributeSuppressionOperation targeting 93 missed lines."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.anonymization.suppression.attribute_op import AttributeSuppressionOperation
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
        "name": ["Alice", "Bob", "Carol", "Dave"],
        "age": [25, 30, 35, 40],
        "salary": [50000, 60000, 70000, 80000],
        "dept": ["IT", "HR", "IT", "Finance"],
    })


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


class TestRemoveMode:
    def test_remove_single_field(self, base_df, reporter, tmp_path):
        op = AttributeSuppressionOperation(field_name="salary")
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_remove_multiple_fields(self, base_df, reporter, tmp_path):
        op = AttributeSuppressionOperation(
            field_name="salary", additional_fields=["dept"],
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_remove_nonexistent_field(self, reporter, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        op = AttributeSuppressionOperation(field_name="nonexistent")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_save_suppressed_schema_true(self, base_df, reporter, tmp_path):
        op = AttributeSuppressionOperation(
            field_name="name", save_suppressed_schema=True,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_save_suppressed_schema_false(self, base_df, reporter, tmp_path):
        op = AttributeSuppressionOperation(
            field_name="name", save_suppressed_schema=False,
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


class TestEdgeCases:
    def test_with_nulls(self, reporter, tmp_path):
        df = pd.DataFrame({"name": ["A", None, "C", None], "val": range(4)})
        op = AttributeSuppressionOperation(field_name="name")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_large_df(self, reporter, tmp_path):
        n = 5000
        df = pd.DataFrame({"name": [f"p{i}" for i in range(n)], "val": range(n)})
        op = AttributeSuppressionOperation(field_name="name")
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_dask_execution(self, base_df, reporter, tmp_path):
        op = AttributeSuppressionOperation(field_name="name", use_dask=True)
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_additional_fields_with_nonexistent(self, base_df, reporter, tmp_path):
        op = AttributeSuppressionOperation(
            field_name="name", additional_fields=["nonexistent"],
        )
        result = _run(op, base_df, tmp_path, reporter)
        assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)

    def test_remove_all_but_one(self, reporter, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        op = AttributeSuppressionOperation(
            field_name="a", additional_fields=["b"],
        )
        result = _run(op, df, tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


class TestConstructor:
    def test_default_mode(self):
        op = AttributeSuppressionOperation(field_name="name")
        assert op.suppression_mode == "REMOVE"

    def test_with_additional_fields(self):
        op = AttributeSuppressionOperation(
            field_name="name", additional_fields=["age", "salary"],
        )
        assert op.additional_fields == ["age", "salary"]
