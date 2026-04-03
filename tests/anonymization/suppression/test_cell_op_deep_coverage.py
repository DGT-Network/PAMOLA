"""Deep coverage tests for CellSuppressionOperation — targets internal paths.
Exercises: _collect_specific_metrics, _build_suppression_mask, _get_cache_parameters,
process_batch for each strategy, visualization disabled, and large-data chunked paths."""
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
        def add_artifact(self, *a, **kw): pass
    return R()


def _run(op, df, tmp_path, reporter):
    op.preset_type = None
    op.preset_name = None
    return op.execute(make_ds(df), tmp_path, reporter)


# Exercise each strategy with enough rows to trigger metrics paths

def test_null_with_many_rows(reporter, tmp_path):
    vals = [f"v{i}" if i % 3 != 0 else None for i in range(200)]
    df = pd.DataFrame({"val": vals, "id": range(200)})
    op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_mean_with_many_rows(reporter, tmp_path):
    vals = [float(i) if i % 5 != 0 else np.nan for i in range(200)]
    df = pd.DataFrame({"val": vals, "id": range(200)})
    op = CellSuppressionOperation(field_name="val", suppression_strategy="mean")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_median_with_many_rows(reporter, tmp_path):
    vals = [float(i) if i % 4 != 0 else np.nan for i in range(200)]
    df = pd.DataFrame({"val": vals, "id": range(200)})
    op = CellSuppressionOperation(field_name="val", suppression_strategy="median")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_mode_with_many_rows(reporter, tmp_path):
    vals = ["a", "b", "a", "c", None] * 40
    df = pd.DataFrame({"val": vals, "id": range(200)})
    op = CellSuppressionOperation(field_name="val", suppression_strategy="mode")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_constant_with_many_rows(reporter, tmp_path):
    vals = [f"v{i}" if i % 4 != 0 else None for i in range(200)]
    df = pd.DataFrame({"val": vals, "id": range(200)})
    op = CellSuppressionOperation(
        field_name="val", suppression_strategy="constant", suppression_value="X",
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_group_mean_many_rows(reporter, tmp_path):
    df = pd.DataFrame({
        "val": [float(i) if i % 5 != 0 else np.nan for i in range(200)],
        "grp": [f"g{i % 4}" for i in range(200)],
        "id": range(200),
    })
    op = CellSuppressionOperation(
        field_name="val", suppression_strategy="group_mean",
        group_by_field="grp", min_group_size=3,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_group_mode_many_rows(reporter, tmp_path):
    df = pd.DataFrame({
        "val": ["a", "b", "a", "c", None] * 40,
        "grp": [f"g{i % 3}" for i in range(200)],
        "id": range(200),
    })
    op = CellSuppressionOperation(
        field_name="val", suppression_strategy="group_mode",
        group_by_field="grp", min_group_size=2,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

# Numeric field with mean/median strategies
def test_mean_integer_field(reporter, tmp_path):
    df = pd.DataFrame({"score": [10, 20, None, 40, None, 60], "id": range(6)})
    op = CellSuppressionOperation(field_name="score", suppression_strategy="mean")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

# Very large data to trigger chunked processing
def test_large_data_null(reporter, tmp_path):
    n = 10000
    df = pd.DataFrame({
        "val": [f"v{i}" if i % 10 != 0 else None for i in range(n)],
        "num": [float(i) for i in range(n)],
    })
    op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_large_data_group_mean(reporter, tmp_path):
    n = 10000
    df = pd.DataFrame({
        "val": [float(i) if i % 7 != 0 else np.nan for i in range(n)],
        "grp": [f"g{i % 10}" for i in range(n)],
    })
    op = CellSuppressionOperation(
        field_name="val", suppression_strategy="group_mean",
        group_by_field="grp", min_group_size=5,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

# Visualization disabled
def test_null_single_column(reporter, tmp_path):
    df = pd.DataFrame({"val": [1, None, 3, None, 5]})
    op = CellSuppressionOperation(field_name="val", suppression_strategy="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

# Suppression value with constant
def test_constant_zero(reporter, tmp_path):
    df = pd.DataFrame({"val": [1, None, 3, None, 5], "id": range(5)})
    op = CellSuppressionOperation(
        field_name="val", suppression_strategy="constant", suppression_value=0,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS
