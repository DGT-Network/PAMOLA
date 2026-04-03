"""Deep coverage tests for RecordSuppressionOperation — targets 114 missed lines.
Exercises: save_suppressed_records, metrics collection, large data, visualization,
various conditions with enough rows to trigger internal branching."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.anonymization.suppression.record_op import RecordSuppressionOperation
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


# --- Null condition with many rows (metrics paths) ---
def test_null_200_rows(reporter, tmp_path):
    vals = [f"v{i}" if i % 5 != 0 else None for i in range(200)]
    df = pd.DataFrame({"name": vals, "val": range(200)})
    op = RecordSuppressionOperation(field_name="name", suppression_condition="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_null_save_suppressed(reporter, tmp_path):
    vals = [f"v{i}" if i % 3 != 0 else None for i in range(100)]
    df = pd.DataFrame({"name": vals, "val": range(100)})
    op = RecordSuppressionOperation(
        field_name="name", suppression_condition="null",
        save_suppressed_records=True,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_null_no_suppressed(reporter, tmp_path):
    df = pd.DataFrame({"name": ["a", "b", "c"] * 30, "val": range(90)})
    op = RecordSuppressionOperation(field_name="name", suppression_condition="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS


# --- Value condition with many rows ---
def test_value_200_rows(reporter, tmp_path):
    df = pd.DataFrame({
        "dept": [f"d{i % 5}" for i in range(200)],
        "val": range(200),
    })
    op = RecordSuppressionOperation(
        field_name="dept", suppression_condition="value",
        suppression_values=["d0", "d1"],
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_value_save_suppressed(reporter, tmp_path):
    df = pd.DataFrame({
        "cat": ["keep", "remove", "keep", "remove"] * 25,
        "val": range(100),
    })
    op = RecordSuppressionOperation(
        field_name="cat", suppression_condition="value",
        suppression_values=["remove"],
        save_suppressed_records=True,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_value_no_matches(reporter, tmp_path):
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 30, "val": range(90)})
    op = RecordSuppressionOperation(
        field_name="cat", suppression_condition="value",
        suppression_values=["nonexistent"],
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS


# --- Range condition ---
def test_range_200_rows(reporter, tmp_path):
    df = pd.DataFrame({"age": list(range(200)), "val": range(200)})
    op = RecordSuppressionOperation(
        field_name="age", suppression_condition="range",
        suppression_range=[50, 150],
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_range_save_suppressed(reporter, tmp_path):
    df = pd.DataFrame({"score": list(range(100)), "val": range(100)})
    op = RecordSuppressionOperation(
        field_name="score", suppression_condition="range",
        suppression_range=[20, 80],
        save_suppressed_records=True,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS


# --- Risk condition ---
def test_risk_200_rows(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({
        "name": [f"p{i}" for i in range(200)],
        "k_anon_risk": np.random.uniform(0, 10, 200),
    })
    op = RecordSuppressionOperation(
        field_name="name", suppression_condition="risk",
        ka_risk_field="k_anon_risk", risk_threshold=5.0,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_risk_save_suppressed(reporter, tmp_path):
    df = pd.DataFrame({
        "name": [f"p{i}" for i in range(100)],
        "k_anon_risk": [1.0 if i % 3 == 0 else 8.0 for i in range(100)],
    })
    op = RecordSuppressionOperation(
        field_name="name", suppression_condition="risk",
        ka_risk_field="k_anon_risk", risk_threshold=5.0,
        save_suppressed_records=True,
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS


# --- Large data ---
def test_null_10k_rows(reporter, tmp_path):
    n = 10000
    vals = [f"v{i}" if i % 10 != 0 else None for i in range(n)]
    df = pd.DataFrame({"name": vals, "val": range(n)})
    op = RecordSuppressionOperation(field_name="name", suppression_condition="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS


# --- Visualization disabled ---
def test_null_mostly_null(reporter, tmp_path):
    vals = [None if i % 2 == 0 else f"v{i}" for i in range(100)]
    df = pd.DataFrame({"name": vals, "val": range(100)})
    op = RecordSuppressionOperation(field_name="name", suppression_condition="null")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

# --- Suppression reason field ---
def test_custom_reason_field(reporter, tmp_path):
    df = pd.DataFrame({"name": [None, "a", None, "b"] * 25, "val": range(100)})
    op = RecordSuppressionOperation(
        field_name="name", suppression_condition="null",
        suppression_reason_field="_my_reason",
    )
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS
