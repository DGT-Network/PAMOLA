"""Deep coverage tests for CurrencyOperation — targets 184 missed lines.
Exercises: statistical analysis, normality tests, distribution detection,
semantic notes, large data, and visualization paths."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.profiling.analyzers.currency import CurrencyOperation
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


# Diverse numeric distributions to trigger different statistical branches
def test_uniform_distribution(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"price": np.random.uniform(10, 100, 200), "id": range(200)})
    op = CurrencyOperation(field_name="price")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_normal_distribution(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"amount": np.random.normal(50, 10, 300), "id": range(300)})
    op = CurrencyOperation(field_name="amount")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_skewed_distribution(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"income": np.random.exponential(5000, 200), "id": range(200)})
    op = CurrencyOperation(field_name="income")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_integers_only(reporter, tmp_path):
    df = pd.DataFrame({"amount": list(range(0, 500, 5)), "id": range(100)})
    op = CurrencyOperation(field_name="amount")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_with_zeros(reporter, tmp_path):
    vals = [0, 0, 0, 10, 20, 30, 0, 50, 0, 100] * 20
    df = pd.DataFrame({"price": vals, "id": range(200)})
    op = CurrencyOperation(field_name="price")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_with_negatives(reporter, tmp_path):
    vals = [-100, -50, 0, 50, 100, 200, -30, 75, 150, -10] * 20
    df = pd.DataFrame({"balance": vals, "id": range(200)})
    op = CurrencyOperation(field_name="balance")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_with_nulls(reporter, tmp_path):
    vals = [10.0, np.nan, 30.0, np.nan, 50.0, 60.0, np.nan, 80.0] * 25
    df = pd.DataFrame({"price": vals, "id": range(200)})
    op = CurrencyOperation(field_name="price")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_small_values(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"micro": np.random.uniform(0.001, 0.1, 200), "id": range(200)})
    op = CurrencyOperation(field_name="micro")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_large_values(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"big": np.random.uniform(1e6, 1e9, 200), "id": range(200)})
    op = CurrencyOperation(field_name="big")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_constant_values(reporter, tmp_path):
    df = pd.DataFrame({"fixed": [99.99] * 200, "id": range(200)})
    op = CurrencyOperation(field_name="fixed")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_bimodal_distribution(reporter, tmp_path):
    np.random.seed(42)
    vals = list(np.random.normal(20, 3, 100)) + list(np.random.normal(80, 3, 100))
    df = pd.DataFrame({"price": vals, "id": range(200)})
    op = CurrencyOperation(field_name="price")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_high_precision_decimals(reporter, tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"rate": np.random.uniform(0.01, 0.99, 200).round(6), "id": range(200)})
    op = CurrencyOperation(field_name="rate")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_field_not_found(reporter, tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    op = CurrencyOperation(field_name="nonexistent")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.ERROR

def test_very_large_dataset(reporter, tmp_path):
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({"amount": np.random.lognormal(5, 1, n), "id": range(n)})
    op = CurrencyOperation(field_name="amount")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_repeated_values(reporter, tmp_path):
    df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0] * 40, "id": range(200)})
    op = CurrencyOperation(field_name="price")
    result = _run(op, df, tmp_path, reporter)
    assert result.status == OperationStatus.SUCCESS

def test_all_nan(reporter, tmp_path):
    df = pd.DataFrame({"price": [np.nan] * 50, "id": range(50)})
    op = CurrencyOperation(field_name="price")
    result = _run(op, df, tmp_path, reporter)
    assert result.status in (OperationStatus.SUCCESS, OperationStatus.ERROR)
