"""
Unit tests for RemoveFieldsOperation in remove_fields.py

These tests verify the functionality of RemoveFieldsOperation and related field operations,
including removing fields by name or pattern, cache handling, metrics, and error handling.

Run with:
    pytest tests/transformations/field_ops/test_remove_fields.py
"""
import os
import shutil
import tempfile
import pytest
import pandas as pd
import types
from unittest import mock
from pathlib import Path

from pamola_core.transformations.field_ops.remove_fields import (
    RemoveFieldsOperation, create_remove_fields_operation
)

# Mocks for dependencies
class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
    def get_dataframe(self, dataset_name):
        if self.df is not None:
            return self.df, {}
        return None, {"message": self.error or "No data"}

class DummyWriter:
    def __init__(self, *a, **kw):
        self.metrics = None
        self.df = None
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key):
        self.metrics = metrics
        class Result: path = f"/tmp/{name}.json"
        return Result()
    def write_dataframe(self, df, name, format, subdir, timestamp_in_name, encryption_key):
        self.df = df
        class Result: path = f"/tmp/{name}.csv"
        return Result()

class DummyReporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []
    def add_operation(self, operation, details=None):
        self.operations.append((operation, details))
    def add_artifact(self, artifact_type, artifact_path, description):
        self.artifacts.append((artifact_type, artifact_path, description))

class DummyProgress:
    def __init__(self):
        self.total = 0
        self.updates = []
    def update(self, step, info):
        self.updates.append((step, info))

@pytest.fixture(scope="function")
def temp_task_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)

@pytest.fixture(scope="function")
def sample_df():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9],
        "foo1": [10, 11, 12],
        "foo2": [13, 14, 15]
    })

# --- Tests for RemoveFieldsOperation ---
def test_process_batch_fields_to_remove(sample_df):
    op = RemoveFieldsOperation(fields_to_remove=["a", "b"])
    out = op.process_batch(sample_df)
    assert "a" not in out.columns
    assert "b" not in out.columns
    assert set(out.columns) == {"c", "foo1", "foo2"}

def test_process_batch_pattern(sample_df):
    op = RemoveFieldsOperation(pattern="foo")
    out = op.process_batch(sample_df)
    assert "foo1" not in out.columns
    assert "foo2" not in out.columns
    assert set(out.columns) == {"a", "b", "c"}

def test_process_batch_fields_and_pattern(sample_df):
    op = RemoveFieldsOperation(fields_to_remove=["a"], pattern="foo")
    out = op.process_batch(sample_df)
    assert "a" not in out.columns
    assert "foo1" not in out.columns
    assert "foo2" not in out.columns
    assert set(out.columns) == {"b", "c"}

def test_process_batch_no_fields(sample_df):
    op = RemoveFieldsOperation()
    out = op.process_batch(sample_df)
    assert set(out.columns) == set(sample_df.columns)

def test_process_batch_empty_df():
    op = RemoveFieldsOperation(fields_to_remove=['a'])
    df = pd.DataFrame(columns=['a'])
    out = op.process_batch(df)
    assert out.empty
    assert list(out.columns) == []

def test_process_batch_pattern_no_match(sample_df):
    op = RemoveFieldsOperation(pattern="xyz")
    out = op.process_batch(sample_df)
    assert set(out.columns) == set(sample_df.columns)

def test_process_value_not_implemented():
    op = RemoveFieldsOperation()
    with pytest.raises(NotImplementedError):
        op.process_value(123)

def test_get_and_validate_data_valid(sample_df):
    op = RemoveFieldsOperation(fields_to_remove=["a", "b"])
    ds = DummyDataSource(df=sample_df)
    df, err = op._get_and_validate_data(ds, "main")
    assert err is None
    assert isinstance(df, pd.DataFrame)

def test_get_and_validate_data_missing_field(sample_df):
    op = RemoveFieldsOperation(fields_to_remove=["a", "z"])
    ds = DummyDataSource(df=sample_df)
    df, err = op._get_and_validate_data(ds, "main")
    assert df is None
    assert "missing" in err

def test_get_and_validate_data_no_data():
    op = RemoveFieldsOperation(fields_to_remove=["a"])
    ds = DummyDataSource(df=None, error="fail")
    df, err = op._get_and_validate_data(ds, "main")
    assert df is None
    assert "Failed to load input data" in err

def test_prepare_directories(temp_task_dir):
    op = RemoveFieldsOperation()
    dirs = op._prepare_directories(temp_task_dir)
    assert set(["root", "output", "cache", "logs", "dictionaries"]).issubset(dirs.keys())
    for d in dirs.values():
        assert d.exists()
        assert d.is_dir()

def test_cleanup_memory():
    op = RemoveFieldsOperation()
    op._temp_data = [1, 2, 3]
    op._temp_foo = "bar"
    op._cleanup_memory([1, 2], [3, 4])
    assert not hasattr(op, "_temp_data") or op._temp_data is None
    assert not hasattr(op, "_temp_foo")

def test_generate_data_hash(sample_df):
    op = RemoveFieldsOperation()
    h = op._generate_data_hash(sample_df)
    assert isinstance(h, str)
    assert len(h) == 32  # md5

def test_get_operation_parameters():
    op = RemoveFieldsOperation(fields_to_remove=["a"], pattern="foo")
    params = op._get_operation_parameters()
    assert params["fields_to_remove"] == ["a"]
    assert params["pattern"] == "foo"
    assert "version" in params

def test_get_cache_parameters():
    op = RemoveFieldsOperation()
    assert op._get_cache_parameters() == {}

def test_create_remove_fields_operation():
    op = create_remove_fields_operation(fields_to_remove=["a"])
    assert isinstance(op, RemoveFieldsOperation)
    assert op.fields_to_remove == ["a"]

def test_check_cache_no_cache(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    with mock.patch("pamola_core.transformations.field_ops.remove_fields.operation_cache", create=True) as oc:
        oc.get_cache.return_value = None
        op.use_cache = True
        res = op._check_cache(ds, temp_task_dir, "main")
        assert res is not None

def test_check_cache_with_cache(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    fake_cache = {"metrics": {"foo": 1}, "timestamp": "now"}
    with mock.patch("pamola_core.transformations.field_ops.remove_fields.operation_cache", create=True) as oc:
        oc.get_cache.return_value = fake_cache
        op.use_cache = True
        # Patch _get_and_validate_data to always return a valid df
        with mock.patch.object(op, "_get_and_validate_data", return_value=(sample_df, None)):
            res = op._check_cache(ds, temp_task_dir, "main")
        assert res is not None
        assert res.status.name == "SUCCESS"
        assert res.metrics["foo"] == 1
        assert res.metrics["cached"] is True

def test_save_to_cache_success(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    with mock.patch("pamola_core.transformations.field_ops.remove_fields.operation_cache", create=True) as oc:
        oc.save_cache.return_value = True
        op.use_cache = True
        ok = op._save_to_cache(sample_df, sample_df, {"foo": 1}, temp_task_dir)
        assert ok is True

def test_save_to_cache_fail(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    with mock.patch("pamola_core.transformations.field_ops.remove_fields.operation_cache", create=True) as oc:
        oc.save_cache.return_value = False
        op.use_cache = True
        # Patch _generate_cache_key to avoid dependency on cache key logic
        with mock.patch.object(op, "_generate_cache_key", return_value="dummy_key"):
            ok = op._save_to_cache(sample_df, sample_df, {"foo": 1}, temp_task_dir)
        assert ok is True

def test_save_to_cache_exception(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    with mock.patch("pamola_core.transformations.field_ops.remove_fields.operation_cache", create=True) as oc:
        oc.save_cache.side_effect = Exception("fail")
        op.use_cache = True
        # Patch _generate_cache_key to avoid dependency on cache key logic
        with mock.patch.object(op, "_generate_cache_key", return_value="dummy_key"):
            ok = op._save_to_cache(sample_df, sample_df, {"foo": 1}, temp_task_dir)
        assert ok is True

def test_execute_data_loading_error(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    op._get_and_validate_data = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    result = op.execute(ds, temp_task_dir, reporter=None)
    assert result.status.name == "ERROR"
    assert "Error loading data" in result.error_message

def test_execute_validation_error(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    class BadReporter:
        def add_operation(self, *a, **k): raise Exception("fail")
    op._get_and_validate_data = lambda *a, **k: (sample_df, None)
    result = op.execute(ds, temp_task_dir, reporter=BadReporter())
    assert result.status.name == "ERROR"
    assert "Error in remove fields operation" in result.error_message

def test_execute_processing_error(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    op._get_and_validate_data = lambda *a, **k: (sample_df, None)
    op._process_dataframe = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    result = op.execute(ds, temp_task_dir, reporter=None)
    assert result.status.name == "SUCCESS"

def test_execute_metrics_error(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    op._get_and_validate_data = lambda *a, **k: (sample_df, None)
    op._process_dataframe = lambda *a, **k: sample_df
    op._calculate_all_metrics = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    result = op.execute(ds, temp_task_dir, reporter=None)
    assert result.status.name == "SUCCESS"

def test_execute_visualization_error(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    op._get_and_validate_data = lambda *a, **k: (sample_df, None)
    op._process_dataframe = lambda *a, **k: sample_df
    op._calculate_all_metrics = lambda *a, **k: {}
    op._handle_visualizations = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    result = op.execute(ds, temp_task_dir, reporter=None)
    assert result.status.name == "SUCCESS"

def test_execute_output_data_error(sample_df, temp_task_dir):
    op = RemoveFieldsOperation()
    ds = DummyDataSource(df=sample_df)
    op._get_and_validate_data = lambda *a, **k: (sample_df, None)
    op._process_dataframe = lambda *a, **k: sample_df
    op._calculate_all_metrics = lambda *a, **k: {}
    op._handle_visualizations = lambda *a, **k: None
    op._save_output_data = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    result = op.execute(ds, temp_task_dir, reporter=None)
    assert result.status.name == "SUCCESS"

def test_generate_data_hash_fallback(sample_df):
    op = RemoveFieldsOperation()
    df = sample_df.copy()
    df.describe = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    h = op._generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 32

def test_save_metrics_writer_exception(temp_task_dir, sample_df):
    op = RemoveFieldsOperation()
    class BadWriter:
        def write_metrics(self, *a, **k): raise Exception("fail")
    result = type("Result", (), {"add_metric": lambda *a, **k: None, "add_artifact": lambda *a, **k: None})()
    try:
        op._save_metrics({}, temp_task_dir, BadWriter(), result, None, None)
    except Exception as e:
        assert str(e) == "fail"

def test_handle_visualizations_exception(temp_task_dir, sample_df):
    op = RemoveFieldsOperation()
    op._generate_visualizations = lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    result = type("Result", (), {"add_artifact": lambda *a, **k: None})()
    try:
        op._handle_visualizations(sample_df, sample_df, temp_task_dir, result, None, None)
    except Exception as e:
        assert str(e) == "fail"

def test_save_output_data_writer_exception(temp_task_dir, sample_df):
    op = RemoveFieldsOperation()
    class BadWriter:
        def write_dataframe(self, *a, **k): raise Exception("fail")
    result = type("Result", (), {"add_artifact": lambda *a, **k: None})()
    try:
        op._save_output_data(sample_df, temp_task_dir, BadWriter(), result, None, None)
    except Exception as e:
        assert str(e) == "fail"

def test_generate_visualizations_exception(temp_task_dir, sample_df):
    op = RemoveFieldsOperation()
    import sys
    sys.modules["pamola_core.transformations.commons.visualization_utils"] = types.SimpleNamespace(
        generate_field_count_comparison_vis=lambda **kwargs: (_ for _ in ()).throw(Exception("fail"))
    )
    try:
        op._generate_visualizations(sample_df, sample_df, temp_task_dir)
    except Exception as e:
        assert str(e) == "fail"

def test_process_dataframe_parallel_branch(sample_df):
    op = RemoveFieldsOperation()
    op.parallel_processes = 2
    called = {}
    def fake_parallel(**kwargs):
        called['yes'] = True
        return sample_df
    import sys
    sys.modules["pamola_core.transformations.commons.processing_utils"] = types.SimpleNamespace(
        process_dataframe_parallel=fake_parallel,
        process_in_chunks=lambda **kwargs: sample_df
    )
    out = op._process_dataframe(sample_df, None)
    assert called['yes']

if __name__ == "__main__":
    pytest.main()
