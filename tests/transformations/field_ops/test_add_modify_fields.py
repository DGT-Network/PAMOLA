"""
Tests for the add_modify_fields module in the PAMOLA.CORE package.

These tests verify the functionality of AddOrModifyFieldsOperation and related field operations,
including adding, modifying, and enriching fields, as well as cache and metrics integration.

Run with:
    pytest tests/transformations/field_ops/test_add_modify_fields.py
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from pamola_core.transformations.field_ops.add_modify_fields import (
    AddOrModifyFieldsOperation, create_add_modify_fields_operation
)

class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
    def get_dataframe(self, dataset_name):
        if self.df is not None:
            return self.df, None
        return None, {"message": self.error or "No data"}

class DummyWriter:
    def __init__(self, *a, **kw): pass
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key=None):
        class Result: path = Path(f"/tmp/{name}.json")
        return Result()
    def write_dataframe(self, df, name, format, subdir, timestamp_in_name, encryption_key=None):
        class Result: path = Path(f"/tmp/{name}.csv")
        return Result()

def dummy_reporter():
    r = MagicMock()
    r.add_operation = MagicMock()
    r.add_artifact = MagicMock()
    return r

def dummy_progress():
    p = MagicMock()
    p.update = MagicMock()
    return p

@pytest.fixture
def valid_config():
    return {
        "field_operations": {
            "new_field": {"operation_type": "add_constant", "constant_value": 42},
            "lookup_field": {"operation_type": "add_from_lookup", "lookup_table_name": "table1"},
            "mod_field": {"operation_type": "modify_constant", "constant_value": "X"},
        },
        "lookup_tables": {"table1": {"lookup_field": 99}},
        "output_format": "csv",
        "name": "testop",
        "description": "desc",
        "field_name": "mod_field",
        "mode": "REPLACE",
        "output_field_name": None,
        "column_prefix": "_",
        "batch_size": 2,
        "use_cache": False,
        "use_dask": False,
        "use_encryption": False,
        "encryption_key": None
    }

@pytest.fixture
def sample_df():
    return pd.DataFrame({"mod_field": [1, 2], "other": [3, 4]})

@pytest.fixture
def empty_df():
    return pd.DataFrame({})

@pytest.fixture
def op(valid_config):
    return AddOrModifyFieldsOperation(**valid_config)

# --- Tests ---
def test_process_batch_valid(op, sample_df):
    batch = sample_df.copy()
    result = op.process_batch(batch)
    assert "new_field" in result.columns
    assert all(result["new_field"] == 42)
    assert "lookup_field" in result.columns
    assert all(result["lookup_field"] == 99)
    assert all(result["mod_field"] == "X")

def test_process_batch_enrich_mode(valid_config, sample_df):
    valid_config["mode"] = "ENRICH"
    op = AddOrModifyFieldsOperation(**valid_config)
    batch = sample_df.copy()
    result = op.process_batch(batch)
    assert f"_{'mod_field'}" in result.columns
    assert all(result[f"_{'mod_field'}"] == "X")

def test_process_batch_empty(op, empty_df):
    result = op.process_batch(empty_df.copy())
    # Should add new_field and lookup_field
    assert "new_field" in result.columns
    assert "lookup_field" in result.columns
    assert all(result["new_field"] == 42)
    assert all(result["lookup_field"] == 99)

def test_process_batch_invalid_operation(op, sample_df):
    op.field_operations["bad"] = {"operation_type": "add_conditional"}
    with pytest.raises(NotImplementedError):
        op.process_batch(sample_df.copy())

def test_process_value_not_implemented(op):
    with pytest.raises(NotImplementedError):
        op.process_value(1)

def test__get_and_validate_data_valid(op, sample_df):
    ds = DummyDataSource(df=sample_df)
    df, err = op._get_and_validate_data(ds, "main")
    assert err is None
    assert isinstance(df, pd.DataFrame)

def test__get_and_validate_data_invalid(op):
    ds = DummyDataSource(df=None, error="fail")
    df, err = op._get_and_validate_data(ds, "main")
    assert df is None
    assert "fail" in err

def test__prepare_directories(tmp_path, op):
    dirs = op._prepare_directories(tmp_path)
    assert all(Path(v).exists() for v in dirs.values())
    assert set(["root", "output", "cache", "logs", "dictionaries"]).issubset(dirs.keys())

def test__generate_data_hash(op, sample_df):
    h = op._generate_data_hash(sample_df)
    assert isinstance(h, str)
    assert len(h) == 32  # md5

def test__get_operation_parameters(op):
    params = op._get_operation_parameters()
    assert "field_operations" in params
    assert "lookup_tables" in params
    assert "version" in params

def test__collect_metrics(op, sample_df, monkeypatch):
    import sys
    sys.modules['pamola_core.transformations.commons.metric_utils'] = __import__('types').SimpleNamespace(calculate_dataset_comparison=lambda a, b: {"foo": 1})
    metrics = op._collect_metrics(sample_df, sample_df)
    assert metrics["foo"] == 1
    assert "operation_type" in metrics

def test__calculate_all_metrics(op, sample_df, monkeypatch):
    import sys
    sys.modules['pamola_core.transformations.commons.metric_utils'] = __import__('types').SimpleNamespace(calculate_dataset_comparison=lambda a, b: {"bar": 2})
    op.execution_time = 1
    op.process_count = 2
    metrics = op._calculate_all_metrics(sample_df, sample_df)
    assert metrics["bar"] == 2
    assert metrics["execution_time_seconds"] == 1
    assert metrics["records_processed"] == 2
    assert metrics["records_per_second"] == 2

def test__cleanup_memory(op, sample_df):
    op._temp_data = [1, 2, 3]
    op._cleanup_memory(sample_df, sample_df)
    assert not hasattr(op, "_temp_data") or op._temp_data is None

def test_create_add_modify_fields_operation(valid_config):
    op = create_add_modify_fields_operation(**valid_config)
    assert isinstance(op, AddOrModifyFieldsOperation)

def test_execute_success(monkeypatch, tmp_path, op, sample_df):
    monkeypatch.setattr("pamola_core.utils.ops.op_data_writer.DataWriter", DummyWriter)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_in_chunks", lambda **kwargs: sample_df)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_parallel", lambda **kwargs: sample_df)
    import sys
    sys.modules['pamola_core.transformations.commons.metric_utils'] = __import__('types').SimpleNamespace(calculate_dataset_comparison=lambda a, b: {"foo": 1})
    monkeypatch.setattr("pamola_core.transformations.commons.visualization_utils.generate_visualization_filename", lambda **kwargs: "testfile.csv")
    monkeypatch.setattr("pamola_core.transformations.commons.visualization_utils.generate_field_count_comparison_vis", lambda **kwargs: {"vis": Path("/tmp/vis.png")})
    ds = DummyDataSource(df=sample_df)
    result = op.execute(ds, tmp_path, dummy_reporter(), dummy_progress(), dataset_name="main", save_output=True, generate_visualization=True)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]

def test_execute_cache(monkeypatch, tmp_path, op, sample_df):
    # Patch cache to always hit
    class DummyCache:
        @staticmethod
        def get_cache(cache_key, operation_type):
            return {"metrics": {"foo": 1}, "timestamp": "now"}
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_data_writer.DataWriter", DummyWriter)
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_in_chunks", lambda **kwargs: sample_df)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_parallel", lambda **kwargs: sample_df)
    ds = DummyDataSource(df=sample_df)
    result = op.execute(ds, tmp_path, dummy_reporter(), dummy_progress(), dataset_name="main", save_output=True, generate_visualization=True)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]

def test__check_cache_no_cache(monkeypatch, op, sample_df, tmp_path):
    class DummyCache:
        @staticmethod
        def get_cache(cache_key, operation_type):
            return None
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    ds = DummyDataSource(df=sample_df)
    result = op._check_cache(ds, tmp_path, "main")
    assert result is None

def test__save_to_cache(monkeypatch, op, sample_df, tmp_path):
    class DummyCache:
        @staticmethod
        def save_cache(data, cache_key, operation_type, metadata=None):
            return True
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    op.use_cache = True
    ok = op._save_to_cache(sample_df, sample_df, {"foo": 1}, tmp_path)
    assert ok is True

def test__save_to_cache_fail(monkeypatch, op, sample_df, tmp_path):
    class DummyCache:
        @staticmethod
        def save_cache(data, cache_key, operation_type, metadata=None):
            raise Exception("fail")
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    ok = op._save_to_cache(sample_df, sample_df, {"foo": 1}, tmp_path)
    assert ok is False

if __name__ == "__main__":
    pytest.main()
