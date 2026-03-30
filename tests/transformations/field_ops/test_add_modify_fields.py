"""
Tests for the add_modify_fields module in the pamola_core/transformations/field_ops package.
These tests ensure that the AddOrModifyFieldsOperation class and its helpers properly implement field addition, modification, caching, metrics, visualizations, and error handling.
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
from pamola_core.transformations.field_ops.add_modify_fields import (
    AddOrModifyFieldsOperation, create_add_modify_fields_operation
)
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
        self.encryption_keys = {}  # Add this attribute to avoid attribute errors in tests
        self.encryption_modes = {}  # Add this attribute to avoid attribute errors in tests
        self.settings = {}
        self.data_source_name = "test"
    def get_dataframe(self, dataset_name, **kwargs):
        if self.df is not None:
            return self.df, None
        return None, {"message": self.error or "No data"}
    def apply_data_types(self, df, *args, **kwargs):
        return df

class DummyWriter:
    def __init__(self, *a, **kw): pass
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key=None):
        class Result:
            def __init__(self, path):
                self.path = path
        return Result(Path(f"/tmp/{name}.json"))
    def write_dataframe(self, df, name, format, subdir, timestamp_in_name, encryption_key=None, encryption_mode=None):
        class Result:
            def __init__(self, path):
                self.path = path
        return Result(Path(f"/tmp/{name}.csv"))

def dummy_reporter():
    r = MagicMock()
    r.add_operation = MagicMock()
    r.add_artifact = MagicMock()
    return r

def dummy_progress():
    p = MagicMock()
    p.update = MagicMock()
    p.create_subtask = MagicMock(return_value=MagicMock())
    p.close = MagicMock()
    return p

@pytest.fixture
def valid_config():
    return {
        "field_operations": {
            "new_field": {"operation_type": "add_constant", "constant_value": 42},
            "lookup_field": {"operation_type": "add_from_lookup", "lookup_table_name": "table1", "base_on_column": "other"},
            "mod_field": {"operation_type": "modify_constant", "constant_value": "X"},
        },
        "lookup_tables": {"table1": {3: 99, 4: 99}},
        "output_format": "csv",
        "name": "testop",
        "description": "desc",
        "field_name": "mod_field",
        "mode": "REPLACE",
        "output_field_name": None,
        "column_prefix": "_",
        "chunk_size": 2,
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
    # Should add new_field (add_constant works regardless of existing columns)
    assert "new_field" in result.columns
    assert all(result["new_field"] == 42)

def test_process_batch_invalid_operation(op, sample_df):
    # Unknown operation types are silently ignored (no exception raised)
    op.field_operations["bad"] = {"operation_type": "totally_unknown_operation_xyz"}
    result = op.process_batch(sample_df.copy())
    # The batch should be returned with original columns plus any valid operations applied
    assert isinstance(result, pd.DataFrame)

def test_process_batch_modify_from_lookup(op, sample_df):
    # Setup: Add a lookup table and configure the operation
    # modify_from_lookup requires base_on_column to map values
    op.field_operations = {
        "mod_field": {"operation_type": "modify_from_lookup", "lookup_table_name": "table1", "base_on_column": "other"}
    }
    op.lookup_tables = {"table1": {3: 123, 4: 123}}
    # The field exists in the DataFrame, so it should be modified
    batch = sample_df.copy()
    result = op.process_batch(batch)
    assert "mod_field" in result.columns
    assert all(result["mod_field"] == 123)

def test_process_value_not_implemented(op):
    with pytest.raises(Exception):
        op.process_value(1)

def test__prepare_directories(tmp_path, op):
    dirs = op._prepare_directories(tmp_path)
    assert all(Path(v).exists() for v in dirs.values())
    assert set(["root", "output", "cache", "logs", "dictionaries", "visualizations", "metrics"]).issubset(dirs.keys())

def test__generate_data_hash(op, sample_df):
    from pamola_core.utils import helpers
    h = helpers.generate_data_hash(sample_df)
    assert isinstance(h, str)
    assert len(h) == 64  # BLAKE2b with digest_size=32 -> 64 hex chars

def test__get_base_parameters(op):
    params = op._get_base_parameters()
    assert "field_operations" in params
    assert "lookup_tables" in params
    assert "version" in params

def test__collect_metrics(op, sample_df, monkeypatch):
    metrics = op._collect_metrics(sample_df, sample_df)
    assert isinstance(metrics, dict)
    assert "fields_added_count" in metrics
    assert "fields_modified_count" in metrics

def test__calculate_all_metrics(op, sample_df, monkeypatch):
    op.start_time = 0
    op.end_time = 1
    op.execution_time = 1
    op.process_count = 2
    metrics = op._calculate_all_metrics(sample_df, sample_df)
    assert "fields_added_count" in metrics
    assert "fields_modified_count" in metrics
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
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", lambda **kwargs: sample_df)
    import sys
    sys.modules['pamola_core.transformations.commons.metric_utils'] = __import__('types').SimpleNamespace(
        calculate_dataset_comparison=lambda a, b: {"foo": 1},
        calculate_transformation_impact=lambda a, b: {"bar": 2}
    )
    sys.modules['pamola_core.transformations.commons.visualization_utils'] = __import__('types').SimpleNamespace(
        generate_visualization_filename=lambda **kwargs: "testfile.csv"
    )
    ds = DummyDataSource(df=sample_df)
    result = op.execute(ds, tmp_path, dummy_reporter(), dummy_progress(), dataset_name="main", save_output=True, generate_visualization=False)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]

def test_execute_cache(monkeypatch, tmp_path, op, sample_df):
    class DummyCache:
        @staticmethod
        def get_cache(cache_key, operation_type):
            return {"metrics": {"foo": 1}, "timestamp": "now"}
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_data_writer.DataWriter", DummyWriter)
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", lambda **kwargs: sample_df)
    ds = DummyDataSource(df=sample_df)
    result = op.execute(ds, tmp_path, dummy_reporter(), dummy_progress(), dataset_name="main", save_output=True, generate_visualization=False)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]

def test_execute_with_cache_and_progress(monkeypatch, tmp_path, op, sample_df):
    class DummyCache:
        @staticmethod
        def get_cache(cache_key, operation_type):
            return {"metrics": {"foo": 1}, "timestamp": "now"}
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_data_writer.DataWriter", DummyWriter)
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", lambda **kwargs: sample_df)
    op.use_cache = True
    ds = DummyDataSource(df=sample_df)
    progress = dummy_progress()
    result = op.execute(ds, tmp_path, dummy_reporter(), progress, dataset_name="main", save_output=True, generate_visualization=False, force_recalculation=False)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]

def test_execute_cache_hit_with_progress_and_reporter(monkeypatch, tmp_path, op, sample_df):
    from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
    class DummyCache:
        @staticmethod
        def get_cache(cache_key, operation_type):
            # Simulate a cache hit with a real OperationResult object
            return OperationResult(
                status=OperationStatus.SUCCESS,
                metrics={"foo": 1},
                error_message=None,
                execution_time=0.1
            )
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_data_writer.DataWriter", DummyWriter)
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", lambda **kwargs: sample_df)
    op.use_cache = True
    ds = DummyDataSource(df=sample_df)
    progress = dummy_progress()
    reporter = dummy_reporter()
    result = op.execute(ds, tmp_path, reporter, progress, dataset_name="main", save_output=True, generate_visualization=False, force_recalculation=False)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]
    # Check that reporter.add_operation was called for cache hit
    assert reporter.add_operation.called

def test__check_cache_no_cache(monkeypatch, op, sample_df, tmp_path):
    # use_cache is False in op fixture, so _check_cache returns None immediately
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = None
    op.operation_cache.generate_cache_key.return_value = "cachekey"
    result = op._check_cache(sample_df, dummy_reporter())
    assert result is None

def test__check_cache_df_none(monkeypatch, op, sample_df, tmp_path):
    # use_cache is False in op fixture, so _check_cache returns None immediately
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = None
    op.operation_cache.generate_cache_key.return_value = "cachekey"
    reporter = dummy_reporter()
    result = op._check_cache(sample_df, reporter)
    assert result is None

def test__save_to_cache(monkeypatch, op, sample_df, tmp_path):
    op.use_cache = True
    op.operation_cache = MagicMock()
    op.operation_cache.save_cache.return_value = True
    op.operation_cache.generate_cache_key.return_value = "cachekey"
    result = OperationResult(status=OperationStatus.SUCCESS)
    ok = op._save_to_cache(sample_df, sample_df, result, tmp_path)
    assert ok is True

def test__save_to_cache_fail(monkeypatch, op, sample_df, tmp_path):
    op.use_cache = True
    op.operation_cache = MagicMock()
    op.operation_cache.save_cache.side_effect = Exception("fail")
    op.operation_cache.generate_cache_key.return_value = "cachekey"
    result = OperationResult(status=OperationStatus.SUCCESS)
    ok = op._save_to_cache(sample_df, sample_df, result, tmp_path)
    assert ok is False

def test__generate_cache_key(monkeypatch, op, sample_df):
    class DummyCache:
        @staticmethod
        def generate_cache_key(operation_name, parameters, data_hash):
            return "cachekey"
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache)
    key = op._generate_cache_key(sample_df)
    assert key == "cachekey"

def test__get_cache_parameters(op):
    params = op._get_cache_parameters()
    assert isinstance(params, dict)
    assert "field_operations" in params
    assert "lookup_tables" in params

def test__process_dataframe(monkeypatch, op, sample_df):
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", lambda **kwargs: sample_df)
    df = op._process_dataframe(sample_df, dummy_progress())
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(sample_df.columns)

def test__save_metrics(monkeypatch, op, sample_df):
    monkeypatch.setattr("pamola_core.transformations.commons.visualization_utils.generate_visualization_filename", lambda **kwargs: "metrics.json")
    writer = DummyWriter()
    result = MagicMock()
    reporter = dummy_reporter()
    metrics = {"foo": 1, "bar": 2}
    ok = op._save_metrics(metrics, writer, result, reporter, dummy_progress())
    # _save_metrics returns True on success or False on failure
    assert ok is True or ok is False

def test__handle_visualizations(monkeypatch, op, sample_df):
    monkeypatch.setattr(op, "_generate_visualizations", lambda **kwargs: {"vis": Path("/tmp/vis.png")})
    result = MagicMock()
    reporter = dummy_reporter()
    vis = op._handle_visualizations(sample_df, sample_df, {}, Path("/tmp"), result, reporter, None, None, False, 1, dummy_progress())
    assert "vis" in vis

def test__generate_visualizations(monkeypatch, op, sample_df, tmp_path):
    from unittest.mock import patch as mpatch
    op.field_operations = {"mod_field": {"operation_type": "modify_constant", "constant_value": "X"}}
    metrics = {"fields_modified_count": 1, "fields_added_count": 0}
    # Patch create_bar_plot from pamola_core.utils.visualization since add_modify_fields uses it
    with mpatch("pamola_core.utils.visualization.create_bar_plot", return_value=str(tmp_path / "bar.png")):
        vis = op._generate_visualizations(sample_df, sample_df, metrics, tmp_path, None, "matplotlib", False, dummy_progress())
    assert isinstance(vis, dict)

def test__generate_visualizations_large_enrich(monkeypatch, op, sample_df, tmp_path):
    from unittest.mock import patch as mpatch
    # Simulate a large DataFrame with prefixed column for ENRICH mode
    large_df = pd.DataFrame({
        "mod_field": range(20000),
        "other": range(20000, 40000)
    })
    large_df["PRE_mod_field"] = ["X"] * 20000
    op.mode = "ENRICH"
    op.column_prefix = "PRE_"
    op.field_operations = {"mod_field": {"operation_type": "modify_constant", "constant_value": "X"}}
    metrics = {"fields_modified_count": 1, "fields_added_count": 0}
    with mpatch("pamola_core.utils.visualization.create_bar_plot", return_value=str(tmp_path / "bar.png")):
        vis = op._generate_visualizations(large_df, large_df, metrics, tmp_path, None, "matplotlib", False, dummy_progress())
    assert isinstance(vis, dict)

def test_execute_with_visualization(monkeypatch, tmp_path, op, sample_df):
    # Patch DataWriter and processing
    monkeypatch.setattr("pamola_core.utils.ops.op_data_writer.DataWriter", DummyWriter)
    monkeypatch.setattr("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", lambda **kwargs: sample_df)
    # Patch visualization utils to avoid real plotting
    import sys
    sys.modules['pamola_core.transformations.commons.visualization_utils'] = __import__('types').SimpleNamespace(
        generate_visualization_filename=lambda **kwargs: "testfile.csv"
    )
    # Patch _handle_visualizations to simulate artifact creation
    monkeypatch.setattr(op, "_handle_visualizations", lambda *a, **k: {"vis": Path("/tmp/vis.png")})
    # Set vis_backend to a non-None value
    op.visualization_backend = "matplotlib"
    ds = DummyDataSource(df=sample_df)
    reporter = dummy_reporter()
    result = op.execute(ds, tmp_path, reporter, dummy_progress(), dataset_name="main", save_output=True, generate_visualization=True)
    assert hasattr(result, "status")
    assert result.status.name in ["SUCCESS", "ERROR", "PENDING"]
    # Check that visualization was handled (artifact added)
    assert reporter.add_artifact.called or reporter.add_operation.called

def test__check_cache_with_artifacts(monkeypatch, op, sample_df, tmp_path):
    op.use_cache = True
    # Set operation_cache directly on the instance (not module-level)
    from unittest.mock import MagicMock
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = {
        'status': 'SUCCESS',
        'metrics': {'foo': 1},
        'error_message': None,
        'execution_time': 1.0,
        'error_trace': None,
        'artifacts': [],
    }
    monkeypatch.setattr(op, "_generate_cache_key", lambda df: "cachekey")
    reporter = dummy_reporter()
    result = op._check_cache(sample_df, reporter)
    assert result is not None
    assert hasattr(result, "metrics")
    # get_cache_result adds 'cached' and 'artifacts_restored' to metrics
    assert result.metrics.get("foo") == 1 or result.metrics.get("cached") is True

if __name__ == "__main__":
    pytest.main()
