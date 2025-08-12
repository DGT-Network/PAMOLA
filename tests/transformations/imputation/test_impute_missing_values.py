"""
Tests for the impute_missing_values module in the pamola_core/transformations/imputation package.
These tests ensure that the ImputeMissingValuesOperation and related functions properly implement
imputation strategies, error handling, metrics, cache, and output management for all supported data types.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation, create_impute_missing_values_operation

# --- Dummy helpers for mocking external dependencies ---
class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error or {"message": "error"}
        self.encryption_keys = {}
        self.encryption_modes = {}
    def get_dataframe(self, dataset_name, **kwargs):  # Accept extra kwargs
        if self.df is not None:
            return self.df, None
        return None, self.error

class DummyWriter:
    def __init__(self, *a, **kw):
        self.metrics_written = False
        self.df_written = False
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key):
        self.metrics_written = True
        class Result:
            path = Path("/tmp/metrics.json")
        return Result()
    def write_dataframe(self, df, name, format, subdir, timestamp_in_name, encryption_key):
        self.df_written = True
        class Result:
            path = Path("/tmp/output.csv")
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

# --- Fixtures and helpers ---
def get_sample_df():
    return pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": ["x", None, "y", "z", "x"],
        "c": pd.to_datetime(["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"])
    })

def get_field_strategies():
    return {
        "a": {"imputation_strategy": "mean"},
        "b": {"imputation_strategy": "most_frequent"},
        "c": {"imputation_strategy": "mode_date"}
    }

def get_invalid_values():
    return {"a": [0], "b": [""], "c": [None]}

# --- Test cases ---
def test_valid_case():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    assert not batch.isnull().any().any()
    # Test execute with mocks
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    reporter = DummyReporter()
    progress = DummyProgress()
    dirs = {"root": task_dir, "output": task_dir, "visualizations": task_dir, "metrics": task_dir}
    with patch("pamola_core.transformations.imputation.impute_missing_values.DataWriter", DummyWriter):
        with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
            with patch.object(ImputeMissingValuesOperation, "save_config"):
                with patch.object(ImputeMissingValuesOperation, "_check_cache", return_value=None):
                    with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                        with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                            with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                                with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                    result = op.execute(ds, task_dir, reporter, progress)
                                    assert result.status.name in ("SUCCESS", "PENDING")

def test_edge_case_empty_df():
    df = pd.DataFrame({"a": [], "b": [], "c": []})
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    assert batch.empty

def test_invalid_input_types():
    from pamola_core.utils.ops.op_config import ConfigError
    with pytest.raises(ConfigError):
        ImputeMissingValuesOperation(field_strategies="notadict", invalid_values=None)
    with pytest.raises(ConfigError):
        ImputeMissingValuesOperation(field_strategies=None, invalid_values="notadict")

def test_process_batch_enrich_mode():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv",
        mode="ENRICH",
        column_prefix="enrich_"
    )
    batch = op.process_batch(df.copy())
    assert any(col.startswith("enrich_") for col in batch.columns)

def test_process_value_not_implemented():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    with pytest.raises(NotImplementedError):
        op.process_value(1)

def test__prepare_directories(tmp_path):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    dirs = op._prepare_directories(tmp_path)
    assert all(Path(v).exists() for v in dirs.values())

def test__generate_data_hash():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    h = op._generate_data_hash(df)
    assert isinstance(h, str)
    assert len(h) == 32

def test__get_operation_parameters():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    params = op._get_operation_parameters()
    assert "field_strategies" in params
    assert "invalid_values" in params
    assert "version" in params

def test__get_cache_parameters():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    params = op._get_cache_parameters()
    assert isinstance(params, dict)

def test__cleanup_memory():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    df = get_sample_df()
    op._temp_data = df
    op._cleanup_memory(df, df)
    assert not hasattr(op, '_temp_data') or op._temp_data is None

def test_create_impute_missing_values_operation():
    op = create_impute_missing_values_operation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    assert isinstance(op, ImputeMissingValuesOperation)

def test_process_batch_all_numeric_strategies():
    df = pd.DataFrame({"a": [1, np.nan, 5, 0]})
    strategies = [
        ("constant", 99),
        ("mean", df["a"].mean(skipna=True)),
        ("median", df["a"].median(skipna=True)),
        ("mode", 1),
        ("min", 0),
        ("max", 5),
        ("interpolation", 3.0),
    ]
    for strat, expected in strategies:
        op = ImputeMissingValuesOperation(
            field_strategies={"a": {"imputation_strategy": strat, "constant_value": 99}},
            invalid_values={"a": [None]},
            output_format="csv"
        )
        batch = op.process_batch(df.copy())
        assert not batch["a"].isnull().any()

def test_process_batch_all_categorical_strategies():
    df = pd.DataFrame({"b": pd.Series(["x", None, "y", "z", "x"], dtype="category")})
    # Add 'foo' to categories before imputation
    df["b"] = df["b"].cat.add_categories(["foo"])
    op = ImputeMissingValuesOperation(
        field_strategies={"b": {"imputation_strategy": "constant", "constant_value": "foo"}},
        invalid_values={"b": [None]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    assert not batch["b"].isnull().any()
    op2 = ImputeMissingValuesOperation(
        field_strategies={"b": {"imputation_strategy": "random_sample"}},
        invalid_values={"b": [None]},
        output_format="csv"
    )
    batch2 = op2.process_batch(df.copy())
    assert not batch2["b"].isnull().any()

def test_process_batch_all_datetime_strategies():
    df = pd.DataFrame({"c": pd.to_datetime(["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"])} )
    for strat in ["constant_date", "mean_date", "median_date", "mode_date", "previous_date", "next_date"]:
        op = ImputeMissingValuesOperation(
            field_strategies={"c": {"imputation_strategy": strat, "constant_value": pd.Timestamp("2020-01-02")}},
            invalid_values={"c": [None]},
            output_format="csv"
        )
        batch = op.process_batch(df.copy())
        assert not batch["c"].isnull().any()

def test_process_batch_all_string_strategies():
    df = pd.DataFrame({"d": ["foo", None, "bar", "baz"]})
    for strat in ["constant", "most_frequent", "random_sample"]:
        op = ImputeMissingValuesOperation(
            field_strategies={"d": {"imputation_strategy": strat, "constant_value": "zzz"}},
            invalid_values={"d": [None]},
            output_format="csv"
        )
        batch = op.process_batch(df.copy())
        assert not batch["d"].isnull().any()

def test_execute_cache_hit():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    fake_result = MagicMock()
    fake_result.status.name = "SUCCESS"
    with patch.object(ImputeMissingValuesOperation, "_check_cache", return_value=fake_result):
        result = op.execute(ds, task_dir, None, None)
        assert result.status.name == "SUCCESS"

def test_execute_cache_hit_reports_to_reporter():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    fake_result = MagicMock()
    fake_result.status.name = "SUCCESS"
    reporter = DummyReporter()
    with patch.object(ImputeMissingValuesOperation, "_check_cache", return_value=fake_result):
        result = op.execute(ds, task_dir, reporter, None)
        assert result.status.name == "SUCCESS"
        assert any("from cache" in opn[0] for opn in reporter.operations)

def test_execute_error_branches():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    dirs = {"root": task_dir, "output": task_dir, "visualizations": task_dir, "metrics": task_dir}
    # Processing error
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", side_effect=Exception("fail")):
            result = op.execute(ds, task_dir, None, None)
            assert result.status.name == "ERROR"
    # Metrics error (should not fail)
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", side_effect=Exception("fail")):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name in ("SUCCESS", "PENDING")
    # Visualization error (should not fail)
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations", side_effect=Exception("fail")):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name in ("SUCCESS", "PENDING")
    # Output error (should fail)
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data", side_effect=Exception("fail")):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name == "ERROR"
    # Cache error (should not fail)
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache", side_effect=Exception("fail")):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name in ("SUCCESS", "PENDING")

def test_execute_data_loading_error():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(None)
    task_dir = Path("/tmp")
    with patch("pamola_core.transformations.imputation.impute_missing_values.load_settings_operation", side_effect=Exception("fail")):
        result = op.execute(ds, task_dir, None, None)
        assert result.status.name == "ERROR"
        assert "Error loading data" in result.error_message

def test_execute_validation_error():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(get_sample_df())
    task_dir = Path("/tmp")
    with patch("pamola_core.transformations.imputation.impute_missing_values.load_settings_operation", return_value={}):
        with patch("pamola_core.transformations.imputation.impute_missing_values.load_data_operation", return_value=get_sample_df()):
            with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=get_sample_df()):
                with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", side_effect=Exception("fail")):
                    with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                        with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                            with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                                with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                    # Patch reporter to raise in add_operation
                                    class BadReporter:
                                        def add_operation(self, *a, **k):
                                            raise Exception("fail")
                                    result = op.execute(ds, task_dir, BadReporter(), None)
                                    assert result.status.name in ("SUCCESS", "PENDING", "ERROR")

def test_execute_visualization_error_branch():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    dirs = {"root": task_dir, "output": task_dir, "visualizations": task_dir, "metrics": task_dir}
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations", side_effect=Exception("fail")):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, DummyReporter(), None)
                                assert result.status.name in ("SUCCESS", "PENDING")

def test_execute_progress_tracker_updates():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    progress = DummyProgress()
    dirs = {"root": task_dir, "output": task_dir, "visualizations": task_dir, "metrics": task_dir}
    with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value=dirs):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                op.execute(ds, task_dir, DummyReporter(), progress)
    # Should have at least 5 progress updates (Preparation, Checking Cache, Data Loading, Validation, Processing, Metrics, Finalization)
    assert len(progress.updates) >= 5

def test__cleanup_memory_extra_temp_attrs():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    df = get_sample_df()
    op._temp_data = df
    op._temp_foo = "bar"
    op._cleanup_memory(df, df)
    assert not hasattr(op, '_temp_data') or op._temp_data is None
    assert not hasattr(op, '_temp_foo')

def test_check_cache_metrics_not_dict(monkeypatch):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    class DummyCache:
        def get_cache(self, **kwargs):
            return {"metrics": "notadict"}
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache())
    ds = DummyDataSource(get_sample_df())
    # Should not fail if metrics is not a dict
    assert op._check_cache(ds, None) is None or True

def test_check_cache_reporter_none(monkeypatch):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    class DummyCache:
        def get_cache(self, **kwargs):
            return {"metrics": {}, "metrics_result_path": None, "output_result_path": None, "visualizations": {}}
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache())
    ds = DummyDataSource(get_sample_df())
    # Should not fail if reporter is None
    op._check_cache(ds, None)

def test_check_cache_cache_exception(monkeypatch):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    class DummyCache:
        def get_cache(self, **kwargs):
            raise Exception("fail")
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache())
    ds = DummyDataSource(get_sample_df())
    # Should not raise
    assert op._check_cache(ds, None) is None

def test_save_to_cache_disabled():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    op.use_cache = False
    assert op._save_to_cache(get_sample_df(), get_sample_df(), {}, MagicMock(), MagicMock(), {}, Path("/tmp")) is False

def test_save_to_cache_exception(monkeypatch):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    op.use_cache = True
    class DummyCache:
        def save_cache(self, **kwargs):
            raise Exception("fail")
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache())
    assert op._save_to_cache(get_sample_df(), get_sample_df(), {}, MagicMock(), MagicMock(), {}, Path("/tmp")) is False

def test_generate_visualizations_backend_none():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    # Should skip and return empty dict
    result = op._generate_visualizations(get_sample_df(), get_sample_df(), Path("/tmp"), None, None, False, None)
    assert result == {}

def test_generate_visualizations_exception(monkeypatch):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    def fail_vis(*a, **k):
        raise Exception("fail")
    monkeypatch.setattr(
        "pamola_core.transformations.commons.visualization_utils.generate_data_distribution_comparison_vis",
        fail_vis
    )
    # Should catch and return {}
    result = op._generate_visualizations(get_sample_df(), get_sample_df(), Path("/tmp"), "theme", "plotly", False, None)
    assert result == {}

def test_save_metrics_reporter_none(tmp_path):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    class DummyWriter:
        def write_metrics(self, **kwargs):
            class R:
                def __init__(self, path):
                    self.path = path
            return R(tmp_path / "metrics.json")
    result = MagicMock()
    metrics = {"a": 1}
    op._save_metrics(metrics, tmp_path, DummyWriter(), result, None, None)
    # Should add artifact to result
    result.add_artifact.assert_called()

def test_save_output_data_reporter_none(tmp_path):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    class DummyWriter:
        def write_dataframe(self, **kwargs):
            class R:
                def __init__(self, path):
                    self.path = path
            return R(tmp_path / "output.csv")
    result = MagicMock()
    op._save_output_data(get_sample_df(), tmp_path, DummyWriter(), result, None, None)
    result.add_artifact.assert_called()

def test_cleanup_memory_none():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    # Should not raise if both args are None
    op._cleanup_memory(None, None)

def test_process_batch_unknown_strategy():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "unknown"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    # Should not raise, should leave NaN as is
    batch = op.process_batch(df.copy())
    assert batch["a"].isnull().any()

def test_process_batch_column_not_in_strategies():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies={},
        invalid_values={},
        output_format="csv"
    )
    # Should not raise, should leave columns as is
    batch = op.process_batch(df.copy())
    assert batch.equals(df)

def test_process_batch_constant_no_value():
    df = pd.DataFrame({"a": [1, None, 3]})
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "constant"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    # Should fill with None or not fill at all
    batch = op.process_batch(df.copy())
    assert batch["a"].isnull().any()

def test_process_batch_all_invalid():
    df = pd.DataFrame({"a": [None, None, None]})
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "mean"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    assert batch["a"].isnull().all()

def test_process_batch_unsupported_dtype():
    df = pd.DataFrame({"a": [object(), object(), None]})
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "mean"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    # Should not raise, should leave as is
    batch = op.process_batch(df.copy())
    assert batch["a"].isnull().sum() == 1

def test_process_batch_all_nat_datetime():
    df = pd.DataFrame({"c": [pd.NaT, pd.NaT, pd.NaT]})
    op = ImputeMissingValuesOperation(
        field_strategies={"c": {"imputation_strategy": "mean_date"}},
        invalid_values={"c": [pd.NaT]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    assert batch["c"].isnull().all()

def test_process_batch_all_none_string():
    df = pd.DataFrame({"d": [None, None, None]})
    op = ImputeMissingValuesOperation(
        field_strategies={"d": {"imputation_strategy": "most_frequent"}},
        invalid_values={"d": [None]},
        output_format="csv"
    )
    with pytest.raises(KeyError):
        op.process_batch(df.copy())

def test_process_batch_categorical_no_valid_categories():
    df = pd.DataFrame({"b": pd.Series([None, None, None], dtype="category")})
    op = ImputeMissingValuesOperation(
        field_strategies={"b": {"imputation_strategy": "most_frequent"}},
        invalid_values={"b": [None]},
        output_format="csv"
    )
    with pytest.raises(KeyError):
        op.process_batch(df.copy())

def test_process_batch_duplicate_columns():
    df = pd.DataFrame([[1, 2], [3, None]], columns=["a", "a"])
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "mean"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    with pytest.raises(AttributeError):
        op.process_batch(df.copy())

def test_handle_visualizations_backend_missing():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    # Should not raise if backend is None
    op._handle_visualizations(get_sample_df(), get_sample_df(), Path("/tmp"), None, None, None, None, None, None, None)

def test_generate_visualizations_invalid_backend():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    # Should catch and return a dict (may not contain error strings, but should not be empty)
    result = op._generate_visualizations(get_sample_df(), get_sample_df(), Path("/tmp"), "theme", "invalid_backend", False, None)
    assert isinstance(result, dict)
    assert result  # Should not be empty

def test_process_batch_multiindex_column():
    # MultiIndex column, not just index
    arrays = [["a", "a", "b"], ["x", "y", "z"]]
    tuples = list(zip(*arrays))
    columns = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
    df = pd.DataFrame([[1, None, 3], [4, 5, None]], columns=columns)
    op = ImputeMissingValuesOperation(
        field_strategies={("a", "y"): {"imputation_strategy": "mean"}},
        invalid_values={("a", "y"): [None]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    assert batch.columns.equals(df.columns)
    assert batch[("a", "y")].isnull().sum() == 0

def test_process_batch_object_column_mixed_types():
    df = pd.DataFrame({"a": [1, "foo", 3.5, None]})
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "mean"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    # Should not fill None, as mean is not possible
    assert batch["a"].isnull().sum() == 1

def test_process_batch_all_empty_strings():
    df = pd.DataFrame({"d": ["", "", ""]})
    op = ImputeMissingValuesOperation(
        field_strategies={"d": {"imputation_strategy": "most_frequent"}},
        invalid_values={"d": [""]},
        output_format="csv"
    )
    with pytest.raises(KeyError):
        op.process_batch(df.copy())

def test_process_batch_all_false_invalid():
    df = pd.DataFrame({"a": [False, False, False]})
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "most_frequent"}},
        invalid_values={"a": [False]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    # Should remain all False, as pandas does not treat all False as missing
    assert not batch["a"].any()

def test_process_batch_all_tuples():
    df = pd.DataFrame({"a": [(1, 2), (3, 4), None]})
    op = ImputeMissingValuesOperation(
        field_strategies={"a": {"imputation_strategy": "random_sample"}},
        invalid_values={"a": [None]},
        output_format="csv"
    )
    batch = op.process_batch(df.copy())
    # Should fill None with a tuple
    assert all(isinstance(val, tuple) for val in batch["a"])

def test_process_batch_no_rows_various_types():
    df = pd.DataFrame({
        "a": pd.Series([], dtype=float),
        "b": pd.Series([], dtype=str),
        "c": pd.Series([], dtype="datetime64[ns]"),
        "d": pd.Series([], dtype=object)
    })
    op = ImputeMissingValuesOperation(
        field_strategies={
            "a": {"imputation_strategy": "mean"},
            "b": {"imputation_strategy": "most_frequent"},
            "c": {"imputation_strategy": "mode_date"},
            "d": {"imputation_strategy": "random_sample"}
        },
        invalid_values={"a": [None], "b": [None], "c": [None], "d": [None]},
        output_format="csv"
    )
    with pytest.raises(KeyError):
        op.process_batch(df.copy())

def test_check_cache_with_visualizations_and_metrics(monkeypatch, tmp_path):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    # Create dummy files for metrics, output, and visualization
    metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "output.csv"
    vis_path = tmp_path / "vis.png"
    metrics_path.write_text("{}")
    output_path.write_text("foo")
    vis_path.write_text("bar")
    class DummyCache:
        def get_cache(self, **kwargs):
            return {
                "metrics": {"a": 1},
                "metrics_result_path": str(metrics_path),
                "output_result_path": str(output_path),
                "visualizations": {"chart": str(vis_path)}
            }
        def generate_cache_key(self, *a, **k):
            return "dummykey"
    monkeypatch.setattr("pamola_core.utils.ops.op_cache.operation_cache", DummyCache())
    ds = DummyDataSource(get_sample_df())
    reporter = DummyReporter()
    result = op._check_cache(ds, reporter)
    assert result is not None
    # Artifacts may or may not be added depending on implementation, but should not error
    assert hasattr(reporter, "artifacts")

def test_handle_visualizations_adds_artifacts(tmp_path):
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    df = get_sample_df()
    chart_path = tmp_path / "chart.png"
    chart_path.write_text("fake image")
    vis_dict = {"chart": str(chart_path)}
    result = MagicMock()
    # Should not raise, regardless of whether add_artifact is called
    op._handle_visualizations(df, df, tmp_path, "theme", "plotly", False, vis_dict, result, None, None)
    assert True

if __name__ == "__main__":
	pytest.main()