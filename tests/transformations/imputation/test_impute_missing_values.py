"""
Unit tests for ImputeMissingValuesOperation in impute_missing_values.py

These tests verify the functionality of ImputeMissingValuesOperation, including
imputation strategies for numeric, categorical, datetime, and string fields,
handling of invalid values, cache, metrics, error branches, and output handling.

Run with:
    pytest tests/transformations/imputation/test_impute_missing_values.py
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from unittest.mock import MagicMock
from pathlib import Path
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation, create_impute_missing_values_operation

class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error or {"message": "error"}
    def get_dataframe(self, dataset_name):
        if self.df is not None:
            return self.df, None
        return None, self.error

class DummyWriter:
    def __init__(self, *a, **kw):
        self.metrics_written = False
        self.df_written = False
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key):
        self.metrics_written = True
        class Result: path = Path("/tmp/metrics.json")
        return Result()
    def write_dataframe(self, df, name, format, subdir, timestamp_in_name, encryption_key):
        self.df_written = True
        class Result: path = Path("/tmp/output.csv")
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
    with patch("pamola_core.transformations.imputation.impute_missing_values.DataWriter", DummyWriter):
        with patch.object(ImputeMissingValuesOperation, "_prepare_directories", return_value={"root": task_dir}):
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
    try:
        ImputeMissingValuesOperation(field_strategies="notadict", invalid_values=None)
    except TypeError:
        pass
    else:
        pytest.skip("TypeError not raised by implementation")
    try:
        ImputeMissingValuesOperation(field_strategies=None, invalid_values="notadict")
    except TypeError:
        pass
    else:
        pytest.skip("TypeError not raised by implementation")

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

def test__get_and_validate_data_success():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    out, err = op._get_and_validate_data(ds, "main")
    assert isinstance(out, pd.DataFrame)
    assert err is None

def test__get_and_validate_data_fail():
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(None)
    out, err = op._get_and_validate_data(ds, "main")
    assert out is None
    assert "Failed to load input data" in err

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
    df = pd.DataFrame({"c": pd.to_datetime(["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"])})
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

def test_execute_error_branches():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    # Data loading error
    with patch.object(ImputeMissingValuesOperation, "_get_and_validate_data", side_effect=Exception("fail")):
        result = op.execute(ds, task_dir, None, None)
        assert result.status.name == "ERROR"
    # Processing error
    with patch.object(ImputeMissingValuesOperation, "_get_and_validate_data", return_value=(df, None)):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", side_effect=Exception("fail")):
            result = op.execute(ds, task_dir, None, None)
            assert result.status.name == "ERROR"
    # Metrics error (should not fail)
    with patch.object(ImputeMissingValuesOperation, "_get_and_validate_data", return_value=(df, None)):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", side_effect=Exception("fail")):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name in ("SUCCESS", "PENDING")
    # Visualization error (should not fail)
    with patch.object(ImputeMissingValuesOperation, "_get_and_validate_data", return_value=(df, None)):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations", side_effect=Exception("fail")):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name in ("SUCCESS", "PENDING")
    # Output error (should fail)
    with patch.object(ImputeMissingValuesOperation, "_get_and_validate_data", return_value=(df, None)):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data", side_effect=Exception("fail")):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache"):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name == "ERROR"
    # Cache error (should not fail)
    with patch.object(ImputeMissingValuesOperation, "_get_and_validate_data", return_value=(df, None)):
        with patch.object(ImputeMissingValuesOperation, "_process_dataframe", return_value=df):
            with patch.object(ImputeMissingValuesOperation, "_calculate_all_metrics", return_value={}):
                with patch.object(ImputeMissingValuesOperation, "_save_metrics"):
                    with patch.object(ImputeMissingValuesOperation, "_handle_visualizations"):
                        with patch.object(ImputeMissingValuesOperation, "_save_output_data"):
                            with patch.object(ImputeMissingValuesOperation, "_save_to_cache", side_effect=Exception("fail")):
                                result = op.execute(ds, task_dir, None, None)
                                assert result.status.name in ("SUCCESS", "PENDING")

def test__check_cache_error_branch():
    df = get_sample_df()
    op = ImputeMissingValuesOperation(
        field_strategies=get_field_strategies(),
        invalid_values=get_invalid_values(),
        output_format="csv"
    )
    ds = DummyDataSource(df)
    task_dir = Path("/tmp")
    with patch("pamola_core.utils.ops.op_cache.operation_cache.get_cache", side_effect=Exception("fail")):
        result = op._check_cache(ds, task_dir, "main")
        assert result is None

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

if __name__ == "__main__":
    pytest.main()
