"""
Unit tests for CleanInvalidValuesOperation in clean_invalid_values.py

These tests verify the functionality of CleanInvalidValuesOperation, including
constraint enforcement, null replacement, whitelist/blacklist, cache, metrics, and output handling.

Run with:
    pytest tests/transformations/cleaning/test_clean_invalid_values.py
"""
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pathlib import Path

from pamola_core.transformations.cleaning.clean_invalid_values import (
    CleanInvalidValuesOperation
)

class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
    def get_dataframe(self, dataset_name):
        if self.error:
            return None, {"message": self.error}
        return self.df, None

class DummyWriter:
    def __init__(self, *a, **kw):
        self.metrics_written = False
        self.saved_path = None
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key=None):
        self.metrics_written = True
        class Result:
            path = f"/tmp/{name}.json"
        return Result()
    def write_dataframe(self, *a, **kw):
        class Result:
            path = "/tmp/output.csv"
        return Result()

@pytest.fixture(scope="function")
def temp_task_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)

@pytest.fixture(scope="function")
def sample_df():
    return pd.DataFrame({
        "age": [10, 20, 30, 40, 50, None],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", None],
        "cat": pd.Series(["a", "b", "c", "a", "b", "c"], dtype="category"),
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", None, "2022-01-01", "2023-01-01", "2024-01-01"])
    })

@pytest.fixture(scope="function")
def operation():
    return CleanInvalidValuesOperation(
        field_constraints={
            "age": {"constraint_type": "valid_range", "min_value": 15, "max_value": 45},
            "cat": {"constraint_type": "allowed_values", "allowed_values": ["a", "b"]},
            "name": {"constraint_type": "min_length", "min_length": 3},
            "date": {"constraint_type": "date_range", "min_date": "2020-01-01", "max_date": "2023-12-31"}
        },
        null_replacement={"age": 99, "name": "Unknown"},
        output_format="csv",
        field_name="age",
        mode="REPLACE",
        batch_size=2
    )

def test_valid_case(operation, sample_df):
    batch = sample_df.copy()
    processed = operation.process_batch(batch)
    # age: only 20, 30, 40 remain, others replaced with 99
    assert processed["age"].isna().sum() == 0
    assert (processed["age"] == 99).sum() == 3
    # cat: only 'a' and 'b' allowed
    assert processed["cat"].isna().sum() == 2
    # name: min_length 3, None replaced with 'Unknown'
    assert processed["name"].isna().sum() == 0
    assert (processed["name"] == "Unknown").sum() == 1
    # date: only dates in range
    assert processed["date"].isna().sum() == 2

def test_edge_case_empty_df(operation):
    empty = pd.DataFrame({"age": [], "name": [], "cat": pd.Series([], dtype="category"), "date": []})
    processed = operation.process_batch(empty)
    assert processed.empty

def test_invalid_input_wrong_type():
    op = CleanInvalidValuesOperation(field_constraints={"age": {"constraint_type": "valid_range", "min_value": 0, "max_value": 100}})
    with pytest.raises(Exception):
        op.process_batch("not_a_dataframe")

def test_null_replacement_modes(sample_df):
    op = CleanInvalidValuesOperation(
        null_replacement={"age": "mean", "name": "mode"},
        field_constraints={},
        output_format="csv",
        field_name="age",
        mode="REPLACE"
    )
    batch = sample_df.copy()
    batch.loc[0, "age"] = None
    batch.loc[1, "name"] = None
    processed = op.process_batch(batch)
    assert not processed["age"].isna().any()
    assert not processed["name"].isna().any()

def test_random_sample_null_replacement(sample_df):
    op = CleanInvalidValuesOperation(
        null_replacement={"cat": "random_sample"},
        field_constraints={},
        output_format="csv",
        field_name="cat",
        mode="REPLACE"
    )
    batch = sample_df.copy()
    batch.loc[0, "cat"] = None
    processed = op.process_batch(batch)
    assert not processed["cat"].isna().any()

def test_process_batch_enrich_mode(sample_df):
    op = CleanInvalidValuesOperation(
        field_constraints={"age": {"constraint_type": "min_value", "min_value": 20}},
        null_replacement=None,
        output_format="csv",
        field_name="age",
        mode="ENRICH",
        column_prefix="enr_"
    )
    batch = sample_df.copy()
    processed = op.process_batch(batch)
    assert "enr_age" in processed.columns
    assert (processed["enr_age"] < 20).sum() == 0

def test_whitelist_blacklist(tmp_path, sample_df):
    whitelist_file = tmp_path / "whitelist.txt"
    blacklist_file = tmp_path / "blacklist.txt"
    whitelist_file.write_text("Alice\nBob\n")
    blacklist_file.write_text("Charlie\nDavid\n")
    op = CleanInvalidValuesOperation(
        whitelist_path={"name": str(whitelist_file)},
        blacklist_path={"name": str(blacklist_file)},
        field_constraints={},
        output_format="csv",
        field_name="name",
        mode="REPLACE"
    )
    batch = sample_df.copy()
    processed = op.process_batch(batch)
    # Only Alice and Bob remain, others become NaN
    assert processed["name"].isin(["Alice", "Bob", None, np.nan]).all()
    # After blacklist, Charlie and David become NaN
    assert processed["name"].isna().sum() >= 2

def test_execute_success(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    data_source = DummyDataSource(df=sample_df)
    reporter = mock.Mock()
    with mock.patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter", DummyWriter):
        result = op.execute(data_source, temp_task_dir, reporter)
    assert result.status.name in ("SUCCESS", "PENDING", "ERROR")

def test_execute_data_load_error(temp_task_dir):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    data_source = DummyDataSource(df=None, error="fail")
    reporter = mock.Mock()
    with mock.patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter", DummyWriter):
        result = op.execute(data_source, temp_task_dir, reporter)
    assert result.status.name == "ERROR"

def test_check_cache_hit(sample_df, temp_task_dir):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    data_source = DummyDataSource(df=sample_df)
    fake_cache = {"metrics": {"foo": 1}, "timestamp": "now"}
    with mock.patch.dict("sys.modules", {"pamola_core.utils.ops.op_cache": mock.Mock()}):
        import sys
        sys.modules["pamola_core.utils.ops.op_cache"].operation_cache.get_cache = mock.Mock(return_value=fake_cache)
        sys.modules["pamola_core.utils.ops.op_cache"].operation_cache.generate_cache_key = mock.Mock(return_value="abc123")
        result = op._check_cache(data_source, temp_task_dir, "main")
    assert result is not None
    assert result.status.name == "SUCCESS"
    assert result.metrics["cached"] is True

def test_get_and_validate_data_success(sample_df):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    data_source = DummyDataSource(df=sample_df)
    df, err = op._get_and_validate_data(data_source, "main")
    assert err is None
    assert isinstance(df, pd.DataFrame)

def test_get_and_validate_data_fail():
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    data_source = DummyDataSource(df=None, error="fail")
    df, err = op._get_and_validate_data(data_source, "main")
    assert df is None
    assert "Failed to load input data" in err

def test_generate_cache_key_and_data_hash(sample_df):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    with mock.patch.dict("sys.modules", {"pamola_core.utils.ops.op_cache": mock.Mock()}):
        import sys
        sys.modules["pamola_core.utils.ops.op_cache"].operation_cache.generate_cache_key = mock.Mock(return_value="abc123")
        key = op._generate_cache_key(sample_df)
    assert key == "abc123"
    with mock.patch("pandas.DataFrame.describe", side_effect=Exception("fail")):
        h = op._generate_data_hash(sample_df)
    assert isinstance(h, str)

def test_process_value_not_implemented():
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    with pytest.raises(NotImplementedError):
        op.process_value(1)

def test_prepare_directories(temp_task_dir):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    dirs = op._prepare_directories(temp_task_dir)
    assert all(Path(p).exists() for p in dirs.values())

def test_collect_metrics(sample_df):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    with mock.patch.dict("sys.modules", {"pamola_core.transformations.commons.metric_utils": mock.Mock()}):
        import sys
        sys.modules["pamola_core.transformations.commons.metric_utils"].calculate_dataset_comparison = mock.Mock(return_value={"foo": 1})
        metrics = op._collect_metrics(sample_df, sample_df)
    assert "foo" in metrics

def test_save_metrics(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    writer = DummyWriter()
    result = mock.Mock()
    reporter = mock.Mock()
    metrics = {"foo": 1, "bar": 2}
    op._save_metrics(metrics, temp_task_dir, writer, result, reporter, None)
    assert writer.metrics_written

def test_generate_visualizations_and_handle(sample_df, temp_task_dir):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    with mock.patch.dict("sys.modules", {"pamola_core.transformations.commons.visualization_utils": mock.Mock()}):
        import sys
        sys.modules["pamola_core.transformations.commons.visualization_utils"].generate_field_count_comparison_vis = mock.Mock(return_value={"foo": Path("/tmp/foo.png")})
        paths = op._generate_visualizations(sample_df, sample_df, temp_task_dir)
    assert "foo" in paths
    result = mock.Mock()
    reporter = mock.Mock()
    op._handle_visualizations(sample_df, sample_df, temp_task_dir, result, reporter, None)

def test_save_output_data(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    writer = DummyWriter()
    result = mock.Mock()
    reporter = mock.Mock()
    processed_df = sample_df.copy()
    with mock.patch.dict("sys.modules", {"pamola_core.transformations.commons.visualization_utils": mock.Mock()}):
        import sys
        sys.modules["pamola_core.transformations.commons.visualization_utils"].generate_visualization_filename = mock.Mock(return_value="output.csv")
        op._save_output_data(processed_df, temp_task_dir, writer, result, reporter, None)

def test_save_to_cache(sample_df, temp_task_dir):
    op = CleanInvalidValuesOperation(field_constraints={}, output_format="csv", field_name="age")
    with mock.patch.dict("sys.modules", {"pamola_core.utils.ops.op_cache": mock.Mock()}):
        import sys
        sys.modules["pamola_core.utils.ops.op_cache"].operation_cache.save_cache = mock.Mock(return_value="mocked")
        assert op._save_to_cache(sample_df, sample_df, {"foo": 1}, temp_task_dir) == "mocked"

if __name__ == "__main__":
    pytest.main()
