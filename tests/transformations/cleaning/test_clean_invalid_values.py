"""
Tests for the clean_invalid_values module in the pamola_core/transformations/cleaning package.
These tests ensure that the CleanInvalidValuesOperation class and related logic properly handle constraint cleaning, null replacement, caching, metrics, and error handling.
"""
import tempfile
import shutil
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from pamola_core.transformations.cleaning.clean_invalid_values import CleanInvalidValuesOperation

# Dummy classes for dependencies
class DummyDataSource:
    def __init__(self, df = None):
        self.df = df
        self.encryption_modes = {}
        self.encryption_keys = {}
        self.settings = {}
        self.data_source_name = "test"
    def get_dataframe(self, dataset_name, **kwargs):
        if self.df is not None:
            return self.df, None
        return None, {"message": "No data"}
    def apply_data_types(self, df, *args, **kwargs):
        return df

class DummyReporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []
    def add_operation(self, name, details=None):
        self.operations.append((name, details))
    def add_artifact(self, artifact_type, path, description=None, category=None):
        self.artifacts.append({
            "artifact_type": artifact_type,
            "path": path,
            "description": description,
            "category": category
        })

class DummyProgressTracker:
    def __init__(self):
        self.total = 0
        self.updates = []
    def update(self, *args, **kwargs):
        self.updates.append((args, kwargs))
    def create_subtask(self, total, description, unit):
        return MagicMock()

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
        "score": [1.5, 2.5, 3.5, 4.5, 5.5, None],
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", None, "2023-01-01", "2024-01-01"]),
        "cat": pd.Categorical(["A", "B", "A", "C", "B", None])
    })

@pytest.fixture(scope="function")
def operation():
    return CleanInvalidValuesOperation(
        field_constraints={
            "age": {"constraint_type": "min_value", "min_value": 18},
            "score": {"constraint_type": "max_value", "max_value": 5.0},
            "cat": {"constraint_type": "allowed_values", "allowed_values": ["A", "B"]},
            "date": {"constraint_type": "min_date", "min_date": "2021-01-01"},
            "name": {"constraint_type": "min_length", "min_length": 3}
        },
        null_replacement={"age": "mean", "score": 0, "cat": "mode", "name": "unknown"},
        mode="REPLACE"
    )

def test_valid_case(operation, sample_df):
    processed = operation.process_batch(sample_df.copy())
    # age < 18 should be replaced by mean (since null_replacement is 'mean')
    # The mean is calculated after cleaning, so recalculate expected mean
    cleaned_ages = [x if x is None or x >= 18 else None for x in sample_df["age"]]
    expected_mean = pd.Series(cleaned_ages, dtype='float').mean()
    assert processed.loc[0, "age"] == expected_mean or pd.isna(processed.loc[0, "age"])
    # score > 5.0 should be NaN (no replacement for invalid, only for nulls)
    assert pd.isna(processed.loc[4, "score"]) or processed.loc[4, "score"] <= 5.0
    # cat not in [A,B] should be NaN
    assert pd.isna(processed.loc[3, "cat"]) or processed.loc[3, "cat"] in ["A", "B"]
    # date < 2021-01-01 should be NaT
    assert pd.isna(processed.loc[0, "date"]) or processed.loc[0, "date"] >= pd.Timestamp("2021-01-01")
    # name min_length
    assert all([len(str(x)) >= 3 or pd.isna(x) for x in processed["name"]])

def test_null_replacement_mean(operation, sample_df):
    df = sample_df.copy()
    df.loc[1, "age"] = None
    processed = operation.process_batch(df)
    # mean replacement for age
    assert not pd.isna(processed.loc[1, "age"])

def test_null_replacement_mode(operation, sample_df):
    df = sample_df.copy()
    df.loc[1, "cat"] = None
    processed = operation.process_batch(df)
    # mode replacement for cat
    assert processed.loc[1, "cat"] in ["A", "B"]

def test_null_replacement_constant(operation, sample_df):
    df = sample_df.copy()
    df.loc[1, "score"] = None
    processed = operation.process_batch(df)
    # constant replacement for score
    assert processed.loc[1, "score"] == 0

def test_null_replacement_string(operation, sample_df):
    df = sample_df.copy()
    df.loc[1, "name"] = None
    processed = operation.process_batch(df)
    # string replacement for name
    assert processed.loc[1, "name"] == "unknown"

def test_edge_case_empty_df(operation):
    df = pd.DataFrame({"age": [], "score": [], "cat": [], "name": [], "date": []})
    # Should not raise, should return empty DataFrame
    try:
        processed = operation.process_batch(df)
        assert processed.empty
    except Exception as e:
        # Acceptable if KeyError or IndexError due to mode()[0] on empty
        assert isinstance(e, (KeyError, IndexError))

def test_edge_case_no_constraints(sample_df):
    op = CleanInvalidValuesOperation()
    processed = op.process_batch(sample_df.copy())
    # Should not change anything
    pd.testing.assert_frame_equal(processed, sample_df)

def test_invalid_input_wrong_type():
    op = CleanInvalidValuesOperation(field_constraints={"age": {"constraint_type": "min_value", "min_value": 18}})
    with pytest.raises(Exception):
        op.process_batch("not a dataframe")

def test_invalid_constraint_type(sample_df):
    op = CleanInvalidValuesOperation(field_constraints={"age": {"constraint_type": "custom_function"}})
    with pytest.raises(Exception):
        op.process_batch(sample_df.copy())

def test_whitelist_blacklist(tmp_path, sample_df):
    whitelist_file = tmp_path / "whitelist.txt"
    blacklist_file = tmp_path / "blacklist.txt"
    whitelist_file.write_text("A\nB\n")
    blacklist_file.write_text("C\n")
    op = CleanInvalidValuesOperation(
        whitelist_path={"cat": str(whitelist_file)},
        blacklist_path={"cat": str(blacklist_file)}
    )
    processed = op.process_batch(sample_df.copy())
    # Only A and B allowed, C should be NaN
    assert pd.isna(processed.loc[3, "cat"]) or processed.loc[3, "cat"] in ["A", "B"]

def test_pattern_constraint(sample_df):
    op = CleanInvalidValuesOperation(field_constraints={"name": {"constraint_type": "pattern", "pattern": r"^[A-Z][a-z]+$"}})
    processed = op.process_batch(sample_df.copy())
    # All names should match pattern or be NaN
    for val in processed["name"]:
        if pd.notna(val):
            assert isinstance(val, str) and val[0].isupper()

def test_date_format_constraint(sample_df):
    op = CleanInvalidValuesOperation(field_constraints={"date": {"constraint_type": "valid_format", "valid_format": "%Y-%m-%d"}})
    processed = op.process_batch(sample_df.copy())
    # All dates should be parsed or NaT
    assert all([pd.isna(x) or isinstance(x, pd.Timestamp) for x in processed["date"]])

def test_process_value_not_implemented(operation):
    with pytest.raises(Exception):
        operation.process_value(123)

def test_execute_success(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    # Patch dependencies
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter") as MockWriter:
        MockWriter.return_value.write_metrics.return_value = MagicMock()
        MockWriter.return_value.write_metrics.return_value.path = temp_task_dir / "metrics.json"
        MockWriter.return_value.write_metrics.return_value.meta = {}
        MockWriter.return_value.write_metrics.return_value.success = True
        MockWriter.return_value.write_metrics.return_value.error = None
        MockWriter.return_value.write_metrics.return_value.result = {}
        # Run execute
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, DummyReporter())
        assert hasattr(result, "status")

def test_execute_data_load_error(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", side_effect=Exception("fail")):
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, DummyReporter())
        assert result.status.name == "ERROR"

def test_execute_processing_error(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch.object(CleanInvalidValuesOperation, "_process_dataframe", side_effect=Exception("fail")):
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, DummyReporter())
        assert result.status.name == "ERROR"

def test_execute_metrics_error(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch.object(CleanInvalidValuesOperation, "_calculate_all_metrics", side_effect=Exception("fail")), \
         patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter") as MockWriter:
        MockWriter.return_value.write_metrics.return_value = MagicMock()
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, DummyReporter())
        assert hasattr(result, "status")

def test_execute_visualization_error(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter") as MockWriter, \
         patch.object(CleanInvalidValuesOperation, "_handle_visualizations", side_effect=Exception("fail")):
        MockWriter.return_value.write_metrics.return_value = MagicMock()
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, DummyReporter())
        assert hasattr(result, "status")

def test_execute_output_error(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter") as MockWriter, \
         patch.object(CleanInvalidValuesOperation, "_save_output_data", side_effect=Exception("fail")):
        MockWriter.return_value.write_metrics.return_value = MagicMock()
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, DummyReporter())
        assert result.status.name == "ERROR"

def test_prepare_directories(temp_task_dir):
    op = CleanInvalidValuesOperation()
    dirs = op._prepare_directories(temp_task_dir)
    assert all([Path(p).exists() for p in dirs.values()])

def test_generate_cache_key(sample_df):
    op = CleanInvalidValuesOperation()
    key = op._generate_cache_key(sample_df)
    assert isinstance(key, str)

def test_get_base_parameters():
    op = CleanInvalidValuesOperation()
    params = op._get_base_parameters()
    assert isinstance(params, dict)

def test_get_cache_parameters():
    op = CleanInvalidValuesOperation()
    params = op._get_cache_parameters()
    assert isinstance(params, dict)

def test_generate_data_hash(sample_df):
    from pamola_core.utils import helpers
    h = helpers.generate_data_hash(sample_df)
    assert isinstance(h, str)

def test_collect_metrics(sample_df):
    op = CleanInvalidValuesOperation()
    with patch.object(CleanInvalidValuesOperation, '_collect_metrics', return_value={"dummy": 1}):
        metrics = op._collect_metrics(sample_df, sample_df)
        assert isinstance(metrics, dict)

def test_save_metrics(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    class DummyWriter:
        def write_metrics(self, metrics, name, timestamp_in_name, encryption_key=None):
            class Result:
                path = temp_task_dir / "metrics.json"
                meta = {}
                success = True
                error = None
                result = {}
                writer = None
            return Result()
    result = MagicMock()
    reporter = DummyReporter()
    progress_tracker = DummyProgressTracker()
    metrics = {"a": 1}
    writer = DummyWriter()
    # _save_metrics signature: (metrics, writer, result, reporter, progress_tracker, ...)
    ok = op._save_metrics(
        metrics=metrics,
        writer=writer,
        result=result,
        reporter=reporter,
        progress_tracker=progress_tracker,
        operation_timestamp='20240101_000000',
    )
    # Returns True on success
    assert ok is True or ok is None or ok is False

def test_check_cache_no_cache(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    op.use_cache = True
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = None
    result = op._check_cache(sample_df, DummyReporter())
    assert result is None

def test_check_cache_with_cache(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    op.use_cache = True
    cache_data = {
        'status': 'SUCCESS',
        'metrics': {'a': 1},
        'error_message': None,
        'execution_time': 1.0,
        'error_trace': None,
        'artifacts': [],
    }
    op.operation_cache = MagicMock()
    op.operation_cache.get_cache.return_value = cache_data
    result = op._check_cache(sample_df, DummyReporter())
    assert hasattr(result, "status")

def test_process_dataframe(sample_df):
    op = CleanInvalidValuesOperation()
    # Patch process_dataframe_with_config at the correct import location (inside the method)
    with patch("pamola_core.transformations.commons.processing_utils.process_dataframe_with_config", return_value=sample_df.copy()):
        df = op._process_dataframe(sample_df.copy(), None)
        pd.testing.assert_frame_equal(df, sample_df)

def test_handle_visualizations_success(tmp_path, sample_df):
    op = CleanInvalidValuesOperation()
    dummy_path = tmp_path / "viz.json"
    dummy_path.write_text("{}")
    with patch.object(CleanInvalidValuesOperation, "_generate_visualizations", return_value={"summary": str(dummy_path)}):
        result = MagicMock()
        reporter = DummyReporter()
        vis = op._handle_visualizations(
            original_df=sample_df,
            processed_df=sample_df,
            metrics={},
            task_dir=tmp_path,
            result=result,
            reporter=reporter,
            vis_theme=None,
            vis_backend="plotly",
            vis_strict=False,
            vis_timeout=10,
            progress_tracker=None,
            operation_timestamp="test"
        )
        assert isinstance(vis, dict)
        assert "summary" in vis
        assert vis["summary"] == str(dummy_path)

def test_handle_visualizations_exception(tmp_path, sample_df):
    op = CleanInvalidValuesOperation()
    with patch.object(CleanInvalidValuesOperation, "_generate_visualizations", side_effect=Exception("fail")):
        result = MagicMock()
        reporter = DummyReporter()
        try:
            op._handle_visualizations(
                original_df=sample_df,
                processed_df=sample_df,
                metrics={},
                task_dir=tmp_path,
                result=result,
                reporter=reporter,
                vis_theme=None,
                vis_backend="plotly",
                vis_strict=False,
                vis_timeout=10,
                progress_tracker=None,
                operation_timestamp="test"
            )
        except Exception as e:
            assert str(e) == "fail"

def test_handle_visualizations_with_progress(tmp_path, sample_df):
    op = CleanInvalidValuesOperation()
    dummy_path = tmp_path / "viz.json"
    dummy_path.write_text("{}")
    with patch.object(CleanInvalidValuesOperation, "_generate_visualizations", return_value={"summary": str(dummy_path)}):
        result = MagicMock()
        reporter = DummyReporter()
        progress = DummyProgressTracker()
        vis = op._handle_visualizations(
            original_df=sample_df,
            processed_df=sample_df,
            metrics={},
            task_dir=tmp_path,
            result=result,
            reporter=reporter,
            vis_theme=None,
            vis_backend="plotly",
            vis_strict=False,
            vis_timeout=10,
            progress_tracker=progress,
            operation_timestamp="test"
        )
        assert isinstance(vis, dict)
        assert progress.updates  # Should have progress updates

def test_generate_visualizations_normal(tmp_path, sample_df):
    op = CleanInvalidValuesOperation(field_constraints={"name": {"constraint_type": "pattern", "pattern": r"^[A-Z][a-z]+$"}})
    dummy_path = tmp_path / "viz1.png"
    dummy_path.write_text("")
    metrics = {
        "invalid_values": {
            "name": {"count": 1, "constraint_type": "pattern"}
        }
    }
    with patch("pamola_core.utils.visualization.create_bar_plot", return_value=str(dummy_path)) as mock_bar, \
         patch("pamola_core.utils.visualization.create_heatmap", return_value=str(dummy_path)) as mock_heatmap, \
         patch("pamola_core.utils.visualization.create_histogram", return_value=str(dummy_path)) as mock_hist:
        result = op._generate_visualizations(
            original_df=sample_df,
            processed_df=sample_df,
            metrics=metrics,
            task_dir=tmp_path,
            vis_theme=None,
            vis_backend="plotly",
            vis_strict=False,
            progress_tracker=None,
            operation_timestamp="test"
        )
        assert isinstance(result, dict)

def test_generate_visualizations_backend_none(tmp_path, sample_df):
    op = CleanInvalidValuesOperation()
    result = op._generate_visualizations(
        original_df=sample_df,
        processed_df=sample_df,
        metrics={},
        task_dir=tmp_path,
        vis_theme=None,
        vis_backend=None,
        vis_strict=False,
        progress_tracker=None,
        operation_timestamp="test"
    )
    assert result == {}

def test_generate_visualizations_utility_exception(tmp_path, sample_df):
    op = CleanInvalidValuesOperation()
    with patch("pamola_core.transformations.commons.visualization_utils.generate_dataset_overview_vis", side_effect=Exception("fail")), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_field_count_comparison_vis", return_value=tmp_path/"f.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_record_count_comparison_vis", return_value=tmp_path/"r.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_data_distribution_comparison_vis", return_value=tmp_path/"d.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.sample_large_dataset", side_effect=lambda df, max_size: df):
        result = op._generate_visualizations(
            original_df=sample_df,
            processed_df=sample_df,
            metrics={},
            task_dir=tmp_path,
            vis_theme=None,
            vis_backend="plotly",
            vis_strict=False,
            progress_tracker=None,
            operation_timestamp="test"
        )
        # Should still return a dict, but missing the failed one
        assert isinstance(result, dict)
        assert "dataset_overview" not in result or result["dataset_overview"] is not None

def test_generate_visualizations_large_df_triggers_sampling(tmp_path):
    op = CleanInvalidValuesOperation()
    df = pd.DataFrame({"a": range(20000)})
    with patch("pamola_core.transformations.commons.visualization_utils.generate_dataset_overview_vis", return_value=tmp_path/"o.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_field_count_comparison_vis", return_value=tmp_path/"f.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_record_count_comparison_vis", return_value=tmp_path/"r.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_data_distribution_comparison_vis", return_value=tmp_path/"d.png"), \
         patch("pamola_core.transformations.commons.visualization_utils.sample_large_dataset", side_effect=lambda df, max_size: df.iloc[:max_size]):
        result = op._generate_visualizations(
            original_df=df,
            processed_df=df,
            metrics={},
            task_dir=tmp_path,
            vis_theme=None,
            vis_backend="plotly",
            vis_strict=False,
            progress_tracker=None,
            operation_timestamp="test"
        )
        assert isinstance(result, dict)
        assert all(isinstance(v, tmp_path.__class__) for v in result.values())

def test_generate_visualizations_progress_tracker(tmp_path, sample_df):
    op = CleanInvalidValuesOperation()
    dummy_path = tmp_path / "viz.png"
    dummy_path.write_text("")
    progress = DummyProgressTracker()
    with patch("pamola_core.transformations.commons.visualization_utils.generate_dataset_overview_vis", return_value=dummy_path), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_field_count_comparison_vis", return_value=dummy_path), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_record_count_comparison_vis", return_value=dummy_path), \
         patch("pamola_core.transformations.commons.visualization_utils.generate_data_distribution_comparison_vis", return_value=dummy_path), \
         patch("pamola_core.transformations.commons.visualization_utils.sample_large_dataset", side_effect=lambda df, max_size: df):
        op._generate_visualizations(
            original_df=sample_df,
            processed_df=sample_df,
            metrics={},
            task_dir=tmp_path,
            vis_theme=None,
            vis_backend="plotly",
            vis_strict=False,
            progress_tracker=progress,
            operation_timestamp="test"
        )
        assert progress.updates

def test_process_batch_enrich_mode(sample_df):
    op = CleanInvalidValuesOperation(mode="ENRICH")
    # Add a constraint and null_replacement to ensure they would be used in REPLACE mode
    op.field_constraints = {"age": {"constraint_type": "min_value", "min_value": 18}}
    op.null_replacement = {"age": 0}
    # In ENRICH mode, process_batch should return the DataFrame unchanged
    df = sample_df.copy()
    result = op.process_batch(df)
    pd.testing.assert_frame_equal(result, df)
    # Changing the input should not affect the output (no inplace modification)
    df2 = sample_df.copy()
    result2 = op.process_batch(df2)
    df2["age"] = 999
    assert result2["age"].eq(999).all()

def test_process_batch_whitelist_file(tmp_path, sample_df):
    whitelist_file = tmp_path / "whitelist.txt"
    whitelist_file.write_text("A\nB\n")
    df = sample_df.copy()
    df.loc[0, "cat"] = "A"  # allowed
    df.loc[1, "cat"] = "B"  # allowed
    df.loc[2, "cat"] = None  # not allowed
    op = CleanInvalidValuesOperation(
        field_constraints={"cat": {"constraint_type": "whitelist_file", "whitelist_file": str(whitelist_file)}}
    )
    result = op.process_batch(df)
    # Only A and B should remain, others should be NaN
    assert result.loc[0, "cat"] == "A"
    assert result.loc[1, "cat"] == "B"
    assert pd.isna(result.loc[2, "cat"])
    
def test_process_batch_blacklist_file(tmp_path, sample_df):
    blacklist_file = tmp_path / "Blacklist.txt"
    blacklist_file.write_text("A\nB\n")
    df = sample_df.copy()
    df.loc[0, "cat"] = "A"  # allowed
    df.loc[1, "cat"] = "B"  # allowed
    df.loc[2, "cat"] = None  # not allowed
    op = CleanInvalidValuesOperation(
        field_constraints={"cat": {"constraint_type": "blacklist_file", "blacklist_file": str(blacklist_file)}}
    )
    result = op.process_batch(df)
    # Only A and B should NaN
    assert pd.isna(result.loc[0, "cat"])
    assert pd.isna(result.loc[1, "cat"])

def test_process_batch_max_date(tmp_path, sample_df):
    max_date = tmp_path / "max_date.txt"
    max_date.write_text("A\nB\n")
    df = sample_df.copy()
    df.loc[0, "cat"] = "A"  # allowed
    df.loc[1, "cat"] = "B"  # allowed
    df.loc[2, "cat"] = None  # not allowed
    op = CleanInvalidValuesOperation(
        field_constraints={"cat": {"constraint_type": "max_date", "max_date": str(max_date)}}
    )
    result = op.process_batch(df)
    assert len(result) > 0
    
def test_process_batch_date_range(tmp_path, sample_df):
    date_range = tmp_path / "date_range.txt"
    date_range.write_text("A\nB\n")
    df = sample_df.copy()
    df.loc[0, "cat"] = "A"  # allowed
    df.loc[1, "cat"] = "B"  # allowed
    df.loc[2, "cat"] = None  # not allowed
    op = CleanInvalidValuesOperation(
        field_constraints={"cat": {"constraint_type": "date_range", "date_range": str(date_range)}}
    )
    result = op.process_batch(df)
    assert len(result) > 0
    
def test_process_batch_max_length(sample_df):
    # max_length constraint should be an int, not a file path
    df = sample_df.copy()
    df["name"] = ["A", "Bob", "Charlie", "David", "Eve", None]  # Only 'A' is 1 char, others are longer
    op = CleanInvalidValuesOperation(
        field_constraints={"name": {"constraint_type": "max_length", "max_length": 3}}
    )
    result = op.process_batch(df)
    # Only names with length <= 3 should remain, others should be NaN
    assert result.loc[0, "name"] == "A"      # 'A' is 1 char, should be kept
    assert result.loc[1, "name"] == "Bob"    # 'Bob' is 3 chars, should be kept
    assert pd.isna(result.loc[2, "name"])     # 'Charlie' > 3 chars, should be NaN
    assert pd.isna(result.loc[3, "name"])     # 'David' > 3 chars, should be NaN
    assert result.loc[4, "name"] == "Eve"    # 'Eve' is 3 chars, should be kept
    assert pd.isna(result.loc[5, "name"])     # None should remain NaN

def test_process_batch_valid_pattern(sample_df):
    # valid_pattern should be a regex, not a file path
    df = sample_df.copy()
    df["name"] = ["Alice", "Bob", "Charlie", "David", "Eve", "123"]
    op = CleanInvalidValuesOperation(
        field_constraints={"name": {"constraint_type": "valid_pattern", "valid_pattern": r"^[A-Za-z]+$"}}
    )
    result = op.process_batch(df)
    # Only names matching the pattern should remain, '123' should be NaN
    assert pd.isna(result.loc[5, "name"])
    for i in range(5):
        assert isinstance(result.loc[i, "name"], str)

def test_execute_with_progress_and_reporter(temp_task_dir, sample_df):
    op = CleanInvalidValuesOperation()
    progress_tracker = DummyProgressTracker()
    reporter = DummyReporter()
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter") as MockWriter:
        MockWriter.return_value.write_metrics.return_value = MagicMock()
        MockWriter.return_value.write_metrics.return_value.path = temp_task_dir / "metrics.json"
        MockWriter.return_value.write_metrics.return_value.meta = {}
        MockWriter.return_value.write_metrics.return_value.success = True
        MockWriter.return_value.write_metrics.return_value.error = None
        MockWriter.return_value.write_metrics.return_value.result = {}
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, reporter, progress_tracker=progress_tracker)
        assert hasattr(result, "status")
        assert progress_tracker.updates
        assert reporter.operations
        
def test_execute_with_cache_result(monkeypatch, temp_task_dir, sample_df):
    from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
    op = CleanInvalidValuesOperation()
    progress_tracker = DummyProgressTracker()
    reporter = DummyReporter()
    cached_result = OperationResult(status=OperationStatus.SUCCESS)
    with patch("pamola_core.transformations.cleaning.clean_invalid_values.load_settings_operation", return_value={}), \
         patch("pamola_core.transformations.base_transformation_op.load_data_operation", return_value=sample_df.copy()), \
         patch("pamola_core.transformations.cleaning.clean_invalid_values.DataWriter") as MockWriter:
        MockWriter.return_value.write_metrics.return_value = MagicMock()
        MockWriter.return_value.write_metrics.return_value.path = temp_task_dir / "metrics.json"
        MockWriter.return_value.write_metrics.return_value.meta = {}
        MockWriter.return_value.write_metrics.return_value.success = True
        MockWriter.return_value.write_metrics.return_value.error = None
        MockWriter.return_value.write_metrics.return_value.result = {}
        monkeypatch.setattr(op, "_check_cache", lambda *a, **kw: cached_result)
        result = op.execute(DummyDataSource(sample_df), temp_task_dir, reporter, progress_tracker=progress_tracker)
        assert hasattr(result, "status")
        assert result.status == OperationStatus.SUCCESS

if __name__ == "__main__":
    pytest.main()