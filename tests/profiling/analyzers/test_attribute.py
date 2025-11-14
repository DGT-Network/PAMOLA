"""
Tests for the attribute module in the pamola_core/profiling/analyzers package.
These tests ensure that the DataAttributeProfilerOperation class properly implements attribute profiling,
configuration, error handling, caching, and visualization logic.
"""
import os
import sys
import unittest
from unittest import mock
import pytest
import pandas as pd
from pathlib import Path
from pamola_core.profiling.analyzers.attribute import DataAttributeProfilerOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus


class DummyDataSource:
    def __init__(self, df=None, error=None):
        self.df = df
        self.error = error
        self.encryption_keys = {}
        self.encryption_modes = {}
    def get_dataframe(self, dataset_name, **kwargs):
        if self.df is not None:
            return self.df, None
        return None, {"message": self.error or "No data"}

class DummyReporter:
    def __init__(self):
        self.operations = []
        self.artifacts = []

    def add_operation(self, *args, **kwargs):
        self.operations.append((args, kwargs))

    def add_artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))


class DummyProgress:
    def __init__(self):
        self.updates = []
        self.total = 0

    def update(self, step, info):
        self.updates.append((step, info))

    def create_subtask(self, total, description, unit):
        return DummyProgress()

    def close(self):
        pass


def minimal_profiler(**kwargs):
    return DataAttributeProfilerOperation(**kwargs)

@pytest.fixture
def kwargs():
        return {
            "name": "DataAttributeProfiler",
            "description": "Automatic profiling of dataset attributes",
            "dictionary_path": None,
            "language": "en",
            "sample_size": 10,
            "max_columns": None,
            "use_encryption": False,
            "encryption_key": None,
            "use_dask": False,
            "use_cache": False,
            "use_vectorization": False,
            "chunk_size": 10000,
            "parallel_processes": 1,
            "npartitions": 1,
            "visualization_theme": None,
            "visualization_backend": None,
            "visualization_strict": False,
            "visualization_timeout": 120,
            "encryption_mode": None
        }

@pytest.fixture
def tmp_task_dir():
    task_dir = Path("test_task_dir/unittest/profiling/analyzers/attribute")
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


@pytest.fixture
def dummy_reporter():
    return DummyReporter()


@pytest.fixture
def dummy_progress():
    return DummyProgress()


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['NY', 'LA', 'SF']
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def patched_load_data(monkeypatch, valid_df):
    monkeypatch.setattr(
        'pamola_core.utils.io.load_data_operation',
        lambda *a, **k: valid_df
    )
    monkeypatch.setattr(
        'pamola_core.utils.io.load_settings_operation',
        lambda *a, **k: {}
    )
    return True


@pytest.fixture
def patched_load_data_empty(monkeypatch, empty_df):
    monkeypatch.setattr(
        'pamola_core.utils.io.load_data_operation',
        lambda *a, **k: empty_df
    )
    monkeypatch.setattr(
        'pamola_core.utils.io.load_settings_operation',
        lambda *a, **k: {}
    )
    return True


@pytest.fixture
def patched_dictionary(monkeypatch):
    monkeypatch.setattr(
        'pamola_core.profiling.commons.attribute_utils.load_attribute_dictionary',
        lambda *a, **k: {'columns': {}, 'summary': {}, 'column_groups': {}}
    )
    monkeypatch.setattr(
        'pamola_core.profiling.commons.attribute_utils.analyze_dataset_attributes',
        lambda **kwargs: {
            'columns': {
                'id': {'role': 'DIRECT_IDENTIFIER', 'statistics': {'entropy': 0.0, 'normalized_entropy': 0.0, 'uniqueness_ratio': 1.0, 'missing_rate': 0.0, 'inferred_type': 'int', 'samples': [1, 2, 3]}},
                'name': {'role': 'QUASI_IDENTIFIER', 'statistics': {'entropy': 1.0, 'normalized_entropy': 1.0, 'uniqueness_ratio': 1.0, 'missing_rate': 0.0, 'inferred_type': 'str', 'samples': ['Alice', 'Bob', 'Charlie']}}
            },
            'summary': {
                'DIRECT_IDENTIFIER': 1,
                'QUASI_IDENTIFIER': 1,
                'SENSITIVE_ATTRIBUTE': 0,
                'INDIRECT_IDENTIFIER': 0,
                'NON_SENSITIVE': 0
            },
            'column_groups': {'QUASI_IDENTIFIER': ['name']},
            'dataset_metrics': {'avg_entropy': 0.5, 'avg_uniqueness': 1.0},
            'conflicts': []
        }
    )
    return True


@pytest.fixture
def patched_write_json(monkeypatch):
    monkeypatch.setattr('pamola_core.utils.io.write_json', lambda *a, **k: None)
    monkeypatch.setattr('pamola_core.utils.io.write_dataframe_to_csv', lambda *a, **k: None)
    monkeypatch.setattr('pamola_core.utils.io.get_timestamped_filename', lambda prefix, ext, ts: f"{prefix}.{ext}")
    monkeypatch.setattr('pamola_core.utils.io.ensure_directory', lambda *a, **k: None)
    return True


@pytest.fixture
def patched_encryption(monkeypatch):
    monkeypatch.setattr('pamola_core.utils.io_helpers.crypto_utils.get_encryption_mode', lambda *a, **k: None)
    return True


@pytest.fixture
def patched_constants(monkeypatch):
    class DummyConstants:
        Artifact_Category_Output = 'output'
        Artifact_Category_Dictionary = 'dict'
        Artifact_Category_Metrics = 'metrics'
        Artifact_Category_Visualization = 'viz'
    monkeypatch.setattr('pamola_core.common.constants.Constants', DummyConstants)
    return True


@pytest.fixture
def patched_visualization(monkeypatch):
    monkeypatch.setattr('pamola_core.utils.visualization.create_pie_chart', lambda *a, **k: 'ok')
    monkeypatch.setattr('pamola_core.utils.visualization.create_bar_plot', lambda *a, **k: 'ok')
    monkeypatch.setattr('pamola_core.utils.visualization.create_scatter_plot', lambda *a, **k: 'ok')
    return True


@pytest.fixture
def patched_cache(monkeypatch):
    class DummyCache:
        def get_cache(self, **kwargs):
            return None

        def save_cache(self, **kwargs):
            return True

        def generate_cache_key(self, **kwargs):
            return 'cachekey'
    monkeypatch.setattr('pamola_core.utils.ops.op_cache.operation_cache', DummyCache())
    return True


def test_valid_case(
    patched_load_data,
    patched_dictionary,
    patched_write_json,
    patched_encryption,
    patched_constants,
    patched_visualization,
    patched_cache,
    tmp_task_dir,
    dummy_reporter,
    dummy_progress,
    valid_df,
    kwargs
):
    profiler = minimal_profiler(**kwargs)
    result = profiler.execute(
        data_source=DummyDataSource(df=valid_df),
        task_dir=tmp_task_dir,
        reporter=dummy_reporter,
        progress_tracker=dummy_progress,
        include_timestamp=False
    )
    assert isinstance(result, OperationResult)
    assert result.status == OperationStatus.SUCCESS
    assert any(a.description == 'Attribute roles analysis' for a in result.artifacts)
    assert any(m[0] == 'quasi_identifiers' for m in result.metrics.items())
    assert result.metrics['total_columns'] == 4
    assert result.metrics['direct_identifiers_count'] == 1
    assert result.metrics['quasi_identifiers_count'] == 1
    assert result.metrics['sensitive_attributes_count'] == 0
    assert result.metrics['indirect_identifiers_count'] == 0
    assert result.metrics['non_sensitive_count'] == 2
    assert result.metrics['avg_uniqueness'] == 1.0
    assert result.metrics['conflicts_count'] == 2


def test_edge_case_empty_df(
    monkeypatch,
    patched_load_data_empty,
    patched_dictionary,
    patched_write_json,
    patched_encryption,
    patched_constants,
    patched_visualization,
    patched_cache,
    tmp_task_dir,
    dummy_reporter,
    dummy_progress,
    empty_df,
    kwargs
):
    monkeypatch.setattr('pamola_core.profiling.analyzers.attribute.load_settings_operation', lambda *a, **k: {})
    monkeypatch.setattr('pamola_core.profiling.analyzers.attribute.load_data_operation', lambda *a, **k: None)
    profiler = minimal_profiler(**kwargs)
    result = profiler.execute(
        data_source=DummyDataSource(df=empty_df),
        task_dir=tmp_task_dir,
        reporter=dummy_reporter,
        progress_tracker=dummy_progress,
        include_timestamp=False
    )
    assert isinstance(result, OperationResult)
    assert result.status == OperationStatus.ERROR
    assert 'No valid DataFrame' in result.error_message


def test_invalid_input(monkeypatch, tmp_task_dir, dummy_reporter, dummy_progress, valid_df, kwargs):
    profiler = minimal_profiler(**kwargs)
    # Patch load_data_operation to raise exception
        # Patch to raise an Exception when called
    def raise_exc(*a, **k):
        raise RuntimeError("This is a test exception")
    monkeypatch.setattr('pamola_core.profiling.analyzers.attribute.load_attribute_dictionary', raise_exc)
    result = profiler.execute(
        data_source=DummyDataSource(df=valid_df),
        task_dir=tmp_task_dir,
        reporter=dummy_reporter,
        progress_tracker=dummy_progress,
        include_timestamp=False
    )
    assert isinstance(result, OperationResult)
    assert result.status == OperationStatus.ERROR
    assert 'Error in attribute profiling' in result.error_message


def test_cache_hit(monkeypatch,
                   patched_load_data,
                   patched_dictionary,
                   patched_write_json,
                   patched_encryption,
                   patched_constants,
                   patched_visualization,
                   tmp_task_dir,
                   dummy_reporter,
                   dummy_progress,
                   valid_df,
                   kwargs):
    profiler = minimal_profiler(**kwargs)
    profiler.use_cache = True
    class DummyCache:
        def get_cache(self, **kwargs):
            return {
                'metrics': {'cached': True, 'cache_key': 'abc', 'cache_timestamp': 'now'},
                'artifacts': [
                    {'artifact_type': 'json', 'path': 'foo.json', 'description': 'desc', 'category': 'output'}
                ]
            }
        def save_cache(self, **kwargs):
            return True
        def generate_cache_key(self, **kwargs):
            return 'cachekey'
    monkeypatch.setattr('pamola_core.utils.ops.op_cache.operation_cache', DummyCache())
    result = profiler.execute(
        data_source=DummyDataSource(df=valid_df),
        task_dir=tmp_task_dir,
        reporter=dummy_reporter,
        progress_tracker=dummy_progress,
        include_timestamp=False
    )
    assert isinstance(result, OperationResult)
    assert result.metrics['cached'] is True
    assert result.metrics['cache_key'] == 'cachekey'
    assert any(a.description == 'desc' for a in result.artifacts)


def test_save_to_cache(monkeypatch, patched_constants, valid_df):
    profiler = minimal_profiler()
    class DummyCache:
        def get_cache(self, **kwargs):
            return None
        def save_cache(self, **kwargs):
            return True
        def generate_cache_key(self, **kwargs):
            return 'cachekey'
    monkeypatch.setattr('pamola_core.utils.ops.op_cache.operation_cache', DummyCache())
    profiler.use_cache = True
    artifacts = []
    metrics = {'foo': 1}
    assert profiler._save_to_cache(valid_df, artifacts, metrics, Path('.')) is True


def test_generate_cache_key(monkeypatch, patched_constants, valid_df):
    profiler = minimal_profiler()
    class DummyCache:
        def get_cache(self, **kwargs):
            return None
        def save_cache(self, **kwargs):
            return True
        def generate_cache_key(self, **kwargs):
            return 'cachekey'
    monkeypatch.setattr('pamola_core.utils.ops.op_cache.operation_cache', DummyCache())
    key = profiler._generate_cache_key(valid_df)
    assert key == 'cachekey'


def test_generate_data_hash(valid_df):
    profiler = minimal_profiler()
    h = profiler._generate_data_hash(valid_df)
    assert isinstance(h, str)
    assert len(h) == 32


def test_prepare_directories(tmp_path):
    profiler = minimal_profiler()
    dirs = profiler._prepare_directories(tmp_path)
    assert set(dirs.keys()) == {'output', 'visualizations', 'dictionaries'}
    for d in dirs.values():
        assert Path(d).exists()


def test_handle_visualizations(monkeypatch, patched_constants, patched_visualization, tmp_path, dummy_reporter, kwargs):
    profiler = minimal_profiler(**kwargs)
    analysis_results = {
        'summary': {'DIRECT_IDENTIFIER': 1, 'QUASI_IDENTIFIER': 1, 'SENSITIVE_ATTRIBUTE': 0, 'INDIRECT_IDENTIFIER': 0, 'NON_SENSITIVE': 0},
        'columns': {
            'id': {'role': 'DIRECT_IDENTIFIER', 'statistics': {'entropy': 0.0, 'uniqueness_ratio': 1.0, 'inferred_type': 'int'}},
            'name': {'role': 'QUASI_IDENTIFIER', 'statistics': {'entropy': 1.0, 'uniqueness_ratio': 1.0, 'inferred_type': 'str'}}
        }
    }
    result = OperationResult(status=OperationStatus.SUCCESS)
    profiler._handle_visualizations(
        analysis_results=analysis_results,
        vis_dir=tmp_path,
        include_timestamp=False,
        result=result,
        reporter=dummy_reporter,
        vis_theme=None,
        vis_backend=None,
        vis_strict=False,
        vis_timeout=2
    )
    # Should not raise and should add artifacts
    assert any(a.category == 'visualization' for a in result.artifacts)


if __name__ == "__main__":
    unittest.main()