"""
Tests for the processing_utils module in the PAMOLA.CORE package.

These tests verify the functionality of data processing utilities including
chunked processing, parallel processing, dataframe splitting/merging, and aggregation.

Run with:
    pytest tests/transformations/commons/test_processing_utils.py
"""
import pytest
import pandas as pd
from unittest import mock
from pamola_core.transformations.commons import processing_utils as pu

class DummyProgressTracker:
    def __init__(self):
        self.total = 0
        self.updates = []
    def update(self, val, info=None):
        self.updates.append((val, info))

# ---- process_in_chunks ----
def test_process_in_chunks_valid(monkeypatch):
    df = pd.DataFrame({'a': range(20)})
    def func(chunk, **kwargs):
        chunk['a'] = chunk['a'] + 1
        return chunk
    tracker = DummyProgressTracker()
    result = pu.process_in_chunks(df, func, batch_size=5, progress_tracker=tracker)
    assert isinstance(result, pd.DataFrame)
    assert (result['a'] == pd.Series(range(1, 21))).all()
    assert tracker.total == 4
    assert tracker.updates[-1][1]['processed_rows'] == 20

def test_process_in_chunks_empty():
    df = pd.DataFrame({'a': []})
    def func(chunk, **kwargs):
        return chunk
    result = pu.process_in_chunks(df, func)
    assert result.empty

def test_process_in_chunks_single_batch():
    df = pd.DataFrame({'a': [1, 2, 3]})
    def func(chunk, **kwargs):
        return chunk * 2
    result = pu.process_in_chunks(df, func, batch_size=10)
    assert (result['a'] == pd.Series([2, 4, 6])).all()

def test_process_in_chunks_func_error():
    df = pd.DataFrame({'a': range(10)})
    def func(chunk, **kwargs):
        if chunk['a'].iloc[0] == 5:
            raise ValueError('fail')
        return chunk
    result = pu.process_in_chunks(df, func, batch_size=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

# ---- _get_dataframe_chunks ----
def test_get_dataframe_chunks_normal():
    df = pd.DataFrame({'a': range(10)})
    chunks = list(pu._get_dataframe_chunks(df, chunk_size=3))
    assert len(chunks) == 4
    assert all(isinstance(c, pd.DataFrame) for c in chunks)
    assert chunks[0]['a'].tolist() == [0, 1, 2]
    assert chunks[-1]['a'].tolist() == [9]

def test_get_dataframe_chunks_empty():
    df = pd.DataFrame({'a': []})
    chunks = list(pu._get_dataframe_chunks(df, chunk_size=3))
    assert len(chunks) == 1
    assert chunks[0].empty

# ---- process_dataframe_parallel ----
def test_process_dataframe_parallel_valid(monkeypatch):
    df = pd.DataFrame({'a': range(10)})
    def func(chunk, **kwargs):
        chunk['a'] = chunk['a'] * 2
        return chunk
    tracker = DummyProgressTracker()
    result = pu.process_dataframe_parallel(df, func, n_jobs=2, batch_size=3, progress_tracker=tracker)
    assert isinstance(result, pd.DataFrame)
    assert set(result['a']) == set([i*2 for i in range(10)])
    assert tracker.total == 4

def test_process_dataframe_parallel_empty():
    df = pd.DataFrame({'a': []})
    def func(chunk, **kwargs):
        return chunk
    result = pu.process_dataframe_parallel(df, func)
    assert result.empty

def test_process_dataframe_parallel_single_batch():
    df = pd.DataFrame({'a': [1, 2, 3]})
    def func(chunk, **kwargs):
        return chunk * 2
    result = pu.process_dataframe_parallel(df, func, batch_size=10)
    assert (result['a'] == pd.Series([2, 4, 6])).all()

def test_process_dataframe_parallel_all_fail(monkeypatch):
    df = pd.DataFrame({'a': range(10)})
    def func(chunk, **kwargs):
        raise Exception('fail')
    result = pu.process_dataframe_parallel(df, func, n_jobs=2, batch_size=3)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

# ---- split_dataframe ----
def test_split_dataframe_valid():
    df = pd.DataFrame({'id': [1,2], 'x': [3,4], 'y': [5,6]})
    field_groups = {'g1': ['x'], 'g2': ['y']}
    result = pu.split_dataframe(df, field_groups, 'id')
    assert set(result.keys()) == {'g1', 'g2'}
    assert all('id' in v.columns for v in result.values())
    assert result['g1'].shape[1] == 2
    assert result['g2'].shape[1] == 2

def test_split_dataframe_empty():
    df = pd.DataFrame({'id': [], 'x': [], 'y': []})
    field_groups = {'g1': ['x'], 'g2': ['y']}
    result = pu.split_dataframe(df, field_groups, 'id')
    assert result == {}

def test_split_dataframe_invalid_field():
    df = pd.DataFrame({'id': [1], 'x': [2]})
    field_groups = {'g1': ['z']}
    with pytest.raises(ValueError):
        pu.split_dataframe(df, field_groups, 'id')

def test_split_dataframe_no_valid_fields():
    df = pd.DataFrame({'id': [1], 'x': [2]})
    field_groups = {'g1': ['z']}
    with pytest.raises(ValueError):
        pu.split_dataframe(df, field_groups, 'id')

def test_split_dataframe_exclude_id():
    df = pd.DataFrame({'id': [1,2], 'x': [3,4]})
    field_groups = {'g1': ['x']}
    result = pu.split_dataframe(df, field_groups, 'id', include_id_field=False)
    assert 'id' not in result['g1'].columns

# ---- merge_dataframes ----
def test_merge_dataframes_valid():
    left = pd.DataFrame({'id': [1,2], 'x': [3,4]})
    right = pd.DataFrame({'id': [1,2], 'y': [5,6]})
    result = pu.merge_dataframes(left, right, 'id')
    assert 'y' in result.columns
    assert result.shape[0] == 2

def test_merge_dataframes_right_key():
    left = pd.DataFrame({'lid': [1,2], 'x': [3,4]})
    right = pd.DataFrame({'rid': [1,2], 'y': [5,6]})
    result = pu.merge_dataframes(left, right, 'lid', right_key='rid')
    assert 'y' in result.columns
    assert result.shape[0] == 2

def test_merge_dataframes_invalid_key():
    left = pd.DataFrame({'id': [1], 'x': [2]})
    right = pd.DataFrame({'id': [2], 'y': [3]})
    with pytest.raises(Exception):
        pu.merge_dataframes(left, right, 'not_a_key')

def test_merge_dataframes_invalid_join_type():
    left = pd.DataFrame({'id': [1], 'x': [2]})
    right = pd.DataFrame({'id': [1], 'y': [3]})
    with pytest.raises(Exception):
        pu.merge_dataframes(left, right, 'id', join_type='bad')

# ---- aggregate_dataframe ----
def test_aggregate_dataframe_valid():
    df = pd.DataFrame({'id': [1,1,2], 'x': [1,2,3]})
    result = pu.aggregate_dataframe(df, ['id'], {'x': ['sum']})
    assert 'x_sum' in result.columns or ('x','sum') in result.columns
    assert result.shape[0] == 2

def test_aggregate_dataframe_custom(monkeypatch):
    df = pd.DataFrame({'id': [1,1,2], 'x': [1,2,3]})
    def fake_flatten(cols):
        return ['id', 'x_sum']
    monkeypatch.setattr(pu, 'flatten_multiindex_columns', fake_flatten)
    result = pu.aggregate_dataframe(df, ['id'], {'x': ['sum']})
    assert 'x_sum' in result.columns

def test_aggregate_dataframe_invalid_group():
    df = pd.DataFrame({'id': [1], 'x': [2]})
    with pytest.raises(Exception):
        pu.aggregate_dataframe(df, ['not_a_col'], {'x': ['sum']})

def test_aggregate_dataframe_invalid_agg():
    df = pd.DataFrame({'id': [1], 'x': [2]})
    with pytest.raises(Exception):
        pu.aggregate_dataframe(df, ['id'], {'not_a_col': ['sum']})

def test_aggregate_dataframe_empty():
    df = pd.DataFrame({'id': [], 'x': []})
    result = pu.aggregate_dataframe(df, ['id'], {'x': ['sum']})
    assert isinstance(result, pd.DataFrame)

# ---- _determine_partitions ----
def test_determine_partitions_default():
    df = pd.DataFrame({'a': range(1000000)})
    parts = pu._determine_partitions(df)
    assert parts == 10

def test_determine_partitions_custom():
    df = pd.DataFrame({'a': range(10)})
    parts = pu._determine_partitions(df, npartitions=3)
    assert parts == 3

if __name__ == "__main__":
    pytest.main()
