"""
Tests for the metric_utils module in the PAMOLA.CORE package.

These tests verify the functionality of data metric utilities including
dataset comparison, field statistics, transformation impact, and performance metrics.

Run with:
    pytest tests/transformations/commons/test_metric_utils.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from pamola_core.transformations.commons import metric_utils
from unittest.mock import patch

class DummyWriter:
    def write_metrics(self, metrics, name, timestamp_in_name, encryption_key=None):
        class Result:
            path = f"/tmp/{name}.json"
        return Result()

class DummyProgress:
    def update(self, val, meta):
        self.last = (val, meta)

@pytest.fixture
def sample_dfs():
    df1 = pd.DataFrame({
        'a': [1, 2, 3, np.nan],
        'b': ['x', 'y', 'z', 'x'],
        'c': [1.1, 2.2, 3.3, 4.4]
    })
    df2 = pd.DataFrame({
        'a': [1, 2, 4, np.nan],
        'b': ['x', 'y', 'w', 'x'],
        'c': [1.1, 2.0, 3.3, 4.5],
        'd': [10, 20, 30, 40]
    })
    return df1, df2

def test_calculate_dataset_comparison_valid(sample_dfs):
    df1, df2 = sample_dfs
    result = metric_utils.calculate_dataset_comparison(df1, df2)
    assert 'row_counts' in result
    assert 'column_counts' in result
    assert 'value_changes' in result
    assert 'null_changes' in result
    assert 'memory_usage' in result
    assert isinstance(result['row_counts'], dict)
    assert isinstance(result['column_counts'], dict)

def test_calculate_dataset_comparison_edge_empty():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    result = metric_utils.calculate_dataset_comparison(df1, df2)
    assert result['row_counts']['original'] == 0
    assert result['column_counts']['original'] == 0

def test_calculate_dataset_comparison_invalid():
    with pytest.raises(ValueError):
        metric_utils.calculate_dataset_comparison(None, pd.DataFrame())
    with pytest.raises(ValueError):
        metric_utils.calculate_dataset_comparison(pd.DataFrame(), None)

def test_compare_row_counts():
    df1 = pd.DataFrame({'a': [1, 2, 3]})
    df2 = pd.DataFrame({'a': [1, 2, 3, 4]})
    res = metric_utils._compare_row_counts(df1, df2)
    assert res['difference'] == 1
    assert res['percent_change'] > 0

def test_compare_column_counts():
    df1 = pd.DataFrame({'a': [1], 'b': [2]})
    df2 = pd.DataFrame({'a': [1], 'c': [3]})
    res, common, added, removed = metric_utils._compare_column_counts(df1, df2)
    assert 'original' in res
    assert 'transformed' in res
    assert 'a' in common
    assert 'c' in added
    assert 'b' in removed

def test_compare_values_and_nulls():
    df1 = pd.DataFrame({'a': [1, 2, np.nan]})
    df2 = pd.DataFrame({'a': [1, 3, np.nan]})
    value_changes, null_changes = metric_utils._compare_values_and_nulls(df1, df2)
    assert 'a' in value_changes
    assert 'a' in null_changes
    assert value_changes['a']['changed'] >= 0

def test_count_value_changes_numeric():
    s1 = pd.Series([1.0, 2.0, 3.0])
    s2 = pd.Series([1.0, 2.1, 3.0])
    res = metric_utils._count_value_changes(s1, s2)
    assert 'changed' in res
    assert res['changed'] == 1

def test_count_value_changes_non_numeric():
    s1 = pd.Series(['a', 'b', 'c'])
    s2 = pd.Series(['a', 'x', 'c'])
    res = metric_utils._count_value_changes(s1, s2)
    assert res['changed'] == 1

def test_count_null_changes():
    s1 = pd.Series([1, None, 3])
    s2 = pd.Series([1, 2, None])
    res = metric_utils._count_null_changes(s1, s2)
    assert 'original_nulls' in res
    assert 'transformed_nulls' in res

def test_compare_memory_usage():
    df1 = pd.DataFrame({'a': [1, 2, 3]})
    df2 = pd.DataFrame({'a': [1, 2, 3, 4]})
    res = metric_utils._compare_memory_usage(df1, df2)
    assert 'original_mb' in res
    assert 'transformed_mb' in res

def test_calculate_field_statistics_valid(sample_dfs):
    df1, _ = sample_dfs
    stats = metric_utils.calculate_field_statistics(df1)
    assert 'a' in stats
    assert 'b' in stats
    assert 'c' in stats
    assert 'data_type' in stats['a']
    assert 'basic_stats' in stats['a']

def test_calculate_field_statistics_empty():
    df = pd.DataFrame()
    stats = metric_utils.calculate_field_statistics(df)
    assert stats == {}

def test_calculate_field_statistics_invalid():
    with pytest.raises(ValueError):
        metric_utils.calculate_field_statistics(None)

def test_validate_fields():
    df = pd.DataFrame({'a': [1], 'b': [2]})
    fields = metric_utils._validate_fields(df, ['a', 'c'])
    assert 'a' in fields
    assert 'c' not in fields

def test_calculate_basic_stats_numeric():
    s = pd.Series([1, 2, 3, np.nan])
    stats = metric_utils._calculate_basic_stats(s)
    assert 'mean' in stats
    assert stats['count'] == 3

def test_calculate_basic_stats_non_numeric():
    s = pd.Series(['a', 'bb', 'ccc'])
    stats = metric_utils._calculate_basic_stats(s)
    assert 'length_mean' in stats
    assert stats['count'] == 3

def test_calculate_null_stats():
    s = pd.Series([1, None, 3])
    stats = metric_utils._calculate_null_stats(s, 3)
    assert stats['count'] == 1
    assert stats['percentage'] > 0

def test_calculate_unique_stats():
    s = pd.Series([1, 2, 2, 3])
    stats = metric_utils._calculate_unique_stats(s, 4)
    assert stats['count'] == 3
    assert stats['percentage'] > 0

def test_calculate_top_values():
    s = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'])
    stats = metric_utils._calculate_top_values(s)
    assert 'a' in stats
    assert stats['a'] == 3

def test_detect_patterns():
    s = pd.Series(['2020-01-01', '2021-02-02', '2022-03-03'])
    patterns = metric_utils._detect_patterns(s)
    assert 'potential_date' in patterns
    s2 = pd.Series(['123', '456', '789'])
    patterns2 = metric_utils._detect_patterns(s2)
    assert patterns2['numeric_strings']
    s3 = pd.Series(['test@example.com', 'foo@bar.com'])
    patterns3 = metric_utils._detect_patterns(s3)
    assert patterns3['email_pattern']

def test_calculate_transformation_impact(sample_dfs):
    df1, df2 = sample_dfs
    result = metric_utils.calculate_transformation_impact(df1, df2)
    assert 'data_quality' in result
    assert 'data_completeness' in result
    assert 'data_distribution' in result
    assert 'correlation_changes' in result
    assert 'field_impact' in result
    assert 'elapsed_time' in result

def test_calculate_null_percentage():
    df = pd.DataFrame({'a': [1, None, 3]})
    pct = metric_utils._calculate_null_percentage(df)
    assert pct > 0

def test_calculate_duplicate_percentage():
    df = pd.DataFrame({'a': [1, 1, 2]})
    pct = metric_utils._calculate_duplicate_percentage(df)
    assert pct > 0

def test_calculate_completeness():
    df = pd.DataFrame({'a': [1, None, 3]})
    comp = metric_utils._calculate_completeness(df)
    assert 'a' in comp
    assert comp['a'] < 100

def test_calculate_distribution_metrics():
    s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s2 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    res = metric_utils._calculate_distribution_metrics(s1, s2)
    assert 'mean' in res
    assert 'distribution_test' in res

def test_calculate_correlation_changes():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 7]})
    res = metric_utils._calculate_correlation_changes(df1, df2, ['a', 'b'])
    assert 'a_b' in res
    assert 'original' in res['a_b']

def test_calculate_field_impact():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 2]})
    df2 = pd.DataFrame({'a': [1, 2, 4], 'b': [1, 2, 3]})
    res = metric_utils._calculate_field_impact(df1, df2, {'a', 'b'})
    assert 'a' in res
    assert 'b' in res
    assert 'null_values' in res['a']
    assert 'unique_values' in res['a']
    assert 'data_type' in res['a']

def test_calculate_performance_metrics_valid():
    res = metric_utils.calculate_performance_metrics(0, 2, 100, 80)
    assert 'elapsed_time' in res
    assert 'rows_per_second' in res
    assert 'throughput_ratio' in res
    assert 'performance_rating' in res

def test_calculate_performance_metrics_invalid():
    with pytest.raises(ValueError):
        metric_utils.calculate_performance_metrics(None, 1, 10, 10)
    with pytest.raises(ValueError):
        metric_utils.calculate_performance_metrics(2, 1, 10, 10)

def test_format_time_duration():
    assert metric_utils.format_time_duration(30).endswith('seconds')
    assert metric_utils.format_time_duration(120).endswith('minutes')
    assert metric_utils.format_time_duration(7200).endswith('hours')

def test_save_metrics_json_writer(tmp_path):
    metrics = {'foo': 'bar'}
    task_dir = tmp_path
    writer = DummyWriter()
    progress = DummyProgress()
    path = metric_utils.save_metrics_json(metrics, task_dir, 'op', 'field', writer, progress, False)
    assert path.exists() or str(path).endswith('.json')

def test_save_metrics_json_no_writer(tmp_path):
    metrics = {'foo': 'bar'}
    task_dir = tmp_path
    with patch('pamola_core.transformations.commons.metric_utils.ensure_directory') as ed, \
         patch('pamola_core.transformations.commons.metric_utils.write_json') as wj:
        ed.return_value = None
        wj.return_value = tmp_path / 'file.json'
        path = metric_utils.save_metrics_json(metrics, task_dir, 'op', 'field', None, None, False)
        assert str(path).endswith('.json')

def test_save_metrics_json_writer_exception(tmp_path):
    metrics = {'foo': 'bar'}
    task_dir = tmp_path
    class BadWriter:
        def write_metrics(self, *a, **kw):
            raise Exception('fail')
    with patch('pamola_core.transformations.commons.metric_utils.ensure_directory') as ed, \
         patch('pamola_core.transformations.commons.metric_utils.write_json') as wj:
        ed.return_value = None
        wj.return_value = tmp_path / 'file.json'
        path = metric_utils.save_metrics_json(metrics, task_dir, 'op', 'field', BadWriter(), None, False)
        assert str(path).endswith('.json')

def test_save_metrics_json_file_exception(tmp_path):
    metrics = {'foo': 'bar'}
    task_dir = tmp_path
    with patch('pamola_core.transformations.commons.metric_utils.ensure_directory') as ed, \
         patch('pamola_core.transformations.commons.metric_utils.write_json', side_effect=Exception('fail')):
        ed.return_value = None
        path = metric_utils.save_metrics_json(metrics, task_dir, 'op', 'field', None, None, False)
        assert str(path).endswith('.json')

if __name__ == "__main__":
    pytest.main()
