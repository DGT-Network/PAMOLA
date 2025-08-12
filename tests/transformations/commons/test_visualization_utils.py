"""
Tests for the visualization_utils module in the PAMOLA.CORE package.

These tests verify the functionality of data visualization utilities including
field count comparison, record count comparison, data distribution comparison, dataset overview, and visualization generation.

Run with:
    pytest tests/transformations/commons/test_visualization_utils.py
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from pamola_core.transformations.commons import visualization_utils as vu

@pytest.fixture
def sample_dfs():
    df_orig = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['x', 'y', 'z', 'x'],
        'C': [True, False, True, False],
        'D': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
    })
    df_trans = pd.DataFrame({
        'A': [10, 20, 30],
        'B': ['y', 'y', 'z'],
        'C': [False, True, True],
        'E': [100, 200, 300]
    })
    return df_orig, df_trans

@pytest.fixture
def sample_series():
    orig = pd.Series([1, 2, 2, 3, 4, None])
    trans = pd.Series([2, 2, 3, 3, 4, 4])
    return orig, trans

@pytest.fixture
def sample_cat_series():
    orig = pd.Series(['a', 'b', 'a', 'c', None])
    trans = pd.Series(['a', 'b', 'b', 'c', 'c'])
    return orig, trans

@pytest.fixture
def tmp_path_obj(tmp_path):
    return tmp_path

def test_generate_visualization_filename_basic():
    fname = vu.generate_visualization_filename('op', 'hist', 'jpg', 'join', True)
    assert fname.startswith('op_hist_join_') and fname.endswith('.jpg')

def test_generate_visualization_filename_no_timestamp():
    fname = vu.generate_visualization_filename('op', 'bar', 'png', None, False)
    assert fname == 'op_bar.png'

def test_generate_visualization_filename_defaults():
    fname = vu.generate_visualization_filename('op', 'pie')
    assert fname.startswith('op_pie') and fname.endswith('.png')

def test_create_field_count_comparison_normal(sample_dfs, tmp_path_obj):
    df_orig, df_trans = sample_dfs
    result = vu.create_field_count_comparison(df_orig, df_trans, 'testop', tmp_path_obj)
    assert result['original_field_count'] == 4
    assert result['transformed_field_count'] == 4
    assert 'added_fields' in result and 'removed_fields' in result
    assert isinstance(result['added_fields'], list)
    assert isinstance(result['removed_fields'], list)
    assert 'chart_recommendation' in result

def test_create_field_count_comparison_empty():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame({'A': [1]})
    result = vu.create_field_count_comparison(df1, df2, 'op', Path('.'))
    assert result['original_field_count'] == 0
    assert result['transformed_field_count'] == 1
    assert result['percent_change'] == 0

def test_create_field_count_comparison_invalid():
    with pytest.raises(AttributeError):
        vu.create_field_count_comparison(None, None, 'op', Path('.'))

def test_create_record_count_comparison_normal(sample_dfs, tmp_path_obj):
    df_orig, df_trans = sample_dfs
    result = vu.create_record_count_comparison(df_orig, {'out1': df_trans}, 'op', tmp_path_obj)
    assert result['original_record_count'] == 4
    assert result['transformed_record_counts']['out1'] == 3
    assert 'chart_recommendation' in result

def test_create_record_count_comparison_multiple(sample_dfs, tmp_path_obj):
    df_orig, df_trans = sample_dfs
    result = vu.create_record_count_comparison(df_orig, {'a': df_trans, 'b': df_orig}, 'op', tmp_path_obj)
    assert 'additional_chart_recommendation' in result

def test_create_record_count_comparison_empty():
    df = pd.DataFrame()
    result = vu.create_record_count_comparison(df, {}, 'op', Path('.'))
    assert result['original_record_count'] == 0
    assert result['total_transformed_records'] == 0
    assert result['percent_change'] == 0.0

def test_create_data_distribution_comparison_numeric(sample_series, tmp_path_obj):
    orig, trans = sample_series
    result = vu.create_data_distribution_comparison(orig, trans, 'A', 'op', tmp_path_obj)
    assert result['is_numeric'] is True
    assert 'original' in result['statistics']
    assert 'Transformed' in result['plot_data']
    assert isinstance(result['plot_data']['Original'], list)

def test_create_data_distribution_comparison_categorical(sample_cat_series, tmp_path_obj):
    orig, trans = sample_cat_series
    result = vu.create_data_distribution_comparison(orig, trans, 'B', 'op', tmp_path_obj)
    assert result['is_numeric'] is False
    assert isinstance(result['plot_data']['Original'], dict)
    assert 'chart_recommendation' in result

def test_create_data_distribution_comparison_empty(tmp_path_obj):
    orig = pd.Series([], dtype=float)
    trans = pd.Series([], dtype=float)
    result = vu.create_data_distribution_comparison(orig, trans, 'A', 'op', tmp_path_obj)
    assert result['statistics']['original']['count'] == 0
    assert result['plot_data']['Original'] == []

def test_create_data_distribution_comparison_invalid():
    with pytest.raises(Exception):
        vu.create_data_distribution_comparison(None, None, 'A', 'op', Path('.'))

def test_create_dataset_overview_normal(sample_dfs, tmp_path_obj):
    df_orig, _ = sample_dfs
    result = vu.create_dataset_overview(df_orig, 'title', tmp_path_obj)
    assert result['record_count'] == 4
    assert 'numeric_stats' in result
    assert 'categorical_stats' in result
    assert 'datetime_stats' in result
    assert 'boolean_stats' in result
    assert isinstance(result['chart_recommendations'], list)

def test_create_dataset_overview_empty(tmp_path_obj):
    df = pd.DataFrame()
    result = vu.create_dataset_overview(df, 'title', tmp_path_obj)
    assert result['record_count'] == 0
    assert result['field_count'] == 0
    assert result['numeric_stats'] == {}

def test_generate_dataset_overview_vis_mocks(sample_dfs, tmp_path_obj):
    df_orig, _ = sample_dfs
    with mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='mock_path'):
        paths = vu.generate_dataset_overview_vis(df_orig, 'op', 'original', 'field', tmp_path_obj, 'ts')
        assert isinstance(paths, dict)
        assert any('bar_chart' in k for k in paths)

def test_generate_data_distribution_comparison_vis_numeric(sample_series, tmp_path_obj):
    orig, trans = sample_series
    with mock.patch('pamola_core.utils.visualization.create_histogram', return_value='mock_hist_path'):
        paths = vu.generate_data_distribution_comparison_vis(orig, trans, 'field', 'op', tmp_path_obj, 'ts')
        assert 'numeric_comparison_histogram' in paths

def test_generate_data_distribution_comparison_vis_categorical(sample_cat_series, tmp_path_obj):
    orig, trans = sample_cat_series
    with mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='mock_bar_path'):
        paths = vu.generate_data_distribution_comparison_vis(orig, trans, 'field', 'op', tmp_path_obj, 'ts')
        assert 'category_comparison_bar_chart' in paths

def test_generate_record_count_comparison_vis_pie(sample_dfs, tmp_path_obj):
    df_orig, df_trans = sample_dfs
    with mock.patch('pamola_core.utils.visualization.create_pie_chart', return_value='mock_pie_path'):
        paths = vu.generate_record_count_comparison_vis(df_orig, {'a': df_trans, 'b': df_orig}, 'field', 'op', tmp_path_obj, 'ts')
        assert 'record_count_distribution_pie_chart' in paths

def test_generate_record_count_comparison_vis_bar(sample_dfs, tmp_path_obj):
    df_orig, df_trans = sample_dfs
    with mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='mock_bar_path'):
        paths = vu.generate_record_count_comparison_vis(df_orig, {'a': df_trans}, 'field', 'op', tmp_path_obj, 'ts')
        assert 'record_count_comparison_bar_chart' in paths

def test_generate_field_count_comparison_vis(sample_dfs, tmp_path_obj):
    df_orig, df_trans = sample_dfs
    with mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='mock_bar_path'):
        paths = vu.generate_field_count_comparison_vis(df_orig, df_trans, 'field', 'op', tmp_path_obj, 'ts')
        assert 'field_count_comparison_bar_chart' in paths

def test_sample_large_dataset_small():
    s = pd.Series([1, 2, 3])
    out = vu.sample_large_dataset(s, max_samples=10)
    assert out.equals(s)

def test_sample_large_dataset_large():
    s = pd.Series(range(10000))
    out = vu.sample_large_dataset(s, max_samples=100)
    assert len(out) == 100
    assert set(out).issubset(set(s))

def test_sample_large_dataset_invalid():
    with pytest.raises(Exception):
        vu.sample_large_dataset(None)

if __name__ == "__main__":
    pytest.main()
