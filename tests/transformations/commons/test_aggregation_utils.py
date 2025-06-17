"""
Tests for the aggregation_utils module in the PAMOLA.CORE package.

These tests verify the functionality of data aggregation utilities including
grouping, aggregation, and visualization helpers.

Run with:
    pytest tests/transformations/commons/test_aggregation_utils.py
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from pamola_core.transformations.commons import aggregation_utils

class TestAggregationUtils:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.tmp_path = tmp_path
        self.df = pd.DataFrame({
            'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
            'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
            'C': [1, 2, 3, 4, 5, 6, 7, 8],
            'D': [2.0, 5., 8., 1., 2., 9., 3., 6.]
        })
        self.agg_df = self.df.groupby(['A', 'B'], as_index=False).agg(count=('C', 'count'), sum_C=('C', 'sum'))
        self.group_by_fields = ['A', 'B']
        self.agg_fields = ['sum_C']
        self.operation_name = 'sum'
        self.field_label = 'C'
        self.task_dir = tmp_path
        self.timestamp = '20250101'

    def test_create_record_count_per_group_data_valid(self):
        result = aggregation_utils.create_record_count_per_group_data(
            self.agg_df, self.group_by_fields, self.operation_name, self.task_dir
        )
        assert 'group_labels' in result
        assert 'record_count_per_group' in result
        assert isinstance(result['group_labels'], list)
        assert isinstance(result['record_count_per_group'], dict)
        assert result['chart_recommendation']

    def test_create_record_count_per_group_data_edge_empty(self):
        empty_df = pd.DataFrame(columns=['A', 'B', 'count'])
        with mock.patch('pandas.DataFrame.__getitem__', side_effect=lambda key: pd.DataFrame() if isinstance(key, list) else pd.Series(dtype=object)):
            result = aggregation_utils.create_record_count_per_group_data(
                empty_df, ['A', 'B'], 'sum', self.task_dir
            )
        assert result['group_labels'] == []
        assert result['record_count_per_group'] == {}

    def test_create_record_count_per_group_data_invalid(self):
        with pytest.raises(KeyError):
            aggregation_utils.create_record_count_per_group_data(
                pd.DataFrame({'X': [1]}), ['A'], 'sum', self.task_dir
            )

    @mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='Error saving file: Please upgrade kaleido\n')
    def test_generate_record_count_per_group_vis_valid(self, mock_bar):
        vis_paths = aggregation_utils.generate_record_count_per_group_vis(
            self.agg_df, self.group_by_fields, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'record_count_per_group_bar_chart' in vis_paths
        assert isinstance(vis_paths['record_count_per_group_bar_chart'], str)

    @mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='Error saving file: Please upgrade kaleido\n')
    def test_generate_record_count_per_group_vis_empty(self, mock_bar):
        empty_df = pd.DataFrame(columns=['A', 'B', 'count'])
        with mock.patch('pandas.DataFrame.__getitem__', side_effect=lambda key: pd.DataFrame() if isinstance(key, list) else pd.Series(dtype=object)):
            vis_paths = aggregation_utils.generate_record_count_per_group_vis(
                empty_df, ['A', 'B'], self.field_label, self.operation_name, self.task_dir, self.timestamp
            )
        assert 'record_count_per_group_bar_chart' in vis_paths
        assert isinstance(vis_paths['record_count_per_group_bar_chart'], str)

    def test_create_aggregation_comparison_data_valid(self):
        agg_df = self.df.groupby(['A', 'B'], as_index=False).agg(sum_C=('C', 'sum'), mean_D=('D', 'mean'))
        result = aggregation_utils.create_aggregation_comparison_data(
            agg_df, ['A', 'B'], ['sum_C', 'mean_D'], 'sum', self.task_dir
        )
        assert 'group_labels' in result
        assert 'agg_comparison' in result
        assert isinstance(result['agg_comparison'], dict)
        assert result['chart_recommendation']

    def test_create_aggregation_comparison_data_edge_empty(self):
        empty_df = pd.DataFrame(columns=['A', 'B', 'sum_C'])
        with mock.patch('pandas.DataFrame.__getitem__', side_effect=lambda key: pd.DataFrame() if isinstance(key, list) else pd.Series(dtype=object)):
            result = aggregation_utils.create_aggregation_comparison_data(
                empty_df, ['A', 'B'], ['sum_C'], 'sum', self.task_dir
            )
        assert result['group_labels'] == []
        assert result['agg_comparison'] == {'sum_C': []}

    def test_create_aggregation_comparison_data_invalid(self):
        with pytest.raises(KeyError):
            aggregation_utils.create_aggregation_comparison_data(
                pd.DataFrame({'X': [1]}), ['A'], ['sum_C'], 'sum', self.task_dir
            )

    @mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='Error saving file: Please upgrade kaleido\n')
    def test_generate_aggregation_comparison_vis_valid(self, mock_bar):
        agg_df = self.df.groupby(['A', 'B'], as_index=False).agg(sum_C=('C', 'sum'))
        vis_paths = aggregation_utils.generate_aggregation_comparison_vis(
            agg_df, ['A', 'B'], ['sum_C'], self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert any(k.startswith('agg_comparison_sum_C_bar_chart') for k in vis_paths)

    @mock.patch('pamola_core.utils.visualization.create_bar_plot', return_value='Error saving file: Please upgrade kaleido\n')
    def test_generate_aggregation_comparison_vis_empty(self, mock_bar):
        empty_df = pd.DataFrame(columns=['A', 'B', 'sum_C'])
        with mock.patch('pandas.DataFrame.__getitem__', side_effect=lambda key: pd.DataFrame() if isinstance(key, list) else pd.Series(dtype=object)):
            vis_paths = aggregation_utils.generate_aggregation_comparison_vis(
                empty_df, ['A', 'B'], ['sum_C'], self.field_label, self.operation_name, self.task_dir, self.timestamp
            )
        assert any(k.startswith('agg_comparison_sum_C_bar_chart') for k in vis_paths)
        for v in vis_paths.values():
            assert isinstance(v, str)

    def test_create_group_size_distribution_data_valid(self):
        result = aggregation_utils.create_group_size_distribution_data(
            self.agg_df, self.group_by_fields, self.operation_name, self.task_dir
        )
        assert 'group_size_distribution' in result
        assert isinstance(result['group_size_distribution'], dict)
        assert result['chart_recommendation']

    def test_create_group_size_distribution_data_edge_empty(self):
        empty_df = pd.DataFrame(columns=['A', 'B', 'count'])
        result = aggregation_utils.create_group_size_distribution_data(
            empty_df, ['A', 'B'], self.operation_name, self.task_dir
        )
        assert result['group_size_distribution'] == {}

    def test_create_group_size_distribution_data_invalid(self):
        with pytest.raises(KeyError):
            aggregation_utils.create_group_size_distribution_data(
                pd.DataFrame({'X': [1]}), ['A'], self.operation_name, self.task_dir
            )

    @mock.patch('pamola_core.utils.visualization.create_histogram', return_value='Error saving file: Please upgrade kaleido\n')
    def test_generate_group_size_distribution_vis_valid(self, mock_hist):
        vis_paths = aggregation_utils.generate_group_size_distribution_vis(
            self.agg_df, self.group_by_fields, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'group_size_distribution_histogram' in vis_paths
        assert isinstance(vis_paths['group_size_distribution_histogram'], str)

    @mock.patch('pamola_core.utils.visualization.create_histogram', return_value='Error saving file: Please upgrade kaleido\n')
    def test_generate_group_size_distribution_vis_empty(self, mock_hist):
        empty_df = pd.DataFrame(columns=['A', 'B', 'count'])
        vis_paths = aggregation_utils.generate_group_size_distribution_vis(
            empty_df, ['A', 'B'], self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'group_size_distribution_histogram' in vis_paths

    def test_build_aggregation_dict_valid(self):
        aggs = {'C': ['sum', 'mean']}
        # Only use valid string names for aggregation functions
        result = aggregation_utils.build_aggregation_dict(aggs)
        assert 'C' in result
        assert any(callable(f) for f in result['C'])

    def test_build_aggregation_dict_edge_empty(self):
        result = aggregation_utils.build_aggregation_dict()
        assert result == {}

    def test_build_aggregation_dict_invalid(self):
        aggs = {'C': ['not_a_func']}
        with pytest.raises(ValueError):
            aggregation_utils.build_aggregation_dict(aggs)

    def test_flatten_multiindex_columns_valid(self):
        columns = pd.MultiIndex.from_tuples([('A', 'sum'), ('B', 'mean')])
        result = aggregation_utils.flatten_multiindex_columns(columns)
        assert result == ['A_sum', 'B_mean']

    def test_flatten_multiindex_columns_edge(self):
        columns = pd.MultiIndex.from_tuples([('A', '')])
        result = aggregation_utils.flatten_multiindex_columns(columns)
        assert result == ['A']

    def test_get_aggregation_function_valid(self):
        func = aggregation_utils._get_aggregation_function('sum')
        assert callable(func)

    def test_get_aggregation_function_invalid(self):
        with pytest.raises(ValueError):
            aggregation_utils._get_aggregation_function('not_a_func')

    def test_is_dask_compatible_function_str(self):
        assert aggregation_utils.is_dask_compatible_function('sum') is True
        assert aggregation_utils.is_dask_compatible_function('not_a_func') is False

    def test_is_dask_compatible_function_callable(self):
        import numpy as np
        assert aggregation_utils.is_dask_compatible_function(np.sum) is True
        assert aggregation_utils.is_dask_compatible_function(lambda x: x) is False

    def test_apply_custom_aggregations_post_dask_valid(self):
        df = pd.DataFrame({'A': ['foo', 'foo', 'bar'], 'C': [1, 2, 3]})
        result_df = df.groupby('A', as_index=False).agg({'C': 'sum'})
        custom_agg_dict = {'C': [lambda x: x.max()]}
        out = aggregation_utils.apply_custom_aggregations_post_dask(
            df, result_df, custom_agg_dict, ['A']
        )
        assert 'C_<lambda>' in out.columns

    def test_apply_custom_aggregations_post_dask_empty(self):
        df = pd.DataFrame({'A': [], 'C': []})
        result_df = pd.DataFrame({'A': [], 'C': []})
        custom_agg_dict = {'C': [lambda x: x.max()]}
        out = aggregation_utils.apply_custom_aggregations_post_dask(
            df, result_df, custom_agg_dict, ['A']
        )
        assert isinstance(out, pd.DataFrame)

    def test_apply_custom_aggregations_post_dask_invalid(self):
        df = pd.DataFrame({'A': [1, 2], 'C': [3, 4]})
        result_df = pd.DataFrame({'A': [1, 2], 'C': [3, 4]})
        custom_agg_dict = {'C': ['not_callable']}
        out = aggregation_utils.apply_custom_aggregations_post_dask(
            df, result_df, custom_agg_dict, ['A']
        )
        assert isinstance(out, pd.DataFrame)

if __name__ == "__main__":
    pytest.main()
