"""
Tests for the merging_utils module in the PAMOLA.CORE package.

These tests verify the functionality of data merging utilities including
record overlap, dataset size comparison, field overlap, and join type distribution.

Run with:
    pytest tests/transformations/commons/test_merging_utils.py
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest import mock
from pamola_core.transformations.commons import merging_utils

class TestMergingUtils:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        self.left_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'val': ['a', 'b', 'c', 'd'],
            'common': [10, 20, 30, 40],
        })
        self.right_df = pd.DataFrame({
            'key': [3, 4, 5, 6],
            'val2': ['x', 'y', 'z', 'w'],
            'common': [30, 40, 50, 60],
        })
        self.merged_df = pd.merge(self.left_df, self.right_df, left_on='id', right_on='key', how='outer', suffixes=('_l', '_r'))
        self.task_dir = tmp_path
        self.timestamp = '20250101T000000'
        self.operation_name = 'test_op'
        self.field_label = 'test_field'
        self.left_key = 'id'
        self.right_key = 'key'
        self.join_type = 'outer'
        yield

    def test_create_record_overlap_data_valid(self):
        result = merging_utils.create_record_overlap_data(
            self.left_df, self.right_df, 'id', 'key', self.operation_name, self.task_dir
        )
        assert result['operation_name'] == self.operation_name
        assert result['left_count'] == 4
        assert result['right_count'] == 4
        assert result['overlap_count'] == 2
        assert result['only_left_count'] == 2
        assert result['only_right_count'] == 2
        assert isinstance(result['left_keys'], set)
        assert isinstance(result['right_keys'], set)
        assert 'chart_recommendation' in result

    def test_create_record_overlap_data_empty(self):
        left = pd.DataFrame({'id': []})
        right = pd.DataFrame({'key': []})
        result = merging_utils.create_record_overlap_data(left, right, 'id', 'key', self.operation_name, self.task_dir)
        assert result['left_count'] == 0
        assert result['right_count'] == 0
        assert result['overlap_count'] == 0
        assert result['only_left_count'] == 0
        assert result['only_right_count'] == 0

    def test_create_record_overlap_data_invalid_column(self):
        with pytest.raises(KeyError):
            merging_utils.create_record_overlap_data(self.left_df, self.right_df, 'bad', 'key', self.operation_name, self.task_dir)
        with pytest.raises(KeyError):
            merging_utils.create_record_overlap_data(self.left_df, self.right_df, 'id', 'bad', self.operation_name, self.task_dir)

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_venn_diagram', return_value=Path('dummy.png'))
    def test_generate_record_overlap_vis_valid(self, mock_venn):
        vis_paths = merging_utils.generate_record_overlap_vis(
            self.left_df, self.right_df, 'id', 'key', self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'record_overlap_venn' in vis_paths
        assert vis_paths['record_overlap_venn'] == Path('dummy.png')
        mock_venn.assert_called_once()

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_venn_diagram', return_value=Path('dummy.png'))
    def test_generate_record_overlap_vis_empty(self, mock_venn):
        left = pd.DataFrame({'id': []})
        right = pd.DataFrame({'key': []})
        vis_paths = merging_utils.generate_record_overlap_vis(
            left, right, 'id', 'key', self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'record_overlap_venn' in vis_paths
        mock_venn.assert_called_once()

    def test_create_dataset_size_comparison_valid(self):
        result = merging_utils.create_dataset_size_comparison(
            self.left_df, self.right_df, self.merged_df, self.operation_name, self.task_dir
        )
        assert result['left_size'] == 4
        assert result['right_size'] == 4
        assert result['merged_size'] == 6
        assert 'chart_recommendation' in result

    def test_create_dataset_size_comparison_empty(self):
        left = pd.DataFrame({'id': []})
        right = pd.DataFrame({'key': []})
        merged = pd.DataFrame({'id': [], 'key': []})
        result = merging_utils.create_dataset_size_comparison(left, right, merged, self.operation_name, self.task_dir)
        assert result['left_size'] == 0
        assert result['right_size'] == 0
        assert result['merged_size'] == 0

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_bar_plot', return_value=Path('dummy_bar.png'))
    def test_generate_dataset_size_comparison_vis_valid(self, mock_bar):
        vis_paths = merging_utils.generate_dataset_size_comparison_vis(
            self.left_df, self.right_df, self.merged_df, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'dataset_size_comparison_bar_chart' in vis_paths
        assert vis_paths['dataset_size_comparison_bar_chart'] == Path('dummy_bar.png')
        mock_bar.assert_called_once()

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_bar_plot', return_value=Path('dummy_bar.png'))
    def test_generate_dataset_size_comparison_vis_empty(self, mock_bar):
        left = pd.DataFrame({'id': []})
        right = pd.DataFrame({'key': []})
        merged = pd.DataFrame({'id': [], 'key': []})
        vis_paths = merging_utils.generate_dataset_size_comparison_vis(
            left, right, merged, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'dataset_size_comparison_bar_chart' in vis_paths
        mock_bar.assert_called_once()

    def test_create_field_overlap_data_valid(self):
        result = merging_utils.create_field_overlap_data(
            self.left_df, self.right_df, self.operation_name, self.task_dir
        )
        assert result['left_field_count'] == 3
        assert result['right_field_count'] == 3
        assert result['overlap_field_count'] == 1
        assert 'chart_recommendation' in result
        assert isinstance(result['left_fields'], set)
        assert isinstance(result['right_fields'], set)

    def test_create_field_overlap_data_empty(self):
        left = pd.DataFrame()
        right = pd.DataFrame()
        result = merging_utils.create_field_overlap_data(left, right, self.operation_name, self.task_dir)
        assert result['left_field_count'] == 0
        assert result['right_field_count'] == 0
        assert result['overlap_field_count'] == 0

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_venn_diagram', return_value=Path('dummy_field_venn.png'))
    def test_generate_field_overlap_vis_valid(self, mock_venn):
        vis_paths = merging_utils.generate_field_overlap_vis(
            self.left_df, self.right_df, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'field_overlap_venn' in vis_paths
        assert vis_paths['field_overlap_venn'] == Path('dummy_field_venn.png')
        mock_venn.assert_called_once()

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_venn_diagram', return_value=Path('dummy_field_venn.png'))
    def test_generate_field_overlap_vis_empty(self, mock_venn):
        left = pd.DataFrame()
        right = pd.DataFrame()
        vis_paths = merging_utils.generate_field_overlap_vis(
            left, right, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'field_overlap_venn' in vis_paths
        mock_venn.assert_called_once()

    def test_create_join_type_distribution_data_valid(self):
        merged = pd.DataFrame({
            'id': [1, 2, 3, 4, None],
            'key': [None, 2, 3, None, 5],
        })
        result = merging_utils.create_join_type_distribution_data(
            merged, 'id', 'key', 'outer', self.operation_name, self.task_dir
        )
        assert result['matched_count'] == 2
        assert result['only_left_count'] == 2
        assert result['only_right_count'] == 1
        assert result['join_type'] == 'outer'
        assert 'chart_recommendation' in result

    def test_create_join_type_distribution_data_empty(self):
        merged = pd.DataFrame({'id': [], 'key': []})
        result = merging_utils.create_join_type_distribution_data(
            merged, 'id', 'key', 'outer', self.operation_name, self.task_dir
        )
        assert result['matched_count'] == 0
        assert result['only_left_count'] == 0
        assert result['only_right_count'] == 0

    def test_create_join_type_distribution_data_invalid_column(self):
        with pytest.raises(KeyError):
            merging_utils.create_join_type_distribution_data(self.merged_df, 'bad', 'key', 'outer', self.operation_name, self.task_dir)
        with pytest.raises(KeyError):
            merging_utils.create_join_type_distribution_data(self.merged_df, 'id', 'bad', 'outer', self.operation_name, self.task_dir)

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_pie_chart', return_value=Path('dummy_pie.png'))
    def test_generate_join_type_distribution_vis_valid(self, mock_pie):
        vis_paths = merging_utils.generate_join_type_distribution_vis(
            self.merged_df, 'id', 'key', self.join_type, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'join_type_distribution_pie_chart' in vis_paths
        assert vis_paths['join_type_distribution_pie_chart'] == Path('dummy_pie.png')
        mock_pie.assert_called_once()

    @mock.patch('pamola_core.transformations.commons.merging_utils.create_pie_chart', return_value=Path('dummy_pie.png'))
    def test_generate_join_type_distribution_vis_empty(self, mock_pie):
        merged = pd.DataFrame({'id': [], 'key': []})
        vis_paths = merging_utils.generate_join_type_distribution_vis(
            merged, 'id', 'key', self.join_type, self.field_label, self.operation_name, self.task_dir, self.timestamp
        )
        assert 'join_type_distribution_pie_chart' in vis_paths
        mock_pie.assert_called_once()

    def test_generate_record_overlap_vis_invalid_column(self):
        with pytest.raises(KeyError):
            merging_utils.generate_record_overlap_vis(
                self.left_df, self.right_df, 'bad', 'key', self.field_label, self.operation_name, self.task_dir, self.timestamp
            )
        with pytest.raises(KeyError):
            merging_utils.generate_record_overlap_vis(
                self.left_df, self.right_df, 'id', 'bad', self.field_label, self.operation_name, self.task_dir, self.timestamp
            )

    def test_generate_field_overlap_vis_invalid_input(self):
        # Should not raise, but should handle empty DataFrames
        left = pd.DataFrame()
        right = pd.DataFrame()
        with mock.patch('pamola_core.transformations.commons.merging_utils.create_venn_diagram', return_value=Path('dummy_field_venn.png')) as mock_venn:
            vis_paths = merging_utils.generate_field_overlap_vis(
                left, right, self.field_label, self.operation_name, self.task_dir, self.timestamp
            )
            assert 'field_overlap_venn' in vis_paths
            mock_venn.assert_called_once()

    def test_generate_dataset_size_comparison_vis_invalid_input(self):
        left = pd.DataFrame()
        right = pd.DataFrame()
        merged = pd.DataFrame()
        with mock.patch('pamola_core.transformations.commons.merging_utils.create_bar_plot', return_value=Path('dummy_bar.png')) as mock_bar:
            vis_paths = merging_utils.generate_dataset_size_comparison_vis(
                left, right, merged, self.field_label, self.operation_name, self.task_dir, self.timestamp
            )
            assert 'dataset_size_comparison_bar_chart' in vis_paths
            mock_bar.assert_called_once()

    def test_generate_join_type_distribution_vis_invalid_column(self):
        with pytest.raises(KeyError):
            merging_utils.generate_join_type_distribution_vis(
                self.merged_df, 'bad', 'key', self.join_type, self.field_label, self.operation_name, self.task_dir, self.timestamp
            )
        with pytest.raises(KeyError):
            merging_utils.generate_join_type_distribution_vis(
                self.merged_df, 'id', 'bad', self.join_type, self.field_label, self.operation_name, self.task_dir, self.timestamp
            )

if __name__ == "__main__":
    pytest.main()
