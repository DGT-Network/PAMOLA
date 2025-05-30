import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from pamola_core.profiling.commons import correlation_utils as cu

class TestCorrelationUtils:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Simple DataFrames for most tests
        self.df_num = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [1, 1, 2, 2, 3],
            'd': ['x', 'y', 'x', 'y', 'z'],
            'e': [1, 0, 1, 0, 1],
            'f': [np.nan, 1, 2, 3, 4],
        })
        self.df_empty = pd.DataFrame({'a': [], 'b': []})
        self.df_single = pd.DataFrame({'a': [1], 'b': [2]})
        self.df_nulls = pd.DataFrame({'a': [1, None, 3], 'b': [None, 2, 3]})
        self.df_cat = pd.DataFrame({'cat1': ['a', 'b', 'a', 'b', 'c'], 'cat2': ['x', 'x', 'y', 'y', 'z']})

    def test_analyze_correlation_valid_numeric(self):
        result = cu.analyze_correlation(self.df_num, 'a', 'b')
        assert result['method'] == 'pearson'
        assert np.isclose(result['correlation_coefficient'], 1.0)
        assert result['field1'] == 'a'
        assert result['field2'] == 'b'
        assert result['sample_size'] == 5
        assert 'plot_data' in result

    def test_analyze_correlation_valid_categorical(self):
        result = cu.analyze_correlation(self.df_cat, 'cat1', 'cat2')
        assert result['method'] == 'cramers_v'
        assert 0.0 <= result['correlation_coefficient'] <= 1.0
        assert result['field1_type'] == 'categorical'
        assert result['field2_type'] == 'categorical'

    def test_analyze_correlation_point_biserial(self):
        # The default method is 'pearson' for two numeric fields, even if one is binary.
        # To test 'point_biserial', we must specify the method explicitly.
        result = cu.analyze_correlation(self.df_num, 'e', 'a', method='point_biserial')
        assert result['method'] == 'point_biserial'
        assert isinstance(result['correlation_coefficient'], float)
        assert 'p_value' in result

    def test_analyze_correlation_correlation_ratio(self):
        result = cu.analyze_correlation(self.df_num, 'd', 'a')
        assert result['method'] == 'correlation_ratio'
        assert 0.0 <= result['correlation_coefficient'] <= 1.0

    def test_analyze_correlation_invalid_field(self):
        result = cu.analyze_correlation(self.df_num, 'not_a_field', 'b')
        assert 'error' in result
        result2 = cu.analyze_correlation(self.df_num, 'a', 'not_b')
        assert 'error' in result2

    def test_analyze_correlation_empty_df(self):
        result = cu.analyze_correlation(self.df_empty, 'a', 'b')
        assert 'error' in result

    def test_analyze_correlation_all_nulls(self):
        df = pd.DataFrame({'a': [None, None], 'b': [None, None]})
        result = cu.analyze_correlation(df, 'a', 'b')
        assert 'error' in result

    def test_analyze_correlation_with_mvf_parser(self):
        mvf_parser = Mock(return_value=['x', 'y'])
        df = pd.DataFrame({'a': ['1,2', '3,4'], 'b': ['x', 'y']})
        result = cu.analyze_correlation(df, 'a', 'b', mvf_parser=mvf_parser)
        assert 'plot_data' in result
        assert mvf_parser.called

    def test_analyze_correlation_matrix_valid(self):
        fields = ['a', 'b', 'c']
        result = cu.analyze_correlation_matrix(self.df_num, fields)
        assert 'correlation_matrix' in result
        assert 'methods' in result
        assert 'significant_correlations' in result
        assert result['fields_analyzed'] == 3

    def test_analyze_correlation_matrix_missing_field(self):
        result = cu.analyze_correlation_matrix(self.df_num, ['a', 'not_a'])
        assert 'error' in result

    def test_analyze_correlation_matrix_max_fields(self):
        fields = [f'f{i}' for i in range(25)]
        df = pd.DataFrame({f'f{i}': np.random.rand(10) for i in range(25)})
        result = cu.analyze_correlation_matrix(df, fields, max_fields=10)
        assert result['fields_analyzed'] == 10

    def test_detect_correlation_type(self):
        assert cu.detect_correlation_type(self.df_num, 'a', 'b') == 'pearson'
        assert cu.detect_correlation_type(self.df_cat, 'cat1', 'cat2') == 'cramers_v'
        # For two numeric fields, even if one is binary, the default is 'pearson'.
        # To test 'point_biserial', use a categorical binary field and a numeric field.
        df_bin_cat = pd.DataFrame({'cat': ['yes', 'no', 'yes', 'no'], 'num': [1, 2, 3, 4]})
        assert cu.detect_correlation_type(df_bin_cat, 'cat', 'num') == 'point_biserial'
        assert cu.detect_correlation_type(self.df_num, 'd', 'a') == 'correlation_ratio'

    def test_calculate_correlation_methods(self):
        # Pearson
        res = cu.calculate_correlation(self.df_num, 'a', 'b', method='pearson')
        assert np.isclose(res['coefficient'], 1.0)
        # Spearman
        res = cu.calculate_correlation(self.df_num, 'a', 'b', method='spearman')
        assert np.isclose(res['coefficient'], 1.0)
        # Cramer's V
        res = cu.calculate_correlation(self.df_cat, 'cat1', 'cat2', method='cramers_v')
        assert 0.0 <= res['coefficient'] <= 1.0
        # Point biserial
        res = cu.calculate_correlation(self.df_num, 'e', 'a', method='point_biserial')
        assert isinstance(res['coefficient'], float)
        # Correlation ratio
        res = cu.calculate_correlation(self.df_num, 'd', 'a', method='correlation_ratio')
        assert 0.0 <= res['coefficient'] <= 1.0
        # Unknown method
        res = cu.calculate_correlation(self.df_num, 'a', 'b', method='unknown')
        assert res['method'] in ['pearson', 'unknown']

    def test_calculate_cramers_v(self):
        v = cu.calculate_cramers_v(self.df_cat['cat1'], self.df_cat['cat2'])
        assert 0.0 <= v <= 1.0
        # Edge: one category only
        v2 = cu.calculate_cramers_v(pd.Series(['a', 'a', 'a']), pd.Series(['x', 'y', 'z']))
        assert v2 == 0.0

    def test_calculate_correlation_ratio(self):
        eta = cu.calculate_correlation_ratio(self.df_cat['cat1'], pd.Series([1, 2, 3, 4, 5]))
        assert 0.0 <= eta <= 1.0
        # Edge: all values same
        eta2 = cu.calculate_correlation_ratio(self.df_cat['cat1'], pd.Series([1, 1, 1, 1, 1]))
        assert eta2 == 0.0

    def test_calculate_point_biserial(self):
        coef, p = cu.calculate_point_biserial(pd.Series([0, 1, 0, 1]), pd.Series([1, 2, 3, 4]))
        assert isinstance(coef, float)
        assert isinstance(p, float)

    def test_interpret_correlation(self):
        assert cu.interpret_correlation(0.05, 'pearson').startswith('Negligible')
        assert cu.interpret_correlation(0.25, 'pearson').startswith('Weak')
        assert cu.interpret_correlation(0.6, 'pearson').startswith('Strong')
        assert cu.interpret_correlation(-0.95, 'pearson').startswith('Near')
        assert cu.interpret_correlation(0.05, 'cramers_v').startswith('Negligible')
        assert cu.interpret_correlation(0.5, 'cramers_v').startswith('Relatively')

    def test_handle_null_values(self):
        df = pd.DataFrame({'a': [1, None, 3], 'b': [None, 2, 3]})
        # Drop
        df1, stats1 = cu.handle_null_values(df, 'drop')
        assert len(df1) == 1
        assert stats1['rows_removed'] == 2
        # Fill
        df2, stats2 = cu.handle_null_values(df, 'fill')
        assert len(df2) == 3
        assert stats2['rows_removed'] == 0
        # Pairwise
        df3, stats3 = cu.handle_null_values(df, 'pairwise')
        assert len(df3) == 3
        # Unknown method
        df4, stats4 = cu.handle_null_values(df, 'unknown')
        assert len(df4) == 1

    def test_find_significant_correlations(self):
        mat = pd.DataFrame({
            'a': [1.0, 0.5, 0.2],
            'b': [0.5, 1.0, 0.4],
            'c': [0.2, 0.4, 1.0]
        }, index=['a', 'b', 'c'])
        pvals = {'a_b': 0.01, 'a_c': 0.2, 'b_c': 0.03}
        sig = cu.find_significant_correlations(mat, threshold=0.3, p_values=pvals)
        assert any(s['statistically_significant'] for s in sig if 'statistically_significant' in s)

    def test_prepare_mvf_fields(self):
        mvf_parser = Mock(return_value=['x', 'y'])
        df = pd.DataFrame({'a': ['1,2', '3,4'], 'b': ['x', 'y']})
        df2 = cu.prepare_mvf_fields(df, 'a', 'b', mvf_parser)
        assert 'a' in df2.columns
        assert mvf_parser.called

    def test_prepare_plot_data(self):
        # Scatter
        data = cu.prepare_plot_data(self.df_num, 'a', 'b', True, True)
        assert data['type'] == 'scatter'
        # Boxplot
        data2 = cu.prepare_plot_data(self.df_num, 'a', 'd', True, False)
        assert data2['type'] == 'boxplot'
        # Heatmap
        data3 = cu.prepare_plot_data(self.df_cat, 'cat1', 'cat2', False, False)
        assert data3['type'] == 'heatmap'

    def test_estimate_resources(self):
        res = cu.estimate_resources(self.df_num, 'a', 'b')
        assert 'estimated_memory_mb' in res
        assert 'estimated_time_seconds' in res
        assert 'potential_issues' in res
        # Edge: missing field
        res2 = cu.estimate_resources(self.df_num, 'a', 'not_a')
        assert 'error' in res2

if __name__ == "__main__":
    pytest.main()