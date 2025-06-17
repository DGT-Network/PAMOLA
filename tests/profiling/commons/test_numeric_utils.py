import pytest
import pandas as pd
import numpy as np
from unittest import mock
from pamola_core.profiling.commons import numeric_utils

class TestNumericUtils:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Setup before each test
        yield
        # Teardown after each test (if needed)

    # calculate_skewness
    def test_calculate_skewness_valid(self):
        data = [1, 2, 3, 4, 5]
        result = numeric_utils.calculate_skewness(data)
        assert isinstance(result, float)

    def test_calculate_skewness_edge_empty(self):
        assert numeric_utils.calculate_skewness([]) == 0.0

    def test_calculate_skewness_invalid_type(self):
        assert numeric_utils.calculate_skewness('invalid') == 0.0

    def test_calculate_skewness_zero_variance(self):
        data = [2, 2, 2, 2]
        assert numeric_utils.calculate_skewness(data) == 0.0

    # calculate_kurtosis
    def test_calculate_kurtosis_valid(self):
        data = [1, 2, 3, 4, 5, 6]
        result = numeric_utils.calculate_kurtosis(data)
        assert isinstance(result, float)

    def test_calculate_kurtosis_edge_empty(self):
        assert numeric_utils.calculate_kurtosis([]) == 0.0

    def test_calculate_kurtosis_invalid_type(self):
        assert numeric_utils.calculate_kurtosis('invalid') == 0.0

    def test_calculate_kurtosis_not_enough_samples(self):
        data = [1, 2, 3]
        assert numeric_utils.calculate_kurtosis(data) == 0.0

    # count_values_by_condition
    def test_count_values_by_condition_zero(self):
        s = pd.Series([0, 1, 0, 2])
        assert numeric_utils.count_values_by_condition(s, 'zero') == 2

    def test_count_values_by_condition_positive(self):
        s = pd.Series([-1, 0, 2, 3])
        assert numeric_utils.count_values_by_condition(s, 'positive') == 2

    def test_count_values_by_condition_negative(self):
        s = pd.Series([-1, 0, 2, -3])
        assert numeric_utils.count_values_by_condition(s, 'negative') == 2

    def test_count_values_by_condition_near_zero(self):
        s = pd.Series([1e-12, 0, 1e-11, 1e-9])
        assert numeric_utils.count_values_by_condition(s, 'near_zero', near_zero_threshold=1e-10) == 2

    def test_count_values_by_condition_invalid_type(self):
        assert numeric_utils.count_values_by_condition('invalid', 'zero') == 0

    def test_count_values_by_condition_unknown_condition(self):
        s = pd.Series([1, 2, 3])
        assert numeric_utils.count_values_by_condition(s, 'unknown') == 0

    # calculate_basic_stats
    def test_calculate_basic_stats_valid(self):
        s = pd.Series([1, 2, 3, 4, 5])
        stats = numeric_utils.calculate_basic_stats(s)
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        assert stats['sum'] == 15.0

    def test_calculate_basic_stats_empty(self):
        s = pd.Series([], dtype=float)
        stats = numeric_utils.calculate_basic_stats(s)
        assert stats['min'] is None or (isinstance(stats['min'], float) and np.isnan(stats['min']))
        assert stats['max'] is None or (isinstance(stats['max'], float) and np.isnan(stats['max']))

    # calculate_extended_stats
    def test_calculate_extended_stats_valid(self):
        s = pd.Series([1, 2, 3, 4, 5])
        stats = numeric_utils.calculate_extended_stats(s)
        assert stats['count'] == 5
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        assert 'zero_count' in stats

    def test_calculate_extended_stats_empty(self):
        s = pd.Series([], dtype=float)
        stats = numeric_utils.calculate_extended_stats(s)
        assert stats['count'] == 0
        assert 'min' in stats
        assert stats['min'] is None or (isinstance(stats['min'], float) and np.isnan(stats['min']))

    # calculate_percentiles
    def test_calculate_percentiles_valid(self):
        s = pd.Series(np.arange(100))
        percentiles = numeric_utils.calculate_percentiles(s)
        assert 'p0.1' in percentiles
        assert 'p99.9' in percentiles

    def test_calculate_percentiles_empty(self):
        s = pd.Series([], dtype=float)
        percentiles = numeric_utils.calculate_percentiles(s)
        for v in percentiles.values():
            assert v == 0.0 or (isinstance(v, float) and np.isnan(v))

    # calculate_histogram
    def test_calculate_histogram_valid(self):
        s = pd.Series([1, 2, 2, 3, 3, 3, 4])
        hist = numeric_utils.calculate_histogram(s, bins=3)
        assert 'bins' in hist and 'counts' in hist
        assert len(hist['bins']) == 3

    def test_calculate_histogram_empty(self):
        s = pd.Series([], dtype=float)
        hist = numeric_utils.calculate_histogram(s)
        assert isinstance(hist['bins'], list)
        assert isinstance(hist['counts'], list)
        assert len(hist['bins']) == 0 or all(isinstance(b, str) for b in hist['bins'])
        assert len(hist['counts']) == 0 or all(isinstance(c, int) for c in hist['counts'])

    # detect_outliers
    def test_detect_outliers_valid(self):
        s = pd.Series([1, 2, 2, 3, 100])
        outliers = numeric_utils.detect_outliers(s)
        assert 'count' in outliers
        assert isinstance(outliers['count'], int)

    def test_detect_outliers_empty(self):
        s = pd.Series([], dtype=float)
        outliers = numeric_utils.detect_outliers(s)
        assert outliers['count'] == 0

    # test_normality
    def test_test_normality_valid(self):
        s = pd.Series(np.random.normal(0, 1, 100))
        result = numeric_utils.test_normality(s)
        assert 'is_normal' in result
        assert 'shapiro' in result
        assert 'anderson' in result
        assert 'ks' in result

    def test_test_normality_empty(self):
        s = pd.Series([], dtype=float)
        result = numeric_utils.test_normality(s)
        assert result['is_normal'] is False

    # analyze_numeric_chunk
    def test_analyze_numeric_chunk_valid(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        stats = numeric_utils.analyze_numeric_chunk(df, 'a')
        assert stats['count'] == 5
        assert 'skewness' in stats

    def test_analyze_numeric_chunk_missing_field(self):
        df = pd.DataFrame({'b': [1, 2, 3]})
        stats = numeric_utils.analyze_numeric_chunk(df, 'a')
        assert stats == {}

    def test_analyze_numeric_chunk_empty(self):
        df = pd.DataFrame({'a': []})
        stats = numeric_utils.analyze_numeric_chunk(df, 'a')
        assert stats == {}

    # combine_chunk_results
    def test_combine_chunk_results_valid(self):
        chunk1 = {'count': 2, 'min': 1, 'max': 3, 'mean': 2, 'median': 2, 'std': 1, 'var': 1, 'sum': 4, 'skewness': 0, 'kurtosis': 0, 'zero_count': 0, 'negative_count': 0, 'positive_count': 2, 'near_zero_count': 0}
        chunk2 = {'count': 3, 'min': 2, 'max': 5, 'mean': 4, 'median': 4, 'std': 1, 'var': 1, 'sum': 12, 'skewness': 0, 'kurtosis': 0, 'zero_count': 0, 'negative_count': 0, 'positive_count': 3, 'near_zero_count': 0}
        combined = numeric_utils.combine_chunk_results([chunk1, chunk2])
        assert combined['count'] == 5
        assert combined['min'] == 1
        assert combined['max'] == 5
        assert combined['zero_count'] == 0
        assert combined['positive_count'] == 5

    def test_combine_chunk_results_empty(self):
        combined = numeric_utils.combine_chunk_results([])
        assert combined['count'] == 0
        assert combined['min'] is None

    # create_empty_stats
    def test_create_empty_stats(self):
        stats = numeric_utils.create_empty_stats()
        assert stats['count'] == 0
        assert stats['min'] is None
        assert stats['histogram']['bins'] == []
        assert stats['normality']['is_normal'] is False

    # prepare_numeric_data
    def test_prepare_numeric_data_valid(self):
        df = pd.DataFrame({'a': [1, 2, None, 4]})
        valid, null_count, non_null_count = numeric_utils.prepare_numeric_data(df, 'a')
        assert null_count == 1
        assert non_null_count == 3
        assert all(~pd.isna(valid))

    # handle_large_dataframe
    def test_handle_large_dataframe_valid(self):
        df = pd.DataFrame({'a': np.arange(100)})
        result = numeric_utils.handle_large_dataframe(df, 'a', numeric_utils.analyze_numeric_chunk, chunk_size=10)
        assert 'count' in result
        assert result['count'] == 100

    def test_handle_large_dataframe_empty(self):
        df = pd.DataFrame({'a': []})
        result = numeric_utils.handle_large_dataframe(df, 'a', numeric_utils.analyze_numeric_chunk, chunk_size=10)
        assert result['count'] == 0

if __name__ == "__main__":
    pytest.main()
