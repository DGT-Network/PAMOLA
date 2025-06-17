import pytest
import pandas as pd
import numpy as np
from unittest import mock

import pamola_core.profiling.commons.statistical_analysis as sa

# Mock calculate_skewness and calculate_kurtosis for import in tested functions
@pytest.fixture(autouse=True)
def patch_numeric_utils(monkeypatch):
    monkeypatch.setattr(
        'pamola_core.profiling.commons.numeric_utils.calculate_skewness',
        lambda x: float(np.nan) if x.empty else float(pd.Series(x).skew())
    )
    monkeypatch.setattr(
        'pamola_core.profiling.commons.numeric_utils.calculate_kurtosis',
        lambda x: float(np.nan) if x.empty else float(pd.Series(x).kurt())
    )

class TestCalculateDistributionMetrics:
    def test_valid_case(self):
        data = pd.Series([1, 2, 3, 4, 5])
        result = sa.calculate_distribution_metrics(data)
        assert isinstance(result, dict)
        assert 'mean' in result and result['mean'] == 3.0
        assert 'median' in result and result['median'] == 3.0
        assert 'std' in result and pytest.approx(result['std'], 0.01) == np.std([1,2,3,4,5], ddof=1)
        assert 'variance' in result and pytest.approx(result['variance'], 0.01) == np.var([1,2,3,4,5], ddof=1)
        assert 'range' in result and result['range'] == 4.0
        assert 'cv' in result

    def test_edge_empty_series(self):
        data = pd.Series([], dtype=float)
        result = sa.calculate_distribution_metrics(data)
        assert isinstance(result, dict)
        assert all(np.isnan(v) or v == 0 or v == {} for v in result.values()) or result == {}

    def test_invalid_input(self):
        result = sa.calculate_distribution_metrics('not_a_series')
        assert result == {}

class TestTestNormality:
    def test_valid_case_all(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        result = sa.test_normality(data, test_method='all')
        assert isinstance(result, dict)
        assert 'shapiro' in result
        assert 'anderson' in result
        assert 'ks' in result
        assert 'skewness' in result
        assert 'kurtosis' in result
        assert 'is_normal' in result

    def test_edge_empty_series(self):
        data = pd.Series([], dtype=float)
        result = sa.test_normality(data)
        assert isinstance(result, dict)
        assert result.get('is_normal') is False or 'error' in result

    def test_invalid_input(self):
        result = sa.test_normality('not_a_series')
        assert result.get('is_normal') is False or 'error' in result

    def test_sample_limit(self):
        data = pd.Series(np.random.normal(0, 1, 6000))
        result = sa.test_normality(data, sample_limit=5000)
        assert isinstance(result, dict)
        assert 'shapiro' in result

    def test_method_selection(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        for method in ['shapiro', 'anderson', 'ks']:
            result = sa.test_normality(data, test_method=method)
            assert method in result

class TestDetectOutliers:
    def test_valid_case_iqr(self):
        data = pd.Series([1, 2, 2, 2, 2, 100])
        result = sa.detect_outliers(data, method='iqr')
        assert result['method'] == 'iqr'
        assert 'count' in result
        assert result['count'] >= 0

    def test_invalid_method(self):
        data = pd.Series([1, 2, 3])
        result = sa.detect_outliers(data, method='unknown')
        assert result['method'] == 'iqr'

    def test_invalid_input(self):
        result = sa.detect_outliers('not_a_series')
        assert result['method'] == 'iqr' or 'error' in result

class TestDetectOutliersIQR:
    def test_valid_case(self):
        data = pd.Series([1, 2, 2, 2, 2, 100])
        result = sa.detect_outliers_iqr(data)
        assert result['method'] == 'iqr'
        assert 'count' in result

    def test_edge_empty_series(self):
        data = pd.Series([], dtype=float)
        result = sa.detect_outliers_iqr(data)
        assert result['count'] == 0

    def test_invalid_input(self):
        result = sa.detect_outliers_iqr('not_a_series')
        assert result['count'] == 0 or 'error' in result

if __name__ == "__main__":
    pytest.main()
