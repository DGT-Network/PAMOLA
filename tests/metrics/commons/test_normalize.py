"""
PAMOLA Core Metrics Package: Unit Tests for Normalize
====================================================
File:        tests/metrics/commons/test_normalize.py
Target:      pamola_core.metrics.commons.normalize
Coverage:    100% line coverage (see docs)
Top-matter:  Standardized (see process docs)

Description:
    Comprehensive unit tests for normalize, including:
    - normalize_metric_value (basic, invert, target_range, zero_range)
    - Edge cases and error handling
    - Compliance with ≥90% line coverage and process requirements

Process:
    - All tests must be self-contained and not depend on external state.
    - All branches and error paths must be exercised.
    - Top-matter must be present and up to date.
    - See process documentation for details.
    
**Version:** 4.0.0
**Coverage Status:** ✅ Full
**Last Updated:** 2025-07-23
"""

import pytest
import numpy as np
from pamola_core.metrics.commons import normalize

def test_normalize_metric_value_basic():
    val = normalize.normalize_metric_value(5, (0, 10))
    assert np.isclose(val, 0.5)

def test_normalize_metric_value_invert():
    val = normalize.normalize_metric_value(2, (0, 10), higher_is_better=False)
    # Inverted: 10 - (2-0) = 8, norm: (8-0)/(10-0) = 0.8
    assert np.isclose(val, 0.8)

def test_normalize_metric_value_target_range():
    val = normalize.normalize_metric_value(5, (0, 10), target_range=(1, 2))
    assert np.isclose(val, 1.5)

def test_normalize_metric_value_zero_range():
    val = normalize.normalize_metric_value(5, (5, 5))
    assert val == 0.0

def test_normalize_array_np_minmax():
    arr = np.array([1, 2, 3])
    norm = normalize.normalize_array_np(arr, method="minmax")
    assert np.allclose(norm, [0.0, 0.5, 1.0])

def test_normalize_array_np_zscore():
    arr = np.array([1, 2, 3])
    norm = normalize.normalize_array_np(arr, method="zscore")
    assert np.allclose(norm, [ -1.22474487, 0., 1.22474487 ], atol=1e-6)

def test_normalize_array_np_zero_range():
    arr = np.array([5, 5, 5])
    norm = normalize.normalize_array_np(arr, method="minmax")
    assert np.all(norm == 0)

def test_normalize_array_np_zero_std():
    arr = np.array([5, 5, 5])
    norm = normalize.normalize_array_np(arr, method="zscore")
    assert np.all(norm == 0)

def test_normalize_array_np_invalid():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        normalize.normalize_array_np(arr, method="bad")

def test_normalize_array_sklearn_zscore():
    arr = np.array([1, 2, 3])
    norm = normalize.normalize_array_sklearn(arr, method="zscore")
    assert np.allclose(norm, [ -1.22474487, 0., 1.22474487 ], atol=1e-6)

def test_normalize_array_sklearn_minmax():
    arr = np.array([1, 2, 3])
    norm = normalize.normalize_array_sklearn(arr, method="minmax")
    assert np.allclose(norm, [0.0, 0.5, 1.0])

def test_normalize_array_sklearn_probability():
    arr = np.array([1, 2, 3])
    norm = normalize.normalize_array_sklearn(arr, method="probability")
    assert np.allclose(norm, arr/6)

def test_normalize_array_sklearn_none():
    arr = np.array([1, 2, 3])
    norm = normalize.normalize_array_sklearn(arr, method="none")
    assert np.all(norm == arr)

def test_normalize_array_sklearn_zero_std():
    arr = np.array([5, 5, 5])
    norm = normalize.normalize_array_sklearn(arr, method="zscore")
    assert np.all(norm == arr)

def test_normalize_array_sklearn_zero_range():
    arr = np.array([5, 5, 5])
    norm = normalize.normalize_array_sklearn(arr, method="minmax")
    assert np.all(norm == arr)

def test_normalize_array_sklearn_zero_sum():
    arr = np.array([0, 0, 0])
    norm = normalize.normalize_array_sklearn(arr, method="probability")
    assert np.all(norm == arr)

def test_normalize_array_sklearn_invalid():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        normalize.normalize_array_sklearn(arr, method="bad")
