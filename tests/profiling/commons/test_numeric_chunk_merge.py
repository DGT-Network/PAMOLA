"""
CDX-R2-9: chunked numeric statistics must equal the unchunked ones.

Pre-fix, ``combine_chunk_results`` took the MIDDLE chunk's median and averaged
per-chunk std/variance (ignoring between-chunk variance): 10,000 zeros followed
by 10,000 hundreds gave median 100 and std 0 instead of 50 and 50.00125.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.profiling.analyzers.numeric import NumericAnalyzer
from pamola_core.profiling.commons.numeric_utils import (
    analyze_numeric_chunk,
    calculate_kurtosis,
    calculate_skewness,
    combine_chunk_results,
    handle_large_dataframe,
)


def _chunked(df, size):
    return handle_large_dataframe(df, "x", analyze_numeric_chunk, chunk_size=size)


def test_zeros_then_hundreds_keeps_between_chunk_variation():
    df = pd.DataFrame({"x": [0.0] * 10000 + [100.0] * 10000})
    out = _chunked(df, 10000)

    assert out["median"] == 50.0
    assert out["std"] == pytest.approx(50.00125, abs=1e-5)
    assert out["variance"] == pytest.approx(2500.125, abs=1e-4)
    assert out["mean"] == pytest.approx(50.0)


def test_default_numeric_analyzer_reports_correct_chunked_stats():
    df = pd.DataFrame({"x": [0.0] * 10000 + [100.0] * 10000})
    r = NumericAnalyzer().analyze(
        df, "x", chunk_size=10000, should_detect_outliers=False,
        should_test_normality=False, track_progress=False,
    )
    stats = r.get("stats", r)
    assert stats["median"] == 50.0
    assert stats["std"] == pytest.approx(50.00125, abs=1e-5)


@pytest.mark.parametrize("size", [7, 100, 999, 5000])
def test_chunked_equals_unchunked_for_every_moment(size):
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(0, 1, 3000), rng.exponential(3, 2000), [1e4, -5e3]])
    df = pd.DataFrame({"x": values})
    series = df["x"]

    out = _chunked(df, size)

    assert out["count"] == len(series)
    assert out["mean"] == pytest.approx(series.mean(), rel=1e-9)
    assert out["median"] == pytest.approx(series.median(), rel=1e-12)
    assert out["std"] == pytest.approx(series.std(), rel=1e-9)
    assert out["variance"] == pytest.approx(series.var(), rel=1e-9)
    assert out["skewness"] == pytest.approx(calculate_skewness(series), rel=1e-7)
    assert out["kurtosis"] == pytest.approx(calculate_kurtosis(series), rel=1e-7)
    assert out["min"] == series.min() and out["max"] == series.max()
    assert out["sum"] == pytest.approx(series.sum(), rel=1e-9)


def test_internal_merge_state_never_leaks_into_the_result():
    out = _chunked(pd.DataFrame({"x": np.arange(50.0)}), 10)
    assert not [k for k in out if k.startswith("_")]


def test_chunks_without_values_report_a_named_approximation():
    chunks = [
        {"count": 3, "min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0, "std": 1.0,
         "var": 1.0, "sum": 6.0},
        {"count": 3, "min": 4.0, "max": 6.0, "mean": 5.0, "median": 5.0, "std": 1.0,
         "var": 1.0, "sum": 15.0},
    ]
    out = combine_chunk_results(chunks)
    assert out["median"] is None
    assert out["median_approx"] in (2.0, 5.0)
    # variance still merges exactly from the per-chunk variances
    assert out["variance"] == pytest.approx(pd.Series([1, 2, 3, 4, 5, 6.0]).var())
