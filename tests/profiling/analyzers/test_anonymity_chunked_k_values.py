"""
CDX-R2-10: chunked k-value computation must match the unchunked one.

Pre-fix, the first pass keyed groups by scalars (single field) while the second
pass looked up tuples, so every lookup missed and chunked single-field runs
produced nulls.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.profiling.analyzers.anonymity import KAnonymityProfilerOperation


def _k(df, fields, chunk_size):
    op = KAnonymityProfilerOperation(
        quasi_identifiers=fields, use_cache=False
    )
    op.chunk_size = chunk_size
    return op._calculate_k_values(df, fields)


def test_single_field_chunked_matches_unchunked():
    df = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    assert _k(df, ["x"], 10).tolist() == [2, 2, 2, 2]
    chunked = _k(df, ["x"], 2)
    assert chunked.tolist() == [2, 2, 2, 2]
    assert not chunked.isna().any()


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7])
def test_multi_field_with_missing_values_matches_unchunked(chunk_size):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "a": rng.choice(["p", "q", None], size=40),
            "b": rng.choice([1.0, 2.0, np.nan], size=40),
            "c": rng.choice(["u", "v"], size=40),
        }
    )
    expected = _k(df, ["a", "b"], 10_000)
    got = _k(df, ["a", "b"], chunk_size)

    assert got.tolist() == expected.tolist()
    assert got.index.equals(df.index)


def test_single_field_with_missing_group_and_duplicate_index():
    df = pd.DataFrame({"x": ["a", None, "a", None, "b"]}, index=[0, 0, 1, 1, 1])
    chunked = _k(df, ["x"], 2)
    assert chunked.tolist() == _k(df, ["x"], 100).tolist() == [2, 2, 2, 2, 1]
    assert chunked.index.tolist() == [0, 0, 1, 1, 1]
