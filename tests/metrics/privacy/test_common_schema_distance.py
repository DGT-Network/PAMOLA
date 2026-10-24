"""CDX-R2-6: DCR/NNDR decide every column's representation from BOTH inputs.

Before: each input was inferred on its own, so identical records stored as category in one
input and int in the other were compared as ``[0, 1]`` vs ``[10, 20]`` (DCR 9 / 19, not 0),
and a date column against a text column was inferred as datetime on one side only.
"""

import pandas as pd
import pytest

from pamola_core.errors.exceptions import ValidationError
from pamola_core.metrics.commons.preprocessing import prepare_pair_for_distance_metrics
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord


def _mixed():
    original = pd.DataFrame({"v": pd.Series([10, 20] * 6, dtype="category"), "w": [1.0, 2.0] * 6})
    transformed = pd.DataFrame({"v": [10, 20] * 6, "w": [1.0, 2.0] * 6})
    return original, transformed


def test_identical_records_in_different_representations_are_identical():
    o, t = prepare_pair_for_distance_metrics(*_mixed())
    assert o["v"].tolist() == t["v"].tolist() == [10.0, 20.0] * 6


def test_dcr_is_zero_for_identical_records_stored_differently():
    original, transformed = _mixed()
    result = DistanceToClosestRecord(normalize_features=False).calculate_metric(original, transformed)
    flat = repr(result)
    assert "'max': 0.0" in flat or "'mean': 0.0" in flat, flat[:400]


def test_datetimes_are_decided_jointly():
    original = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    transformed = pd.DataFrame({"d": ["2020-01-01", "2020-01-02"]})
    o, t = prepare_pair_for_distance_metrics(original, transformed)
    assert o["d"].tolist() == t["d"].tolist()


def test_text_against_dates_is_categorical_on_both_sides():
    original = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    transformed = pd.DataFrame({"d": ["alpha", "beta"]})
    o, t = prepare_pair_for_distance_metrics(original, transformed)
    assert not set(o["d"]) & set(t["d"])


def test_different_column_sets_raise_a_typed_error():
    with pytest.raises(ValidationError, match="different columns"):
        prepare_pair_for_distance_metrics(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
