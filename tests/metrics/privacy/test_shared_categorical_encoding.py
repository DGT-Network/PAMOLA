"""CDX-AC-11: DCR/NNDR encode both datasets in ONE categorical space.

Each dataset used to get its own ``OrdinalEncoder``: original ``['alice', 'bob']``
and transformed ``['xavier', 'zoe']`` both became codes 0 and 1, so the distances
were all 0 - "maximal risk" for datasets that share nothing.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.commons.preprocessing import prepare_data_for_distance_metrics
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord
from pamola_core.metrics.privacy.neighbor import NearestNeighborDistanceRatio


def _disjoint():
    original = pd.DataFrame({"name": ["alice", "bob"] * 10, "n": np.arange(20.0)})
    transformed = pd.DataFrame({"name": ["xavier", "zoe"] * 10, "n": np.arange(20.0)})
    return original, transformed


def test_unrelated_categories_get_different_codes():
    original, transformed = _disjoint()
    o = prepare_data_for_distance_metrics(original, reference=transformed)
    t = prepare_data_for_distance_metrics(transformed, reference=original)
    assert set(o["name"]) == {0.0, 1.0}      # alice, bob
    assert set(t["name"]) == {2.0, 3.0}      # xavier, zoe
    assert not set(o["name"]) & set(t["name"])


def test_the_same_category_has_the_same_code_in_both_datasets():
    original = pd.DataFrame({"c": ["a", "b", "c", "a"]})
    transformed = pd.DataFrame({"c": ["c", "d", "a", "a"]})
    o = prepare_data_for_distance_metrics(original, reference=transformed)
    t = prepare_data_for_distance_metrics(transformed, reference=original)
    code = {}
    for frame, source in ((o, original), (t, transformed)):
        for value, enc in zip(source["c"], frame["c"]):
            assert code.setdefault(value, enc) == enc, f"{value!r} encoded inconsistently"


def test_dcr_is_not_zero_for_datasets_with_no_category_in_common():
    original, transformed = _disjoint()
    for kwargs in ({"normalize_features": False}, {}):
        stats = DistanceToClosestRecord(**kwargs).calculate_metric(original, transformed)[
            "dcr_statistics"
        ]
        assert stats["min"] > 0, "unrelated categories must not collapse to distance 0"
        assert stats["mean"] > 0


def test_dcr_is_still_zero_for_identical_datasets():
    original, _ = _disjoint()
    stats = DistanceToClosestRecord(normalize_features=False).calculate_metric(
        original, original.copy()
    )["dcr_statistics"]
    assert stats["max"] == pytest.approx(0.0, abs=1e-9)


def test_nndr_uses_the_same_encoding():
    original, transformed = _disjoint()
    NearestNeighborDistanceRatio().calculate_metric(original, transformed)  # runs
    o = prepare_data_for_distance_metrics(original, reference=transformed)
    t = prepare_data_for_distance_metrics(transformed, reference=original)
    assert not set(o["name"]) & set(t["name"])


def test_numeric_only_data_is_unchanged_by_the_reference():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
    other = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
    pd.testing.assert_frame_equal(
        prepare_data_for_distance_metrics(df),
        prepare_data_for_distance_metrics(df, reference=other),
    )


def test_single_dataset_call_keeps_working():
    df = pd.DataFrame({"c": ["x", "y", "x"], "n": [1.0, 2.0, 3.0]})
    out = prepare_data_for_distance_metrics(df)
    assert out["c"].tolist() == [0.0, 1.0, 0.0]
