"""Nominal categoricals in DCR/NNDR: every mismatch is equally far, ordinal only when declared."""

import numpy as np
import pandas as pd
import pytest

from pamola_core.errors.exceptions import ResourceError, ValidationError
from pamola_core.metrics.commons.preprocessing import (
    MAX_ONE_HOT_FEATURES,
    NOMINAL_ONE_HOT_WEIGHT,
    prepare_pair_for_distance_metrics,
    scale_pair_for_distance,
)
from pamola_core.metrics.privacy import distance as distance_module
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord
from pamola_core.metrics.privacy.neighbor import NearestNeighborDistanceRatio


def _dcr(orig, synth, **kwargs):
    return DistanceToClosestRecord(use_faiss=False, **kwargs).calculate_metric(orig, synth)[
        "dcr_statistics"
    ]["min"]


def test_three_nominal_categories_are_pairwise_equidistant():
    cats = ["A", "B", "C"]
    for metric, expected in (("euclidean", 1.0), ("manhattan", 2 * NOMINAL_ONE_HOT_WEIGHT)):
        d = {}
        for x in cats:
            for y in cats:
                if x < y:
                    d[(x, y)] = _dcr(
                        pd.DataFrame({"c": [x]}), pd.DataFrame({"c": [y]}), distance_metric=metric
                    )
        assert list(d.values()) == [pytest.approx(expected)] * 3, d


def test_a_mismatch_costs_the_same_for_a_rare_and_a_common_category():
    common = pd.DataFrame({"c": ["A"] * 98 + ["B", "C"]})
    for target in ("B", "C"):  # B and C are equally rare; "A" is common
        d_rare = _dcr(common[common.c == "A"].head(1), pd.DataFrame({"c": [target]}))
        d_common = _dcr(common[common.c == target].head(1), pd.DataFrame({"c": ["A"]}))
        assert d_rare == pytest.approx(1.0) == pytest.approx(d_common)


def test_mismatch_is_one_unit_commensurate_with_one_standard_deviation():
    rng = np.random.default_rng(3)
    orig = pd.DataFrame({"n": rng.normal(size=400), "c": rng.choice(["A", "B", "C"], 400)})
    o, t = prepare_pair_for_distance_metrics(orig, orig.copy())
    a, _ = scale_pair_for_distance(o, t, normalize_features=True)
    cols = list(o.columns)
    nom = [i for i, c in enumerate(cols) if c.startswith("c=")]
    # two records equal in "n" and different in "c": distance exactly 1
    i = int(np.argmax(a[:, nom[0]] > 0))
    j = int(np.argmax(a[:, nom[1]] > 0))
    v = a[i].copy()
    w = a[i].copy()
    w[nom] = a[j, nom]
    assert np.linalg.norm(v - w) == pytest.approx(1.0)
    # one standard deviation of the standardized numeric column is also 1
    assert a[:, cols.index("n")].std() == pytest.approx(1.0, abs=1e-9)


def test_ordinal_is_used_only_when_declared():
    orig = pd.DataFrame({"s": ["low"]})
    near, far = pd.DataFrame({"s": ["mid"]}), pd.DataFrame({"s": ["high"]})
    order = {"s": ["low", "mid", "high"]}
    # nominal by default: both mismatches are equal
    assert _dcr(orig, near, normalize_features=False) == pytest.approx(
        _dcr(orig, far, normalize_features=False)
    )
    # declared ordinal: the rank distance is used (1 vs 2)
    d_near = _dcr(orig, near, normalize_features=False, ordinal_columns=order)
    d_far = _dcr(orig, far, normalize_features=False, ordinal_columns=order)
    assert (d_near, d_far) == (pytest.approx(1.0), pytest.approx(2.0))


def test_a_declared_order_must_list_every_category():
    with pytest.raises(ValidationError, match="does not list"):
        prepare_pair_for_distance_metrics(
            pd.DataFrame({"s": ["low", "extra"]}),
            pd.DataFrame({"s": ["mid"]}),
            ordinal_columns={"s": ["low", "mid"]},
        )


def test_ordinal_names_without_an_order_sort_numbers_as_numbers():
    o, _ = prepare_pair_for_distance_metrics(
        pd.DataFrame({"g": ["10", "9", "100"]}),
        pd.DataFrame({"g": ["9"]}),
        ordinal_columns=["g"],
    )
    assert o["g"].tolist() == [1.0, 0.0, 2.0]
    assert o.attrs["nominal_columns"] == []


def test_too_many_one_hot_features_are_refused_with_the_fix():
    n = MAX_ONE_HOT_FEATURES + 1
    df = pd.DataFrame({"id": [f"v{i}" for i in range(n)]})
    with pytest.raises(ResourceError, match="ordinal"):
        prepare_pair_for_distance_metrics(df, df.copy())


def _mixed_pair(n=400, seed=0):
    rng = np.random.default_rng(seed)

    def make(m):
        return pd.DataFrame(
            {
                "age": rng.normal(40, 12, m),
                "city": rng.choice(["Paris", "Rome", "Oslo", "Kyiv"], m),
                "plan": rng.choice(["a", "b", "c"], m),
            }
        )

    return make(n), make(n)


def test_dense_blocked_and_faiss_agree_on_mixed_data(monkeypatch):
    orig, synth = _mixed_pair()
    o, t = prepare_pair_for_distance_metrics(orig, synth)
    a, b = scale_pair_for_distance(o, t, normalize_features=True)
    dense = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(axis=1)

    dcr = DistanceToClosestRecord(use_faiss=False)
    monkeypatch.setattr(distance_module, "_WORKING_MEMORY_MIB", 1)  # force several blocks
    blocked = dcr._calculate_dcr_sklearn(a, b)
    np.testing.assert_allclose(blocked, dense, rtol=1e-6, atol=1e-8)  # float64 dot-product form

    if distance_module.faiss is None:
        pytest.skip("FAISS is not installed")
    fast = DistanceToClosestRecord(use_faiss=True)._calculate_dcr_faiss(a, b)
    np.testing.assert_allclose(fast, dense, rtol=1e-4, atol=1e-3)  # float32 squared-L2


def test_statistics_do_not_depend_on_the_backend_on_mixed_data():
    if distance_module.faiss is None:
        pytest.skip("FAISS is not installed")
    orig, synth = _mixed_pair(200, seed=5)
    exact = DistanceToClosestRecord(use_faiss=False).calculate_metric(orig, synth)
    fast = DistanceToClosestRecord(use_faiss=True).calculate_metric(orig, synth)
    for key in ("min", "mean", "max"):
        assert fast["dcr_statistics"][key] == pytest.approx(
            exact["dcr_statistics"][key], rel=1e-4, abs=1e-3
        )


def test_nndr_treats_nominal_categories_symmetrically():
    orig = pd.DataFrame({"c": ["A", "B", "C"] * 4})
    synth = pd.DataFrame({"c": ["A", "B", "C"]})
    res = NearestNeighborDistanceRatio(n_neighbors=3).calculate_metric(orig, synth)
    # every synthetic record has equal distances to the 1st..3rd neighbour pool of its own
    # category copies and to the others: the three values are identical
    values = res["nndr_values"]
    assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])


def test_dcr_result_carries_no_privacy_score():
    orig, synth = _mixed_pair(50, seed=1)
    res = DistanceToClosestRecord().calculate_metric(orig, synth)
    assert "privacy_score" not in res
    assert not hasattr(DistanceToClosestRecord, "_calculate_privacy_score")
