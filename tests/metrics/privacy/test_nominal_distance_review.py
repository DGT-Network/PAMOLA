"""Review findings on the nominal-aware distance preprocessing (R1-01 ... R1-07, Mahalanobis).

Each test is built from a minimal reproduction of the finding.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pamola_core.errors.exceptions import ValidationError
from pamola_core.metrics.commons.preprocessing import (
    _column_kind,
    mahalanobis_inverse_covariance,
    prepare_data_for_distance_metrics,
    prepare_pair_for_distance_metrics,
    scale_pair_for_distance,
)
from pamola_core.metrics.privacy import distance as distance_module
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord
from pamola_core.metrics.privacy.neighbor import NearestNeighborDistanceRatio


def _dcr_values(orig, synth, **kwargs):
    kwargs.setdefault("use_faiss", False)
    kwargs.setdefault("normalize_features", False)
    dcr = DistanceToClosestRecord(**kwargs)
    o, t = prepare_pair_for_distance_metrics(orig, synth, kwargs.get("ordinal_columns"))
    a, b = scale_pair_for_distance(o, t, kwargs["normalize_features"])
    return dcr._calculate_dcr_sklearn(a, b)


# --- R1-01: the dtype decides first ------------------------------------------------------
def test_r1_01_categorical_dtype_with_numeric_labels_stays_nominal():
    orig = pd.DataFrame({"c": pd.Categorical([1])})
    synth = pd.DataFrame({"c": pd.Categorical([2, 10])})
    np.testing.assert_allclose(_dcr_values(orig, synth), [1.0, 1.0])  # was 1 and 9
    renamed = pd.DataFrame({"c": pd.Categorical(["A"])}), pd.DataFrame({"c": pd.Categorical(["B", "C"])})
    np.testing.assert_allclose(_dcr_values(*renamed), _dcr_values(orig, synth))


def test_r1_01_nndr_does_not_depend_on_how_nominal_levels_are_named():
    def nndr(o, s):
        return NearestNeighborDistanceRatio().calculate_metric(o, s)["nndr_values"]

    numbers = nndr(pd.DataFrame({"c": pd.Categorical([1, 2])}), pd.DataFrame({"c": pd.Categorical([10])}))
    letters = nndr(pd.DataFrame({"c": pd.Categorical(["A", "B"])}), pd.DataFrame({"c": pd.Categorical(["C"])}))
    assert numbers == pytest.approx(letters)
    assert numbers[0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("n_true", [5, 1])  # balanced and 10% prevalence
def test_r1_01_booleans_are_nominal_one_unit_mismatch(n_true):
    frame = pd.DataFrame({"b": [True] * n_true + [False] * (10 - n_true)})
    o, t = prepare_pair_for_distance_metrics(frame, frame.copy())
    a, _ = scale_pair_for_distance(o, t, normalize_features=True)
    mismatch = np.linalg.norm(a[0] - a[-1])  # a True row against a False row
    assert mismatch == pytest.approx(1.0)  # was 2 (balanced) or 3.33 (10% prevalence)


def test_r1_01_object_columns_of_python_numbers_are_numeric_never_dates():
    objects = pd.Series([1, 2, 3], dtype=object)
    assert _column_kind(objects, objects.copy()) == "numeric"
    orig = pd.DataFrame({"n": pd.Series([0, 10], dtype=object)})
    synth = pd.DataFrame({"n": pd.Series([4], dtype=object)})
    expected = _dcr_values(orig.astype(float), synth.astype(float))
    np.testing.assert_allclose(_dcr_values(orig, synth), expected)  # 4, not a date difference


def test_r1_01_an_ordinal_declaration_controls_a_categorical_with_numeric_labels():
    orig = pd.DataFrame({"c": pd.Categorical([10])})
    near, far = pd.DataFrame({"c": pd.Categorical([2])}), pd.DataFrame({"c": pd.Categorical([1])})
    order = {"c": [10, 2, 1]}
    assert _dcr_values(orig, near, ordinal_columns=order)[0] == pytest.approx(1.0)
    assert _dcr_values(orig, far, ordinal_columns=order)[0] == pytest.approx(2.0)


# --- R1-02: generated names never collide with input columns ---------------------------
def test_r1_02_a_numeric_column_named_like_a_generated_feature_is_still_standardized():
    original = pd.DataFrame({"c": ["A", "B"], "c=A": [0.0, 100.0]})
    synthetic = pd.DataFrame({"c": ["A"], "c=A": [20.0]})
    for columns in (["c", "c=A"], ["c=A", "c"]):
        values = DistanceToClosestRecord(use_faiss=False).calculate_metric(
            original[columns], synthetic[columns]
        )
        assert values["dcr_statistics"]["mean"] == pytest.approx(0.4)  # was 14.14 in one order


def test_r1_02_generated_names_are_unique_and_the_types_are_kept_by_position():
    original = pd.DataFrame({"c": ["A", "B"], "c=A": [0.0, 100.0]})
    o, _ = prepare_pair_for_distance_metrics(original, original.copy())
    assert len(set(o.columns)) == o.shape[1]
    assert o.attrs["nominal_positions"] == [i for i, c in enumerate(o.columns) if c in o.attrs["nominal_columns"]]
    assert "c=A" in o.columns  # the input numeric column keeps its own name


def test_r1_02_duplicate_input_names_are_refused_before_scaling():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(ValidationError, match="unique"):
        prepare_pair_for_distance_metrics(frame, frame.copy())


# --- R1-04: forced FAISS equals the exact search ---------------------------------------
@pytest.fixture
def _mixed_for_faiss():
    orig = pd.DataFrame({"n": [0.0, 2.0, 10.0], "c": ["A", "B", "C"]})
    synth = pd.DataFrame({"n": [0.0], "c": ["D"]})
    return orig, synth


def test_r1_04_forced_faiss_mean_k_is_the_mean_of_distances(_mixed_for_faiss):
    if distance_module.faiss is None:
        pytest.skip("FAISS is not installed")
    orig, synth = _mixed_for_faiss
    kwargs = dict(aggregation="mean_k", k_neighbors=3, normalize_features=False)
    exact = DistanceToClosestRecord(use_faiss=False, **kwargs).calculate_metric(orig, synth)
    forced = DistanceToClosestRecord(use_faiss=True, **kwargs).calculate_metric(orig, synth)
    assert exact["dcr_statistics"]["mean"] == pytest.approx(4.4286478662, rel=1e-6)
    assert forced["dcr_statistics"]["mean"] == pytest.approx(
        exact["dcr_statistics"]["mean"], rel=1e-5
    )  # was the root of the mean of squares, 5.97


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_r1_04_forced_faiss_percentile_falls_back_to_the_exact_search_visibly(
    _mixed_for_faiss, metric
):
    if distance_module.faiss is None:
        pytest.skip("FAISS is not installed")
    orig, synth = _mixed_for_faiss
    kwargs = dict(aggregation="percentile", distance_metric=metric, normalize_features=False)
    exact = DistanceToClosestRecord(use_faiss=False, **kwargs).calculate_metric(orig, synth)
    forced = DistanceToClosestRecord(use_faiss=True, **kwargs).calculate_metric(orig, synth)
    assert forced["dcr_statistics"]["mean"] == pytest.approx(exact["dcr_statistics"]["mean"])
    assert forced["backend_fallback"]["from"] == "faiss"
    assert "percentile" in forced["backend_fallback"]["reason"]
    assert forced["backend"] == "sklearn"


# --- R1-05: the ordinal declaration is validated up front ------------------------------
@pytest.mark.parametrize(
    ("data", "declared", "message"),
    [
        (pd.DataFrame({"c": ["A", "B", "C"]}), ["typo"], "unknown column"),
        (pd.DataFrame({"c": ["A", "B", "C"]}), {"c": ["A", "B", "A", "C"]}, "repeats"),
        (pd.DataFrame({"c": ["A", "B", "C"]}), {"c": []}, "empty"),
        (pd.DataFrame({"c": [1, 2, 10]}), {"c": [10, 2, 1]}, "already ordered"),
        (pd.DataFrame({"c": pd.to_datetime(["2020-01-01", "2020-01-02"])}), ["c"], "already ordered"),
        (pd.DataFrame({"c": ["A", "B", "C"]}), "c", "single string"),
    ],
)
def test_r1_05_bad_ordinal_declarations_raise(data, declared, message):
    with pytest.raises(ValidationError, match=message):
        prepare_pair_for_distance_metrics(data, data.copy(), declared)


def test_r1_05_the_check_happens_before_encoding_other_columns():
    data = pd.DataFrame({"big": [f"v{i}" for i in range(2000)], "c": ["A"] * 2000})
    with pytest.raises(ValidationError, match="unknown column"):  # not ResourceError
        prepare_pair_for_distance_metrics(data, data.copy(), ["typo"])


# --- R1-06: missing values are their own category ----------------------------------------
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (None, "None", 1.0),
        (np.nan, "nan", 1.0),
        (None, np.nan, 0.0),
        (pd.NA, None, 0.0),
    ],
)
def test_r1_06_missing_values_do_not_impersonate_labels(left, right, expected):
    orig = pd.DataFrame({"c": pd.Series(["A", left], dtype=object)})
    synth = pd.DataFrame({"c": pd.Series([right], dtype=object)})
    values = _dcr_values(orig, synth)
    if expected == 0.0:
        assert values[0] == pytest.approx(0.0)  # the missing category exists on both sides
    else:
        assert values[0] == pytest.approx(1.0)  # "None" / "nan" is a real label unknown to the original


# --- R1-07: the standalone preprocessor is nominal-aware too -----------------------------
def test_r1_07_standalone_preparation_does_not_ordinal_encode_nominal_data():
    out = prepare_data_for_distance_metrics(pd.DataFrame({"c": ["A", "B", "C"]}))
    assert out.shape == (3, 3)
    a, _ = scale_pair_for_distance(out, out, normalize_features=True)
    dist = cdist(a, a)
    assert dist[0, 1] == pytest.approx(dist[0, 2]) == pytest.approx(dist[1, 2]) == pytest.approx(1.0)


# --- Mahalanobis ---------------------------------------------------------------------------
def test_mahalanobis_dcr_equals_the_pseudo_inverse_reference_on_mixed_data():
    rng = np.random.default_rng(2)
    orig = pd.DataFrame(
        {"n": rng.normal(size=60), "m": rng.normal(size=60), "c": rng.choice(["A", "B", "C"], 60)}
    )
    synth = pd.DataFrame(
        {"n": rng.normal(size=10), "m": rng.normal(size=10), "c": rng.choice(["A", "B", "C"], 10)}
    )
    o, t = prepare_pair_for_distance_metrics(orig, synth)
    a, b = scale_pair_for_distance(o, t, normalize_features=True)
    vi = mahalanobis_inverse_covariance(a)
    expected = cdist(b, a, "mahalanobis", VI=vi).min(axis=1)
    got = DistanceToClosestRecord(distance_metric="mahalanobis", use_faiss=False)._calculate_dcr_sklearn(a, b)
    np.testing.assert_allclose(got, expected, rtol=1e-8)
    stats = DistanceToClosestRecord(distance_metric="mahalanobis").calculate_metric(orig, synth)
    assert stats["dcr_statistics"]["mean"] == pytest.approx(float(expected.mean()), rel=1e-8)
    res = NearestNeighborDistanceRatio(distance_metric="mahalanobis").calculate_metric(orig, synth)
    assert len(res["nndr_values"]) == 10
