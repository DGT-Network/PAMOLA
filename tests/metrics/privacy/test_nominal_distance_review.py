"""Review findings on the nominal-aware distance preprocessing (R1-01 ... R1-07, Mahalanobis).

Each test is built from a minimal reproduction of the finding.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pamola_core.common.enum.distance_metric_type import DistanceMetricType
from pamola_core.errors.exceptions import InvalidParameterError, ValidationError
from pamola_core.metrics.commons.preprocessing import (
    _column_kind,
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


# --- R2 -------------------------------------------------------------------------------------
def test_r2_01_float_and_string_labels_are_distinct_nominal_levels():
    original = pd.DataFrame({"c": pd.Categorical([1.0, "other"])})
    synthetic = pd.DataFrame({"c": pd.Categorical(["1"])})
    assert DistanceToClosestRecord(use_faiss=False).calculate_metric(original, synthetic)[
        "dcr_statistics"
    ]["mean"] == pytest.approx(1.0)  # was 0
    nndr = NearestNeighborDistanceRatio().calculate_metric(original, synthetic)["nndr_values"]
    assert nndr[0] == pytest.approx(1.0, abs=1e-6)  # was 0


def test_r2_01_one_categorical_with_three_distinct_levels_gives_three_columns():
    frame = pd.DataFrame({"c": pd.Categorical([1.0, "1", 2.0])})
    o, _ = prepare_pair_for_distance_metrics(frame, frame.copy())
    assert o.shape[1] == 3
    assert o.attrs["nominal_positions"] == [0, 1, 2]


def test_r2_01_the_same_identity_rules_apply_to_ordinal_checks():
    data = pd.DataFrame({"c": pd.Categorical(["1", "x"])})
    with pytest.raises(ValidationError, match="does not list"):
        prepare_pair_for_distance_metrics(data, data.copy(), {"c": [1.0, "x"]})  # 1.0 is not "1"
    with pytest.raises(ValidationError, match="repeats"):
        prepare_pair_for_distance_metrics(data, data.copy(), {"c": [1, 1.0, "x"]})  # 1 is 1.0


def test_r2_01_integers_and_floats_with_the_same_value_are_one_category():
    orig = pd.DataFrame({"c": pd.Categorical([1, 2])})
    synth = pd.DataFrame({"c": pd.Categorical([1.0])})
    assert DistanceToClosestRecord(use_faiss=False).calculate_metric(orig, synth)[
        "dcr_statistics"
    ]["min"] == pytest.approx(0.0)


@pytest.mark.parametrize("cls", [DistanceToClosestRecord, NearestNeighborDistanceRatio])
def test_r2_02_nullable_boolean_keeps_a_missing_category(cls):
    original = pd.DataFrame({"c": pd.Series([True, False, pd.NA], dtype="boolean")})
    synthetic = pd.DataFrame({"c": [True]})
    result = cls().calculate_metric(original, synthetic)
    value = result["dcr_statistics"]["mean"] if "dcr_statistics" in result else result["nndr_values"][0]
    assert value == pytest.approx(0.0, abs=1e-9)
    o, _ = prepare_pair_for_distance_metrics(original, synthetic)
    assert o.shape == (3, 3)  # False, True, <missing>
    assert o.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_r2_02_nullable_integers_against_a_categorical_do_not_crash():
    original = pd.DataFrame({"c": pd.Series([1, 2, pd.NA], dtype="Int64")})
    synthetic = pd.DataFrame({"c": pd.Categorical([1, 2, 3])})
    o, t = prepare_pair_for_distance_metrics(original, synthetic)
    assert list(o.columns) == list(t.columns) == ["c=1", "c=2", "c=3", "c=<missing>"]
    nullable = pd.DataFrame({"c": pd.Series([True, pd.NA], dtype="boolean")})
    out = prepare_data_for_distance_metrics(nullable)  # the standalone helper too
    assert "c=<missing>" in out.columns


@pytest.mark.parametrize("cls", [DistanceToClosestRecord, NearestNeighborDistanceRatio])
@pytest.mark.parametrize("form", ["mahalanobis", DistanceMetricType.MAHALANOBIS])
def test_r2_03_mahalanobis_is_rejected_before_any_computation(cls, form):
    with pytest.raises(InvalidParameterError, match="not supported for"):
        cls(distance_metric=form)


def test_r2_03_the_other_metrics_are_unaffected_by_the_guard():
    for metric in ("euclidean", "manhattan", "cosine"):
        DistanceToClosestRecord(distance_metric=metric)
        NearestNeighborDistanceRatio(distance_metric=metric)
    DistanceToClosestRecord(distance_metric=DistanceMetricType.EUCLIDEAN)


def test_r2_05_reciprocal_reference_calls_agree_on_the_feature_order():
    a = pd.DataFrame({"c": ["A", "B"], "n": [0.0, 2.0]})
    b = pd.DataFrame({"n": [0.0], "c": ["A"]})
    x = prepare_data_for_distance_metrics(a, reference=b)
    y = prepare_data_for_distance_metrics(b, reference=a)
    assert list(x.columns) == list(y.columns)
    u, v = scale_pair_for_distance(x, y, False)
    np.testing.assert_allclose(cdist(v, u), [[0.0, np.sqrt(5)]])  # was [[1, 2]]


def test_r2_05_scaling_refuses_frames_with_different_features():
    x = pd.DataFrame({"a": [1.0], "b": [2.0]})
    y = pd.DataFrame({"b": [2.0], "a": [1.0]})
    with pytest.raises(ValidationError, match="same features"):
        scale_pair_for_distance(x, y, False)


def test_r2_06_datetime_resolutions_are_unified_before_extraction():
    days = pd.to_datetime(["2020-01-01", "2020-01-02"])
    original = pd.DataFrame({"d": pd.Series(days).astype("datetime64[s]")})
    transformed = pd.DataFrame({"d": pd.Series(days).astype("datetime64[ns]")})
    o, t = prepare_pair_for_distance_metrics(original, transformed)
    np.testing.assert_array_equal(o["d"].to_numpy(), t["d"].to_numpy())
    values = DistanceToClosestRecord(use_faiss=False, normalize_features=False).calculate_metric(
        original, transformed
    )
    assert values["dcr_statistics"]["max"] == pytest.approx(0.0)  # was 1.58e18


def test_r2_06_timezone_aware_values_are_compared_in_utc():
    aware = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 12:00"]).tz_localize("UTC")})
    shifted = pd.DataFrame({"d": pd.to_datetime(["2020-01-01 13:00"]).tz_localize("Europe/Paris")})
    o, t = prepare_pair_for_distance_metrics(aware, shifted)
    np.testing.assert_array_equal(o["d"].to_numpy(), t["d"].to_numpy())
