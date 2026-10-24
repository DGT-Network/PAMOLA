"""CDX-AC-11b: utility metrics encode both datasets in one feature space.

``pd.get_dummies`` was applied to each dataset on its own, so a transformed frame
with a missing or extra category had different columns than the original, and the
model trained on one was scored on a misaligned other.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.commons.preprocessing import encode_pair_with_dummies
from pamola_core.metrics.utility.classification import ClassificationUtility
from pamola_core.metrics.utility.regression import RegressionUtility


def _frames(extra=False, missing=False):
    rng = np.random.default_rng(0)
    n = 120
    cats = np.array(["red", "green", "blue"])[rng.integers(0, 3, n)]
    original = pd.DataFrame({"colour": cats, "x": rng.normal(size=n)})
    original["y"] = original["x"] * 2 + (original["colour"] == "red") * 3
    original["label"] = (original["y"] > original["y"].median()).astype(int)
    transformed = original.copy()
    if missing:
        transformed["colour"] = transformed["colour"].replace({"blue": "green"})
    if extra:
        transformed.loc[transformed.index[:10], "colour"] = "purple"
    return original, transformed


@pytest.mark.parametrize("case", ["missing", "extra", "both"])
def test_columns_are_identical_and_in_the_same_order(case):
    original, transformed = _frames(extra=case in ("extra", "both"), missing=case in ("missing", "both"))
    o, t = encode_pair_with_dummies(original, transformed, exclude=["y"])
    assert list(o.columns) == list(t.columns)
    assert {"colour_blue", "colour_green", "colour_red"} <= set(o.columns)
    if case in ("extra", "both"):
        assert "colour_purple" in o.columns and not o["colour_purple"].any()
    if case in ("missing", "both"):
        assert not t["colour_blue"].any()  # absent in transformed: all False, not missing
    assert list(o.index) == list(original.index) and list(t.index) == list(transformed.index)
    assert o["y"].equals(original["y"]) and t["y"].equals(transformed["y"])


def test_without_the_fix_the_columns_would_differ():
    original, transformed = _frames(missing=True, extra=True)
    assert list(pd.get_dummies(original).columns) != list(pd.get_dummies(transformed).columns)


@pytest.mark.parametrize("variant", [{"missing": True}, {"extra": True}])
def test_regression_runs_and_scores_on_aligned_features(variant):
    original, transformed = _frames(**variant)
    result = RegressionUtility(models=["linear"], metrics=["r2", "mae"], cv_folds=3).calculate_metric(
        original.drop(columns="label"), transformed.drop(columns="label"), value_field="y"
    )
    r2 = result["linear"]["r2"]
    # y depends on x and colour; a model fed the right columns explains most of it
    assert r2 > 0.8, f"features misaligned? r2={r2}"


@pytest.mark.parametrize("variant", [{"missing": True}, {"extra": True}])
def test_classification_runs_on_aligned_features(variant):
    original, transformed = _frames(**variant)
    result = ClassificationUtility(models=["logistic"], metrics=["accuracy"], cv_folds=3).calculate_metric(
        original.drop(columns="y"), transformed.drop(columns="y"), value_field="label"
    )
    assert result["logistic"]["accuracy"] > 0.7


def test_identical_frames_give_the_same_result_as_before():
    original, _ = _frames()
    o, t = encode_pair_with_dummies(original, original.copy(), exclude=["y"])
    pd.testing.assert_frame_equal(o, t)
    pd.testing.assert_frame_equal(
        o.drop(columns="y"), pd.get_dummies(original.drop(columns="y"))
    )
