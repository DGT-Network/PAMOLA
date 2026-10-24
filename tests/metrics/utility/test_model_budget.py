"""PERF-FIX-2: kernel / ensemble models are bounded, with the sampling stated in the result.

Before: UtilityMetricOperation (default regression models linear/rf/svr) did not finish within
1500 s on 100K rows - the RBF SVR is quadratic (40K rows: 74 s fit + 36 s predict).
"""

import time

import numpy as np
import pandas as pd

from pamola_core.metrics.commons import model_budget
from pamola_core.metrics.commons.model_budget import apply_budget, cap_rows
from pamola_core.metrics.utility.classification import ClassificationUtility
from pamola_core.metrics.utility.regression import RegressionUtility
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "target": 2 * x + rng.normal(scale=0.1, size=n)})


def test_cap_is_seeded_aligned_and_a_noop_below_the_budget():
    X, y = _frame(1000)[["x"]], _frame(1000)["target"]
    a = cap_rows(X, y, 100)
    b = cap_rows(X, y, 100)
    assert len(a[0]) == len(a[1]) == 100 and a[2] == 1000
    assert a[0].index.equals(b[0].index) and a[0].index.equals(a[1].index)
    assert cap_rows(X, y, None)[0] is X and cap_rows(X, y, 5000)[0] is X


def test_budget_applies_to_svr_not_to_linear():
    df = _frame(60_000)
    X, y = df[["x"]], df["target"]
    *_, note = apply_budget(SVR(), X, y, X, y)
    assert note and note["train_rows_used"] == 10_000 and note["train_rows_available"] == 60_000
    assert apply_budget(LinearRegression(), X, y, X, y)[-1] is None


def test_regression_default_models_finish_on_a_large_frame_and_report_the_sampling(monkeypatch):
    monkeypatch.setitem(model_budget.ROW_BUDGETS, "SVR", (500, 1000))
    monkeypatch.setitem(model_budget.ROW_BUDGETS, "RandomForestRegressor", (2000, 4000))
    original, transformed = _frame(30_000, 1), _frame(30_000, 2)
    started = time.monotonic()
    result = RegressionUtility(cv_folds=3).calculate_metric(original, transformed, value_field="target")
    assert time.monotonic() - started < 120
    assert result["linear"]["r2"] > 0.9 and "sampling" not in result["linear"]
    assert result["svr"]["sampling"]["train_rows_used"] == 500
    assert result["rf"]["sampling"]["evaluation_rows_used"] == 4000
    assert np.isfinite(result["svr"]["r2"])


def test_small_inputs_are_unchanged():
    result = RegressionUtility(cv_folds=3).calculate_metric(_frame(300, 1), _frame(300, 2), value_field="target")
    assert all("sampling" not in block for block in result.values() if isinstance(block, dict))


def test_classification_svm_is_budgeted_too(monkeypatch):
    monkeypatch.setitem(model_budget.ROW_BUDGETS, "SVC", (300, 600))
    rng = np.random.default_rng(3)
    def make(seed):
        r = np.random.default_rng(seed)
        x = r.normal(size=2000)
        return pd.DataFrame({"x": x, "y": (x + r.normal(scale=0.3, size=2000) > 0).astype(int)})
    result = ClassificationUtility(models=["svm"], cv_folds=3).calculate_metric(make(1), make(2), value_field="y")
    assert result["svm"]["sampling"]["train_rows_used"] == 300
