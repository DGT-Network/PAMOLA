"""
Row budgets for the model-based utility metrics.

A kernel SVM is quadratic in the number of rows (an RBF ``SVR`` took ~15 s to fit on 20K rows
and its prediction another ~5 s; on the 66K training rows of a 100K dataset it did not finish
in 25 minutes) and a random forest is expensive beyond a few hundred thousand rows. Those two
model families are therefore trained and evaluated on a seeded random sample of the rows of
each fold when a fold is larger than the budget. The metric keeps its meaning (a model fitted
on the original data and scored on the transformed data); only the estimate is computed from a
sample, and the result says so (``"sampling"`` in the model's result block). Linear models and
folds under the budget are untouched.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

#: (rows used for fitting, rows used for scoring) per model class name; ``None`` = no cap.
ROW_BUDGETS: Dict[str, Tuple[Optional[int], Optional[int]]] = {
    "SVR": (10_000, 50_000),
    "SVC": (10_000, 50_000),
    "RandomForestRegressor": (200_000, 500_000),
    "RandomForestClassifier": (200_000, 500_000),
}

SAMPLE_SEED = 42


def budget_for(model: Any) -> Tuple[Optional[int], Optional[int]]:
    return ROW_BUDGETS.get(type(model).__name__, (None, None))


def _take(data: Any, index: np.ndarray) -> Any:
    return data.iloc[index] if hasattr(data, "iloc") else data[index]


def cap_rows(X: Any, y: Any, cap: Optional[int]) -> Tuple[Any, Any, int]:
    """Seeded sample of at most ``cap`` aligned rows; returns ``(X, y, rows_before)``."""
    n = len(X)
    if cap is None or n <= cap:
        return X, y, n
    index = np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, size=cap, replace=False))
    return _take(X, index), _take(y, index), n


def apply_budget(model: Any, X_train, y_train, X_test, y_test):
    """Cap the fold for ``model``; returns the (possibly reduced) data and a note or ``None``."""
    train_cap, eval_cap = budget_for(model)
    if train_cap is None and eval_cap is None:
        return X_train, y_train, X_test, y_test, None
    X_train, y_train, n_train = cap_rows(X_train, y_train, train_cap)
    X_test, y_test, n_test = cap_rows(X_test, y_test, eval_cap)
    note = None
    if len(X_train) < n_train or len(X_test) < n_test:
        note = {
            "train_rows_used": int(len(X_train)), "train_rows_available": int(n_train),
            "evaluation_rows_used": int(len(X_test)), "evaluation_rows_available": int(n_test),
            "seed": SAMPLE_SEED,
        }
    return X_train, y_train, X_test, y_test, note
