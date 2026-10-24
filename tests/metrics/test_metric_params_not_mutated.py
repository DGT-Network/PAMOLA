"""ASB-OPS-11: the metric operations never write into the caller's ``metric_params``.

``FidelityOperation`` added ``confidence_level`` and ``normalize`` to the caller's dictionary with
``setdefault``; the same caller dict reused for another call then carried those values along.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.metrics.operations.privacy_ops import PrivacyMetricOperation
from pamola_core.metrics.operations.utility_ops import UtilityMetricOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frames(n=120):
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    return a, a + rng.normal(scale=0.1, size=(n, 2))


_CASES = {
    "fidelity": (
        lambda params: FidelityOperation(fidelity_metrics=["ks", "kl"], metric_params=params, use_cache=False),
        {"ks": {"alpha_note": 1}, "kl": {}},
    ),
    "privacy": (
        lambda params: PrivacyMetricOperation(privacy_metrics=["dcr", "nndr"], metric_params=params, use_cache=False),
        {"dcr": {"normalize_features": True}, "nndr": {"threshold": 0.4}},
    ),
    "utility": (
        lambda params: UtilityMetricOperation(utility_metrics=["regression"], metric_params=params, use_cache=False),
        {"regression": {"value_field": "y", "models": ["linear"], "cv_folds": 3}},
    ),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_callers_metric_params_are_left_as_they_were(case, tmp_path):
    make, params = _CASES[case]
    before = copy.deepcopy(params)
    original, transformed = _frames()
    op = make(params)
    result = op.execute(
        DataSource(dataframes={"o": original, "t": transformed}), tmp_path, None,
        original_dataset_name="o", transformed_dataset_name="t",
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    assert params == before
