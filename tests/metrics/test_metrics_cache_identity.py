"""METR-CACHE-01: a cached metric is keyed on BOTH inputs.

The cache was checked before the transformed dataset was loaded and keyed on the
original DataFrame alone, so with ``use_cache=True`` a second run with the same
original and configuration but a different transformed dataset returned the first
run's score.
"""

import numpy as np
import pandas as pd

from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _run(original, transformed, task_dir, **op_kwargs):
    op = FidelityOperation(
        fidelity_metrics=["ks"], use_cache=True, columns=["x"], **op_kwargs
    )
    source = DataSource(dataframes={"main": original, "synth": transformed})
    result = op.execute(
        source,
        task_dir,
        None,
        original_dataset_name="main",
        transformed_dataset_name="synth",
    )
    assert result.status != OperationStatus.ERROR, result.error_message
    return result


def _frames():
    rng = np.random.default_rng(0)
    original = pd.DataFrame({"x": rng.normal(size=300)})
    close = pd.DataFrame({"x": rng.normal(size=300)})
    far = pd.DataFrame({"x": rng.normal(loc=5.0, size=300)})
    return original, close, far


def test_different_transformed_data_does_not_hit_the_cache(tmp_path):
    original, close, far = _frames()

    first = _run(original, close, tmp_path)
    second = _run(original, far, tmp_path)

    assert not first.metrics.get("cached")
    # Same original, same configuration, different transformed data: recomputed.
    assert not second.metrics.get("cached")
    assert second.status == OperationStatus.SUCCESS


def test_identical_inputs_still_hit_the_cache(tmp_path):
    original, close, _ = _frames()

    _run(original, close, tmp_path)
    second = _run(original, close.copy(), tmp_path)

    assert second.metrics.get("cached") is True
