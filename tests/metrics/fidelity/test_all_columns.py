"""ASB-OPS-1: KS / KL evaluate EVERY common column, in a fixed order, with a stated aggregate.

Before: only ``list(set(common))[0]`` was evaluated - which column depended on PYTHONHASHSEED,
so ``FidelityOperation(columns=[a, b])`` covered one of them and different runs covered different ones.
"""

import numpy as np
import pandas as pd
import pytest

from pamola_core.errors.exceptions import ValidationError
from pamola_core.metrics.fidelity.distribution.kl_divergence import KLDivergence
from pamola_core.metrics.fidelity.distribution.ks_test import KolmogorovSmirnovTest
from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


def _frames(shift_b=0.0, shift_a=0.0):
    rng = np.random.default_rng(0)
    original = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "name": ["x", "y"] * 250})
    transformed = pd.DataFrame(
        {"a": original["a"] + shift_a, "b": original["b"] + shift_b, "name": ["x", "y"] * 250}
    )
    return original, transformed


def test_ks_evaluates_every_numeric_column_and_names_the_aggregate():
    original, transformed = _frames(shift_b=5.0)
    result = KolmogorovSmirnovTest(normalize=False).calculate_metric(original, transformed)
    assert list(result["per_column"]) == ["a", "b"]            # order of the original dataset
    assert result["per_column"]["a"]["ks_statistic"] == pytest.approx(0.0)
    assert result["per_column"]["b"]["ks_statistic"] > 0.9
    assert result["aggregate_column"] == "b" and result["ks_statistic"] == result["per_column"]["b"]["ks_statistic"]
    assert "max ks_statistic" in result["aggregate_rule"]
    assert result["skipped_columns"] == {"name": "not numeric in both datasets"}
    assert result["min_p_value_column"] == "b"


def test_the_result_does_not_depend_on_which_column_comes_first():
    original, transformed = _frames(shift_a=5.0)
    swapped = KolmogorovSmirnovTest(normalize=False).calculate_metric(original[["b", "a"]], transformed[["b", "a"]])
    assert swapped["aggregate_column"] == "a"
    assert list(swapped["per_column"]) == ["b", "a"]


def test_ks_with_no_numeric_common_column_is_a_typed_error():
    frame = pd.DataFrame({"name": ["x", "y"]})
    with pytest.raises(ValidationError, match="No numeric common columns"):
        KolmogorovSmirnovTest().calculate_metric(frame, frame)


def test_kl_evaluates_every_common_column():
    original, transformed = _frames()
    transformed["name"] = ["x"] * 400 + ["y"] * 100       # only the categorical column drifts
    result = KLDivergence(normalize=True).calculate_metric(original[["name", "a"]], transformed[["name", "a"]])
    assert set(result["per_column"]) == {"name", "a"}
    assert result["aggregate_column"] == "name"
    assert result["kl_divergence"] == result["per_column"]["name"]["kl_divergence"]
    assert "max kl_divergence" in result["aggregate_rule"]


def test_fidelity_operation_covers_both_requested_columns(tmp_path):
    original, transformed = _frames(shift_b=5.0)
    op = FidelityOperation(columns=["a", "b"], use_cache=False)
    result = op.execute(
        DataSource(dataframes={"orig": original, "synth": transformed}), tmp_path, None,
        original_dataset_name="orig", transformed_dataset_name="synth",
    )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    import json
    metrics = json.loads(next((tmp_path / "metrics").glob("FidelityOperation_*.json")).read_text(encoding="utf-8"))["metrics"]
    assert set(metrics["ks"]["per_column"]) == {"a", "b"} and set(metrics["kl"]["per_column"]) == {"a", "b"}
    assert metrics["ks"]["aggregate_column"] == "b"
