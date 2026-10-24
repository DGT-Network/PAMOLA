"""ASB-OPS-2/3/4/5: what DCR and NNDR report, and what they do not say.

* no self-relative percentile "risk" classes (an always-5% "high risk");
* neutral descriptive text, no verdicts or recommendations;
* NNDR honours ``n_neighbors`` and nothing per-record reaches the log;
* a FAISS failure is reported in the result (``backend_fallback``) with one warning, not hidden.
"""

import logging
import types

import numpy as np
import pandas as pd
import pytest

from pamola_core.errors.exceptions import InvalidParameterError, ValidationError
from pamola_core.metrics.operations.privacy_ops import PrivacyMetricOperation
from pamola_core.metrics.privacy import distance as distance_module
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord
from pamola_core.metrics.privacy.neighbor import NearestNeighborDistanceRatio
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus

VERDICT_WORDS = ("good privacy", "poor privacy", "moderate privacy", "recommendation", "high risk", "caution", "warning")


def _frames(n=60, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("xyz"))
    return a, a + shift


def test_dcr_has_no_self_relative_risk_classes():
    original, synthetic = _frames(shift=0.5)
    result = DistanceToClosestRecord().calculate_metric(original, synthetic)
    assert "risk_assessment" not in result and "proportion_at_risk" not in result
    assert "threshold_assessment" not in result          # nothing the caller did not ask for
    assert {"min", "max", "mean", "std", "p5", "p95"} <= set(result["dcr_statistics"])


def test_dcr_thresholds_come_from_the_caller():
    original, synthetic = _frames(shift=0.0)              # identical records: every DCR is 0
    result = DistanceToClosestRecord(distance_thresholds=[0.5, 2.0]).calculate_metric(original, synthetic)
    assessment = result["threshold_assessment"]
    assert assessment["records_below"] == {"0.5": 60, "2.0": 60}
    assert assessment["proportion_below"]["0.5"] == 1.0


@pytest.mark.parametrize("shift", [0.0, 0.3, 5.0])
def test_interpretations_state_numbers_not_verdicts(shift):
    original, synthetic = _frames(shift=shift)
    dcr = DistanceToClosestRecord(distance_thresholds=[0.1]).calculate_metric(original, synthetic)
    nndr = NearestNeighborDistanceRatio(threshold=0.5).calculate_metric(original, synthetic)
    for text in (dcr["interpretation"], nndr["interpretation"]):
        assert text and not any(word in text.lower() for word in VERDICT_WORDS), text


def test_nndr_honours_n_neighbors():
    original = pd.DataFrame({"x": [0.0, 1.0, 3.0, 7.0, 15.0]})
    synthetic = pd.DataFrame({"x": [0.2]})
    second = NearestNeighborDistanceRatio(n_neighbors=2).calculate_metric(original, synthetic)
    fourth = NearestNeighborDistanceRatio(n_neighbors=4).calculate_metric(original, synthetic)
    assert second["nndr_values"][0] == pytest.approx(0.2 / 0.8, rel=1e-6)
    assert fourth["nndr_values"][0] == pytest.approx(0.2 / 6.8, rel=1e-6)
    assert fourth["n_neighbors"] == 4


def test_nndr_rejects_an_unusable_n_neighbors():
    with pytest.raises(InvalidParameterError):
        NearestNeighborDistanceRatio(n_neighbors=1)
    with pytest.raises(ValidationError, match="at least that many"):
        NearestNeighborDistanceRatio(n_neighbors=10).calculate_metric(
            pd.DataFrame({"x": [1.0, 2.0]}), pd.DataFrame({"x": [1.5]})
        )


def test_per_record_values_never_reach_the_log(tmp_path, caplog):
    original, synthetic = _frames(n=400, shift=0.1)
    op = PrivacyMetricOperation(privacy_metrics=["nndr"], use_cache=False)
    with caplog.at_level(logging.DEBUG):
        result = op.execute(
            DataSource(dataframes={"orig": original, "synth": synthetic}), tmp_path, None,
            original_dataset_name="orig", transformed_dataset_name="synth",
        )
    assert result.status == OperationStatus.SUCCESS, result.error_message
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "<400 values>" in text
    assert len(text) < 20_000      # 400 floats printed would alone be longer than a few KB


class _BrokenFaiss(types.SimpleNamespace):
    def IndexFlatL2(self, dimension):  # noqa: N802 - faiss API name
        raise RuntimeError("index build failed")


def test_a_faiss_failure_is_reported_not_hidden(monkeypatch, caplog):
    monkeypatch.setattr(distance_module, "faiss", _BrokenFaiss())
    original, synthetic = _frames(shift=0.2)
    with caplog.at_level(logging.WARNING):
        result = DistanceToClosestRecord(use_faiss=True).calculate_metric(original, synthetic)
    assert result["backend"] == "sklearn"
    assert result["backend_fallback"]["from"] == "faiss" and result["backend_fallback"]["to"] == "sklearn"
    assert "index build failed" in result["backend_fallback"]["reason"]
    assert len([r for r in caplog.records if "FAISS backend not used" in r.getMessage()]) == 1
    exact = DistanceToClosestRecord(use_faiss=False).calculate_metric(original, synthetic)
    assert result["dcr_statistics"]["mean"] == pytest.approx(exact["dcr_statistics"]["mean"])
    assert "backend_fallback" not in exact and exact["backend"] == "sklearn"


def test_an_unsupported_metric_for_faiss_is_reported_too(monkeypatch):
    class Working(types.SimpleNamespace):
        pass

    monkeypatch.setattr(distance_module, "faiss", Working())
    original, synthetic = _frames()
    result = DistanceToClosestRecord(use_faiss=True, distance_metric="manhattan").calculate_metric(original, synthetic)
    assert "manhattan" in result["backend_fallback"]["reason"]
