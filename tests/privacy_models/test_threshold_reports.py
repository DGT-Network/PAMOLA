"""
DOC-02: privacy-model reports state threshold results, never regulatory verdicts.

The library does not map GDPR/HIPAA/CCPA/... to thresholds and does not label a
dataset "compliant". Thresholds are chosen by the caller; results are named
``threshold_met``.
"""

import inspect
import pathlib

import pandas as pd
import pytest

from pamola_core.privacy_models.k_anonymity import calculation as ka_calc
from pamola_core.privacy_models.k_anonymity import ka_reporting
from pamola_core.privacy_models.k_anonymity.calculation import KAnonymityProcessor
from pamola_core.privacy_models.l_diversity.interpretation import (
    RiskInterpreter,
    interpret_risk_metrics,
)
from pamola_core.privacy_models.l_diversity.reporting import LDiversityReport

PKG = pathlib.Path(ka_calc.__file__).resolve().parents[2]


def _report_data(min_k, k_value=5):
    return {
        "k_anonymity_configuration": {"k_value": k_value},
        "privacy_evaluation": {
            "min_k": min_k,
            "at_risk_records": 4,
            "percentage_at_risk": 40.0,
        },
    }


# --- k-anonymity --------------------------------------------------------------


def test_k_threshold_report_has_neutral_fields_only():
    report = ka_reporting.generate_threshold_report(_report_data(min_k=3))

    assert report["k_threshold"] == 5
    assert report["actual_min_k"] == 3
    assert report["threshold_met"] is False
    assert report["at_risk_records"] == 4
    assert report["at_risk_percentage"] == 40.0
    for forbidden in ("regulation", "compliance_status", "compliance_details"):
        assert forbidden not in report
    assert "below the threshold" in report["reason"]


def test_k_threshold_is_the_callers_not_a_laws():
    data = _report_data(min_k=7, k_value=2)
    assert ka_reporting.generate_threshold_report(data)["threshold_met"] is True
    assert ka_reporting.generate_threshold_report(data, k_threshold=10)["threshold_met"] is False


def test_k_threshold_report_is_written_when_a_path_is_given(tmp_path):
    out = tmp_path / "r.json"
    ka_reporting.generate_threshold_report(_report_data(6), output_path=str(out))
    assert out.exists() and "threshold_met" in out.read_text(encoding="utf-8")


def test_processor_generate_report_threshold_option():
    df = pd.DataFrame({"age": [1, 1, 2, 2], "city": list("aabb")})
    processor = KAnonymityProcessor(k=2, progress_tracking=False)
    processor.evaluate_privacy(df, ["age", "city"])
    report = processor.generate_report(threshold_report=True)
    assert report["k_threshold"] == 2 and report["threshold_met"] is True
    assert "regulation" not in inspect.signature(processor.generate_report).parameters


def test_k_evaluate_privacy_result_uses_threshold_met():
    df = pd.DataFrame({"age": [1, 1, 2], "city": list("aab")})
    result = KAnonymityProcessor(k=2, progress_tracking=False).evaluate_privacy(
        df, ["age", "city"]
    )
    assert result["threshold_met"] == False  # noqa: E712 - numpy.bool_
    assert "compliant" not in result


def test_text_summary_says_threshold_met():
    data = _report_data(6)
    data["privacy_evaluation"]["threshold_met"] = True
    text = ka_reporting.KAnonymityReport(data).get_summary()
    assert "Threshold met: Yes" in text and "Compliance" not in text


# --- l-diversity ---------------------------------------------------------------


def test_risk_interpreter_threshold_is_caller_chosen():
    result = RiskInterpreter(l_threshold=4).interpret_threshold(3, "distinct")
    assert result["threshold_met"] is False and result["l_threshold"] == 4
    assert RiskInterpreter().interpret_threshold(3, "distinct")["threshold_met"] is True
    assert RiskInterpreter().interpret_threshold(3, "distinct", l_threshold=5)["l_threshold"] == 5
    assert "regulation" not in inspect.signature(RiskInterpreter.__init__).parameters


def test_interpret_risk_metrics_has_threshold_check_not_compliance():
    out = interpret_risk_metrics({"overall_risk": {"min_diversity": 2}}, target_l=3)
    assert "compliance" not in out
    assert out["threshold_check"]["threshold_met"] is False


def test_l_diversity_report_methods_take_thresholds_not_regulations():
    params = inspect.signature(LDiversityReport.generate_threshold_report).parameters
    assert {"l_threshold", "k_threshold", "diversity_type", "c_max"} <= set(params)
    assert "regulation" not in params
    assert not hasattr(LDiversityReport, "generate_compliance_report")


# --- guard: no law is mapped to a threshold anywhere in privacy_models -----------


@pytest.mark.parametrize("relative", ["privacy_models", "configs", "metrics/privacy"])
def test_no_regulation_names_in_privacy_code(relative):
    offenders = []
    for path in (PKG / relative).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("GDPR", "HIPAA", "CCPA", "PIPEDA", "regulatory_requirements",
                       "compliance_status", "regulation:"):
            if needle in text:
                offenders.append(f"{path.name}:{needle}")
    assert offenders == []


def test_threshold_check_visualization_runs_without_a_regulation(tmp_path):
    from pamola_core.privacy_models.l_diversity.advanced_visualization import (
        LDiversityAdvancedVisualizer,
    )

    df = pd.DataFrame(
        {
            "age": [1, 1, 1, 2, 2, 2],
            "city": list("aaabbb"),
            "disease": list("xyzxyz"),
        }
    )
    params = inspect.signature(LDiversityAdvancedVisualizer.visualize_threshold_check).parameters
    assert "regulation" not in params and "l_threshold" in params

    fig, saved = LDiversityAdvancedVisualizer(None).visualize_threshold_check(
        df, ["age", "city"], ["disease"], l_threshold=3, save_path=tmp_path
    )
    assert fig is not None
    assert saved is None or "threshold_check" in str(saved)
