"""Coverage tests for privacy_risk.py — targets t-closeness, attribute inference, edge cases."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.analysis.privacy_risk import calculate_full_risk
from pamola_core.analysis.privacy_risk import (
    _calculate_l_diversity,
    _calculate_t_closeness,
    _simulate_linkage_attack,
    _simulate_attribute_inference,
)
from pamola_core.errors.exceptions import (
    ColumnNotFoundError,
    InvalidParameterError,
    ValidationError,
)


@pytest.fixture
def base_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.randint(20, 70, 200),
        "zip": [f"1000{i % 5}" for i in range(200)],
        "disease": np.random.choice(["flu", "cold", "covid"], 200),
        "salary": np.random.randint(30000, 100000, 200),
    })


class TestCalculateTCloseness:
    def test_basic(self, base_df):
        result = _calculate_t_closeness(
            base_df, quasi_identifiers=["age", "zip"],
            sensitive_attributes=["disease"],
        )
        assert isinstance(result, dict)
        assert "t" in result
        assert result["distance_metric"] == "EMD"
        assert result["num_classes"] > 0

    def test_empty_df(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = _calculate_t_closeness(
            df, quasi_identifiers=["a"], sensitive_attributes=["b"],
        )
        assert result["t"] == 0.0

    def test_no_qi_raises(self, base_df):
        with pytest.raises(ValidationError):
            _calculate_t_closeness(
                base_df, quasi_identifiers=[],
                sensitive_attributes=["disease"],
            )

    def test_no_sensitive_raises(self, base_df):
        with pytest.raises(ValidationError):
            _calculate_t_closeness(
                base_df, quasi_identifiers=["age"],
                sensitive_attributes=[],
            )

    def test_missing_columns_raises(self, base_df):
        with pytest.raises(ColumnNotFoundError):
            _calculate_t_closeness(
                base_df, quasi_identifiers=["nonexist"],
                sensitive_attributes=["disease"],
            )

    def test_invalid_metric_raises(self, base_df):
        with pytest.raises(InvalidParameterError):
            _calculate_t_closeness(
                base_df, quasi_identifiers=["age"],
                sensitive_attributes=["disease"],
                distance_metric="INVALID",
            )


class TestLDiversityEdgeCases:
    def test_empty_df(self):
        df = pd.DataFrame({"a": [], "b": []})
        result, entropy = _calculate_l_diversity(
            df, quasi_identifiers=["a"], sensitive_attributes=["b"],
        )
        assert result["l"] == 0

    def test_no_qi_raises(self, base_df):
        with pytest.raises(ValidationError):
            _calculate_l_diversity(
                base_df, quasi_identifiers=[],
                sensitive_attributes=["disease"],
            )


class TestSimulateLinkageAttack:
    def test_no_group_cols_in_df(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        risk = _simulate_linkage_attack(
            df, quasi_identifiers=["missing"], sensitive_attributes=["also_missing"],
        )
        assert risk == 0.0


class TestSimulateAttributeInference:
    def test_basic(self, base_df):
        risk = _simulate_attribute_inference(
            base_df, quasi_identifiers=["age", "zip"],
            sensitive_attributes=["disease"],
        )
        assert 0.0 <= risk <= 1.0

    def test_empty_df(self):
        df = pd.DataFrame({"a": [], "b": []})
        risk = _simulate_attribute_inference(
            df, quasi_identifiers=["a"], sensitive_attributes=["b"],
        )
        assert risk == 0.0

    def test_no_qi_and_no_sa_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValidationError):
            _simulate_attribute_inference(
                df, quasi_identifiers=[], sensitive_attributes=[],
            )


class TestCalculateFullRiskWeights:
    def test_custom_weights(self, base_df):
        result = calculate_full_risk(
            base_df,
            quasi_identifiers=["age", "zip"],
            sensitive_attributes=["disease"],
            weights={
                "k_anonymity": 0.25,
                "l_diversity": 0.25,
                "attribute_disclosure_risk": 0.25,
                "membership_inference_risk": 0.25,
            },
        )
        assert isinstance(result, dict)
        assert "risk_assessment" in result
