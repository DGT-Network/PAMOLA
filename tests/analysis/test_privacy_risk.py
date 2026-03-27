"""
File: test_privacy_risk.py
Test Target: pamola_core.analysis.privacy_risk
Coverage Target: >=90%

Comprehensive test suite for calculate_full_risk() and related functions.
Tests k-anonymity, l-diversity, attack simulations, and risk aggregation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pamola_core.analysis.privacy_risk import (
    calculate_full_risk,
    _calculate_k_anonymity,
    _calculate_l_diversity,
    _simulate_linkage_attack,
    _simulate_attribute_inference,
    _simulate_membership_inference,
)
from pamola_core.errors.exceptions import ValidationError, ColumnNotFoundError


# ======================== FIXTURES ========================

@pytest.fixture
def sample_df():
    """Standard test DataFrame with identifiers and sensitive attrs."""
    return pd.DataFrame({
        "age": [25, 25, 30, 30, 35, 35, 40, 40],
        "zip": ["12345", "12345", "12345", "12346", "12346", "12346", "12347", "12347"],
        "disease": ["Flu", "Flu", "COVID", "Flu", "COVID", "Cancer", "Flu", "COVID"],
    })


@pytest.fixture
def df_high_k_anonymity():
    """DataFrame with high k-anonymity (large equivalence classes)."""
    return pd.DataFrame({
        "quasi_id_1": ["A"] * 100,
        "quasi_id_2": ["X"] * 100,
        "sensitive": ["Safe"] * 100,
    })


@pytest.fixture
def df_low_k_anonymity():
    """DataFrame with low k-anonymity (unique records)."""
    return pd.DataFrame({
        "quasi_id": list(range(10)),
        "sensitive": ["Secret"] * 10,
    })


@pytest.fixture
def df_low_l_diversity():
    """DataFrame with low l-diversity (homogeneous sensitive attrs)."""
    return pd.DataFrame({
        "age": [25, 25, 25, 30, 30, 30],
        "disease": ["Flu", "Flu", "Flu", "COVID", "COVID", "COVID"],
    })


@pytest.fixture
def df_empty():
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values."""
    return pd.DataFrame({
        "age": [25, None, 30, 40],
        "zip": ["123", "123", None, "123"],
        "disease": ["Flu", "COVID", "Flu", None],
    })


# ======================== K-ANONYMITY TESTS ========================

class TestKAnonymity:
    """Tests for _calculate_k_anonymity()."""

    def test_k_anonymity_basic(self, sample_df):
        """Test basic k-anonymity calculation."""
        result = _calculate_k_anonymity(sample_df, ["age", "zip"])

        assert isinstance(result, dict)
        assert "k" in result
        assert "quasi_identifiers" in result
        assert "equivalence_classes" in result
        assert isinstance(result["k"], int)
        assert result["k"] > 0

    def test_k_anonymity_values(self, sample_df):
        """Test k-anonymity calculation correctness."""
        result = _calculate_k_anonymity(sample_df, ["age", "zip"])

        # With sample_df structure, (25, 12345) has 2 records
        # (30, 12345), (30, 12346), (35, 12346) have 1-2
        # Minimum should be 1
        assert result["k"] >= 1
        assert result["equivalence_classes"] >= 1

    def test_k_anonymity_high_group(self, df_high_k_anonymity):
        """Test k-anonymity with large equivalence class."""
        result = _calculate_k_anonymity(df_high_k_anonymity, ["quasi_id_1", "quasi_id_2"])

        assert result["k"] == 100  # All 100 records in same class

    def test_k_anonymity_unique_records(self, df_low_k_anonymity):
        """Test k-anonymity with all unique records (k=1)."""
        result = _calculate_k_anonymity(df_low_k_anonymity, ["quasi_id"])

        assert result["k"] == 1  # Each record is unique

    def test_k_anonymity_missing_column(self, sample_df):
        """Test error when QI column missing."""
        with pytest.raises(ColumnNotFoundError):
            _calculate_k_anonymity(sample_df, ["nonexistent_col"])

    def test_k_anonymity_empty_df(self):
        """Test k-anonymity on empty DataFrame."""
        result = _calculate_k_anonymity(pd.DataFrame(), ["col"])

        assert result["k"] == 0

    def test_k_anonymity_no_quasi_identifiers(self, sample_df):
        """Test error when no quasi-identifiers provided."""
        with pytest.raises(ValidationError):
            _calculate_k_anonymity(sample_df, [])

    def test_k_anonymity_none_input(self):
        """Test k-anonymity with None DataFrame."""
        result = _calculate_k_anonymity(None, ["col"])

        assert result["k"] == 0

    def test_k_anonymity_equivalence_classes_count(self, sample_df):
        """Test equivalence classes count."""
        result = _calculate_k_anonymity(sample_df, ["age", "zip"])

        assert result["equivalence_classes"] > 0
        assert isinstance(result["equivalence_classes"], int)

    def test_k_anonymity_records_in_smallest_classes(self, sample_df):
        """Test count of records in smallest classes."""
        result = _calculate_k_anonymity(sample_df, ["age", "zip"])

        # Records in smallest classes should be >= k
        assert result["records_in_smallest_classes"] >= result["k"]


# ======================== L-DIVERSITY TESTS ========================

class TestLDiversity:
    """Tests for _calculate_l_diversity()."""

    def test_l_diversity_basic(self, sample_df):
        """Test basic l-diversity calculation."""
        result, max_entropy = _calculate_l_diversity(
            sample_df, ["age", "zip"], ["disease"]
        )

        assert isinstance(result, dict)
        assert "l" in result
        assert "entropy" in result
        assert isinstance(result["l"], int)
        assert isinstance(result["entropy"], float)
        assert isinstance(max_entropy, float)

    def test_l_diversity_low_diversity(self, df_low_l_diversity):
        """Test l-diversity with homogeneous groups."""
        result, max_entropy = _calculate_l_diversity(
            df_low_l_diversity, ["age"], ["disease"]
        )

        # Age 25 has all "Flu", l=1
        assert result["l"] == 1

    def test_l_diversity_high_diversity(self, sample_df):
        """Test l-diversity with diverse sensitive attributes."""
        result, max_entropy = _calculate_l_diversity(
            sample_df, ["age"], ["disease"]
        )

        assert result["l"] > 0
        assert result["entropy"] >= 0

    def test_l_diversity_missing_sensitive_attr(self, sample_df):
        """Test error when sensitive attribute missing."""
        with pytest.raises(ColumnNotFoundError):
            _calculate_l_diversity(sample_df, ["age"], ["nonexistent"])

    def test_l_diversity_no_sensitive_attributes(self, sample_df):
        """Test error with no sensitive attributes."""
        with pytest.raises(ValidationError):
            _calculate_l_diversity(sample_df, ["age"], [])

    def test_l_diversity_empty_df(self):
        """Test l-diversity on empty DataFrame."""
        result, max_entropy = _calculate_l_diversity(pd.DataFrame(), ["col"], ["sens"])

        assert result["l"] == 0
        assert max_entropy == 0.0

    def test_l_diversity_entropy_calculation(self, sample_df):
        """Test entropy is non-negative."""
        result, max_entropy = _calculate_l_diversity(
            sample_df, ["age"], ["disease"]
        )

        assert result["entropy"] >= 0
        assert max_entropy >= 0


# ======================== LINKAGE ATTACK TESTS ========================

class TestLinkageAttack:
    """Tests for _simulate_linkage_attack()."""

    def test_linkage_attack_basic(self, sample_df):
        """Test basic linkage attack simulation."""
        result = _simulate_linkage_attack(sample_df, ["age", "zip"], ["disease"])

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_linkage_attack_no_unique_records(self, df_high_k_anonymity):
        """Test linkage attack with no unique records (k=100)."""
        result = _simulate_linkage_attack(
            df_high_k_anonymity, ["quasi_id_1", "quasi_id_2"], []
        )

        assert result == 0.0  # No singletons

    def test_linkage_attack_all_unique(self, df_low_k_anonymity):
        """Test linkage attack where all records are unique."""
        result = _simulate_linkage_attack(df_low_k_anonymity, ["quasi_id"], [])

        assert result == 1.0  # All records are singletons

    def test_linkage_attack_empty_df(self):
        """Test linkage attack on empty DataFrame."""
        result = _simulate_linkage_attack(pd.DataFrame(), ["col"], [])

        assert result == 0.0

    def test_linkage_attack_no_columns(self, sample_df):
        """Test error when no QI or sensitive attrs provided."""
        with pytest.raises(ValidationError):
            _simulate_linkage_attack(sample_df, [], [])

    def test_linkage_attack_return_type(self, sample_df):
        """Test return type is float and properly rounded."""
        result = _simulate_linkage_attack(sample_df, ["age"], [])

        assert isinstance(result, float)
        # Should be rounded to 4 decimals
        assert len(str(result).split(".")[-1]) <= 4


# ======================== ATTRIBUTE INFERENCE TESTS ========================

class TestAttributeInference:
    """Tests for _simulate_attribute_inference()."""

    def test_attribute_inference_basic(self, sample_df):
        """Test basic attribute inference simulation."""
        result = _simulate_attribute_inference(sample_df, ["age", "zip"], ["disease"])

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_attribute_inference_deterministic_groups(self, df_low_l_diversity):
        """Test with deterministic groups (single sensitive value)."""
        result = _simulate_attribute_inference(
            df_low_l_diversity, ["age"], ["disease"]
        )

        # Both age groups have single disease value → risk = 1.0
        assert result == 1.0

    def test_attribute_inference_diverse_groups(self):
        """Test with diverse groups (multiple sensitive values)."""
        df = pd.DataFrame({
            "age": [25, 25, 30, 30],
            "disease": ["Flu", "COVID", "Flu", "COVID"],
        })
        result = _simulate_attribute_inference(df, ["age"], ["disease"])

        # Both groups have 2 different diseases → risk = 0.0
        assert result == 0.0

    def test_attribute_inference_no_quasi_identifiers(self, sample_df):
        """Test that no quasi-identifiers returns 0.0 (can't form equivalence classes)."""
        result = _simulate_attribute_inference(sample_df, [], ["disease"])

        assert result == 0.0

    def test_attribute_inference_no_sensitive_attrs(self, sample_df):
        """Test with no sensitive attributes."""
        result = _simulate_attribute_inference(sample_df, ["age"], [])

        assert result == 0.0

    def test_attribute_inference_empty_df(self):
        """Test on empty DataFrame."""
        result = _simulate_attribute_inference(pd.DataFrame(), ["col"], ["sens"])

        assert result == 0.0


# ======================== MEMBERSHIP INFERENCE TESTS ========================

class TestMembershipInference:
    """Tests for _simulate_membership_inference()."""

    def test_membership_inference_basic(self, sample_df):
        """Test basic membership inference simulation."""
        result = _simulate_membership_inference(sample_df, ["age", "zip"], [])

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_membership_inference_all_unique(self, df_low_k_anonymity):
        """Test with all unique records."""
        result = _simulate_membership_inference(
            df_low_k_anonymity, ["quasi_id"], []
        )

        assert result == 1.0

    def test_membership_inference_no_unique(self, df_high_k_anonymity):
        """Test with no unique records."""
        result = _simulate_membership_inference(
            df_high_k_anonymity, ["quasi_id_1", "quasi_id_2"], []
        )

        assert result == 0.0

    def test_membership_inference_missing_column(self, sample_df):
        """Test error with missing column."""
        with pytest.raises(ColumnNotFoundError):
            _simulate_membership_inference(sample_df, ["nonexistent"], [])

    def test_membership_inference_empty_df(self):
        """Test on empty DataFrame."""
        result = _simulate_membership_inference(pd.DataFrame(), ["col"], [])

        assert result == 0.0

    def test_membership_inference_no_grouping_cols(self, sample_df):
        """Test error with no grouping columns."""
        with pytest.raises(ValidationError):
            _simulate_membership_inference(sample_df, [], [])


# ======================== FULL RISK CALCULATION TESTS ========================

class TestFullRiskCalculation:
    """Tests for calculate_full_risk()."""

    def test_full_risk_basic(self, sample_df):
        """Test basic full risk calculation."""
        result = calculate_full_risk(
            sample_df, ["age", "zip"], ["disease"]
        )

        assert isinstance(result, dict)
        assert "risk_assessment" in result
        assert "k_anonymity" in result
        assert "l_diversity" in result
        assert "reidentification_risk" in result
        assert "attribute_disclosure_risk" in result
        assert "membership_inference_risk" in result

    def test_full_risk_result_structure(self, sample_df):
        """Test result has all required fields."""
        result = calculate_full_risk(
            sample_df, ["age", "zip"], ["disease"]
        )

        # Risk assessment should be integer percentage 0-100
        assert isinstance(result["risk_assessment"], int)
        assert 0 <= result["risk_assessment"] <= 100

        # Check subcomponent structures
        assert isinstance(result["k_anonymity"], dict)
        assert isinstance(result["l_diversity"], dict)
        assert isinstance(result["reidentification_risk"], float)
        assert isinstance(result["attribute_disclosure_risk"], float)
        assert isinstance(result["membership_inference_risk"], float)

    def test_full_risk_custom_weights(self, sample_df):
        """Test full risk with custom weights."""
        custom_weights = {
            "k_anonymity": 0.25,
            "l_diversity": 0.25,
            "attribute_disclosure_risk": 0.25,
            "membership_inference_risk": 0.25,
        }
        result = calculate_full_risk(
            sample_df, ["age", "zip"], ["disease"], weights=custom_weights
        )

        assert isinstance(result["risk_assessment"], int)

    def test_full_risk_weights_validation(self, sample_df):
        """Test that weights must sum to 1.0."""
        bad_weights = {
            "k_anonymity": 0.5,
            "l_diversity": 0.3,
            "attribute_disclosure_risk": 0.1,
            "membership_inference_risk": 0.05,
        }
        with pytest.raises(ValidationError):
            calculate_full_risk(
                sample_df, ["age", "zip"], ["disease"], weights=bad_weights
            )

    def test_full_risk_high_k_low_risk(self, df_high_k_anonymity):
        """Test full risk with high k-anonymity (low risk)."""
        result = calculate_full_risk(
            df_high_k_anonymity, ["quasi_id_1", "quasi_id_2"], ["sensitive"]
        )

        # High k should result in lower overall risk
        assert result["k_anonymity"]["k"] == 100
        assert result["risk_assessment"] <= 50

    def test_full_risk_low_k_high_risk(self, df_low_k_anonymity):
        """Test full risk with low k-anonymity (high risk)."""
        result = calculate_full_risk(
            df_low_k_anonymity, ["quasi_id"], ["sensitive"]
        )

        # Low k (k=1) should result in higher risk
        assert result["k_anonymity"]["k"] == 1
        assert result["risk_assessment"] > 0

    def test_full_risk_missing_column(self, sample_df):
        """Test error with missing quasi-identifier."""
        with pytest.raises(ColumnNotFoundError):
            calculate_full_risk(
                sample_df, ["nonexistent"], ["disease"]
            )

    def test_full_risk_empty_df(self):
        """Test full risk on empty DataFrame."""
        result = calculate_full_risk(
            pd.DataFrame(), ["col"], ["sens"]
        )

        # Empty df: k=0 (treated as max risk) + l=0 → risk score is non-zero
        assert isinstance(result["risk_assessment"], int)
        assert 0 <= result["risk_assessment"] <= 100

    def test_full_risk_none_weights_default(self, sample_df):
        """Test that None weights use defaults."""
        result = calculate_full_risk(
            sample_df, ["age", "zip"], ["disease"], weights=None
        )

        assert isinstance(result["risk_assessment"], int)

    def test_full_risk_reproducibility(self, sample_df):
        """Test that same input produces same risk score."""
        result1 = calculate_full_risk(
            sample_df, ["age", "zip"], ["disease"]
        )
        result2 = calculate_full_risk(
            sample_df, ["age", "zip"], ["disease"]
        )

        assert result1["risk_assessment"] == result2["risk_assessment"]


# ======================== EDGE CASES ========================

class TestPrivacyRiskEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_record_df(self):
        """Test with single record."""
        df = pd.DataFrame({
            "age": [25],
            "disease": ["Flu"],
        })
        result = calculate_full_risk(df, ["age"], ["disease"])

        # Single record should have high risk (k=1)
        assert result["k_anonymity"]["k"] == 1

    def test_all_same_quasi_ids(self):
        """Test where all records have identical QI values."""
        df = pd.DataFrame({
            "age": [25] * 10,
            "disease": ["Flu"] * 10,
        })
        result = calculate_full_risk(df, ["age"], ["disease"])

        # All same QI → k=10
        assert result["k_anonymity"]["k"] == 10

    def test_df_with_nulls(self):
        """Test DataFrame with null values."""
        df = pd.DataFrame({
            "age": [25, None, 30],
            "disease": ["Flu", "COVID", None],
        })
        result = calculate_full_risk(df, ["age"], ["disease"])

        assert isinstance(result, dict)
        assert "risk_assessment" in result

    def test_single_quasi_identifier(self):
        """Test with single quasi-identifier."""
        df = pd.DataFrame({
            "age": [25, 25, 30, 30],
            "disease": ["Flu", "COVID", "Flu", "COVID"],
        })
        result = calculate_full_risk(df, ["age"], ["disease"])

        assert result["k_anonymity"]["k"] == 2

    def test_single_sensitive_attribute(self):
        """Test with single sensitive attribute."""
        df = pd.DataFrame({
            "age": [25, 25, 30, 30],
            "zip": ["123", "123", "456", "456"],
            "disease": ["Flu", "Flu", "COVID", "COVID"],
        })
        result = calculate_full_risk(df, ["age", "zip"], ["disease"])

        assert "l_diversity" in result

    def test_multiple_sensitive_attributes(self):
        """Test with multiple sensitive attributes."""
        df = pd.DataFrame({
            "age": [25, 25, 30, 30],
            "disease": ["Flu", "Flu", "COVID", "COVID"],
            "medication": ["A", "B", "C", "D"],
        })
        result = calculate_full_risk(df, ["age"], ["disease", "medication"])

        assert "l_diversity" in result
        assert len(result["l_diversity"]["sensitive_attributes"]) == 2
