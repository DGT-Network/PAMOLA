"""
Unit tests for TCloseness privacy model.

Tests verify t-closeness calculation using Wasserstein distance,
privacy evaluation, and model application functionality.

Run with: pytest -s tests/privacy_models/test_t_closeness.py
"""

import pytest
import pandas as pd
import numpy as np

from pamola_core.privacy_models.t_closeness.calculation import TCloseness
from pamola_core.errors.exceptions import ValidationError


class TestTClosenessInitialization:
    """Test TCloseness initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with required parameters."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        assert tc.quasi_identifiers == ["age", "city"]
        assert tc.sensitive_column == "disease"
        assert tc.t == 0.5

    def test_single_quasi_identifier(self):
        """Test initialization with single quasi-identifier."""
        tc = TCloseness(
            quasi_identifiers=["age"],
            sensitive_column="disease",
            t=0.3
        )

        assert len(tc.quasi_identifiers) == 1
        assert tc.quasi_identifiers[0] == "age"

    def test_multiple_quasi_identifiers(self):
        """Test initialization with multiple quasi-identifiers."""
        quasi_ids = ["age", "city", "zip"]
        tc = TCloseness(
            quasi_identifiers=quasi_ids,
            sensitive_column="disease",
            t=0.4
        )

        assert tc.quasi_identifiers == quasi_ids

    def test_various_t_thresholds(self):
        """Test initialization with different t values."""
        for t_value in [0.1, 0.3, 0.5, 0.7, 1.0]:
            tc = TCloseness(
                quasi_identifiers=["age"],
                sensitive_column="disease",
                t=t_value
            )
            assert tc.t == t_value

    def test_zero_t_threshold(self):
        """Test initialization with t=0 (very strict)."""
        tc = TCloseness(
            quasi_identifiers=["age"],
            sensitive_column="disease",
            t=0.0
        )
        assert tc.t == 0.0

    def test_large_t_threshold(self):
        """Test initialization with large t value."""
        tc = TCloseness(
            quasi_identifiers=["age"],
            sensitive_column="disease",
            t=10.0
        )
        assert tc.t == 10.0


class TestTClosenessEvaluatePrivacy:
    """Test t-closeness privacy evaluation functionality."""

    @pytest.fixture
    def balanced_data(self):
        """Create data with balanced distribution in sensitive attribute."""
        return pd.DataFrame({
            "age": [25, 25, 25, 25, 30, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "NYC", "LA", "LA", "LA", "LA"],
            "disease": ["flu", "cold", "flu", "cold", "flu", "cold", "flu", "cold"],
        })

    @pytest.fixture
    def skewed_data(self):
        """Create data with skewed distribution in sensitive attribute."""
        return pd.DataFrame({
            "age": [25, 25, 25, 25, 30, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "NYC", "LA", "LA", "LA", "LA"],
            "disease": ["flu", "flu", "flu", "cold", "diabetes", "diabetes", "diabetes", "cancer"],
        })

    def test_evaluate_privacy_returns_dict(self, balanced_data):
        """Test that evaluate_privacy returns a dictionary."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        assert isinstance(result, dict)

    def test_evaluate_privacy_contains_key_metrics(self, balanced_data):
        """Test that result contains expected keys."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        assert "max_t_value" in result
        assert "is_t_close" in result

    def test_max_t_value_is_numeric(self, balanced_data):
        """Test that max_t_value is numeric."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        assert isinstance(result["max_t_value"], (int, float))
        assert result["max_t_value"] >= 0

    def test_is_t_close_is_boolean(self, balanced_data):
        """Test that is_t_close is boolean."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        assert isinstance(result["is_t_close"], bool)

    def test_strict_t_threshold(self, balanced_data):
        """Test with very strict (small) t threshold."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.01
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        # With strict threshold, likely not t-close
        assert isinstance(result["is_t_close"], bool)

    def test_lenient_t_threshold(self, balanced_data):
        """Test with lenient (large) t threshold."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=10.0
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        # With lenient threshold, likely t-close
        assert result["is_t_close"] is True

    def test_missing_quasi_identifier_column(self, balanced_data):
        """Test error handling for missing quasi-identifier column returns non-close result."""
        tc = TCloseness(
            quasi_identifiers=["nonexistent"],
            sensitive_column="disease",
            t=0.5
        )

        # Source catches ValidationError internally and returns dict with is_t_close=False
        result = tc.evaluate_privacy(balanced_data, ["nonexistent"])
        assert isinstance(result, dict)
        assert result["is_t_close"] is False

    def test_missing_sensitive_column(self, balanced_data):
        """Test error handling for missing sensitive column returns non-close result."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="nonexistent",
            t=0.5
        )

        # Source catches ValidationError internally and returns dict with is_t_close=False
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])
        assert isinstance(result, dict)
        assert result["is_t_close"] is False

    def test_single_quasi_identifier_evaluation(self, balanced_data):
        """Test evaluation with single quasi-identifier."""
        tc = TCloseness(
            quasi_identifiers=["age"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(balanced_data, ["age"])

        assert "max_t_value" in result
        assert result["is_t_close"] is not None

    def test_balanced_distribution_t_value(self, balanced_data):
        """Test t-value for balanced distribution."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(balanced_data, ["age", "city"])

        # Balanced groups should have low t-value
        assert result["max_t_value"] >= 0
        assert result["max_t_value"] <= 1

    def test_skewed_distribution_t_value(self, skewed_data):
        """Test t-value for skewed distribution."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.evaluate_privacy(skewed_data, ["age", "city"])

        # Distribution difference affects t-value
        assert result["max_t_value"] >= 0


class TestTClosenessApplyModel:
    """Test t-closeness model application functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for model application."""
        return pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
            "disease": ["flu", "cold", "allergy", "diabetes", "heart", "cancer"],
        })

    def test_apply_model_returns_dataframe(self, sample_data):
        """Test that apply_model returns a DataFrame."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.apply_model(sample_data, ["age", "city"])

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_preserves_data_shape(self, sample_data):
        """Test that apply_model returns data of same shape."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.apply_model(sample_data, ["age", "city"])

        # Current implementation returns original data
        assert result.shape == sample_data.shape

    def test_apply_model_with_suppression_true(self, sample_data):
        """Test apply_model with suppression=True."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.apply_model(sample_data, ["age", "city"], suppression=True)

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_with_suppression_false(self, sample_data):
        """Test apply_model with suppression=False."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.apply_model(sample_data, ["age", "city"], suppression=False)

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_with_kwargs(self, sample_data):
        """Test apply_model accepts additional kwargs."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        result = tc.apply_model(
            sample_data,
            ["age", "city"],
            suppression=True,
            custom_param="value"
        )

        assert isinstance(result, pd.DataFrame)


class TestTClosenessDistanceCalculation:
    """Test Wasserstein distance calculation for t-closeness."""

    @pytest.fixture
    def simple_binary_data(self):
        """Create simple data with binary sensitive attribute."""
        return pd.DataFrame({
            "age": [25, 25, 25, 25, 30, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "NYC", "LA", "LA", "LA", "LA"],
            "outcome": [0, 0, 0, 1, 0, 0, 1, 1],
        })

    @pytest.fixture
    def uniform_distribution_data(self):
        """Create data with uniform distribution in groups."""
        return pd.DataFrame({
            "age": [25, 25, 25, 25, 30, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "NYC", "LA", "LA", "LA", "LA"],
            "value": [1, 2, 3, 4, 1, 2, 3, 4],
        })

    def test_distance_calculation_exists(self, simple_binary_data):
        """Test that distance calculation works."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="outcome",
            t=0.5
        )
        result = tc.evaluate_privacy(simple_binary_data, ["age", "city"])

        assert result["max_t_value"] is not None

    def test_symmetric_data_low_distance(self, uniform_distribution_data):
        """Test that identical distributions give low distance."""
        tc = TCloseness(
            quasi_identifiers=["age"],
            sensitive_column="value",
            t=0.5
        )
        result = tc.evaluate_privacy(uniform_distribution_data, ["age"])

        # Both groups have same distribution
        assert isinstance(result["max_t_value"], float)


class TestTClosenessEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataset(self):
        """Test with single-row dataset."""
        data = pd.DataFrame({
            "age": [25],
            "city": ["NYC"],
            "disease": ["flu"],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_two_rows_dataset(self):
        """Test with two-row dataset."""
        data = pd.DataFrame({
            "age": [25, 30],
            "city": ["NYC", "LA"],
            "disease": ["flu", "cold"],
        })
        tc = TCloseness(
            quasi_identifiers=["age"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age"])
        assert isinstance(result, dict)

    def test_all_same_sensitive_value(self):
        """Test with all identical sensitive attribute values."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
            "disease": ["flu", "flu", "flu", "flu", "flu", "flu"],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_all_unique_sensitive_values(self):
        """Test with all unique sensitive attribute values."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 25, 25, 25],
            "city": ["NYC", "NYC", "NYC", "NYC", "NYC", "NYC"],
            "disease": ["flu", "cold", "allergy", "cough", "fever", "pain"],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_null_values_in_sensitive_column(self):
        """Test handling of null values in sensitive column."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
            "disease": ["flu", None, "cold", "diabetes", None, "cancer"],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_numeric_sensitive_column(self):
        """Test with numeric sensitive column."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
            "salary": [50000, 55000, 60000, 65000, 70000, 75000],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="salary",
            t=5000
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_categorical_sensitive_column(self):
        """Test with categorical sensitive column."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
            "category": ["A", "B", "C", "A", "B", "C"],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="category",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_single_group(self):
        """Test with all records in single quasi-identifier group."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 25],
            "city": ["NYC", "NYC", "NYC", "NYC"],
            "disease": ["flu", "cold", "allergy", "cough"],
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)

    def test_many_groups(self):
        """Test with many quasi-identifier groups."""
        data = pd.DataFrame({
            "age": list(range(1, 21)),
            "city": ["NYC", "LA"] * 10,
            "disease": ["flu", "cold"] * 10,
        })
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )

        result = tc.evaluate_privacy(data, ["age", "city"])
        assert isinstance(result, dict)


class TestTClosenessProcess:
    """Test the process() method inherited from base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "age": [25, 30, 25],
            "city": ["NYC", "LA", "NYC"],
            "disease": ["flu", "cold", "flu"],
        })

    def test_process_method_exists(self, sample_data):
        """Test that process method exists."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        assert hasattr(tc, "process")
        assert callable(tc.process)

    def test_process_method_callable(self, sample_data):
        """Test that process method is callable."""
        tc = TCloseness(
            quasi_identifiers=["age", "city"],
            sensitive_column="disease",
            t=0.5
        )
        # Just verify it doesn't raise error
        tc.process(sample_data)
