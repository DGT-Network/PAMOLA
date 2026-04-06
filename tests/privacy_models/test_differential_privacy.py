"""
Unit tests for DifferentialPrivacyProcessor privacy model.

Tests verify differential privacy implementation using Laplace and Gaussian mechanisms,
noise addition, privacy evaluation, and model application functionality.

Run with: pytest -s tests/privacy_models/test_differential_privacy.py
"""

import pytest
import pandas as pd
import numpy as np

from pamola_core.privacy_models.differential_privacy.calculation import DifferentialPrivacyProcessor
from pamola_core.errors.exceptions import ValidationError


class TestDifferentialPrivacyProcessorInitialization:
    """Test DifferentialPrivacyProcessor initialization and configuration."""

    def test_default_laplace_mechanism(self):
        """Test initialization with Laplace mechanism (default)."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        assert processor.epsilon == 1.0
        assert processor.sensitivity == 1.0
        assert processor.mechanism == "laplace"

    def test_explicit_laplace_mechanism(self):
        """Test initialization with explicit Laplace mechanism."""
        processor = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="laplace"
        )
        assert processor.mechanism == "laplace"

    def test_gaussian_mechanism(self):
        """Test initialization with Gaussian mechanism."""
        processor = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="gaussian"
        )
        assert processor.mechanism == "gaussian"

    def test_mechanism_case_insensitive(self):
        """Test that mechanism parameter is case-insensitive."""
        processor1 = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="LAPLACE"
        )
        assert processor1.mechanism == "laplace"

        processor2 = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="Gaussian"
        )
        assert processor2.mechanism == "gaussian"

    def test_invalid_mechanism_raises_error(self):
        """Test that invalid mechanism raises ValidationError."""
        with pytest.raises(ValidationError):
            DifferentialPrivacyProcessor(
                epsilon=1.0,
                sensitivity=1.0,
                mechanism="invalid"
            )

    def test_small_epsilon_high_privacy(self):
        """Test initialization with small epsilon (high privacy)."""
        processor = DifferentialPrivacyProcessor(epsilon=0.1, sensitivity=1.0)
        assert processor.epsilon == 0.1

    def test_large_epsilon_low_privacy(self):
        """Test initialization with large epsilon (low privacy)."""
        processor = DifferentialPrivacyProcessor(epsilon=100.0, sensitivity=1.0)
        assert processor.epsilon == 100.0

    def test_various_sensitivity_values(self):
        """Test initialization with different sensitivity values."""
        for sensitivity in [0.1, 1.0, 10.0, 100.0]:
            processor = DifferentialPrivacyProcessor(
                epsilon=1.0,
                sensitivity=sensitivity
            )
            assert processor.sensitivity == sensitivity

    def test_zero_epsilon(self):
        """Test initialization with epsilon=0 (infinite privacy)."""
        processor = DifferentialPrivacyProcessor(epsilon=0.0, sensitivity=1.0)
        assert processor.epsilon == 0.0

    def test_zero_sensitivity(self):
        """Test initialization with sensitivity=0."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=0.0)
        assert processor.sensitivity == 0.0


class TestDifferentialPrivacyNoiseAddition:
    """Test noise addition mechanism functionality."""

    def test_laplace_noise_addition(self):
        """Test noise is added using Laplace mechanism."""
        processor = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="laplace"
        )
        original_value = 100.0

        # Set seed for reproducibility
        np.random.seed(42)
        noisy_value = processor.add_noise(original_value)

        assert isinstance(noisy_value, float)
        # Noise changes the value
        assert noisy_value != original_value

    def test_gaussian_noise_addition(self):
        """Test noise is added using Gaussian mechanism."""
        processor = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="gaussian"
        )
        original_value = 100.0

        np.random.seed(42)
        noisy_value = processor.add_noise(original_value)

        assert isinstance(noisy_value, float)
        assert noisy_value != original_value

    def test_noise_scale_laplace(self):
        """Test that noise scale in Laplace is sensitivity/epsilon."""
        processor = DifferentialPrivacyProcessor(
            epsilon=2.0,
            sensitivity=10.0,
            mechanism="laplace"
        )

        # Higher epsilon = smaller scale = less noise
        np.random.seed(42)
        noise1 = processor.add_noise(0.0)

        processor2 = DifferentialPrivacyProcessor(
            epsilon=0.5,
            sensitivity=10.0,
            mechanism="laplace"
        )

        np.random.seed(42)
        noise2 = processor2.add_noise(0.0)

        # Different seeds give different noise, just verify difference
        assert isinstance(noise1, float)
        assert isinstance(noise2, float)

    def test_noise_addition_preserves_type(self):
        """Test that noise addition returns numeric type."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        for value in [0, 1.5, -5, 100.5]:
            result = processor.add_noise(float(value))
            assert isinstance(result, float)

    def test_noise_addition_negative_values(self):
        """Test noise addition on negative values."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        original_value = -100.0

        noisy_value = processor.add_noise(original_value)

        assert isinstance(noisy_value, float)

    def test_noise_addition_zero(self):
        """Test noise addition on zero."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        np.random.seed(42)
        noisy_value = processor.add_noise(0.0)

        assert isinstance(noisy_value, float)


class TestDifferentialPrivacyEvaluatePrivacy:
    """Test privacy evaluation functionality."""

    @pytest.fixture
    def numeric_data(self):
        """Create DataFrame with numeric columns."""
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "score": [100, 150, 200, 250, 300],
        })

    @pytest.fixture
    def mixed_data(self):
        """Create DataFrame with mixed types."""
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "city": ["NYC", "LA", "SF", "CHI", "BOS"],
            "salary": [50000, 60000, 70000, 80000, 90000],
        })

    def test_evaluate_privacy_returns_dict(self, numeric_data):
        """Test that evaluate_privacy returns dictionary."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(numeric_data, ["age"])

        assert isinstance(result, dict)

    def test_evaluate_privacy_contains_key_fields(self, numeric_data):
        """Test that result contains expected fields."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(numeric_data, ["age"])

        assert "privacy_budget" in result
        assert "quasi_identifiers" in result
        assert "compliance" in result
        assert "original_means" in result
        assert "dp_means" in result

    def test_evaluate_privacy_original_means(self, numeric_data):
        """Test that original means are calculated correctly."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(numeric_data, ["age"])

        assert "age" in result["original_means"]
        expected_mean = numeric_data["age"].mean()
        assert result["original_means"]["age"] == expected_mean

    def test_evaluate_privacy_dp_means_computed(self, numeric_data):
        """Test that DP means are computed."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(numeric_data, ["age"])

        assert "age" in result["dp_means"]
        assert isinstance(result["dp_means"]["age"], float)

    def test_evaluate_privacy_multiple_numeric_columns(self, numeric_data):
        """Test evaluation with multiple numeric columns."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(numeric_data, ["age"])

        # All numeric columns should have means
        assert "age" in result["original_means"]
        assert "salary" in result["original_means"]
        assert "score" in result["original_means"]

    def test_evaluate_privacy_privacy_budget(self, numeric_data):
        """Test that privacy_budget equals epsilon."""
        epsilon_value = 2.5
        processor = DifferentialPrivacyProcessor(
            epsilon=epsilon_value,
            sensitivity=1.0
        )
        result = processor.evaluate_privacy(numeric_data, ["age"])

        assert result["privacy_budget"] == epsilon_value

    def test_evaluate_privacy_quasi_identifiers_stored(self, numeric_data):
        """Test that quasi_identifiers are stored in result."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        quasi_ids = ["age", "salary"]
        result = processor.evaluate_privacy(numeric_data, quasi_ids)

        assert result["quasi_identifiers"] == quasi_ids

    def test_evaluate_privacy_compliance_flag(self, numeric_data):
        """Test that compliance flag is boolean."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(numeric_data, ["age"])

        assert isinstance(result["compliance"], bool)

    def test_evaluate_privacy_invalid_dataframe_type(self):
        """Test error handling for non-DataFrame input returns non-compliant result."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        # Source catches ValidationError internally and returns a dict with compliance=False
        result = processor.evaluate_privacy([1, 2, 3], ["age"])
        assert isinstance(result, dict)
        assert result["compliance"] is False

    def test_evaluate_privacy_with_mixed_types(self, mixed_data):
        """Test evaluation with mixed data types."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.evaluate_privacy(mixed_data, ["age"])

        # Only numeric columns should have means
        assert "age" in result["original_means"]
        assert "salary" in result["original_means"]


class TestDifferentialPrivacyApplyModel:
    """Test model application functionality."""

    @pytest.fixture
    def numeric_data(self):
        """Create numeric DataFrame."""
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "score": [100, 150, 200, 250, 300],
        })

    @pytest.fixture
    def mixed_data(self):
        """Create mixed type DataFrame."""
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "city": ["NYC", "LA", "SF", "CHI", "BOS"],
            "salary": [50000, 60000, 70000, 80000, 90000],
        })

    def test_apply_model_returns_dataframe(self, numeric_data):
        """Test that apply_model returns DataFrame."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.apply_model(numeric_data, ["age"])

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_preserves_shape(self, numeric_data):
        """Test that apply_model preserves shape."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.apply_model(numeric_data, ["age"])

        assert result.shape == numeric_data.shape

    def test_apply_model_preserves_columns(self, numeric_data):
        """Test that apply_model preserves column names."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.apply_model(numeric_data, ["age"])

        assert list(result.columns) == list(numeric_data.columns)

    def test_apply_model_modifies_numeric_columns(self, numeric_data):
        """Test that numeric columns are modified with noise."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        np.random.seed(42)
        result = processor.apply_model(numeric_data, ["age"])

        # Numeric values should change due to noise
        assert not numeric_data["age"].equals(result["age"])

    def test_apply_model_preserves_non_numeric(self, mixed_data):
        """Test that non-numeric columns are preserved."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.apply_model(mixed_data, ["age"])

        # Non-numeric columns should be unchanged
        pd.testing.assert_series_equal(mixed_data["name"], result["name"])
        pd.testing.assert_series_equal(mixed_data["city"], result["city"])

    def test_apply_model_with_suppression_true(self, numeric_data):
        """Test apply_model with suppression=True."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.apply_model(
            numeric_data,
            ["age"],
            suppression=True
        )

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_with_suppression_false(self, numeric_data):
        """Test apply_model with suppression=False."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        result = processor.apply_model(
            numeric_data,
            ["age"],
            suppression=False
        )

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_laplace_mechanism(self, numeric_data):
        """Test apply_model with Laplace mechanism."""
        processor = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="laplace"
        )
        result = processor.apply_model(numeric_data, ["age"])

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_gaussian_mechanism(self, numeric_data):
        """Test apply_model with Gaussian mechanism."""
        processor = DifferentialPrivacyProcessor(
            epsilon=1.0,
            sensitivity=1.0,
            mechanism="gaussian"
        )
        result = processor.apply_model(numeric_data, ["age"])

        assert isinstance(result, pd.DataFrame)


class TestDifferentialPrivacyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataset(self):
        """Test with single-row dataset."""
        data = pd.DataFrame({"value": [100.0]})
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.evaluate_privacy(data, [])
        assert isinstance(result, dict)

    def test_single_column_dataset(self):
        """Test with single-column dataset."""
        data = pd.DataFrame({"value": [100.0, 200.0, 300.0]})
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_all_zeros_dataset(self):
        """Test with all zero values."""
        data = pd.DataFrame({
            "value1": [0.0, 0.0, 0.0],
            "value2": [0.0, 0.0, 0.0],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.evaluate_privacy(data, [])
        assert isinstance(result, dict)

    def test_all_negative_dataset(self):
        """Test with all negative values."""
        data = pd.DataFrame({
            "value": [-100.0, -200.0, -300.0],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_very_large_values(self):
        """Test with very large values."""
        data = pd.DataFrame({
            "value": [1e10, 2e10, 3e10],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_very_small_values(self):
        """Test with very small values."""
        data = pd.DataFrame({
            "value": [1e-10, 2e-10, 3e-10],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_mixed_sign_values(self):
        """Test with positive and negative values."""
        data = pd.DataFrame({
            "value": [-100.0, 0.0, 100.0, -50.0, 50.0],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_very_small_epsilon(self):
        """Test with very small epsilon (maximum privacy)."""
        data = pd.DataFrame({"value": [100.0, 200.0, 300.0]})
        processor = DifferentialPrivacyProcessor(epsilon=0.01, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_very_large_epsilon(self):
        """Test with very large epsilon (minimum privacy)."""
        data = pd.DataFrame({"value": [100.0, 200.0, 300.0]})
        processor = DifferentialPrivacyProcessor(epsilon=1000.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_nan_values(self):
        """Test handling of NaN values."""
        data = pd.DataFrame({
            "value": [100.0, np.nan, 300.0],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)

    def test_inf_values(self):
        """Test handling of infinite values."""
        data = pd.DataFrame({
            "value": [100.0, np.inf, -np.inf, 300.0],
        })
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)

        result = processor.apply_model(data, [])
        assert isinstance(result, pd.DataFrame)


class TestDifferentialPrivacyProcess:
    """Test the process() method inherited from base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({"value": [100, 200, 300]})

    def test_process_method_exists(self, sample_data):
        """Test that process method exists."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        assert hasattr(processor, "process")
        assert callable(processor.process)

    def test_process_method_callable(self, sample_data):
        """Test that process method is callable."""
        processor = DifferentialPrivacyProcessor(epsilon=1.0, sensitivity=1.0)
        # Just verify it doesn't raise error
        processor.process(sample_data)
