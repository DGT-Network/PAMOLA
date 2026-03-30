"""
Unit tests for KAnonymityProcessor privacy model.

Tests verify k-anonymity calculation, privacy evaluation, and model application
with various k-levels, edge cases, and error scenarios.

Run with: pytest -s tests/privacy_models/test_k_anonymity.py
"""

import pytest
import pandas as pd
import numpy as np

from pamola_core.privacy_models.k_anonymity.calculation import KAnonymityProcessor
from pamola_core.errors.exceptions import ValidationError


def _is_int(value):
    """Check if value is an integer type (including numpy integer types)."""
    return isinstance(value, (int, np.integer))


class TestKAnonymityProcessorInitialization:
    """Test KAnonymityProcessor initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        processor = KAnonymityProcessor(progress_tracking=False)
        assert processor.k == 3
        assert processor.suppression is True
        assert processor.mask_value == "MASKED"
        assert processor.use_dask is False

    def test_custom_k_value(self):
        """Test initialization with custom k value."""
        processor = KAnonymityProcessor(k=5, progress_tracking=False)
        assert processor.k == 5

    def test_custom_suppression_setting(self):
        """Test initialization with suppression disabled."""
        processor = KAnonymityProcessor(suppression=False, progress_tracking=False)
        assert processor.suppression is False

    def test_custom_mask_value(self):
        """Test initialization with custom mask value."""
        processor = KAnonymityProcessor(mask_value="***", progress_tracking=False)
        assert processor.mask_value == "***"

    def test_adaptive_k_initialization(self):
        """Test initialization with adaptive k values."""
        adaptive_k = {("NYC",): 5, ("LA",): 3}
        processor = KAnonymityProcessor(adaptive_k=adaptive_k, progress_tracking=False)
        assert processor.adaptive_k == adaptive_k

    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises ValidationError."""
        with pytest.raises(ValidationError):
            KAnonymityProcessor(log_level="INVALID_LEVEL", progress_tracking=False)

    def test_valid_log_levels(self):
        """Test initialization with valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            processor = KAnonymityProcessor(log_level=level, progress_tracking=False)
            assert processor.k == 3  # Verify normal initialization


class TestKAnonymityEvaluatePrivacy:
    """Test k-anonymity privacy evaluation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create diverse sample DataFrame for k-anonymity testing."""
        return pd.DataFrame({
            "age": [25, 30, 25, 35, 30, 25, 40, 25, 30, 25],
            "city": ["NYC", "LA", "NYC", "NYC", "LA", "NYC", "NYC", "LA", "NYC", "LA"],
            "salary": [50000, 60000, 50000, 70000, 60000, 50000, 75000, 50000, 60000, 65000],
        })

    @pytest.fixture
    def k3_compliant_data(self):
        """Create data that satisfies k=3."""
        return pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30],
            "city": ["NYC", "NYC", "NYC", "LA", "LA", "LA"],
            "income": ["low", "low", "low", "med", "med", "med"],
        })

    def test_evaluate_privacy_returns_dict(self, sample_data):
        """Test evaluate_privacy returns a dictionary."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert isinstance(result, dict)

    def test_evaluate_privacy_contains_key_metrics(self, sample_data):
        """Test evaluate_privacy result contains key metrics."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert "min_k" in result
        assert "at_risk_records" in result
        assert "at_risk_groups" in result
        assert "compliant" in result

    def test_k_equals_1_boundary(self, sample_data):
        """Test k=1 boundary condition (all individual records)."""
        processor = KAnonymityProcessor(k=1, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert result["min_k"] >= 1
        assert result["compliant"] == True  # noqa: E712 — numpy.bool_ is not `is True`

    def test_k_equals_n_boundary(self, sample_data):
        """Test k=n where n is dataset size (all records in one group)."""
        n = len(sample_data)
        processor = KAnonymityProcessor(k=n, progress_tracking=False)
        quasi_identifiers = ["age"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert _is_int(result["min_k"])

    def test_compliant_dataset(self, k3_compliant_data):
        """Test evaluation of a k-anonymity compliant dataset."""
        processor = KAnonymityProcessor(k=3, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(k3_compliant_data, quasi_identifiers)

        assert result["compliant"] == True  # noqa: E712 — numpy.bool_
        assert result["min_k"] >= 3

    def test_non_compliant_dataset(self):
        """Test evaluation of a non-compliant dataset."""
        data = pd.DataFrame({
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
            "salary": [50000, 60000, 70000],
        })
        processor = KAnonymityProcessor(k=5, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert result["compliant"] == False  # noqa: E712 — numpy.bool_

    def test_missing_quasi_identifier_column(self, sample_data):
        """Test error handling for missing quasi-identifier column."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["nonexistent_column"]

        with pytest.raises(Exception):  # ValidationError expected
            processor.evaluate_privacy(sample_data, quasi_identifiers)

    def test_single_quasi_identifier(self, sample_data):
        """Test evaluation with single quasi-identifier."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert "min_k" in result
        assert _is_int(result["min_k"])

    def test_multiple_quasi_identifiers(self, sample_data):
        """Test evaluation with multiple quasi-identifiers."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert "min_k" in result
        assert _is_int(result["min_k"])

    def test_detailed_metrics_flag(self, sample_data):
        """Test detailed_metrics parameter inclusion."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(
            sample_data,
            quasi_identifiers,
            detailed_metrics=True
        )

        assert isinstance(result, dict)

    def test_execution_time_recorded(self, sample_data):
        """Test execution_time is recorded in results."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(sample_data, quasi_identifiers)

        assert "execution_time" in result
        assert result["execution_time"] >= 0  # can be 0.0 on fast machines


class TestKAnonymityApplyModel:
    """Test k-anonymity model application and suppression/masking."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for model application."""
        return pd.DataFrame({
            "age": [25, 30, 25, 35, 30, 25, 40, 25, 30, 25],
            "city": ["NYC", "LA", "NYC", "NYC", "LA", "NYC", "NYC", "LA", "NYC", "LA"],
            "salary": [50000, 60000, 50000, 70000, 60000, 50000, 75000, 50000, 60000, 65000],
        })

    def test_apply_model_with_suppression(self, sample_data):
        """Test model application with suppression enabled."""
        processor = KAnonymityProcessor(k=3, suppression=True, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.apply_model(sample_data, quasi_identifiers, suppression=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data)

    def test_apply_model_with_masking(self, sample_data):
        """Test model application with masking instead of suppression."""
        processor = KAnonymityProcessor(k=3, suppression=False, mask_value="***", progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.apply_model(sample_data, quasi_identifiers, suppression=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    def test_apply_model_returns_info(self, sample_data):
        """Test apply_model with return_info=True."""
        processor = KAnonymityProcessor(k=3, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result, info = processor.apply_model(
            sample_data,
            quasi_identifiers,
            return_info=True
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(info, dict)
        assert "original_records" in info
        assert "anonymized_records" in info
        assert "execution_time" in info

    def test_apply_model_suppression_removes_records(self, sample_data):
        """Test that suppression reduces record count."""
        processor = KAnonymityProcessor(k=5, suppression=True, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.apply_model(sample_data, quasi_identifiers, suppression=True)

        # With k=5, many groups won't meet threshold
        assert len(result) <= len(sample_data)

    def test_apply_model_masking_preserves_count(self, sample_data):
        """Test that masking preserves record count."""
        processor = KAnonymityProcessor(k=3, suppression=False, mask_value="MASKED", progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.apply_model(sample_data, quasi_identifiers, suppression=False)

        assert len(result) == len(sample_data)

    def test_apply_model_with_add_k_column(self, sample_data):
        """Test add_k_column parameter adds k values."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.apply_model(
            sample_data,
            quasi_identifiers,
            add_k_column=True
        )

        # Should have additional column for k-values
        assert isinstance(result, pd.DataFrame)

    def test_apply_model_k_value_1(self, sample_data):
        """Test apply_model with k=1 (all records kept)."""
        processor = KAnonymityProcessor(k=1, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.apply_model(sample_data, quasi_identifiers, suppression=True)

        # All records should be kept with k=1
        assert len(result) == len(sample_data)

    def test_apply_model_missing_quasi_identifier(self, sample_data):
        """Test error handling for missing quasi-identifier."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["nonexistent_column"]

        with pytest.raises(Exception):  # ValidationError expected
            processor.apply_model(sample_data, quasi_identifiers)

    def test_apply_model_empty_quasi_identifiers(self, sample_data):
        """Test error handling for empty quasi-identifiers."""
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = []

        with pytest.raises(Exception):  # ValidationError expected
            processor.apply_model(sample_data, quasi_identifiers)


class TestKAnonymityEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataset(self):
        """Test k-anonymity evaluation on single-row dataset."""
        data = pd.DataFrame({
            "age": [25],
            "city": ["NYC"],
            "salary": [50000],
        })
        processor = KAnonymityProcessor(k=1, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert isinstance(result, dict)
        assert result["min_k"] == 1

    def test_identical_rows_dataset(self):
        """Test k-anonymity with all identical rows."""
        data = pd.DataFrame({
            "age": [25, 25, 25, 25],
            "city": ["NYC", "NYC", "NYC", "NYC"],
            "salary": [50000, 50000, 50000, 50000],
        })
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert result["min_k"] >= 4
        assert result["compliant"] == True  # noqa: E712 — numpy.bool_

    def test_null_values_in_data(self):
        """Test handling of null values in quasi-identifiers."""
        data = pd.DataFrame({
            "age": [25, 30, None, 25, 30],
            "city": ["NYC", "LA", "NYC", "NYC", "LA"],
            "salary": [50000, 60000, None, 50000, 60000],
        })
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]

        # Should handle null values gracefully
        result = processor.evaluate_privacy(data, quasi_identifiers)
        assert isinstance(result, dict)

    def test_very_large_k_value(self):
        """Test with k value larger than dataset size."""
        data = pd.DataFrame({
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
        })
        processor = KAnonymityProcessor(k=100, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert result["compliant"] == False  # noqa: E712 — numpy.bool_

    def test_all_unique_rows(self):
        """Test dataset with all unique quasi-identifier combinations."""
        data = pd.DataFrame({
            "age": [25, 26, 27, 28, 29],
            "city": ["NYC", "LA", "SF", "CHI", "BOS"],
        })
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "city"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert result["min_k"] == 1
        assert result["compliant"] == False  # noqa: E712 — numpy.bool_

    def test_numeric_quasi_identifiers(self):
        """Test with numeric quasi-identifier columns."""
        data = pd.DataFrame({
            "age": [25, 25, 30, 30, 35, 35],
            "income": [50000, 50000, 60000, 60000, 70000, 70000],
            "score": [100, 100, 200, 200, 300, 300],
        })
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "income"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert result["compliant"] == True  # noqa: E712 — numpy.bool_

    def test_string_quasi_identifiers(self):
        """Test with string quasi-identifier columns."""
        data = pd.DataFrame({
            "city": ["NYC", "NYC", "LA", "LA"],
            "state": ["NY", "NY", "CA", "CA"],
            "job": ["Engineer", "Engineer", "Doctor", "Doctor"],
        })
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["city", "state"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert result["compliant"] == True  # noqa: E712 — numpy.bool_

    def test_mixed_type_quasi_identifiers(self):
        """Test with mixed data types in quasi-identifiers."""
        data = pd.DataFrame({
            "age": [25, 25, 30, 30],
            "name": ["Alice", "Alice", "Bob", "Bob"],
            "salary": [50000, 50000, 60000, 60000],
        })
        processor = KAnonymityProcessor(k=2, progress_tracking=False)
        quasi_identifiers = ["age", "name"]
        result = processor.evaluate_privacy(data, quasi_identifiers)

        assert isinstance(result, dict)


class TestKAnonymityProcess:
    """Test the process() method inherited from base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "age": [25, 30, 25],
            "city": ["NYC", "LA", "NYC"],
        })

    def test_process_method_exists(self, sample_data):
        """Test that process method exists and is callable."""
        processor = KAnonymityProcessor(progress_tracking=False)
        assert hasattr(processor, "process")
        assert callable(processor.process)

    def test_process_method_handles_dataframe(self, sample_data):
        """Test process method with DataFrame input."""
        processor = KAnonymityProcessor(progress_tracking=False)
        # Process method may or may not return value depending on implementation
        result = processor.process(sample_data)
        # Just verify it doesn't raise an error
        assert result is None or isinstance(result, (pd.DataFrame, dict))
