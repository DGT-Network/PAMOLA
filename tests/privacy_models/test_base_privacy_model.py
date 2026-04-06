"""
Unit tests for BasePrivacyModelProcessor abstract base class.

Tests verify the abstract interface definition and error handling for
abstract method implementation requirements.

Run with: pytest -s tests/privacy_models/test_base_privacy_model.py
"""

import pytest
import pandas as pd

from pamola_core.privacy_models.base import BasePrivacyModelProcessor


class ConcretePrivacyModel(BasePrivacyModelProcessor):
    """Concrete implementation of BasePrivacyModelProcessor for testing."""

    def process(self, data):
        """Process method implementation."""
        return data

    def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: list[str], **kwargs) -> dict:
        """Evaluate privacy method implementation."""
        return {
            "compliant": True,
            "quasi_identifiers": quasi_identifiers,
            "records": len(data),
        }

    def apply_model(
        self,
        data: pd.DataFrame,
        quasi_identifiers: list[str],
        suppression: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Apply model method implementation."""
        return data.copy()


class TestBasePrivacyModelProcessor:
    """Test suite for BasePrivacyModelProcessor abstract base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "age": [25, 30, 25, 35, 30, 25],
            "city": ["NYC", "LA", "NYC", "NYC", "LA", "NYC"],
            "salary": [50000, 60000, 50000, 70000, 60000, 50000],
        })

    def test_cannot_instantiate_abstract_class(self):
        """Verify BasePrivacyModelProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePrivacyModelProcessor()

    def test_concrete_implementation_instantiation(self):
        """Verify concrete implementation can be instantiated."""
        model = ConcretePrivacyModel()
        assert model is not None
        assert isinstance(model, BasePrivacyModelProcessor)

    def test_process_method_required(self):
        """Verify process() is an abstract method."""
        with pytest.raises(TypeError):
            class IncompleteModel(BasePrivacyModelProcessor):
                def evaluate_privacy(self, data, quasi_identifiers, **kwargs):
                    pass

                def apply_model(self, data, quasi_identifiers, suppression=True, **kwargs):
                    pass

            IncompleteModel()

    def test_evaluate_privacy_method_required(self):
        """Verify evaluate_privacy() is an abstract method."""
        with pytest.raises(TypeError):
            class IncompleteModel(BasePrivacyModelProcessor):
                def process(self, data):
                    pass

                def apply_model(self, data, quasi_identifiers, suppression=True, **kwargs):
                    pass

            IncompleteModel()

    def test_apply_model_method_required(self):
        """Verify apply_model() is an abstract method."""
        with pytest.raises(TypeError):
            class IncompleteModel(BasePrivacyModelProcessor):
                def process(self, data):
                    pass

                def evaluate_privacy(self, data, quasi_identifiers, **kwargs):
                    pass

            IncompleteModel()

    def test_process_method_execution(self, sample_data):
        """Test process method execution on concrete implementation."""
        model = ConcretePrivacyModel()
        result = model.process(sample_data)
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_evaluate_privacy_method_execution(self, sample_data):
        """Test evaluate_privacy method with valid inputs."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city"]
        result = model.evaluate_privacy(sample_data, quasi_identifiers)

        assert isinstance(result, dict)
        assert "compliant" in result
        assert result["quasi_identifiers"] == quasi_identifiers
        assert result["records"] == len(sample_data)

    def test_evaluate_privacy_with_kwargs(self, sample_data):
        """Test evaluate_privacy passes kwargs correctly."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city"]
        result = model.evaluate_privacy(
            sample_data,
            quasi_identifiers,
            custom_param="test_value"
        )

        assert result is not None
        assert isinstance(result, dict)

    def test_apply_model_method_execution(self, sample_data):
        """Test apply_model method with valid inputs."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city"]
        result = model.apply_model(sample_data, quasi_identifiers, suppression=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    def test_apply_model_with_suppression_false(self, sample_data):
        """Test apply_model with suppression=False."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city"]
        result = model.apply_model(sample_data, quasi_identifiers, suppression=False)

        assert isinstance(result, pd.DataFrame)

    def test_apply_model_with_kwargs(self, sample_data):
        """Test apply_model passes kwargs correctly."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city"]
        result = model.apply_model(
            sample_data,
            quasi_identifiers,
            suppression=True,
            custom_param="test_value"
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_abstract_methods_signature_types(self, sample_data):
        """Verify abstract methods require correct parameter types."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city"]

        # evaluate_privacy should accept DataFrame and list of strings
        result = model.evaluate_privacy(sample_data, quasi_identifiers)
        assert isinstance(result, dict)

        # apply_model should accept DataFrame and list of strings
        result = model.apply_model(sample_data, quasi_identifiers)
        assert isinstance(result, pd.DataFrame)

    def test_multiple_quasi_identifiers(self, sample_data):
        """Test methods with multiple quasi-identifiers."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age", "city", "salary"]

        result = model.evaluate_privacy(sample_data, quasi_identifiers)
        assert result["quasi_identifiers"] == quasi_identifiers

        anonymized = model.apply_model(sample_data, quasi_identifiers)
        assert isinstance(anonymized, pd.DataFrame)

    def test_single_quasi_identifier(self, sample_data):
        """Test methods with single quasi-identifier."""
        model = ConcretePrivacyModel()
        quasi_identifiers = ["age"]

        result = model.evaluate_privacy(sample_data, quasi_identifiers)
        assert result["quasi_identifiers"] == quasi_identifiers

        anonymized = model.apply_model(sample_data, quasi_identifiers)
        assert isinstance(anonymized, pd.DataFrame)

    def test_empty_quasi_identifiers(self, sample_data):
        """Test behavior with empty quasi-identifiers list."""
        model = ConcretePrivacyModel()
        quasi_identifiers = []

        result = model.evaluate_privacy(sample_data, quasi_identifiers)
        assert isinstance(result, dict)

        anonymized = model.apply_model(sample_data, quasi_identifiers)
        assert isinstance(anonymized, pd.DataFrame)
