"""
Base abstract generator for synthetic data generation.

This module provides the foundation for all generators in the fake data system,
defining a unified interface for the generation of synthetic data while
preserving statistical properties.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseGenerator(ABC):
    """
    Abstract base class for all data generators.

    Defines a common interface for generating synthetic data while preserving
    statistical properties of the original data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize generator with optional configuration.

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    def generate(self, count: int, **params) -> List[str]:
        """
        Generate specified number of synthetic values.

        Args:
            count: Number of values to generate
            **params: Additional parameters for generation

        Returns:
            List of generated values
        """
        pass

    @abstractmethod
    def generate_like(self, original_value: str, **params) -> str:
        """
        Generate a synthetic value similar to the original one.

        Args:
            original_value: Original value to base generation on
            **params: Additional parameters for generation

        Returns:
            Generated value
        """
        pass

    def transform(self, values: List[str], **params) -> List[str]:
        """
        Transform a list of original values into synthetic ones.

        Default implementation calls generate_like for each value.

        Args:
            values: List of original values
            **params: Additional parameters for generation

        Returns:
            List of transformed values
        """
        return [self.generate_like(value, **params) for value in values]

    def validate(self, value: str) -> bool:
        """
        Check if a value is valid according to this generator's rules.

        Args:
            value: Value to validate

        Returns:
            True if value is valid, False otherwise
        """
        # Default implementation accepts any non-empty string
        return bool(value and isinstance(value, str))