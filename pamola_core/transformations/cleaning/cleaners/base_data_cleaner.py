"""
Abstract base class for data cleaners.

Defines a standard interface for cleaning operations, including normalization,
missing value treatment, and text sanitization, while preserving original data semantics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import pandas as pd


class BaseDataCleaner(ABC):
    """
    Abstract base class for all data cleaner.

    Defines a common interface for generating synthetic data while preserving
    statistical properties of the original data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data cleaner with optional configuration.

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    def execute(
            self,
            df: pd.DataFrame,
            **kwargs
    ) -> pd.DataFrame:
        """
        Handling null, missing, or empty values.

        Parameters:
        -----------
        df : pd.DataFrame
            The input data to be processed.
        **kwargs : dict, optional
            Additional parameters for customization.

        Returns:
        --------
        DataFrame
        """
        pass