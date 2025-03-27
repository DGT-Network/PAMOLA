"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Categorical Generalization Processor
---------------------------------------------
This module provides methods for generalizing categorical attributes
to enhance anonymization. It extends the BaseGeneralizationProcessor and
implements techniques such as value grouping, frequency smoothing,
and hierarchical generalization.

Categorical generalization replaces specific categorical values with
more abstract, less specific values while preserving statistical
distribution.

Common approaches include:
- **Value Grouping**: Replacing rare values with more common categories.
- **Frequency Smoothing**: Merging infrequent categories into an "Other" group.
- **Hierarchical Generalization**: Mapping values to a higher-level taxonomy
  (e.g., "Car Brand" → "Automobile").
- **k-Anonymity Constraints**: Ensuring that each generalized category has
  at least k occurrences.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

# Required libraries
from typing import Dict, Optional
import pandas as pd
from abc import ABC
from pamola_core.anonymization.generalization.base import BaseGeneralizationProcessor

class CategoricalGeneralizationProcessor(BaseGeneralizationProcessor, ABC):
    """
    Categorical Generalization Processor for anonymizing categorical attributes.
    This class extends BaseGeneralizationProcessor and provides techniques for
    generalizing categorical data to enhance anonymization.

    Methods:
    --------
    - group_values(): Groups similar values into broader categories.
    - smooth_frequencies(): Merges infrequent categories into an "Other" class.
    - apply_hierarchy(): Uses a predefined hierarchy for generalization.
    """

    def __init__(self, hierarchy: Optional[Dict] = None, min_group_size: int = 5):
        """
        Initialize the categorical generalization processor.

        Parameters:
        -----------
        hierarchy : dict, optional
            A predefined hierarchy for categorical values (default: None).
        min_group_size : int
            Minimum occurrences for a category to remain unchanged (default: 5).
        """
        super().__init__()  # Ensure base class is properly initialized
        self.hierarchy = hierarchy or {}
        self.min_group_size = max(1, min_group_size)  # Ensure min_group_size is at least 1

    def generalize(self, data: pd.DataFrame, column: str, method: str = "grouping") -> pd.DataFrame:
        """
        Apply categorical generalization to a specified column.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data to be generalized.
        column : str
            The column name to be processed.
        method : str, optional
            The generalization method to apply ("grouping", "smoothing", "hierarchy").

        Returns:
        --------
        pd.DataFrame
            The dataset with generalized categorical values.
        """
        if method == "grouping":
            return self.group_values(data, column)
        elif method == "smoothing":
            return self.smooth_frequencies(data, column)
        elif method == "hierarchy":
            return self.apply_hierarchy(data, column)
        else:
            raise ValueError(f"Invalid generalization method: {method}")

    def group_values(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Group rare categorical values into a common category.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with grouped categorical values.
        """
        data = data.copy()
        rare_values = self._get_rare_values(data[column], self.min_group_size)
        data.loc[data[column].isin(rare_values), column] = "Other"
        return data

    def smooth_frequencies(self, data: pd.DataFrame, column: str, threshold: float = 0.05) -> pd.DataFrame:
        """
        Smooth categorical frequencies by merging infrequent categories into 'Other'.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data.
        column : str
            The column name to be processed.
        threshold : float, optional
            Frequency threshold for merging categories (default: 0.05).

        Returns:
        --------
        pd.DataFrame
            The dataset with frequency-smoothed categorical values.
        """
        data = data.copy()
        rare_values = data[column].value_counts(normalize=True).loc[lambda x: x < threshold].index
        data.loc[data[column].isin(rare_values), column] = "Other"
        return data

    def apply_hierarchy(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply predefined hierarchical generalization to a categorical column.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset containing categorical data.
        column : str
            The column name to be processed.

        Returns:
        --------
        pd.DataFrame
            The dataset with hierarchical generalization applied.
        """
        if not self.hierarchy:
            raise ValueError("Hierarchy mapping is required for this method.")

        data = data.copy()
        data[column] = data[column].astype(str).map(self.hierarchy).fillna(data[column])
        return data
    
    def _get_rare_values(self, data: pd.Series, threshold: int) -> pd.Index:
        """
        Identify rare categorical values in a Pandas Series based on a specified frequency threshold.

        Parameters:
        -----------
        data : pd.Series
            The input data series containing categorical values to analyze.
        threshold : int
            The maximum frequency a value can have to be considered rare. Values occurring fewer times than this threshold are deemed rare.

        Returns:
        --------
        pd.Index
            An index object containing the rare values that occur less frequently than the specified threshold.
        """
        return data.value_counts().loc[lambda x: x < threshold].index
