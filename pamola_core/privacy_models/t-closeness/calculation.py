"""
PAMOLA.CORE - t-Closeness Privacy Model
----------------------------------------
This module provides a class for implementing the t-Closeness privacy model.
t-Closeness ensures that the distribution of a sensitive attribute in any
equivalence class is close to the distribution of the attribute in the overall dataset.

Key features:
- Calculates t-Closeness for specified quasi-identifiers and sensitive attributes
- Evaluates privacy risks and compliance with t-Closeness criteria
- Supports suppression of non-compliant records

This processor helps data custodians apply t-Closeness techniques to protect
sensitive information while maintaining data utility.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from typing import List, Dict, Any
from scipy.stats import wasserstein_distance
import logging

from pamola_core.metrics.base import BasePrivacyModelProcessor

# Set up logging configuration
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class TCloseness(BasePrivacyModelProcessor):
    """Class to calculate t-Closeness"""

    def __init__(self, quasi_identifiers: List[str], sensitive_column: str, t: float):
        """
        Parameters:
        -----------
        quasi_identifiers : List[str]
            Attributes used to group the data.
        sensitive_column : str
            Column containing sensitive information.
        t : float
            Threshold for closeness.
        """
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_column = sensitive_column
        self.t = t

    def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calculate t-Closeness.

        Parameters:
        -----------
        data : pd.DataFrame
            The data to check.

        Returns:
        --------
        dict
            {'max_t_value': float, 'is_t_close': bool} - The t-Closeness value and validity status.
        """
        try:
            if not all(col in data.columns for col in self.quasi_identifiers + [self.sensitive_column]):
                raise ValueError("Quasi-identifiers or sensitive column do not exist in the data!")

            global_distribution = data[self.sensitive_column].value_counts(normalize=True)
            max_t_value = 0

            for _, group in data.groupby(self.quasi_identifiers):
                group_distribution = group[self.sensitive_column].value_counts(normalize=True).reindex(
                    global_distribution.index, fill_value=0
                )
                emd_value = wasserstein_distance(global_distribution.values, group_distribution.values)
                max_t_value = max(max_t_value, emd_value)

            return {
                "max_t_value": float(max_t_value),
                "is_t_close": max_t_value <= self.t
            }
        except Exception as e:
            logging.error(f"An error occurred during t-Closeness evaluation: {e}")
            return {
                "max_t_value": None,
                "is_t_close": False
            }

    def apply_model(self, data: pd.DataFrame, quasi_identifiers: List[str], suppression: bool = True, **kwargs) -> pd.DataFrame:
        """
        Apply the t-Closeness model to the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset to be transformed.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        suppression : bool, optional
            Whether to suppress non-compliant records (default: True).
        kwargs : dict
            Additional parameters for model application.

        Returns:
        --------
        pd.DataFrame
            The transformed dataset with t-closeness guarantees applied.
        """
        # Placeholder for actual implementation of applying t-closeness
        # This could involve modifying the dataset based on t-closeness criteria
        return data  # Return the original data for now
