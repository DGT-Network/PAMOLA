"""
PAMOLA.CORE - A utility class for analyzing data distributions.
---------------------------------------------------------
This module provides functions for computing Shannon entropy to measure 
the uncertainty or randomness in a given dataset.

Key Features:
 - Computes Shannon entropy for a Pandas Series.
 - Handles missing values by removing NaNs before computation.
 - Supports different logarithm bases for entropy calculation.

This class is useful for data profiling, feature selection, and privacy analysis.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""


import pandas as pd
import logging
from scipy.stats import entropy

logger = logging.getLogger(__name__)

class DataProfiler:

    @staticmethod
    def compute_shannon_entropy(column: pd.Series, base: int = 2) -> float:
        """
        Compute the Shannon entropy of a Pandas Series.

        Parameters:
        -----------
        column : pd.Series
            The input data column for entropy calculation.
        base : int, optional (default=2)
            The logarithm base used for entropy computation (e.g., 2 for bits, e for nats).

        Returns:
        --------
        float
            The calculated Shannon entropy value. Returns 0.0 if the input column is empty.
        """
        if not isinstance(column, pd.Series):
            raise TypeError("Input must be a Pandas Series.")

        # Remove NaN values
        column_clean = column.dropna()

        if column_clean.empty:
            return 0.0

        # Compute probability distribution
        value_counts = column_clean.value_counts(normalize=True)

        # Compute entropy
        entropy_value = entropy(value_counts, base=base)

        return float(entropy_value)
    