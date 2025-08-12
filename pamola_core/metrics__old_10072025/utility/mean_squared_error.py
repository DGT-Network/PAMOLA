"""
PAMOLA.CORE - Mean Squared Error Metric
---------------------------------------
This module provides a class for calculating the Mean Squared Error (MSE)
between real and synthetic datasets. MSE is a measure of the average squared
difference between the estimated values and the actual value.

Key features:
- Calculates MSE for specified columns in datasets
- Provides overall MSE across multiple columns
- Validates input data for consistency and correctness

This metric helps data custodians assess the accuracy of synthetic data
compared to real data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from pamola_core.metrics__old_10072025.base import UtilityMetric

class MeanSquaredError(UtilityMetric):
    """Class to calculate Mean Squared Error (MSE) between real and synthetic data"""

    def __init__(self, name: str = "Mean Squared Error",
                 description: str = "Calculates the Mean Squared Error between real and synthetic data"):
        super().__init__(name, description)  # Call the parent constructor

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, columns: list, **kwargs) -> Dict[str, Any]:
        """
        Calculate Mean Squared Error (MSE).

        Parameters:
        -----------
        real_data : pd.DataFrame
            The actual data.
        synthetic_data : pd.DataFrame
            The synthetic data.
        columns : list
            List of columns to calculate MSE.

        Returns:
        --------
        dict
            {'mse': Dict[str, float], 'overall_mse': float} - MSE values for each column and overall MSE.
        """
        if not isinstance(real_data, pd.DataFrame) or not isinstance(synthetic_data, pd.DataFrame):
            raise ValueError("Both real_data and synthetic_data must be pandas DataFrames.")

        if not columns:
            raise ValueError("A list of columns must be provided to calculate MSE!")

        mse_values = {}
        overall_mse = 0
        count = 0

        for col in columns:
            if col in real_data.columns and col in synthetic_data.columns:
                mse = np.mean((real_data[col] - synthetic_data[col]) ** 2)
                mse_values[col] = mse
                overall_mse += mse
                count += 1
            else:
                mse_values[col] = None  # Handle missing data case

        # Calculate overall MSE if at least one valid column was processed
        overall_mse = overall_mse / count if count > 0 else None

        return {
            "mse": mse_values,
            "overall_mse": overall_mse
        }