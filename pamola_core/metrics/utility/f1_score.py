"""
PAMOLA.CORE - F1 Score Metric
-----------------------------
This module provides a class for calculating the F1 Score between real and
synthetic datasets. The F1 Score is a measure of a test's accuracy and is
used to evaluate the performance of classification models.

Key features:
- Calculates F1 Score for binary and multi-class classification
- Supports different averaging methods: micro, macro, weighted, and binary
- Validates input data for consistency and correctness

This metric helps data custodians assess the classification performance
of synthetic data compared to real data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from typing import Dict, Any

from pamola.pamola_core.metrics.base import UtilityMetric

import logging

# Configure logging
logger = logging.getLogger(__name__)

class F1Score(UtilityMetric):
    """Class to calculate F1 Score between real and synthetic data"""

    def __init__(self, name: str = "F1 Score", description: str = "Calculates the F1 Score between real and synthetic data"):
        super().__init__(name, description)  # Call the parent constructor

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_column: str, average: str = "weighted", **kwargs) -> Dict[str, Any]:
        logger.info("Calculating F1 Score")
        try:
            # Validate input types
            if not isinstance(real_data, pd.DataFrame) or not isinstance(synthetic_data, pd.DataFrame):
                raise ValueError("Both real_data and synthetic_data must be pandas DataFrames.")

            # Check if the DataFrames are empty
            if real_data.empty or synthetic_data.empty:
                raise ValueError("Real data or synthetic data cannot be empty.")

            # Check if the target column exists in both DataFrames
            if target_column not in real_data.columns or target_column not in synthetic_data.columns:
                raise ValueError(f"Column '{target_column}' does not exist in the data!")

            # Extract true and predicted labels
            y_true = real_data[target_column].to_numpy()
            y_pred = synthetic_data[target_column].to_numpy()

            # Check if the values are numeric
            if not np.issubdtype(y_true.dtype, np.number) or not np.issubdtype(y_pred.dtype, np.number):
                raise ValueError("The target column must contain numeric values.")

            # Calculate F1 Score
            f1 = f1_score(y_true, y_pred, average=average)
            logger.info("F1 Score calculation completed successfully")
            return {"f1_score": f1}
        except Exception as e:
            logger.error(f"Error during F1 Score calculation: {e}")
            raise