"""
PAMOLA.CORE - Differential Privacy Processor
--------------------------------------------
This module provides a class for implementing differential privacy using
Laplace and Gaussian mechanisms. Differential privacy ensures that the
output of a dataset query is statistically indistinguishable, providing
privacy guarantees for individuals within the dataset.

Key features:
- Supports Laplace and Gaussian mechanisms for noise addition
- Configurable privacy level (epsilon) and sensitivity
- Evaluates privacy risks and compliance
- Applies differential privacy transformations to datasets

This processor helps data custodians apply differential privacy techniques
to protect sensitive information while maintaining data utility.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

from pamola_core.privacy_models.base import BasePrivacyModelProcessor

# Set up logging configuration
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class DifferentialPrivacyProcessor(BasePrivacyModelProcessor):
    """Class to calculate Epsilon-Differential Privacy using Laplace and Gaussian mechanisms"""

    def __init__(self, epsilon: float, sensitivity: float, mechanism: str = "laplace"):
        """
        Parameters:
        -----------
        epsilon : float
            Privacy level (smaller -> more private but less accurate).
        sensitivity : float
            Sensitivity of the query.
        mechanism : str
            Data protection mechanism ("laplace" or "gaussian").
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.mechanism = mechanism.lower()
        
        if self.mechanism not in ["laplace", "gaussian"]:
            raise ValueError("Mechanism must be 'laplace' or 'gaussian'")

    def add_noise(self, value: float) -> float:
        """Add noise to a value based on the selected mechanism."""
        if self.mechanism == "laplace":
            scale = self.sensitivity / self.epsilon
            noise = np.random.laplace(0, scale)
        else:  # Gaussian Mechanism
            scale = np.sqrt(2 * np.log(1.25)) * (self.sensitivity / self.epsilon)
            noise = np.random.normal(0, scale)
        
        return value + noise

    def evaluate_privacy(self, data: pd.DataFrame, quasi_identifiers: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate anonymization risks and compliance of the dataset based on
        differential privacy principles and calculate means.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset to be evaluated.
        quasi_identifiers : list[str]
            List of column names used as quasi-identifiers.
        kwargs : dict
            Additional parameters for model evaluation.

        Returns:
        --------
        dict
            A dictionary containing anonymization metrics, evaluation results,
            and means of original and differentially private data.
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame!")

            original_means = {}
            dp_means = {}
            dp_data = {}

            for column in data.select_dtypes(include=[np.number]).columns:
                original_mean = data[column].mean()
                dp_column = data[column].apply(self.add_noise)
                dp_mean = dp_column.mean()

                original_means[column] = float(original_mean)
                dp_means[column] = float(dp_mean)
                dp_data[column] = dp_column.tolist()

            privacy_budget = self.epsilon  # Simplified example
            return {
                "privacy_budget": privacy_budget,
                "quasi_identifiers": quasi_identifiers,
                "compliance": True,  # Placeholder for actual compliance check
                "original_means": original_means,
                "dp_means": dp_means,
                "dp_data": dp_data
            }
        except Exception as e:
            logging.error(f"An error occurred during privacy evaluation: {e}")
            return {
                "privacy_budget": None,
                "quasi_identifiers": quasi_identifiers,
                "compliance": False,
                "original_means": {},
                "dp_means": {},
                "dp_data": {}
            }

    def apply_model(self, data: pd.DataFrame, quasi_identifiers: List[str], suppression: bool = True, **kwargs) -> pd.DataFrame:
        """
        Apply the differential privacy model to transform the dataset.

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
            The transformed dataset with differential privacy guarantees applied.
        """
        # Apply noise to the numeric columns
        dp_data = data.copy()
        for column in dp_data.select_dtypes(include=[np.number]).columns:
            dp_data[column] = dp_data[column].apply(self.add_noise)

        return dp_data
