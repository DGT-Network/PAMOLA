"""
PAMOLA.CORE - l-Diversity Information Loss Metrics
-------------------------------------------------
This module provides specialized metrics for quantifying information loss
due to l-diversity constraints in anonymized datasets. These metrics help
evaluate the utility-privacy tradeoff specific to l-diversity anonymization.

Key features:
- Specialized metrics for different l-diversity types (distinct, entropy, recursive)
- Analysis of utility impact on sensitive attributes
- Assessment of analytical value reduction
- Correlation loss measurement between attributes
- Comprehensive utility evaluation framework for l-diversity

These metrics help data custodians understand how l-diversity protection
affects the utility of their data for various analytical purposes.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import logging
from typing import List
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class LDiversityLossMetric:
    """
    Specialized metric for measuring information loss due to enforcing l-diversity.

    This metric focuses on the loss of utility resulting from ensuring
    diversity of sensitive attribute values within equivalence classes.
    It supports different types of l-diversity (distinct, entropy, recursive).
    """

    def __init__(self, diversity_type: str = "distinct"):
        """
        Initialize the l-diversity loss metric.

        Parameters:
        -----------
        diversity_type : str, optional
            Type of l-diversity for evaluation ("distinct", "entropy", "recursive").
        """
        self.name = f"{diversity_type.capitalize()} l-Diversity Loss"
        self.description = (
            f"Measures information loss due to enforcing {diversity_type} l-diversity"
        )
        self.diversity_type = diversity_type

    def calculate_metric(
        self,
        orig_data: pd.DataFrame,
        anon_data: pd.DataFrame,
        sensitive_attributes: List[str],
        quasi_identifiers: List[str],
    ) -> float:
        """
        Calculate loss for predictive modeling tasks.

        Returns:
        --------
        float
            Loss as a percentage (0-100).
        """
        predictive_losses = []

        for attr in sensitive_attributes:
            loss = self._calculate_predictive_value_loss(
                orig_data, anon_data, attr, quasi_identifiers
            )
            predictive_losses.append(loss)

        return np.mean(predictive_losses) if predictive_losses else 0

    def _calculate_predictive_value_loss(
        self,
        orig_data: pd.DataFrame,
        anon_data: pd.DataFrame,
        sensitive_attr: str,
        quasi_identifiers: List[str],
    ) -> float:
        """
        Estimates information loss for predictive modeling due to l-diversity.

        This function assumes that a sensitive attribute can be predicted
        based on quasi-identifiers. It evaluates how much information loss
        occurs when comparing the original dataset to the anonymized dataset.

        Parameters:
        -----------
        orig_data : pd.DataFrame
            The original dataset.
        anon_data : pd.DataFrame
            The anonymized dataset.
        sensitive_attr : str
            The sensitive attribute for which predictive loss is calculated.
        quasi_identifiers : List[str]
            The list of quasi-identifiers.

        Returns:
        --------
        float
            Loss as a percentage (0-100).
        """
        try:
            if (
                sensitive_attr not in orig_data.columns
                or sensitive_attr not in anon_data.columns
            ):
                return (
                    100  # Maximum loss if the attribute is missing in anonymized data
                )

            # Check if the sensitive attribute is categorical or numerical
            if pd.api.types.is_numeric_dtype(orig_data[sensitive_attr]):
                # Use mean squared error for numerical attributes
                orig_mean = orig_data.groupby(quasi_identifiers)[sensitive_attr].mean()
                anon_mean = anon_data.groupby(quasi_identifiers)[sensitive_attr].mean()

                # Compute the error
                mse = np.mean((orig_mean - anon_mean) ** 2)

                # Normalize loss
                loss = min(100, 100 * mse / (orig_mean.std() + 1e-6))

            else:
                # Use distributional divergence for categorical attributes
                orig_counts = orig_data.groupby(quasi_identifiers)[
                    sensitive_attr
                ].value_counts(normalize=True)
                anon_counts = anon_data.groupby(quasi_identifiers)[
                    sensitive_attr
                ].value_counts(normalize=True)

                # Align indices
                all_indices = orig_counts.index.union(anon_counts.index)
                orig_counts = orig_counts.reindex(all_indices, fill_value=0)
                anon_counts = anon_counts.reindex(all_indices, fill_value=0)

                # Compute KL divergence (with smoothing)
                kl_div = (
                    orig_counts * np.log((orig_counts + 1e-6) / (anon_counts + 1e-6))
                ).sum()

                # Normalize loss
                loss = min(100, 100 * kl_div)

            return loss

        except Exception as e:
            logger.warning(
                f"Error calculating predictive value loss for {sensitive_attr}: {e}"
            )
            return 100  # Assume maximum loss in case of failure
