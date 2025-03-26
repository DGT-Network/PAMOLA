"""
PAMOLA.CORE - Kullback-Leibler Divergence
-----------------------------------------
This module provides the Kullback-Leibler Divergence for measuring information
loss between real and synthetic datasets.

Key features:
- Quantifies divergence between probability distributions
- Provides KLD score for multiple columns
- Handles distribution comparison failures gracefully

These metrics help data custodians evaluate the information loss in synthetic data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from scipy.stats import entropy
from typing import List, Dict
from core.metrics.base import QualityMetric

# Configure logging
import logging
logger = logging.getLogger(__name__)

class KullbackLeiblerDivergence(QualityMetric):
    def __init__(self, name: str = "Kullback-Leibler Divergence", description: str = "Evaluate information loss"):
        super().__init__(name, description)

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, float]:
        logger.info("Calculating Kullback-Leibler Divergence metrics")
        results = {}
        columns_to_evaluate = target_columns if target_columns else real_data.select_dtypes(include=["number"]).columns.tolist()

        try:
            for col in columns_to_evaluate:
                if col in real_data.columns and col in synthetic_data.columns:
                    real_dist = real_data[col].value_counts(normalize=True, bins=10, sort=False)
                    synth_dist = synthetic_data[col].value_counts(normalize=True, bins=10, sort=False)
                    kld_score = entropy(real_dist + 1e-9, synth_dist + 1e-9)
                    results[col] = kld_score

            logger.info("Kullback-Leibler Divergence calculation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error during Kullback-Leibler Divergence calculation: {e}")
            raise