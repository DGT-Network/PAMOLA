"""
PAMOLA.CORE - Wasserstein Distance
----------------------------------
This module provides the Wasserstein Distance for evaluating the similarity
between the distributions of real and synthetic datasets.

Key features:
- Calculates Earth Mover's Distance for distribution comparison
- Supports evaluation of multiple columns
- Handles distance calculation failures gracefully

These metrics help data custodians assess the distributional similarity of synthetic data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from scipy.stats import wasserstein_distance
from typing import List, Dict

# Configure logging
import logging

logger = logging.getLogger(__name__)


class WassersteinDistance:
    def __init__(
        self,
        name: str = "Wasserstein Distance",
        description: str = "Evaluate distribution similarity",
    ):
        """Initialize the Wasserstein Distance metric."""
        self.name = name
        self.description = description

    def calculate_metric(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_columns: List[str] = None,
    ) -> Dict[str, float]:
        logger.info("Calculating Wasserstein Distance metrics")
        results = {}
        columns_to_evaluate = (
            target_columns
            if target_columns
            else real_data.select_dtypes(include=["number"]).columns.tolist()
        )

        try:
            for col in columns_to_evaluate:
                if col in real_data.columns and col in synthetic_data.columns:
                    results[col] = wasserstein_distance(
                        real_data[col], synthetic_data[col]
                    )

            logger.info("Wasserstein Distance calculation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error during Wasserstein Distance calculation: {e}")
            raise
