"""
PAMOLA.CORE - Pearson Correlation
---------------------------------
This module provides the Pearson correlation coefficient for assessing the linear
relationship between real and synthetic datasets.

Key features:
- Calculates correlation coefficient for multiple columns
- Evaluates linear relationship strength and direction
- Handles constant columns gracefully

These metrics help data custodians understand the linear relationships preserved in synthetic data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from scipy.stats import pearsonr
from typing import List, Dict
from pamola_core.metrics__old_10072025.base import QualityMetric

# Configure logging
import logging
logger = logging.getLogger(__name__)

class PearsonCorrelation(QualityMetric):
    def __init__(self, name: str = "Pearson Correlation", description: str = "Evaluate linear relationship"):
        super().__init__(name, description)

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, float]:
        logger.info("Calculating Pearson Correlation metrics")
        results = {}
        columns_to_evaluate = target_columns if target_columns else real_data.select_dtypes(include=["number"]).columns.tolist()

        try:
            for col in columns_to_evaluate:
                if col in real_data.columns and col in synthetic_data.columns:
                    if real_data[col].nunique() == 1 or synthetic_data[col].nunique() == 1:
                        results[col] = None  # Cannot calculate correlation if column is constant
                    else:
                        pearson_corr, _ = pearsonr(real_data[col], synthetic_data[col])
                        results[col] = pearson_corr

            logger.info("Pearson Correlation calculation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error during Pearson Correlation calculation: {e}")
            raise