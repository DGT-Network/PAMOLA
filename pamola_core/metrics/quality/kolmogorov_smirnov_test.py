"""
PAMOLA.CORE - Kolmogorov-Smirnov Test
-------------------------------------
This module provides the Kolmogorov-Smirnov Test for evaluating the similarity
between the distributions of real and synthetic datasets.

Key features:
- Compares empirical distribution functions of two datasets
- Provides KS statistic and p-value for distribution similarity
- Supports evaluation of multiple columns

These metrics help data custodians assess the distributional similarity of synthetic data.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import pandas as pd
from scipy.stats import ks_2samp
from typing import List, Dict
from pamola.pamola_core.metrics.base import QualityMetric

# Configure logging
import logging
logger = logging.getLogger(__name__)

class KolmogorovSmirnovTest(QualityMetric):
    def __init__(self, name: str = "Kolmogorov-Smirnov Test", description: str = "Evaluate distribution similarity"):
        super().__init__(name, description)

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, target_columns: List[str] = None) -> Dict[str, tuple]:
        logger.info("Calculating Kolmogorov-Smirnov Test metrics")
        results = {}
        columns_to_evaluate = target_columns if target_columns else real_data.select_dtypes(include=["number"]).columns.tolist()

        try:
            for col in columns_to_evaluate:
                if col in real_data.columns and col in synthetic_data.columns:
                    ks_stat, ks_pval = ks_2samp(real_data[col], synthetic_data[col])
                    results[f"{col}_ks_stat"] = ks_stat
                    results[f"{col}_ks_pval"] = ks_pval

            logger.info("Kolmogorov-Smirnov Test calculation completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error during Kolmogorov-Smirnov Test calculation: {e}")
            raise