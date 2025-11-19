"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Attack Simulation
-----------------------
This module provides an abstract base class for attack simulation feature
in PAMOLA.CORE. It defines the general structure and required methods for
implementing specific attack simulation

NOTE: This module requires 'numpy' and 'scipy' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, distance
from pamola_core.attacks.preprocess_data import PreprocessData


class DistanceToClosestRecord(PreprocessData):
    """
    DistanceToClosestRecord class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods calculate Distance to Closest Record (DCR) for attack simulation.
    """

    def __init__(self):
        pass

    def calculate_dcr(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        method: str = "kdtree",
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Distance to Closest Record (DCR): The distance from each record in data2
        to its nearest record in data1. Measures how similar two datasets are.

        Parameters
        ----------
        data1 : pd.DataFrame
            First dataset (reference set)
        data2 : pd.DataFrame
            Second dataset (query set)
        method : {"kdtree", "cdist"}, default="kdtree"
            - "kdtree": use scipy.spatial.KDTree (fast for large datasets)
            - "cdist": use scipy.spatial.distance.cdist (good for custom metrics)
        metric : str, default="euclidean"
            Distance metric used when method="cdist".

        Returns
        -------
        dcr_values : np.ndarray
            Array of distances from each point in data2 to its nearest neighbor in data1.
            Larger values indicate greater dissimilarity between the datasets.
        """
        # --- 1. Input validation ---
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        if data1.empty or data2.empty:
            return np.array([])

        # --- 2. Preprocess to numeric arrays ---
        data1_vec, data2_vec = self.preprocess_data(data1, data2)

        # --- 3. Compute DCR ---
        if method == "kdtree":
            tree = KDTree(data1_vec)
            dcr_values, _ = tree.query(data2_vec, k=1)
        elif method == "cdist":
            distances = distance.cdist(data2_vec, data1_vec, metric=metric)
            dcr_values = np.min(distances, axis=1)
        else:
            raise ValueError(
                f"Unknown DCR method: {method}. Must be 'kdtree' or 'cdist'."
            )

        return dcr_values
