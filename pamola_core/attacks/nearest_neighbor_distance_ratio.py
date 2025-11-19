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

NOTE: This module requires 'numpy', 'scipy' and 'scikit-learn' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from pamola_core.attacks.preprocess_data import PreprocessData


class NearestNeighborDistanceRatio(PreprocessData):
    """
    NearestNeighborDistanceRatio class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods calculate Nearest Neighbor Distance Ratio (NNDR) for attack simulation.
    """

    def __init__(self):
        pass

    def calculate_nndr(
        self, data1: pd.DataFrame, data2: pd.DataFrame, method: str = "kdtree"
    ) -> np.ndarray:
        """
        Nearest Neighbor Distance Ratio (NNDR):
        Ratio of distance to the nearest neighbor vs. second nearest neighbor
        for each record in data2, relative to data1.

        A smaller NNDR indicates a more distinct / reliable match.

        Parameters
        ----------
        data1 : pd.DataFrame
            First dataset (reference set)
        data2 : pd.DataFrame
            Second dataset (query set)
        method : {"kdtree", "neighbors"}, default="kdtree"
            - "kdtree": use scipy.spatial.KDTree (fast for numeric data)
            - "neighbors": use sklearn NearestNeighbors (more flexible metrics)

        Returns
        -------
        nndr_values : np.ndarray
            Array of NNDR values for each record in data2.
        """
        # --- 1. Validate input ---
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        if data1.empty or data2.empty:
            return np.array([])

        # --- 2. Preprocess ---
        data1_vec, data2_vec = self.preprocess_data(data1, data2)

        # --- 3. Find two nearest neighbors ---
        if method == "kdtree":
            tree = KDTree(data1_vec)
            distances, _ = tree.query(data2_vec, k=2)
        elif method == "neighbors":
            nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto")
            nbrs.fit(data1_vec)
            distances, _ = nbrs.kneighbors(data2_vec)
        else:
            raise ValueError(
                f"Unknown NNDR method: {method}. Must be 'kdtree' or 'neighbors'."
            )

        # --- 4. Compute NNDR ---
        nndr_values = distances[:, 0] / np.maximum(distances[:, 1], 1e-10)

        return nndr_values
