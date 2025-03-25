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
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from pamola.pamola_core.attacks.preprocess_data import PreprocessData



class NearestNeighborDistanceRatio(PreprocessData):
    """
    NearestNeighborDistanceRatio class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods calculate Nearest Neighbor Distance Ratio (NNDR) for attack simulation.
    """

    def __init__(self):
        pass


    def calculate_nndr_kdtree(self, data1, data2):
        """
        Nearest Neighbor Distance Ratio (NNDR): The ratio of the distance between a given data point of a dataset to the nearest data point and the second nearest data point in the another dataset
        Using library sklearn.neighbors.NearestNeighbors

        Parameters
        -----------
        data1: First data set
        data2: Second data set

        Returns:
        -----------
        nndr_value: Array of the ratio of the distance between a given data point of dataset data2 to the nearest data point and the second nearest data point in the dataset data1
        The smaller the value of nndr_value, the more clearly the data point has a nearest neighbor. More reliable pairing
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        
        data1_transform, data2_transform = self.preprocess_data(data1, data2)

        # Init KDTree and for each data point in data2_transform find the 2 nearest neighbors in data1_transform
        tree = KDTree(data1_transform)
        distances, indices = tree.query(data2_transform, k=2)

        # Calculate the ratio between two nearest neighbors
        nndr_values = distances[:, 0] / np.maximum(distances[:, 1], 1e-10)

        return nndr_values


    def calculate_nndr_neighbors(self, data1, data2):
        """
        Nearest Neighbor Distance Ratio (NNDR): The ratio of the distance between a given data point of a dataset to the nearest data point and the second nearest data point in the another dataset
        Using library sklearn.neighbors.NearestNeighbors

        Parameters
        -----------
        data1: First data set
        data2: Second data set

        Returns:
        -----------
        nndr_value: Array of the ratio of the distance between a given data point of dataset data2 to the nearest data point and the second nearest data point in the dataset data1
        The smaller the value of nndr_value, the more clearly the data point has a nearest neighbor. More reliable pairing
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        
        data1_transform, data2_transform = self.preprocess_data(data1, data2)

        # Init NearestNeighbors and for each data point in data2 find the 2 nearest neighbors in data1
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto')
        nbrs.fit(data1_transform)
        distances, indices = nbrs.kneighbors(data2_transform)

        # Calculate the ratio between two nearest neighbors
        nndr_values = distances[:, 0] / np.maximum(distances[:, 1], 1e-10)

        return nndr_values