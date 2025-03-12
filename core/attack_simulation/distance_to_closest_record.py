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
from scipy.spatial import KDTree, distance
from pamola.core.attack_simulation.preprocess_data import PreprocessData



class DistanceToClosestRecord(PreprocessData):
    """
    DistanceToClosestRecord class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods calculate Distance to Closest Record (DCR) for attack simulation.
    """

    def __init__(self):
        pass


    def calculate_dcr_kdtree(self, data1, data2):
        """
        Distance to Closest Record (DCR): The distance from a data point of dataset to the nearest data point in another dataset
        Using library scipy.spatial.KDTree

        Parameters:
        -----------
        data1: First dataset
        data2: Second dataset

        Returns:
        -----------
        dcr_values: Array of distances from a data point of dataset data2 to the nearest data point in dataset data1
        The larger the values of dcr, the greater the difference between two datasets
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        
        data1_transform, data2_transform = self.preprocess_data(data1, data2)

        tree = KDTree(data1_transform)
        # For each data point in data2 find the nearest neighbors in data1
        dcr_values, indices = tree.query(data2_transform, k=1)

        return dcr_values


    def calculate_dcr_distance(self, data1, data2):
        """
        Distance to Closest Record (DCR): The distance from a data point of dataset to the nearest data point in another dataset
        Using library scipy.spatial.distance

        Parameters:
        -----------
        data1: First data set
        data2: Second data set

        Returns:
        -----------
        dcr_values: Array of distances from a data point of dataset data2 to the nearest data point in dataset data1
        The larger the values of dcr, the greater the difference between two data sets
        """

        # Check that the datasets are valid
        if data1 is None or data2 is None:
            raise ValueError("Input datasets cannot be None.")
        
        data1_transform, data2_transform = self.preprocess_data(data1, data2)

        # Calculate the Euclidean distance between each data point in data2_transform and all data points in data1_transform
        distances = distance.cdist(data2_transform, data1_transform, metric='euclidean')

        # Filter to get the smallest distance corresponding to each data point of data2_transform
        dcr_values = np.min(distances, axis=1)

        return dcr_values