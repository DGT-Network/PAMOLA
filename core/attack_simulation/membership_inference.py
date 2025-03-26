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

NOTE: This module requires 'numpy' and 'scikit-learn' as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from pamola.core.attack_simulation.distance_to_closest_record import DistanceToClosestRecord
from pamola.core.attack_simulation.nearest_neighbor_distance_ratio import NearestNeighborDistanceRatio
from pamola.core.attack_simulation.preprocess_data import PreprocessData



class MembershipInference(PreprocessData):
    """
    MembershipInference class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods membership inference attack (MIA) for attack simulation.
    """

    def __init__(self, dcr_threshold = None, nndr_threshold = None, m_threshold = None):
        """
        The constructor of the MembershipInferenceAttack class

        Parameters:
        -----------
        dcr_threshold: Threshold used for DCR algorithm (MIA)
        nndr_threshold: Threshold used for NNDR algorithm (MIA)
        m_threshold: Threshold used for model trainning (MIA)
        dcr: DistanceToClosestRecord object is used to call the functions of the DistanceToClosestRecord class
        nndr: NearestNeighborDistanceRatio object is used to call the functions of the NearestNeighborDistanceRatio class
        """

        self.dcr_threshold = dcr_threshold
        self.nndr_threshold = nndr_threshold
        self.m_threshold = m_threshold
        self.dcr = DistanceToClosestRecord()
        self.nndr = NearestNeighborDistanceRatio()


    def membership_inference_attack_dcr(self, data_train, data_test):
        """
        The function performs a Membership Inference Attack (MIA) using DCR
        This is an attack that attempts to infer whether a data point is part of the model's training set or not

        Parameters:
        -----------
        data_train: Training dataset
        data_test: Test dataset

        Returns:
        -----------
        prediction_values: list of predictions for each sample(data point) of data_test
            1: sample belongs to the data_train
            0: sample not in data_train
        """

        # Check that the datasets are valid
        if data_train is None or data_test is None:
            raise ValueError("Input datasets cannot be None.")
        
        # Calculate DCR for each samples of data_test vs data_train
        dcr_values = self.dcr.calculate_dcr_distance(data_train, data_test)

        # If dcr_threshold is None, determine threshold based on median
        threshold = (
            self.dcr_threshold if self.dcr_threshold is not None
            else np.median(dcr_values[dcr_values > 0]) if np.any(dcr_values > 0)
            else np.median(dcr_values)
        )

        # Membership prediction: if DCR < threshold, then "member", otherwise "non-member"
        prediction_values = (dcr_values < threshold).astype(int)

        return prediction_values


    def membership_inference_attack_nndr(self, data_train, data_test):
        """
        The function performs a Membership Inference Attack (MIA) using NNDR

        Parameters:
        -----------
        data_train: Training dataset
        data_test: Test dataset

        Returns:
        -----------
        prediction_values: list of predictions for each sample(data point) of data_test
            1: sample belongs to the data_train
            0: sample not in data_train
        """

        # Check that the datasets are valid
        if data_train is None or data_test is None:
            raise ValueError("Input datasets cannot be None.")
        
        # Calculate NNDR for each samples of data_test vs data_train
        nndr_values = self.nndr.calculate_nndr_kdtree(data_train, data_test)

        # If nndr_threshold is None, determine threshold based on median
        threshold = (
            self.nndr_threshold if self.nndr_threshold is not None
            else np.median(nndr_values[nndr_values > 0]) if np.any(nndr_values > 0)
            else np.median(nndr_values)
        )

        # Membership prediction: if NNDR < threshold, then "member", otherwise "non-member"
        prediction_values = (nndr_values < threshold).astype(int)

        return prediction_values


    def membership_inference_attack_model(self, data_train, data_test):
        """
        The function performs a Membership Inference Attack (MIA) using model trainning

        Parameters:
        -----------
        data_train: Training dataset
        data_test: Test dataset

        Returns:
        -----------
        prediction_values: list of predictions for each sample(data point) of data_test
            1: sample belongs to the data_train
            0: sample not in data_train
        """

        # Check that the datasets are valid
        if data_train is None or data_test is None:
            raise ValueError("Input datasets cannot be None.")
        
        data1_transform, data2_transform = self.preprocess_data(data_train, data_test)

        kmeans = KMeans(n_clusters=5, random_state=42)
        label_train = kmeans.fit_predict(data1_transform)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data1_transform, label_train)

        probs_values = model.predict_proba(data2_transform).max(axis=1)

        # If m_threshold is None, determine threshold based on median
        threshold = (
            self.m_threshold if self.m_threshold is not None
            else np.median(probs_values[probs_values > 0]) if np.any(probs_values > 0)
            else np.median(probs_values)
        )

        # Membership prediction: if probs > threshold, then "member", otherwise "non-member"
        prediction_values = (probs_values > threshold).astype(int)

        return prediction_values