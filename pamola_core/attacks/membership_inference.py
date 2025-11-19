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
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from pamola_core.attacks.distance_to_closest_record import DistanceToClosestRecord
from pamola_core.attacks.nearest_neighbor_distance_ratio import (
    NearestNeighborDistanceRatio,
)
from pamola_core.attacks.preprocess_data import PreprocessData


class MembershipInference(PreprocessData):
    """
    MembershipInference class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods membership inference attack (MIA) for attack simulation.
    """

    def __init__(self, dcr_threshold=None, nndr_threshold=None, m_threshold=None):
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

    def membership_inference_attack_dcr(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Perform a Membership Inference Attack using DCR-based distance.
        A test point is classified as a member if its DCR distance to the
        training set is below a chosen threshold.

        Parameters
        ----------
        data_train : pd.DataFrame
            Training dataset.
        data_test : pd.DataFrame
            Test dataset.

        Returns
        -------
        np.ndarray
            Binary predictions for each test sample:
            1 = inferred member, 0 = inferred non-member.
        """
        if (
            data_train is None
            or data_test is None
            or data_train.empty
            or data_test.empty
        ):
            return np.array([], dtype=int)

        # Compute DCR distances for each test sample
        dcr_values = self.dcr.calculate_dcr(data_train, data_test, method="cdist")

        # Determine threshold
        if self.dcr_threshold is not None:
            threshold = self.dcr_threshold
        else:
            positive_vals = dcr_values[dcr_values > 0]
            if len(positive_vals) > 0:
                threshold = np.median(positive_vals)
            else:
                threshold = np.median(dcr_values)

        if threshold <= 0:
            # Fallback to a more robust heuristic (e.g., 50th percentile)
            threshold = np.percentile(dcr_values, 50)

        # Predict membership: 1 if distance < threshold, else 0
        predictions = (dcr_values < threshold).astype(int)

        return predictions

    def membership_inference_attack_nndr(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Perform a Membership Inference Attack using NNDR (Nearest Neighbor Distance Ratio).

        NNDR is defined as d1 / d2, where d1 and d2 are the distances to the 1st and 2nd
        nearest neighbors in the training set. A low NNDR indicates the sample is very
        close to a single training point, suggesting membership.

        Parameters
        ----------
        data_train : pd.DataFrame
            Training dataset.
        data_test : pd.DataFrame
            Test dataset.

        Returns
        -------
        np.ndarray
            Binary predictions for each test sample:
            1 = inferred member, 0 = inferred non-member.
        """
        if (
            data_train is None
            or data_test is None
            or data_train.empty
            or data_test.empty
        ):
            return np.array([], dtype=int)

        # Compute NNDR values using KDTree
        nndr_values = self.nndr.calculate_nndr(data_train, data_test, method="kdtree")

        # Determine threshold
        if self.nndr_threshold is not None:
            threshold = self.nndr_threshold
        else:
            positive_vals = nndr_values[nndr_values > 0]
            if len(positive_vals) > 0:
                threshold = np.median(positive_vals)
            else:
                threshold = np.median(nndr_values)

        if threshold <= 0:
            threshold = np.percentile(nndr_values, 50)  # fallback to median

        predictions = (nndr_values < threshold).astype(int)

        return predictions

    def membership_inference_attack_model(self, data_train, data_test):
        """
        Perform a clustering-based confidence Membership Inference Attack (MIA).

        This attack clusters the training data, trains a classifier to predict cluster labels,
        and uses prediction confidence on test data to decide membership.

        Parameters
        ----------
        data_train : pd.DataFrame
            Training dataset.
        data_test : pd.DataFrame
            Test dataset.

        Returns
        -------
        np.ndarray
            Binary predictions for each sample in data_test:
            1 = predicted as member of data_train, 0 = non-member.
        """
        if data_train is None or data_test is None:
            raise ValueError("Input datasets cannot be None.")

        # Preprocess consistently
        data1_transform, data2_transform = self.preprocess_data(data_train, data_test)

        # Avoid error if data_train is too small
        n_clusters = min(5, len(data_train))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        label_train = kmeans.fit_predict(data1_transform)

        # Train attack model to predict cluster labels
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data1_transform, label_train)

        # Confidence scores on test data
        confidence_scores = model.predict_proba(data2_transform).max(axis=1)

        # Determine threshold
        if self.m_threshold is not None:
            threshold = self.m_threshold
        else:
            threshold = (
                np.median(confidence_scores[confidence_scores > 0])
                if np.any(confidence_scores > 0)
                else np.median(confidence_scores)
            )

        # Predict membership based on confidence
        membership_predictions = (confidence_scores > threshold).astype(int)

        return membership_predictions
