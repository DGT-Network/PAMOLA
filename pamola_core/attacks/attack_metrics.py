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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from pamola_core.attacks.preprocess_data import PreprocessData


class AttackMetrics(PreprocessData):
    """
    AttackMetrics class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods evaluate the performance of an attack for attack simulation.
    """

    def __init__(self):
        pass

    def attack_metrics(self, y_true, y_pred):
        """
        Compute attack performance metrics for Membership Inference Attacks (MIA).

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels for each sample in the test set.
            - 1 → sample is part of the training set (member)
            - 0 → sample is not part of the training set (non-member)

        y_pred : array-like of shape (n_samples,)
            Predicted labels output by the attack algorithm.
            - 1 → predicted as member
            - 0 → predicted as non-member

        Returns
        -------
        dict
            A dictionary containing key attack metrics:
            - "Attack Accuracy": Overall correctness of attack predictions.
            - "Precision": Among predicted members, proportion that are true members.
            - "Recall": Among actual members, proportion correctly identified.
            - "F1-Score": Harmonic mean of precision and recall (balances the two).
            - "AUC-ROC": Ability to distinguish members vs non-members (0.5 = random).
            - "Advantage": How much better the attack is than random guessing.
            Formula: 2 * (Attack Accuracy − 0.5)
        """

        # --- Core metrics ---
        attack_acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC requires both positive and negative classes to exist
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.5  # fallback: if only one class is present, attack ≈ random

        # Advantage = how much better the attack is than random guessing
        advantage = 2 * (attack_acc - 0.5)

        results = {
            "Attack Accuracy": round(attack_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4),
            "Advantage": round(advantage, 4),
        }

        return results

    def attack_success_rate(self, y_true, y_pred):
        """
        Compute Attack Success Rate (ASR) for Membership Inference Attack.

        ASR measures the attacker's ability to correctly identify training set members.
        Essentially, this is equivalent to Recall for the "member" class.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels (1 = member, 0 = non-member).

        y_pred : array-like of shape (n_samples,)
            Predicted labels by the attack algorithm (1 = member, 0 = non-member).

        Returns
        -------
        float
            Attack Success Rate (ASR):
            - High ASR → Attacker can correctly predict many actual members.
            - Low ASR → Attacker struggles to distinguish members from non-members.
        """

        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Total actual members in ground truth
        total_members = np.sum(y_true == 1)

        if total_members == 0:
            return 0.0  # No members present → ASR undefined, return 0

        # True Positives: correctly predicted members
        true_positives = np.sum((y_pred == 1) & (y_true == 1))

        # ASR = TP / (TP + FN)
        asr = true_positives / total_members

        return round(asr, 4)

    def residual_risk_score(self, y_true, y_pred):
        """
        Compute Residual Risk Score (RRS) for Membership Inference Attack.

        RRS estimates the residual privacy risk of the system by measuring
        the difference between:
        - P(Y=1 | X=1): probability of correctly identifying members (TPR)
        - P(Y=1 | X=0): probability of incorrectly identifying non-members as members (FPR)

        RRS = P(Y=1 | X=1) - P(Y=1 | X=0)

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth labels (1 = member, 0 = non-member).

        y_pred : array-like of shape (n_samples,)
            Predicted labels by the attack algorithm (1 = member, 0 = non-member).

        Returns
        -------
        float
            Residual Risk Score (RRS):
            - RRS > 0: attack performs better than random, indicating privacy risk
            - RRS = 0: attack equivalent to random guess (no residual risk)
            - RRS < 0: attack performs worse than random (unlikely but possible)
        """

        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Total members and non-members
        total_members = np.sum(y_true == 1)
        total_non_members = np.sum(y_true == 0)

        # P(Y=1 | X=1): True Positive Rate (TPR)
        if total_members > 0:
            p_y1_given_x1 = np.sum((y_pred == 1) & (y_true == 1)) / total_members
        else:
            p_y1_given_x1 = 0.0

        # P(Y=1 | X=0): False Positive Rate (FPR)
        if total_non_members > 0:
            p_y1_given_x0 = np.sum((y_pred == 1) & (y_true == 0)) / total_non_members
        else:
            p_y1_given_x0 = 0.0

        # Residual Risk Score
        rrs = p_y1_given_x1 - p_y1_given_x0
        return round(rrs, 4)
