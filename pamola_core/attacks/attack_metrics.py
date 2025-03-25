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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pamola.pamola_core.attacks.preprocess_data import PreprocessData



class AttackMetrics(PreprocessData):
    """
    AttackMetrics class for attack simulation in PAMOLA.CORE.
    This class extends PreprocessData and define methods evaluate the performance of an attack for attack simulation.
    """
    
    def __init__(self):
        pass


    def attack_metrics(self, y_true, y_pred):
        """
        Attack Metrics: Calculate the attack performance metrics (Membership Inference Attack (MIA))

        Parameters:
        -----------
        y_true: actual label of each data sample in the test set (1: member, 0: non-member)
        y_pred: predictive label of attack algorithm

        Returns:
        -----------
        dict: Attack Metrics Results
            - Attack Accuracy: the rate of correct predictions of the attack algorithm over the total number of samples. The higher the accuracy → the more successful the attack
            - Precision: Among the predicted "member" samples, how many are actually "member" samples? High precision → less confusion between member and non-member
            - Recall: number of true member samples, how many samples are predicted correctly. High recall → Attack is able to detect many members
            - F1-Score: is the harmonic average of Precision and Recall, which balances precision and detection. F1-Score near 1 → Strong attack (both accurate and good at detection)
            - AUC-ROC: Measures the ability to distinguish between members and non-members of an attack model
                AUC = 0.5 → Attack equivalent to random guess
                AUC > 0.5 → Attack is able to distinguish between members and non-members
                AUC ≈ 1.0 → Very strong attack (almost perfect discrimination)
            - Advantage: measures how much better the attack is than random guess
                Advantage = 0 → Useless attack (just random guess)
                Advantage > 0 → Effective attack
                Advantage ≈ 1 → Very strong attack
        """

        attack_acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred)
        advantage = 2 * (attack_acc - 0.5)

        results = {
            "Attack Accuracy": round(attack_acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4),
            "Advantage": round(advantage, 4)
        }

        return results


    def attack_success_rate(self, y_true, y_pred):
        """
        Attack Success Rate (ASR): measures the ability of the attacker to correctly identify the training set members

        Parameters:
        -----------
        y_true: actual label (1: member, 0: non-member)
        y_pred: predicted label (1: member, 0: non-member)

        Returns:
        -----------
        Ratio of correctly guessed members to the total number of actual members
            - High ASR → Attacker can correctly predict many members
            - Low ASR → Attacker has difficulty distinguishing members from non-members
        """

        # Actual number of members
        total_members = np.sum(y_true)

        # ASR = TP / (TP + FN)
        asr = np.sum(
            (y_pred == 1) & (y_true == 1)) / total_members if total_members > 0 else 0  # Avoid dividing by zero

        return round(asr, 4)  # Round to 4 decimal places


    def residual_risk_score(self, y_true, y_pred):
        """
        Residual Risk Score (RRS): Risk level of the system

        Parameters:
        -----------
        y_true: actual label (1: member, 0: non-member)
        y_pred: predicted label (1: member, 0: non-member)

        Returns:
        -----------
            float: Residual Risk Score (RRS)
        """
        
        # Number of members (X=1) and non-members (X=0)
        total_members = np.sum(y_true)
        total_non_members = len(y_true) - total_members

        # Probability of correctly predicting member: P(Y=1 | X=1)
        p_y1_given_x1 = np.sum((y_pred == 1) & (y_true == 1)) / total_members if total_members > 0 else 0

        # Probability of incorrectly predicting that a non-member is a member: P(Y=1 | X=0)
        p_y1_given_x0 = np.sum((y_pred == 1) & (y_true == 0)) / total_non_members if total_non_members > 0 else 0

        # Residual Risk Score (RRS): P(Y=1 | X=1) - P(Y=1 | X=0)
        rrs = p_y1_given_x1 - p_y1_given_x0
        return round(rrs, 4)  # Round to 4 decimal places