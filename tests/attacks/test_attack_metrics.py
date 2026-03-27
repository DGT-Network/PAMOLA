"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for AttackMetrics

This module contains comprehensive unit tests for the AttackMetrics class.
Tests cover:
- Initialization
- Attack metrics calculation (accuracy, precision, recall, F1, AUC-ROC, advantage)
- Attack success rate (ASR) calculation
- Residual risk score (RRS) calculation
- Error handling for edge cases
- Metric boundary conditions
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from pamola_core.attacks.attack_metrics import AttackMetrics


class TestAttackMetricsInitialization(unittest.TestCase):
    """Test AttackMetrics initialization"""

    def test_initialization(self):
        """Test AttackMetrics can be initialized"""
        metrics = AttackMetrics()
        self.assertIsInstance(metrics, AttackMetrics)


class TestAttackMetrics(unittest.TestCase):
    """Test attack_metrics method"""

    def setUp(self):
        self.metrics = AttackMetrics()

    def test_attack_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        results = self.metrics.attack_metrics(y_true, y_pred)

        self.assertEqual(results['Attack Accuracy'], 1.0)
        self.assertEqual(results['Precision'], 1.0)
        self.assertEqual(results['Recall'], 1.0)
        self.assertEqual(results['F1-Score'], 1.0)
        self.assertEqual(results['AUC-ROC'], 1.0)
        self.assertEqual(results['Advantage'], 1.0)

    def test_attack_metrics_random_predictions(self):
        """Test metrics with random predictions (50% accuracy)"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # Accuracy should be 0.5 (50%), advantage should be 0
        self.assertEqual(results['Attack Accuracy'], 0.5)
        self.assertEqual(results['Advantage'], 0.0)

    def test_attack_metrics_all_positives(self):
        """Test metrics when all predictions are positive"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # Precision = TP/(TP+FP) = 2/4 = 0.5
        # Recall = TP/(TP+FN) = 2/2 = 1.0
        self.assertEqual(results['Precision'], 0.5)
        self.assertEqual(results['Recall'], 1.0)

    def test_attack_metrics_all_negatives(self):
        """Test metrics when all predictions are negative"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # Precision = TP/(TP+FP) = 0/0 -> 0 (zero_division=0)
        # Recall = TP/(TP+FN) = 0/2 = 0.0
        self.assertEqual(results['Precision'], 0.0)
        self.assertEqual(results['Recall'], 0.0)

    def test_attack_metrics_partial_correct(self):
        """Test metrics with partially correct predictions"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # Accuracy = 4/6 ≈ 0.667
        self.assertAlmostEqual(results['Attack Accuracy'], 4/6, places=3)

    def test_attack_metrics_structure(self):
        """Test that metrics dictionary has all required keys"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        required_keys = [
            'Attack Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            'AUC-ROC',
            'Advantage'
        ]
        for key in required_keys:
            self.assertIn(key, results)

    def test_attack_metrics_value_ranges(self):
        """Test that all metric values are in valid ranges"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 1, 0, 0, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        for key, value in results.items():
            if key != 'Advantage':
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            else:
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)

    def test_attack_metrics_f1_score_calculation(self):
        """Test F1 score calculation"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # TP=1, FP=0, FN=1, TN=2
        # Precision = 1/1 = 1.0
        # Recall = 1/2 = 0.5
        # F1 = 2*(1.0*0.5)/(1.0+0.5) = 1.0/1.5 ≈ 0.667
        self.assertAlmostEqual(results['F1-Score'], 2/3, places=3)

    def test_attack_metrics_single_class(self):
        """Test metrics with single class (only positives or only negatives)"""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # Should handle single class gracefully
        self.assertIsNotNone(results['Attack Accuracy'])

    def test_attack_metrics_auc_roc_fallback(self):
        """Test that AUC-ROC handles edge cases gracefully"""
        # Single class case — roc_auc_score raises ValueError, source falls back to 0.5
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        # When only one class present, source returns 0.5 as fallback;
        # accept nan as well in case the sklearn version differs.
        auc = results['AUC-ROC']
        self.assertTrue(
            auc in [0.5, 1.0] or (isinstance(auc, float) and np.isnan(auc)),
            f"Unexpected AUC-ROC value: {auc}"
        )


class TestAttackSuccessRate(unittest.TestCase):
    """Test attack_success_rate method"""

    def setUp(self):
        self.metrics = AttackMetrics()

    def test_asr_perfect_member_detection(self):
        """Test ASR with perfect member detection"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        self.assertEqual(asr, 1.0)

    def test_asr_no_member_detection(self):
        """Test ASR with no member detection"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        self.assertEqual(asr, 0.0)

    def test_asr_partial_member_detection(self):
        """Test ASR with partial member detection"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        # 2 out of 3 members correctly identified
        self.assertAlmostEqual(asr, 2/3, places=3)

    def test_asr_no_members_in_ground_truth(self):
        """Test ASR when there are no members in ground truth"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        # No members present -> ASR undefined, return 0
        self.assertEqual(asr, 0.0)

    def test_asr_only_false_positives(self):
        """Test ASR with only false positives (no true positives)"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        # No members correctly identified
        self.assertEqual(asr, 0.0)

    def test_asr_definition(self):
        """Test that ASR = Recall for positive class"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        # ASR should equal recall for members
        # TP=2, FN=2, so recall = 2/4 = 0.5
        self.assertEqual(asr, 0.5)

    def test_asr_numpy_array_input(self):
        """Test ASR with numpy array input"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        self.assertGreaterEqual(asr, 0.0)
        self.assertLessEqual(asr, 1.0)

    def test_asr_list_input(self):
        """Test ASR with list input"""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        self.assertGreaterEqual(asr, 0.0)
        self.assertLessEqual(asr, 1.0)


class TestResidualRiskScore(unittest.TestCase):
    """Test residual_risk_score method"""

    def setUp(self):
        self.metrics = AttackMetrics()

    def test_rrs_no_residual_risk(self):
        """Test RRS when attack is random (no residual risk)"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # RRS = TPR - FPR
        # TP=1, FP=1, FN=1, TN=1
        # TPR = 1/2 = 0.5, FPR = 1/2 = 0.5
        # RRS = 0.5 - 0.5 = 0.0
        self.assertEqual(rrs, 0.0)

    def test_rrs_perfect_attack(self):
        """Test RRS with perfect attack"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # RRS = TPR - FPR = 1.0 - 0.0 = 1.0
        self.assertEqual(rrs, 1.0)

    def test_rrs_inverse_attack(self):
        """Test RRS with inverse predictions"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # RRS = TPR - FPR = 0.0 - 1.0 = -1.0
        self.assertEqual(rrs, -1.0)

    def test_rrs_partial_attack(self):
        """Test RRS with partial successful attack"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # TP=2, FN=2, FP=2, TN=2
        # TPR = 2/4 = 0.5, FPR = 2/4 = 0.5
        # RRS = 0.5 - 0.5 = 0.0
        self.assertEqual(rrs, 0.0)

    def test_rrs_definition(self):
        """Test that RRS = TPR - FPR"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # TP=3, FN=1, FP=2, TN=2
        # TPR = 3/4 = 0.75, FPR = 2/4 = 0.5
        # RRS = 0.75 - 0.5 = 0.25
        self.assertAlmostEqual(rrs, 0.25, places=3)

    def test_rrs_no_members_or_non_members(self):
        """Test RRS when only one class exists"""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # TPR = 2/4 = 0.5, FPR = 0/0 = 0 (no non-members)
        # RRS = 0.5 - 0.0 = 0.5
        self.assertEqual(rrs, 0.5)

    def test_rrs_numpy_array_input(self):
        """Test RRS with numpy array input"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        self.assertGreaterEqual(rrs, -1.0)
        self.assertLessEqual(rrs, 1.0)

    def test_rrs_list_input(self):
        """Test RRS with list input"""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        self.assertGreaterEqual(rrs, -1.0)
        self.assertLessEqual(rrs, 1.0)


class TestAttackMetricsEdgeCases(unittest.TestCase):
    """Test edge cases for attack metrics"""

    def setUp(self):
        self.metrics = AttackMetrics()

    def test_metrics_single_sample(self):
        """Test metrics with single sample"""
        y_true = np.array([1])
        y_pred = np.array([1])

        results = self.metrics.attack_metrics(y_true, y_pred)

        self.assertIsNotNone(results['Attack Accuracy'])

    def test_metrics_large_dataset(self):
        """Test metrics with large dataset"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 10000)
        y_pred = np.random.randint(0, 2, 10000)

        results = self.metrics.attack_metrics(y_true, y_pred)

        self.assertIsNotNone(results['Attack Accuracy'])

    def test_metrics_imbalanced_dataset(self):
        """Test metrics with highly imbalanced dataset"""
        y_true = np.array([1] * 100 + [0] * 10)
        y_pred = np.array([1] * 100 + [0] * 10)

        results = self.metrics.attack_metrics(y_true, y_pred)

        self.assertEqual(results['Attack Accuracy'], 1.0)

    def test_asr_with_all_correct_predictions(self):
        """Test ASR when all predictions are correct"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        self.assertEqual(asr, 1.0)

    def test_rrs_with_all_correct_predictions(self):
        """Test RRS when all predictions are correct"""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        self.assertEqual(rrs, 1.0)

    def test_metrics_consistency_across_calls(self):
        """Test that metrics are consistent across multiple calls"""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])

        results1 = self.metrics.attack_metrics(y_true, y_pred)
        results2 = self.metrics.attack_metrics(y_true, y_pred)

        self.assertEqual(results1, results2)

    def test_asr_rrs_consistency(self):
        """Test relationship between ASR, RRS, and other metrics"""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1])

        results = self.metrics.attack_metrics(y_true, y_pred)
        asr = self.metrics.attack_success_rate(y_true, y_pred)
        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        # ASR should be recall
        self.assertAlmostEqual(asr, results['Recall'], places=3)


class TestAttackMetricsInterpreteation(unittest.TestCase):
    """Test interpretation of attack metrics from privacy perspective"""

    def setUp(self):
        self.metrics = AttackMetrics()

    def test_high_asr_indicates_privacy_risk(self):
        """Test that high ASR indicates privacy risk (vulnerable members)"""
        # Scenario: attack correctly identifies many members
        y_true = np.array([1] * 100 + [0] * 100)
        y_pred = np.array([1] * 90 + [0] * 10 + [0] * 100)  # 90% TPR

        asr = self.metrics.attack_success_rate(y_true, y_pred)

        self.assertEqual(asr, 0.9)  # High ASR = high privacy risk

    def test_high_rrs_indicates_privacy_risk(self):
        """Test that high RRS indicates privacy risk"""
        # Scenario: attack performs much better than random
        y_true = np.array([1] * 50 + [0] * 50)
        y_pred = np.array([1] * 45 + [0] * 5 + [0] * 45 + [1] * 5)  # 90% TPR, 10% FPR

        rrs = self.metrics.residual_risk_score(y_true, y_pred)

        self.assertEqual(rrs, 0.8)  # High RRS = high privacy risk


if __name__ == '__main__':
    unittest.main()
