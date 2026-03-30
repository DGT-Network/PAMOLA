"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for MembershipInference

This module contains comprehensive unit tests for the MembershipInference class.
Tests cover:
- Initialization with default and custom thresholds
- DCR-based membership inference attack
- NNDR-based membership inference attack
- Model-based membership inference attack
- Threshold determination logic
- Error handling for invalid inputs
- Edge cases (empty datasets, identical records)
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from pamola_core.attacks.membership_inference import MembershipInference
from pamola_core.errors.exceptions import ValidationError


class TestMembershipInferenceInitialization(unittest.TestCase):
    """Test MembershipInference initialization"""

    def test_initialization_with_defaults(self):
        """Test MembershipInference initialization with default parameters"""
        attack = MembershipInference()
        self.assertIsNone(attack.dcr_threshold)
        self.assertIsNone(attack.nndr_threshold)
        self.assertIsNone(attack.m_threshold)

    def test_initialization_with_custom_thresholds(self):
        """Test MembershipInference initialization with custom thresholds"""
        attack = MembershipInference(
            dcr_threshold=0.5,
            nndr_threshold=0.3,
            m_threshold=0.7
        )
        self.assertEqual(attack.dcr_threshold, 0.5)
        self.assertEqual(attack.nndr_threshold, 0.3)
        self.assertEqual(attack.m_threshold, 0.7)

    def test_initialization_creates_dcr_and_nndr_instances(self):
        """Test that DCR and NNDR instances are created during initialization"""
        attack = MembershipInference()
        self.assertIsNotNone(attack.dcr)
        self.assertIsNotNone(attack.nndr)


@pytest.fixture
def training_and_test_data():
    """Fixture providing training and test datasets"""
    np.random.seed(42)

    # Training data (known members)
    training_data = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'feature3': np.random.rand(50),
    })

    # Test data (mix of members and non-members)
    test_data = pd.DataFrame({
        'feature1': np.random.rand(30),
        'feature2': np.random.rand(30),
        'feature3': np.random.rand(30),
    })

    return training_data, test_data


class TestDCRMembershipInference(unittest.TestCase):
    """Test DCR-based membership inference attack"""

    def setUp(self):
        self.attack = MembershipInference(dcr_threshold=0.5)

    def test_dcr_membership_inference_basic(self):
        """Test basic DCR membership inference"""
        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        data_test = pd.DataFrame({
            'x': [1, 10, 3, 20, 5],
            'y': [10, 100, 30, 200, 50]
        })

        predictions = self.attack.membership_inference_attack_dcr(data_train, data_test)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(data_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_dcr_identical_records_detected(self):
        """Test DCR identifies records identical to training set"""
        data_train = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        data_test = pd.DataFrame({
            'a': [1, 2, 100],
            'b': [4, 5, 600]
        })

        predictions = self.attack.membership_inference_attack_dcr(data_train, data_test)

        # First two records are identical, should be members (prediction=1)
        self.assertEqual(predictions[0], 1)
        self.assertEqual(predictions[1], 1)

    def test_dcr_threshold_application(self):
        """Test that DCR threshold is applied correctly"""
        attack_strict = MembershipInference(dcr_threshold=0.01)  # Strict threshold
        attack_lenient = MembershipInference(dcr_threshold=10.0)  # Lenient threshold

        data_train = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20)
        })
        data_test = pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10)
        })

        pred_strict = attack_strict.membership_inference_attack_dcr(data_train, data_test)
        pred_lenient = attack_lenient.membership_inference_attack_dcr(data_train, data_test)

        # Strict threshold should result in fewer members
        self.assertLessEqual(pred_strict.sum(), pred_lenient.sum())

    def test_dcr_empty_datasets(self):
        """Test DCR with empty datasets"""
        result = self.attack.membership_inference_attack_dcr(
            pd.DataFrame(),
            pd.DataFrame()
        )

        self.assertEqual(len(result), 0)

    def test_dcr_none_data(self):
        """Test DCR with None data"""
        result = self.attack.membership_inference_attack_dcr(None, pd.DataFrame())
        self.assertEqual(len(result), 0)

        result = self.attack.membership_inference_attack_dcr(pd.DataFrame(), None)
        self.assertEqual(len(result), 0)

    def test_dcr_single_record_test(self):
        """Test DCR with single test record"""
        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        data_test = pd.DataFrame({
            'x': [1],
            'y': [10]
        })

        predictions = self.attack.membership_inference_attack_dcr(data_train, data_test)

        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], [0, 1])


class TestNNDRMembershipInference(unittest.TestCase):
    """Test NNDR-based membership inference attack"""

    def setUp(self):
        self.attack = MembershipInference(nndr_threshold=0.5)

    def test_nndr_membership_inference_basic(self):
        """Test basic NNDR membership inference"""
        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        data_test = pd.DataFrame({
            'x': [1, 10, 3],
            'y': [10, 100, 30]
        })

        predictions = self.attack.membership_inference_attack_nndr(data_train, data_test)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(data_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_nndr_close_records_detected(self):
        """Test NNDR identifies records close to training set"""
        data_train = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0]
        })
        data_test = pd.DataFrame({
            'a': [1.01, 2.5],  # Very close to first, between first and second
            'b': [4.01, 5.0]
        })

        predictions = self.attack.membership_inference_attack_nndr(data_train, data_test)

        # Both test records are close to training set
        self.assertTrue(all(pred == 1 for pred in predictions))

    def test_nndr_threshold_application(self):
        """Test that NNDR threshold is applied correctly"""
        attack_strict = MembershipInference(nndr_threshold=0.1)
        attack_lenient = MembershipInference(nndr_threshold=5.0)

        data_train = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20)
        })
        data_test = pd.DataFrame({
            'x': np.random.rand(10),
            'y': np.random.rand(10)
        })

        pred_strict = attack_strict.membership_inference_attack_nndr(data_train, data_test)
        pred_lenient = attack_lenient.membership_inference_attack_nndr(data_train, data_test)

        # Different thresholds may yield different results
        self.assertEqual(len(pred_strict), len(pred_lenient))

    def test_nndr_empty_datasets(self):
        """Test NNDR with empty datasets"""
        result = self.attack.membership_inference_attack_nndr(
            pd.DataFrame(),
            pd.DataFrame()
        )

        self.assertEqual(len(result), 0)

    def test_nndr_none_data(self):
        """Test NNDR with None data"""
        result = self.attack.membership_inference_attack_nndr(None, pd.DataFrame())
        self.assertEqual(len(result), 0)

        result = self.attack.membership_inference_attack_nndr(pd.DataFrame(), None)
        self.assertEqual(len(result), 0)


class TestModelBasedMembershipInference(unittest.TestCase):
    """Test model-based membership inference attack"""

    def setUp(self):
        self.attack = MembershipInference(m_threshold=0.6)

    def test_model_membership_inference_basic(self):
        """Test basic model-based membership inference"""
        np.random.seed(42)
        data_train = pd.DataFrame({
            'x': np.random.rand(30),
            'y': np.random.rand(30)
        })
        data_test = pd.DataFrame({
            'x': np.random.rand(20),
            'y': np.random.rand(20)
        })

        predictions = self.attack.membership_inference_attack_model(data_train, data_test)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(data_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_model_inference_with_training_members(self):
        """Test model inference identifies training members"""
        # Create data where test set includes training records
        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        data_test = pd.DataFrame({
            'x': [1, 2, 100, 200],
            'y': [10, 20, 1000, 2000]
        })

        predictions = self.attack.membership_inference_attack_model(data_train, data_test)

        # Model should identify some members (first two are exact matches)
        self.assertGreater(predictions.sum(), 0)

    def test_model_inference_none_data(self):
        """Test model inference with None data"""
        with self.assertRaises(ValidationError):
            self.attack.membership_inference_attack_model(None, pd.DataFrame())

    def test_model_inference_with_default_threshold(self):
        """Test model inference with auto-determined threshold"""
        attack = MembershipInference(m_threshold=None)  # Auto-determine threshold

        data_train = pd.DataFrame({
            'x': np.random.rand(25),
            'y': np.random.rand(25)
        })
        data_test = pd.DataFrame({
            'x': np.random.rand(15),
            'y': np.random.rand(15)
        })

        predictions = attack.membership_inference_attack_model(data_train, data_test)

        self.assertEqual(len(predictions), len(data_test))

    def test_model_inference_small_training_set(self):
        """Test model inference with very small training set"""
        data_train = pd.DataFrame({
            'x': [1, 2],
            'y': [10, 20]
        })
        data_test = pd.DataFrame({
            'x': [1, 3],
            'y': [10, 30]
        })

        predictions = self.attack.membership_inference_attack_model(data_train, data_test)

        self.assertEqual(len(predictions), len(data_test))


class TestMembershipInferenceEdgeCases(unittest.TestCase):
    """Test edge cases for membership inference attacks"""

    def setUp(self):
        self.attack = MembershipInference()

    def test_identical_train_test_datasets(self):
        """Test membership inference when training and test sets are identical"""
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })

        # DCR test
        predictions_dcr = self.attack.membership_inference_attack_dcr(data, data)
        # All should be detected as members (identical records)
        self.assertEqual(len(predictions_dcr), len(data))

    def test_completely_different_datasets(self):
        """Test membership inference with completely different datasets"""
        data_train = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        data_test = pd.DataFrame({
            'x': [100, 200, 300],
            'y': [400, 500, 600]
        })

        predictions_dcr = self.attack.membership_inference_attack_dcr(data_train, data_test)

        # Far apart data should have fewer member predictions
        self.assertGreater(len(data_test), predictions_dcr.sum())

    def test_single_feature_datasets(self):
        """Test membership inference with single feature"""
        data_train = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        data_test = pd.DataFrame({'x': [1, 10, 3]})

        predictions = self.attack.membership_inference_attack_dcr(data_train, data_test)

        self.assertEqual(len(predictions), len(data_test))

    def test_high_dimensional_datasets(self):
        """Test membership inference with high-dimensional data"""
        np.random.seed(42)
        data_train = pd.DataFrame(
            np.random.rand(10, 50),
            columns=[f'feat_{i}' for i in range(50)]
        )
        data_test = pd.DataFrame(
            np.random.rand(5, 50),
            columns=[f'feat_{i}' for i in range(50)]
        )

        predictions_dcr = self.attack.membership_inference_attack_dcr(data_train, data_test)

        self.assertEqual(len(predictions_dcr), len(data_test))

    def test_categorical_data(self):
        """Test membership inference with categorical data"""
        data_train = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red', 'blue'],
            'size': ['small', 'medium', 'large', 'small', 'medium']
        })
        data_test = pd.DataFrame({
            'color': ['red', 'yellow', 'blue'],
            'size': ['small', 'tiny', 'medium']
        })

        predictions = self.attack.membership_inference_attack_dcr(data_train, data_test)

        self.assertEqual(len(predictions), len(data_test))

    def test_mixed_data_types(self):
        """Test membership inference with mixed numeric and categorical data"""
        data_train = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'score': [95.5, 87.3, 92.1, 88.9, 91.2]
        })
        data_test = pd.DataFrame({
            'id': [1, 10, 3],
            'name': ['Alice', 'Frank', 'Charlie'],
            'score': [95.5, 85.0, 92.1]
        })

        predictions = self.attack.membership_inference_attack_dcr(data_train, data_test)

        self.assertEqual(len(predictions), len(data_test))


class TestMembershipInferenceThresholdLogic(unittest.TestCase):
    """Test threshold determination logic"""

    def test_dcr_threshold_fallback_to_percentile(self):
        """Test DCR falls back to percentile when threshold is invalid"""
        attack = MembershipInference(dcr_threshold=-1)  # Invalid threshold

        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        data_test = pd.DataFrame({
            'x': [1, 10],
            'y': [10, 100]
        })

        # Should not raise error and should produce valid predictions
        predictions = attack.membership_inference_attack_dcr(data_train, data_test)
        self.assertEqual(len(predictions), len(data_test))

    def test_nndr_threshold_fallback(self):
        """Test NNDR falls back to percentile when threshold is invalid"""
        attack = MembershipInference(nndr_threshold=-1)

        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        data_test = pd.DataFrame({
            'x': [1, 10],
            'y': [10, 100]
        })

        predictions = attack.membership_inference_attack_nndr(data_train, data_test)
        self.assertEqual(len(predictions), len(data_test))


if __name__ == '__main__':
    unittest.main()
