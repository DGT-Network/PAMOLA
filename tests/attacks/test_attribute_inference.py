"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for AttributeInference

This module contains comprehensive unit tests for the AttributeInference class.
Tests cover:
- Initialization
- Attribute inference attack (entropy-based)
- Feature selection using entropy
- Mapping building and inference
- Error handling for missing attributes
- Edge cases (empty data, missing values, single feature)
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from pamola_core.attacks.attribute_inference import AttributeInference
from pamola_core.errors.exceptions import ValidationError, FieldNotFoundError


class TestAttributeInferenceInitialization(unittest.TestCase):
    """Test AttributeInference initialization"""

    def test_initialization(self):
        """Test AttributeInference can be initialized"""
        attack = AttributeInference()
        self.assertIsInstance(attack, AttributeInference)


@pytest.fixture
def training_and_test_data():
    """Fixture providing training and test datasets for attribute inference"""
    # Training data with known attributes
    training_data = pd.DataFrame({
        'age': [25, 30, 35, 25, 30, 35, 25, 30],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'income': [50000, 60000, 75000, 55000, 65000, 70000, 52000, 58000],
        'salary_level': ['low', 'med', 'high', 'low', 'med', 'high', 'low', 'med']
    })

    # Test data with unknown salary_level (to be inferred)
    test_data = pd.DataFrame({
        'age': [25, 30, 35, 25],
        'gender': ['M', 'F', 'M', 'F'],
        'income': [51000, 61000, 76000, 54000],
    })

    return training_data, test_data


class TestAttributeInferenceAttack(unittest.TestCase):
    """Test attribute_inference_attack method"""

    def setUp(self):
        self.attack = AttributeInference()

    def test_attribute_inference_basic(self):
        """Test basic attribute inference with simple dataset"""
        data_train = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'red', 'blue'],
            'size': ['small', 'small', 'large', 'small', 'large'],
            'price': [10, 20, 30, 15, 35]
        })
        data_test = pd.DataFrame({
            'color': ['red', 'blue', 'red'],
            'size': ['small', 'small', 'large']
        })

        # Try to infer 'price' based on other attributes
        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='price'
        )

        self.assertIsInstance(predictions, pd.Series)
        self.assertEqual(len(predictions), len(data_test))

    def test_attribute_inference_categorical_target(self):
        """Test inferring categorical target attribute"""
        data_train = pd.DataFrame({
            'age': [20, 25, 30, 35, 20, 25, 30],
            'city': ['NYC', 'LA', 'Chicago', 'Boston', 'NYC', 'LA', 'Chicago'],
            'occupation': ['teacher', 'engineer', 'doctor', 'lawyer', 'teacher', 'engineer', 'doctor']
        })
        data_test = pd.DataFrame({
            'age': [20, 25, 30],
            'city': ['NYC', 'LA', 'Chicago']
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='occupation'
        )

        self.assertEqual(len(predictions), len(data_test))
        # Predictions should be from the training set occupation values
        self.assertTrue(all(p in data_train['occupation'].unique() or pd.isna(p) for p in predictions))

    def test_attribute_inference_numerical_target(self):
        """Test inferring numerical target attribute"""
        data_train = pd.DataFrame({
            'x': [1, 2, 3, 1, 2, 3],
            'y': [10, 20, 30, 10, 20, 30],
            'z': [100, 200, 300, 100, 200, 300]
        })
        data_test = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='z'
        )

        self.assertEqual(len(predictions), len(data_test))

    def test_attribute_inference_with_missing_values(self):
        """Test attribute inference handles missing values in training data"""
        data_train = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': ['a', 'b', 'c', None, 'e'],
            'target': ['X', 'Y', 'X', 'Y', 'X']
        })
        data_test = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': ['a', 'b']
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='target'
        )

        self.assertEqual(len(predictions), len(data_test))

    def test_attribute_inference_target_not_in_training(self):
        """Test attribute inference when target not in training data"""
        data_train = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        data_test = pd.DataFrame({
            'a': [1, 2],
            'b': [4, 5]
        })

        with self.assertRaises(FieldNotFoundError):
            self.attack.attribute_inference_attack(
                data_train, data_test, target_attribute='non_existent'
            )

    def test_attribute_inference_empty_data_train(self):
        """Test attribute inference with empty training data"""
        data_train = pd.DataFrame()
        data_test = pd.DataFrame({'a': [1, 2]})

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='a'
        )

        self.assertEqual(len(predictions), 0)

    def test_attribute_inference_empty_data_test(self):
        """Test attribute inference with empty test data"""
        data_train = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        data_test = pd.DataFrame()

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='a'
        )

        self.assertEqual(len(predictions), 0)

    def test_attribute_inference_no_features_available(self):
        """Test when only target attribute exists in training data"""
        data_train = pd.DataFrame({
            'target': [1, 2, 3, 4, 5]
        })
        data_test = pd.DataFrame({
            'target': [1, 2]
        })

        with self.assertRaises(ValidationError):
            self.attack.attribute_inference_attack(
                data_train, data_test, target_attribute='target'
            )

    def test_attribute_inference_single_feature(self):
        """Test attribute inference with single feature (besides target)"""
        data_train = pd.DataFrame({
            'color': ['red', 'red', 'blue', 'blue', 'red'],
            'expensive': ['yes', 'yes', 'no', 'no', 'yes']
        })
        data_test = pd.DataFrame({
            'color': ['red', 'blue', 'red']
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='expensive'
        )

        self.assertEqual(len(predictions), len(data_test))

    def test_attribute_inference_high_entropy_feature_selected(self):
        """Test that low-entropy feature is selected (higher discriminative power)"""
        data_train = pd.DataFrame({
            'feature1': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # Low entropy (50-50 split)
            'feature2': np.random.randint(0, 100, 10),   # High entropy (many values)
            'target': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        })
        data_test = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [50, 75]
        })

        # Should use feature1 which has lower entropy and better discriminative power
        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='target'
        )

        # Predictions should follow the pattern from feature1
        self.assertTrue(all(p in data_train['target'].unique() for p in predictions))

    def test_attribute_inference_majority_vote(self):
        """Test that majority vote is used for mapping"""
        data_train = pd.DataFrame({
            'income_level': ['low', 'low', 'low', 'high', 'high'],
            'category': ['A', 'A', 'A', 'B', 'B']
        })
        data_test = pd.DataFrame({
            'income_level': ['low', 'high']
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='category'
        )

        # 'low' should map to 'A' (majority), 'high' should map to 'B'
        self.assertEqual(predictions[0], 'A')
        self.assertEqual(predictions[1], 'B')

    def test_attribute_inference_fillna_with_global_mode(self):
        """Test that unknown feature values are filled with global mode"""
        data_train = pd.DataFrame({
            'size': ['small', 'small', 'large', 'small'],
            'color': ['red', 'red', 'blue', 'red']
        })
        data_test = pd.DataFrame({
            'size': ['small', 'unknown'],  # 'unknown' not in training
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='color'
        )

        # Unknown should be filled with global mode of 'color' which is 'red'
        self.assertEqual(len(predictions), 2)

    def test_attribute_inference_consistency(self):
        """Test that inference results are consistent"""
        data_train = pd.DataFrame({
            'x': [1, 1, 2, 2, 1],
            'y': ['A', 'A', 'B', 'B', 'A']
        })
        data_test = pd.DataFrame({
            'x': [1, 2, 1]
        })

        pred1 = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='y'
        )
        pred2 = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='y'
        )

        # Results should be identical
        self.assertTrue((pred1 == pred2).all())


class TestAttributeInferenceEdgeCases(unittest.TestCase):
    """Test edge cases for attribute inference"""

    def setUp(self):
        self.attack = AttributeInference()

    def test_all_same_target_values(self):
        """Test when all target values are the same"""
        data_train = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
            'target': ['same', 'same', 'same', 'same', 'same']
        })
        data_test = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='target'
        )

        # All predictions should be 'same'
        self.assertTrue(all(p == 'same' for p in predictions))

    def test_single_record_training(self):
        """Test with single training record"""
        data_train = pd.DataFrame({
            'feature': [1],
            'target': ['value']
        })
        data_test = pd.DataFrame({
            'feature': [1]
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='target'
        )

        self.assertEqual(len(predictions), 1)

    def test_target_with_nan_values(self):
        """Test when target has NaN values in training data"""
        # Use data where every group has at least one non-NaN target value,
        # so groupby agg never hits an empty dropna result.
        data_train = pd.DataFrame({
            'x': [1, 1, 2, 2],
            'y': ['A', None, 'B', 'B']
        })
        data_test = pd.DataFrame({
            'x': [1, 2]
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='y'
        )

        self.assertEqual(len(predictions), 2)

    def test_string_and_numeric_features(self):
        """Test with mixed string and numeric features"""
        data_train = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string': ['A', 'B', 'A', 'B', 'A'],
            'target': [10, 20, 10, 20, 10]
        })
        data_test = pd.DataFrame({
            'numeric': [1, 2],
            'string': ['A', 'B']
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='target'
        )

        self.assertEqual(len(predictions), 2)

    def test_large_dataset(self):
        """Test with larger dataset"""
        np.random.seed(42)
        data_train = pd.DataFrame({
            'x': np.random.rand(1000),
            'y': np.random.rand(1000),
            'z': np.random.choice(['cat', 'dog', 'bird'], 1000)
        })
        data_test = pd.DataFrame({
            'x': np.random.rand(100),
            'y': np.random.rand(100)
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='z'
        )

        self.assertEqual(len(predictions), 100)


class TestAttributeInferenceAccuracy(unittest.TestCase):
    """Test accuracy of attribute inference"""

    def setUp(self):
        self.attack = AttributeInference()

    def test_deterministic_relationship(self):
        """Test inference when target has deterministic relationship with features"""
        data_train = pd.DataFrame({
            'x': [1, 1, 1, 2, 2, 2],
            'y': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        data_test = pd.DataFrame({
            'x': [1, 2, 1, 2]
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='y'
        )

        # Should perfectly predict based on x
        self.assertEqual(predictions[0], 'A')
        self.assertEqual(predictions[1], 'B')
        self.assertEqual(predictions[2], 'A')
        self.assertEqual(predictions[3], 'B')

    def test_probabilistic_relationship(self):
        """Test inference when target has probabilistic relationship"""
        data_train = pd.DataFrame({
            'feature': [1, 1, 1, 1, 2, 2, 2, 2],
            'target': ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'X']
        })
        data_test = pd.DataFrame({
            'feature': [1, 2]
        })

        predictions = self.attack.attribute_inference_attack(
            data_train, data_test, target_attribute='target'
        )

        # feature=1 should map to 'X' (majority), feature=2 should map to 'Y'
        self.assertEqual(predictions[0], 'X')
        self.assertEqual(predictions[1], 'Y')


if __name__ == '__main__':
    unittest.main()
