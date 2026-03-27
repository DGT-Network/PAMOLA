"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for DistanceToClosestRecord

This module contains comprehensive unit tests for the DistanceToClosestRecord class.
Tests cover:
- Initialization
- DCR calculation with different methods (kdtree, cdist)
- DCR calculation with different distance metrics
- Validation of distance values
- Error handling for invalid parameters
- Edge cases (empty data, single records, identical records)
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from pamola_core.attacks.distance_to_closest_record import DistanceToClosestRecord
from pamola_core.errors.exceptions import ValidationError, InvalidParameterError


class TestDistanceToClosestRecordInitialization(unittest.TestCase):
    """Test DistanceToClosestRecord initialization"""

    def test_initialization(self):
        """Test DistanceToClosestRecord can be initialized"""
        attack = DistanceToClosestRecord()
        self.assertIsInstance(attack, DistanceToClosestRecord)


@pytest.fixture
def sample_dataframes():
    """Fixture providing sample datasets for DCR calculations"""
    data1 = pd.DataFrame({
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 2, 3, 4]
    })

    data2 = pd.DataFrame({
        'x': [0.5, 1.5, 2.5],
        'y': [0.5, 1.5, 2.5]
    })

    return data1, data2


class TestDCRCalculation(unittest.TestCase):
    """Test DCR calculation methods"""

    def setUp(self):
        self.attack = DistanceToClosestRecord()

    def test_dcr_kdtree_method(self):
        """Test DCR calculation using KDTree method"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [0.1, 1.1, 2.1],
            'y': [0.1, 1.1, 2.1]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='kdtree')

        self.assertIsInstance(dcr_values, np.ndarray)
        self.assertEqual(len(dcr_values), len(data2))
        self.assertTrue(all(d >= 0 for d in dcr_values))  # Distances should be non-negative

    def test_dcr_cdist_method(self):
        """Test DCR calculation using cdist method"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2]
        })
        data2 = pd.DataFrame({
            'x': [0.5, 1.5],
            'y': [0.5, 1.5]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='cdist')

        self.assertIsInstance(dcr_values, np.ndarray)
        self.assertEqual(len(dcr_values), len(data2))
        self.assertTrue(all(d >= 0 for d in dcr_values))

    def test_dcr_identical_records(self):
        """Test DCR with identical records in both datasets"""
        data1 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        data2 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='kdtree')

        # Identical records should have distance 0
        self.assertTrue(all(abs(d) < 1e-6 for d in dcr_values))

    def test_dcr_completely_different_records(self):
        """Test DCR with completely different records"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2]
        })
        data2 = pd.DataFrame({
            'x': [100, 101, 102],
            'y': [100, 101, 102]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='kdtree')

        # Distant records should have large distances
        self.assertTrue(all(d > 100 for d in dcr_values))

    def test_dcr_single_record_in_data2(self):
        """Test DCR with single record in second dataset"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [2],
            'y': [2]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='kdtree')

        self.assertEqual(len(dcr_values), 1)
        self.assertAlmostEqual(dcr_values[0], 0.0, places=5)

    def test_dcr_single_record_in_data1(self):
        """Test DCR with single record in first dataset (reference)"""
        data1 = pd.DataFrame({
            'x': [0],
            'y': [0]
        })
        data2 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [1, 2, 3]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='kdtree')

        self.assertEqual(len(dcr_values), 3)

    def test_dcr_euclidean_metric(self):
        """Test DCR with euclidean distance metric"""
        data1 = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })
        data2 = pd.DataFrame({
            'x': [3],
            'y': [4]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='cdist', metric='euclidean')

        # preprocess_data applies StandardScaler (fit on data1) before computing distances.
        # data1 scaled: [(-1,-1), (1,1)]; data2 scaled: [(5,7)]
        # dist (5,7)->(-1,-1) = sqrt(36+64)=10, dist (5,7)->(1,1) = sqrt(16+36)=sqrt(52)
        # Closest is sqrt(52) ≈ 7.211
        self.assertAlmostEqual(dcr_values[0], np.sqrt(52), places=3)

    def test_dcr_manhattan_metric(self):
        """Test DCR with manhattan distance metric"""
        data1 = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })
        data2 = pd.DataFrame({
            'x': [3],
            'y': [4]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2, method='cdist', metric='cityblock')

        # preprocess_data applies StandardScaler (fit on data1) before computing distances.
        # data1 scaled: [(-1,-1), (1,1)]; data2 scaled: [(5,7)]
        # Manhattan dist (5,7)->(-1,-1) = 6+8=14, dist (5,7)->(1,1) = 4+6=10
        # Closest is 10
        self.assertAlmostEqual(dcr_values[0], 10.0, places=5)

    def test_dcr_none_data1(self):
        """Test DCR with None as first dataset"""
        with self.assertRaises(ValidationError):
            self.attack.calculate_dcr(None, pd.DataFrame({'x': [1]}))

    def test_dcr_none_data2(self):
        """Test DCR with None as second dataset"""
        with self.assertRaises(ValidationError):
            self.attack.calculate_dcr(pd.DataFrame({'x': [1]}), None)

    def test_dcr_empty_data1(self):
        """Test DCR with empty first dataset"""
        result = self.attack.calculate_dcr(
            pd.DataFrame(),
            pd.DataFrame({'x': [1, 2]})
        )

        self.assertEqual(len(result), 0)

    def test_dcr_empty_data2(self):
        """Test DCR with empty second dataset"""
        result = self.attack.calculate_dcr(
            pd.DataFrame({'x': [1, 2]}),
            pd.DataFrame()
        )

        self.assertEqual(len(result), 0)

    def test_dcr_invalid_method(self):
        """Test DCR with invalid method parameter"""
        data1 = pd.DataFrame({'x': [1, 2]})
        data2 = pd.DataFrame({'x': [3, 4]})

        with self.assertRaises(InvalidParameterError):
            self.attack.calculate_dcr(data1, data2, method='invalid_method')


class TestDCRNumericalProperties(unittest.TestCase):
    """Test numerical properties of DCR calculations"""

    def setUp(self):
        self.attack = DistanceToClosestRecord()

    def test_dcr_non_negative(self):
        """Test that all DCR values are non-negative"""
        np.random.seed(42)
        data1 = pd.DataFrame(np.random.rand(10, 3))
        data2 = pd.DataFrame(np.random.rand(5, 3))

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertTrue(all(d >= 0 for d in dcr_values))

    def test_dcr_triangle_inequality(self):
        """Test that DCR satisfies triangle inequality (indirectly)"""
        data1 = pd.DataFrame({
            'x': [0, 10],
            'y': [0, 10]
        })
        data2a = pd.DataFrame({
            'x': [5],
            'y': [5]
        })
        data2b = pd.DataFrame({
            'x': [3],
            'y': [3]
        })

        dcr_a = self.attack.calculate_dcr(data1, data2a)[0]
        dcr_b = self.attack.calculate_dcr(data1, data2b)[0]

        # Verify that data2b is closer to data1 than data2a
        # (5,5) is at distance sqrt(50) ≈ 7.07 to nearest point
        # (3,3) is at distance sqrt(18) ≈ 4.24 to nearest point
        self.assertGreater(dcr_a, dcr_b)

    def test_dcr_method_consistency(self):
        """Test that kdtree and cdist methods give similar results"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0, 1, 2, 3]
        })
        data2 = pd.DataFrame({
            'x': [0.5, 1.5, 2.5],
            'y': [0.5, 1.5, 2.5]
        })

        dcr_kdtree = self.attack.calculate_dcr(data1, data2, method='kdtree')
        dcr_cdist = self.attack.calculate_dcr(data1, data2, method='cdist', metric='euclidean')

        # Results should be very similar
        np.testing.assert_array_almost_equal(dcr_kdtree, dcr_cdist, decimal=5)

    def test_dcr_scales_with_distance(self):
        """Test that DCR scales appropriately with point separation"""
        data1 = pd.DataFrame({
            'x': [0],
            'y': [0]
        })

        # Test point at distance 1
        data2_near = pd.DataFrame({
            'x': [1],
            'y': [0]
        })
        dcr_near = self.attack.calculate_dcr(data1, data2_near)[0]

        # Test point at distance 10
        data2_far = pd.DataFrame({
            'x': [10],
            'y': [0]
        })
        dcr_far = self.attack.calculate_dcr(data1, data2_far)[0]

        self.assertLess(dcr_near, dcr_far)
        self.assertAlmostEqual(dcr_near, 1.0)
        self.assertAlmostEqual(dcr_far, 10.0)


class TestDCRDataTypes(unittest.TestCase):
    """Test DCR with different data types"""

    def setUp(self):
        self.attack = DistanceToClosestRecord()

    def test_dcr_integer_data(self):
        """Test DCR with integer data"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2]
        })
        data2 = pd.DataFrame({
            'x': [1, 2],
            'y': [1, 2]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)

    def test_dcr_float_data(self):
        """Test DCR with float data"""
        data1 = pd.DataFrame({
            'x': [0.5, 1.5, 2.5],
            'y': [0.5, 1.5, 2.5]
        })
        data2 = pd.DataFrame({
            'x': [1.0, 2.0],
            'y': [1.0, 2.0]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)

    def test_dcr_mixed_numeric_types(self):
        """Test DCR with mixed int and float columns"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0.5, 1.5, 2.5]
        })
        data2 = pd.DataFrame({
            'x': [1, 2],
            'y': [1.5, 2.5]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)

    def test_dcr_categorical_data_vectorized(self):
        """Test DCR with categorical data (converted to numeric by preprocessor)"""
        data1 = pd.DataFrame({
            'color': ['red', 'blue', 'green'],
            'size': ['small', 'medium', 'large']
        })
        data2 = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['small', 'medium']
        })

        # Preprocessor should convert categorical to numeric
        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)


class TestDCREdgeCases(unittest.TestCase):
    """Test edge cases for DCR calculation"""

    def setUp(self):
        self.attack = DistanceToClosestRecord()

    def test_dcr_many_records_one_feature(self):
        """Test DCR with many records and single feature"""
        data1 = pd.DataFrame({'x': np.arange(100)})
        data2 = pd.DataFrame({'x': [50.5, 25.3]})

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)

    def test_dcr_few_records_many_features(self):
        """Test DCR with few records and many features"""
        data1 = pd.DataFrame(np.random.rand(3, 50))
        data2 = pd.DataFrame(np.random.rand(2, 50))

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)

    def test_dcr_with_zeros(self):
        """Test DCR with datasets containing zeros"""
        data1 = pd.DataFrame({
            'x': [0, 0, 1],
            'y': [0, 0, 1]
        })
        data2 = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2)

        # First point matches exactly
        self.assertAlmostEqual(dcr_values[0], 0.0)

    def test_dcr_with_negative_values(self):
        """Test DCR with negative values"""
        data1 = pd.DataFrame({
            'x': [-1, 0, 1],
            'y': [-1, 0, 1]
        })
        data2 = pd.DataFrame({
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2)

        self.assertEqual(len(dcr_values), 2)

    def test_dcr_with_very_small_distances(self):
        """Test DCR with very small distances (numerical precision)"""
        data1 = pd.DataFrame({
            'x': [1e-10, 2e-10],
            'y': [1e-10, 2e-10]
        })
        data2 = pd.DataFrame({
            'x': [1.1e-10],
            'y': [1.1e-10]
        })

        dcr_values = self.attack.calculate_dcr(data1, data2)

        # Should handle very small numbers properly
        self.assertGreater(len(dcr_values), 0)


if __name__ == '__main__':
    unittest.main()
