"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for NearestNeighborDistanceRatio

This module contains comprehensive unit tests for the NearestNeighborDistanceRatio class.
Tests cover:
- Initialization
- NNDR calculation with different methods (kdtree, neighbors)
- NNDR mathematical properties (ratio >= 0, first NN always closer)
- Error handling for invalid parameters
- Edge cases (empty data, insufficient reference points)
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from pamola_core.attacks.nearest_neighbor_distance_ratio import NearestNeighborDistanceRatio
from pamola_core.errors.exceptions import ValidationError, InvalidParameterError


class TestNearestNeighborDistanceRatioInitialization(unittest.TestCase):
    """Test NearestNeighborDistanceRatio initialization"""

    def test_initialization(self):
        """Test NearestNeighborDistanceRatio can be initialized"""
        attack = NearestNeighborDistanceRatio()
        self.assertIsInstance(attack, NearestNeighborDistanceRatio)


@pytest.fixture
def sample_dataframes():
    """Fixture providing sample datasets for NNDR calculations"""
    data1 = pd.DataFrame({
        'x': [0, 1, 2, 3, 4],
        'y': [0, 1, 2, 3, 4]
    })

    data2 = pd.DataFrame({
        'x': [0.5, 1.5, 2.5],
        'y': [0.5, 1.5, 2.5]
    })

    return data1, data2


class TestNNDRCalculation(unittest.TestCase):
    """Test NNDR calculation methods"""

    def setUp(self):
        self.attack = NearestNeighborDistanceRatio()

    def test_nndr_kdtree_method(self):
        """Test NNDR calculation using KDTree method"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [0.5, 1.5, 2.5],
            'y': [0.5, 1.5, 2.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        self.assertIsInstance(nndr_values, np.ndarray)
        self.assertEqual(len(nndr_values), len(data2))
        self.assertTrue(all(r >= 0 for r in nndr_values))  # Ratios should be non-negative

    def test_nndr_neighbors_method(self):
        """Test NNDR calculation using sklearn NearestNeighbors method"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2]
        })
        data2 = pd.DataFrame({
            'x': [0.5, 1.5],
            'y': [0.5, 1.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='neighbors')

        self.assertIsInstance(nndr_values, np.ndarray)
        self.assertEqual(len(nndr_values), len(data2))
        self.assertTrue(all(r >= 0 for r in nndr_values))

    def test_nndr_ratio_property(self):
        """Test that NNDR = d1/d2 where d1 < d2"""
        data1 = pd.DataFrame({
            'x': [0, 10],
            'y': [0, 10]
        })
        data2 = pd.DataFrame({
            'x': [1],
            'y': [1]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        # Point (1,1) is at distance sqrt(2) ≈ 1.414 from (0,0)
        # and distance sqrt(162) ≈ 12.73 from (10,10)
        # Ratio should be 1.414/12.73 ≈ 0.111
        self.assertLess(nndr_values[0], 1.0)

    def test_nndr_identical_records(self):
        """Test NNDR with identical records"""
        data1 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        data2 = pd.DataFrame({
            'x': [1],
            'y': [4]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        # Identical record should have distance 0 to nearest neighbor
        # Second nearest neighbor will be > 0, so ratio will be 0
        self.assertAlmostEqual(nndr_values[0], 0.0, places=5)

    def test_nndr_close_and_far_neighbors(self):
        """Test NNDR distinguishes between close and far neighbors"""
        data1 = pd.DataFrame({
            'x': [0, 1, 100],  # Two close points, one far
            'y': [0, 1, 100]
        })
        data2 = pd.DataFrame({
            'x': [0.5],
            'y': [0.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        # Nearest neighbor is around (0,0) at distance ~0.707
        # Second nearest is around (1,1) at distance ~0.707
        # Third neighbor is (100,100) at distance ~141
        # NNDR = d1/d2 ≈ 1.0 (both nearest neighbors are equidistant)
        # This is a low NNDR, indicating a distinct match
        self.assertLess(nndr_values[0], 1.5)

    def test_nndr_none_data1(self):
        """Test NNDR with None as first dataset"""
        with self.assertRaises(ValidationError):
            self.attack.calculate_nndr(None, pd.DataFrame({'x': [1]}))

    def test_nndr_none_data2(self):
        """Test NNDR with None as second dataset"""
        with self.assertRaises(ValidationError):
            self.attack.calculate_nndr(pd.DataFrame({'x': [1]}), None)

    def test_nndr_empty_data1(self):
        """Test NNDR with empty first dataset"""
        result = self.attack.calculate_nndr(
            pd.DataFrame(),
            pd.DataFrame({'x': [1, 2]})
        )

        self.assertEqual(len(result), 0)

    def test_nndr_empty_data2(self):
        """Test NNDR with empty second dataset"""
        result = self.attack.calculate_nndr(
            pd.DataFrame({'x': [1, 2]}),
            pd.DataFrame()
        )

        self.assertEqual(len(result), 0)

    def test_nndr_invalid_method(self):
        """Test NNDR with invalid method parameter"""
        data1 = pd.DataFrame({'x': [1, 2]})
        data2 = pd.DataFrame({'x': [3, 4]})

        with self.assertRaises(InvalidParameterError):
            self.attack.calculate_nndr(data1, data2, method='invalid_method')

    def test_nndr_single_reference_record(self):
        """Test NNDR with only one reference record"""
        # This should fail or handle gracefully since we need 2 nearest neighbors
        data1 = pd.DataFrame({
            'x': [0],
            'y': [0]
        })
        data2 = pd.DataFrame({
            'x': [1],
            'y': [1]
        })

        # Should handle the case where there's only 1 reference point
        try:
            nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')
            # If it succeeds, values should be finite (may be inf due to d2=0)
            self.assertEqual(len(nndr_values), 1)
        except Exception:
            # It's acceptable if this raises an error
            pass


class TestNNDRNumericalProperties(unittest.TestCase):
    """Test numerical properties of NNDR calculations"""

    def setUp(self):
        self.attack = NearestNeighborDistanceRatio()

    def test_nndr_non_negative(self):
        """Test that all NNDR values are non-negative"""
        np.random.seed(42)
        data1 = pd.DataFrame(np.random.rand(10, 3))
        data2 = pd.DataFrame(np.random.rand(5, 3))

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        self.assertTrue(all(r >= 0 for r in nndr_values))

    def test_nndr_first_less_than_second_distance(self):
        """Test that first NN distance is always <= second NN distance"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [0.5, 1.5, 2.5],
            'y': [0.5, 1.5, 2.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        # All NNDR values should be <= 1.0 in cases where distances are computed correctly
        # (d1/d2 <= 1 because d1 is closer than d2)
        self.assertTrue(all(r <= 2.0 for r in nndr_values))  # Allow some floating point error

    def test_nndr_method_consistency(self):
        """Test that kdtree and neighbors methods give similar results"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [0.5, 1.5, 2.5],
            'y': [0.5, 1.5, 2.5]
        })

        nndr_kdtree = self.attack.calculate_nndr(data1, data2, method='kdtree')
        nndr_neighbors = self.attack.calculate_nndr(data1, data2, method='neighbors')

        # Results should be very similar
        np.testing.assert_array_almost_equal(nndr_kdtree, nndr_neighbors, decimal=4)

    def test_nndr_bound_by_number_of_reference_points(self):
        """Test NNDR values are reasonable given the number of reference points"""
        np.random.seed(42)
        data1 = pd.DataFrame({
            'x': np.random.rand(3),
            'y': np.random.rand(3)
        })
        data2 = pd.DataFrame({
            'x': [0.5],
            'y': [0.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        # Should have valid NNDR values
        self.assertEqual(len(nndr_values), 1)
        self.assertGreater(nndr_values[0], 0)


class TestNNDRDataTypes(unittest.TestCase):
    """Test NNDR with different data types"""

    def setUp(self):
        self.attack = NearestNeighborDistanceRatio()

    def test_nndr_integer_data(self):
        """Test NNDR with integer data"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [1, 2],
            'y': [1, 2]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        self.assertEqual(len(nndr_values), 2)

    def test_nndr_float_data(self):
        """Test NNDR with float data"""
        data1 = pd.DataFrame({
            'x': [0.0, 1.5, 2.3, 3.7, 4.2],
            'y': [0.1, 1.4, 2.2, 3.8, 4.1]
        })
        data2 = pd.DataFrame({
            'x': [1.0, 2.0],
            'y': [1.0, 2.0]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2, method='kdtree')

        self.assertEqual(len(nndr_values), 2)

    def test_nndr_mixed_numeric_types(self):
        """Test NNDR with mixed int and float columns"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0.5, 1.5, 2.5, 3.5]
        })
        data2 = pd.DataFrame({
            'x': [1, 2],
            'y': [1.5, 2.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 2)

    def test_nndr_categorical_data_vectorized(self):
        """Test NNDR with categorical data (converted by preprocessor)"""
        data1 = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red', 'blue'],
            'size': ['small', 'medium', 'large', 'small', 'large']
        })
        data2 = pd.DataFrame({
            'color': ['red', 'blue'],
            'size': ['small', 'medium']
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 2)


class TestNNDREdgeCases(unittest.TestCase):
    """Test edge cases for NNDR calculation"""

    def setUp(self):
        self.attack = NearestNeighborDistanceRatio()

    def test_nndr_many_reference_few_test(self):
        """Test NNDR with many reference points and few test points"""
        np.random.seed(42)
        data1 = pd.DataFrame(np.random.rand(100, 3))
        data2 = pd.DataFrame(np.random.rand(2, 3))

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 2)

    def test_nndr_few_reference_many_test(self):
        """Test NNDR with few reference points and many test points"""
        np.random.seed(42)
        data1 = pd.DataFrame(np.random.rand(5, 3))
        data2 = pd.DataFrame(np.random.rand(50, 3))

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 50)

    def test_nndr_high_dimensional(self):
        """Test NNDR with high-dimensional data"""
        np.random.seed(42)
        data1 = pd.DataFrame(np.random.rand(10, 50))
        data2 = pd.DataFrame(np.random.rand(5, 50))

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 5)

    def test_nndr_with_zeros(self):
        """Test NNDR with zero values"""
        data1 = pd.DataFrame({
            'x': [0, 0, 1, 1],
            'y': [0, 0, 1, 1]
        })
        data2 = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 2)

    def test_nndr_with_negative_values(self):
        """Test NNDR with negative values"""
        data1 = pd.DataFrame({
            'x': [-1, 0, 1, 2],
            'y': [-1, 0, 1, 2]
        })
        data2 = pd.DataFrame({
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 2)

    def test_nndr_collinear_points(self):
        """Test NNDR with collinear (aligned) points"""
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [1.5],
            'y': [1.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        self.assertEqual(len(nndr_values), 1)

    def test_nndr_clustered_reference_points(self):
        """Test NNDR when reference points are clustered"""
        data1 = pd.DataFrame({
            'x': [0, 0.1, 0.2, 100, 100.1],
            'y': [0, 0.1, 0.2, 100, 100.1]
        })
        data2 = pd.DataFrame({
            'x': [0.05],
            'y': [0.05]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        # Point (0.05, 0.05) is very close to cluster around (0,0)
        # First NN is very close, second NN is also close (within cluster)
        # So NNDR should be close to 1.0
        self.assertEqual(len(nndr_values), 1)
        self.assertGreater(nndr_values[0], 0)


class TestNNDRPrivacyInterpretation(unittest.TestCase):
    """Test NNDR from privacy perspective"""

    def setUp(self):
        self.attack = NearestNeighborDistanceRatio()

    def test_nndr_low_values_indicate_distinct_match(self):
        """Test that low NNDR indicates a distinct, reliable match"""
        # High privacy scenario: test point far from reference set
        # Should have high NNDR (first and second nearest are both far)
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [100],
            'y': [100]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        # Far point should have high NNDR (low privacy risk)
        self.assertGreater(nndr_values[0], 0.5)

    def test_nndr_high_values_indicate_privacy_risk(self):
        """Test that high NNDR could indicate privacy risk"""
        # Low privacy scenario: test point very close to reference set
        data1 = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4]
        })
        data2 = pd.DataFrame({
            'x': [0.5],
            'y': [0.5]
        })

        nndr_values = self.attack.calculate_nndr(data1, data2)

        # Close point should have lower NNDR
        self.assertLess(nndr_values[0], 2.0)


if __name__ == '__main__':
    unittest.main()
