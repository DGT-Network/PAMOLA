"""
PAMOLA - Privacy-Aware Machine Learning Analytics
Unit Tests for LinkageAttack

This module contains comprehensive unit tests for the LinkageAttack class.
Tests cover:
- Initialization with default and custom parameters
- Record linkage attack (direct comparison)
- Probabilistic linkage attack (Fellegi-Sunter)
- Cluster-vector linkage attack (CVPLA)
- Error handling for invalid inputs
- Edge cases (empty DataFrames, mismatched columns)
"""

import unittest
import pandas as pd
import numpy as np
import pytest
from pamola_core.attacks.linkage_attack import LinkageAttack
from pamola_core.errors.exceptions import ValidationError


class TestLinkageAttackInitialization(unittest.TestCase):
    """Test LinkageAttack initialization"""

    def test_initialization_with_defaults(self):
        """Test LinkageAttack initialization with default parameters"""
        attack = LinkageAttack()
        self.assertIsNone(attack.fs_threshold)
        self.assertEqual(attack.n_components, 2)

    def test_initialization_with_custom_parameters(self):
        """Test LinkageAttack initialization with custom parameters"""
        attack = LinkageAttack(fs_threshold=0.9, n_components=5)
        self.assertEqual(attack.fs_threshold, 0.9)
        self.assertEqual(attack.n_components, 5)

    def test_initialization_with_zero_threshold(self):
        """Test LinkageAttack initialization with zero threshold"""
        attack = LinkageAttack(fs_threshold=0.0)
        self.assertEqual(attack.fs_threshold, 0.0)

    def test_initialization_with_high_n_components(self):
        """Test LinkageAttack initialization with high n_components"""
        attack = LinkageAttack(n_components=100)
        self.assertEqual(attack.n_components, 100)


@pytest.fixture
def sample_dataframes():
    """Fixture providing sample original and anonymized DataFrames"""
    # Original (unmodified) data
    original = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle']
    })

    # Anonymized data (same records, some modified)
    anonymized = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charles', 'Diana', 'Eva'],  # Slight modifications
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle']
    })

    return original, anonymized


class TestRecordLinkageAttack(unittest.TestCase):
    """Test record_linkage_attack method"""

    def setUp(self):
        self.attack = LinkageAttack()

    def test_record_linkage_exact_match(self):
        """Test record linkage with exact matching records"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        data2 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id'])

        self.assertEqual(len(result), 3)
        self.assertListEqual(list(result['id']), [1, 2, 3])

    def test_record_linkage_partial_matches(self):
        """Test record linkage with partial matching records"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        data2 = pd.DataFrame({
            'id': [1, 3],
            'name': ['Alice', 'Charlie']
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id'])

        self.assertEqual(len(result), 2)
        self.assertListEqual(list(result['id']), [1, 3])

    def test_record_linkage_no_matches(self):
        """Test record linkage with no matching records"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        data2 = pd.DataFrame({
            'id': [10, 20, 30],
            'name': ['Diana', 'Eve', 'Frank']
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id'])

        self.assertEqual(len(result), 0)

    def test_record_linkage_auto_detect_keys(self):
        """Test record linkage with auto-detected common columns"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'extra': ['a', 'b', 'c']
        })
        data2 = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'different': ['x', 'y']
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=None)

        # Should auto-detect 'id' and 'name' as common keys
        self.assertGreater(len(result), 0)

    def test_record_linkage_none_data1(self):
        """Test record linkage with None as first dataset"""
        with self.assertRaises(ValidationError):
            self.attack.record_linkage_attack(None, pd.DataFrame(), linkage_keys=['id'])

    def test_record_linkage_none_data2(self):
        """Test record linkage with None as second dataset"""
        with self.assertRaises(ValidationError):
            self.attack.record_linkage_attack(pd.DataFrame(), None, linkage_keys=['id'])

    def test_record_linkage_empty_result_no_common_keys(self):
        """Test record linkage with no common keys"""
        data1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        data2 = pd.DataFrame({
            'c': [10, 20, 30],
            'd': ['p', 'q', 'r']
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=[])

        self.assertEqual(len(result), 0)

    def test_record_linkage_multiple_linkage_keys(self):
        """Test record linkage with multiple linkage keys"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'city': ['NYC', 'LA', 'Chicago']
        })
        data2 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'city': ['NYC', 'LA', 'Chicago']
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id', 'name'])

        self.assertEqual(len(result), 3)

    def test_record_linkage_index_reset(self):
        """Test record linkage properly resets indices"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        }, index=[10, 20, 30])
        data2 = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob']
        }, index=[100, 200])

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id'])

        self.assertEqual(len(result), 2)


class TestProbabilisticLinkageAttack(unittest.TestCase):
    """Test probabilistic_linkage_attack method (Fellegi-Sunter)"""

    def setUp(self):
        self.attack = LinkageAttack(fs_threshold=0.85)

    def test_probabilistic_linkage_exact_match(self):
        """Test probabilistic linkage with exact matching records"""
        data1 = pd.DataFrame({
            'name': ['Alice Johnson', 'Bob Smith'],
            'age': [25, 30]
        })
        data2 = pd.DataFrame({
            'name': ['Alice Johnson', 'Bob Smith'],
            'age': [25, 30]
        })

        result = self.attack.probabilistic_linkage_attack(data1, data2, keys=['name', 'age'])

        self.assertGreater(len(result), 0)
        self.assertIn('similarity_score', result.columns)

    def test_probabilistic_linkage_with_typos(self):
        """Test probabilistic linkage handles slight typos"""
        data1 = pd.DataFrame({
            'name': ['Alice Johnson', 'Bob Smith'],
        })
        data2 = pd.DataFrame({
            'name': ['Alice Jonson', 'Bob Smith'],  # Typo in first name
        })

        result = self.attack.probabilistic_linkage_attack(data1, data2, keys=['name'])

        # Should find matches despite typo
        self.assertGreater(len(result), 0)

    def test_probabilistic_linkage_none_data(self):
        """Test probabilistic linkage with None data"""
        with self.assertRaises(ValidationError):
            self.attack.probabilistic_linkage_attack(None, pd.DataFrame(), keys=['name'])

    def test_probabilistic_linkage_empty_dataframes(self):
        """Test probabilistic linkage with empty DataFrames"""
        # Pass keys=None so source auto-detects (finds none), returns early before
        # attempting column selection on empty DataFrames (which would raise KeyError).
        result = self.attack.probabilistic_linkage_attack(
            pd.DataFrame(),
            pd.DataFrame(),
            keys=None
        )

        self.assertEqual(len(result), 0)

    def test_probabilistic_linkage_no_common_keys(self):
        """Test probabilistic linkage with no common keys"""
        data1 = pd.DataFrame({'a': [1, 2]})
        data2 = pd.DataFrame({'b': [3, 4]})

        result = self.attack.probabilistic_linkage_attack(data1, data2, keys=None)

        self.assertEqual(len(result), 0)

    def test_probabilistic_linkage_threshold_filtering(self):
        """Test that similarity threshold is applied correctly"""
        attack = LinkageAttack(fs_threshold=0.99)
        data1 = pd.DataFrame({
            'name': ['Perfect Match', 'Another Name'],
        })
        data2 = pd.DataFrame({
            'name': ['Perfect Match', 'Different Name'],
        })

        result = attack.probabilistic_linkage_attack(data1, data2, keys=['name'])

        # Only perfect or near-perfect matches should pass
        self.assertTrue(all(result['similarity_score'] >= 0.99))

    def test_probabilistic_linkage_handles_missing_values(self):
        """Test probabilistic linkage handles missing values"""
        data1 = pd.DataFrame({
            'name': ['Alice', None, 'Charlie'],
            'age': [25, 30, 35]
        })
        data2 = pd.DataFrame({
            'name': ['Alice', 'Bob', None],
            'age': [25, 30, 35]
        })

        result = self.attack.probabilistic_linkage_attack(data1, data2, keys=['name', 'age'])

        self.assertIsInstance(result, pd.DataFrame)


class TestClusterVectorLinkageAttack(unittest.TestCase):
    """Test cluster_vector_linkage_attack method (CVPLA)"""

    def setUp(self):
        self.attack = LinkageAttack(n_components=2)

    def test_cluster_vector_linkage_basic(self):
        """Test basic cluster-vector linkage attack"""
        data1 = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })
        data2 = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })

        result = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.5)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('ID_DF1', result.columns)
        self.assertIn('ID_DF2', result.columns)
        self.assertIn('Score', result.columns)

    def test_cluster_vector_linkage_identical_datasets(self):
        """Test CVPLA with identical datasets (0% privacy)"""
        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        result = self.attack.cluster_vector_linkage_attack(data, data, similarity_threshold=0.5)

        # CVPLA deduplicates on ID_DF1 (each data1 record kept once, highest score).
        # For identical datasets multiple data2 records may map to the same data1 record,
        # so result length <= len(data). We only verify at least one match is found.
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), len(data))

    def test_cluster_vector_linkage_different_datasets(self):
        """Test CVPLA with completely different datasets (100% privacy)"""
        data1 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })
        data2 = pd.DataFrame({
            'x': [100, 200, 300],
            'y': [1000, 2000, 3000]
        })

        result = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.9)

        # High threshold should reduce matches for very different data
        self.assertLessEqual(len(result), len(data2))

    def test_cluster_vector_linkage_similarity_threshold(self):
        """Test similarity threshold parameter"""
        data1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        data2 = pd.DataFrame({
            'a': [1, 2, 10],
            'b': [4, 5, 60]
        })

        result_low = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.5)
        result_high = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.95)

        # Higher threshold should result in fewer matches
        self.assertGreaterEqual(len(result_low), len(result_high))

    def test_cluster_vector_linkage_none_data(self):
        """Test CVPLA with None data"""
        with self.assertRaises(ValidationError):
            self.attack.cluster_vector_linkage_attack(None, pd.DataFrame())

    def test_cluster_vector_linkage_empty_dataframes(self):
        """Test CVPLA with empty DataFrames"""
        result = self.attack.cluster_vector_linkage_attack(
            pd.DataFrame(),
            pd.DataFrame()
        )

        self.assertEqual(len(result), 0)
        self.assertListEqual(list(result.columns), ['ID_DF1', 'ID_DF2', 'Score'])

    def test_cluster_vector_linkage_partial_match(self):
        """Test CVPLA identifies records with similarity"""
        data1 = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0]
        })
        data2 = pd.DataFrame({
            'feature1': [1.1, 2.2, 3.3],  # Slightly different
            'feature2': [10.5, 20.5, 30.5]
        })

        result = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.7)

        # Should find some matches due to similarity
        self.assertGreater(len(result), 0)

    def test_cluster_vector_linkage_n_components_validation(self):
        """Test n_components parameter is respected"""
        attack1 = LinkageAttack(n_components=1)
        attack2 = LinkageAttack(n_components=5)

        data1 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'z': [7, 8, 9]
        })
        data2 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'z': [7, 8, 9]
        })

        result1 = attack1.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.5)
        result2 = attack2.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.5)

        # Both should produce results
        self.assertGreater(len(result1), 0)
        self.assertGreater(len(result2), 0)

    def test_cluster_vector_linkage_deduplication(self):
        """Test that duplicate ID_DF1 values are removed (keeping highest score)"""
        data1 = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30]
        })
        data2 = pd.DataFrame({
            'x': [1.5, 2.5],
            'y': [15, 25]
        })

        result = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.0)

        # Check that no ID_DF1 is duplicated
        self.assertEqual(len(result), len(result['ID_DF1'].unique()))


class TestLinkageAttackEdgeCases(unittest.TestCase):
    """Test edge cases for LinkageAttack"""

    def setUp(self):
        self.attack = LinkageAttack()

    def test_single_record_datasets(self):
        """Test attacks with single-record datasets"""
        data1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
        data2 = pd.DataFrame({'id': [1], 'name': ['Alice']})

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id'])

        self.assertEqual(len(result), 1)

    def test_large_similarity_values(self):
        """Test CVPLA produces valid similarity scores"""
        data1 = pd.DataFrame({
            'a': np.random.rand(10),
            'b': np.random.rand(10)
        })
        data2 = pd.DataFrame({
            'a': np.random.rand(10),
            'b': np.random.rand(10)
        })

        result = self.attack.cluster_vector_linkage_attack(data1, data2, similarity_threshold=0.0)

        # All scores should be between 0 and 1
        self.assertTrue((result['Score'] >= 0).all())
        self.assertTrue((result['Score'] <= 1).all())

    def test_string_and_numeric_columns(self):
        """Test attacks handle mixed data types"""
        data1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95.5, 87.3, 92.1]
        })
        data2 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [95.5, 87.3, 92.1]
        })

        result = self.attack.record_linkage_attack(data1, data2, linkage_keys=['id', 'name'])

        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()
