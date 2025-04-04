import unittest
import pandas as pd
from pamola.pamola_core.core.privacy_models.l_diversity.reporting import LDiversityReport

class TestLDiversityReport(unittest.TestCase):

    def setUp(self):
        # Setup code to initialize LDiversityReport instance
        self.processor = None  # Replace with actual processor instance if available
        self.reporter = LDiversityReport(self.processor)

    def test_extract_entropy_metrics(self):
        # Sample group diversity data
        group_diversity = pd.DataFrame({
            'sa_entropy': [0.5, 1.0, 1.5]
        })
        sensitive_attributes = ['sa']

        # Call the method
        metrics = self.reporter._extract_entropy_metrics(group_diversity, sensitive_attributes)

        # Assertions
        self.assertIn('attribute_entropy', metrics)
        self.assertIn('sa', metrics['attribute_entropy'])
        self.assertEqual(metrics['attribute_entropy']['sa']['min_entropy'], 0.5)
        self.assertEqual(metrics['attribute_entropy']['sa']['max_entropy'], 1.5)
        self.assertEqual(metrics['attribute_entropy']['sa']['avg_entropy'], 1.0)

    def test_extract_recursive_metrics(self):
        # Sample group diversity data
        group_diversity = pd.DataFrame()
        sensitive_attributes = ['sa']

        # Call the method
        metrics = self.reporter._extract_recursive_metrics(group_diversity, sensitive_attributes)

        # Assertions
        self.assertIn('attribute_recursive', metrics)
        self.assertIn('c_value', metrics)
        self.assertEqual(metrics['c_value'], 1.0)

    def test_empty_dataset(self):
        # Test with an empty DataFrame
        data = pd.DataFrame()
        result = self.reporter._calculate_diversity_metrics(data)
        self.assertEqual(result, {})  # Expecting an empty dictionary

    def test_no_quasi_identifiers(self):
        # Test with no quasi-identifiers
        data = pd.DataFrame({
            'sensitive_attr': ['A', 'B', 'A', 'B']
        })
        result = self.reporter._calculate_diversity_metrics(data, sensitive_attributes=['sensitive_attr'])
        self.assertIn('diversity_metrics', result)  # Check if the key exists

    def test_valid_data(self):
        # Test with valid data
        data = pd.DataFrame({
            'quasi_id1': ['Q1', 'Q1', 'Q2', 'Q2'],
            'quasi_id2': ['Q2', 'Q2', 'Q1', 'Q1'],
            'sensitive_attr': ['A', 'B', 'A', 'B']
        })
        result = self.reporter._calculate_diversity_metrics(data, 
                                                             quasi_identifiers=['quasi_id1', 'quasi_id2'], 
                                                             sensitive_attributes=['sensitive_attr'])
        self.assertIsInstance(result, dict)  # Check if the result is a dictionary
        self.assertGreater(len(result), 0)  # Check if the result contains metrics
if __name__ == '__main__':
    unittest.main()