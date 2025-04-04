import unittest
import pandas as pd

from core.metrics.privacy.l_diversity import LDiversity

class TestLDiversity(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'SEX': ['F', 'F', 'M', 'M', 'F', 'M'],
            'AGE': [25, 25, 30, 30, 25, 30],
            'CITY': ['Toronto', 'Toronto', 'Toronto', 'Toronto', 'Toronto', 'Toronto'],
            'Income': [50000, 60000, 70000, 80000, 50000, 70000]
        })
        self.quasi_identifiers = ['SEX', 'AGE', 'CITY']
        self.sensitive_column = 'Income'
        self.l_value = 2
        self.l_diversity_metric = LDiversity(self.quasi_identifiers, self.sensitive_column, self.l_value)

    def test_l_diversity(self):
        result = self.l_diversity_metric.calculate(self.data)
        self.assertIn('l_value', result)
        self.assertIn('is_l_diverse', result)
        self.assertEqual(result['l_value'], 2)  # Expecting l-value to be 2
        self.assertTrue(result['is_l_diverse'])  # Expecting it to be diverse

    def test_invalid_quasi_identifiers(self):
        invalid_quasi_identifiers = ['INVALID_COLUMN']
        l_diversity_metric_invalid = LDiversity(invalid_quasi_identifiers, self.sensitive_column, self.l_value)
        with self.assertRaises(ValueError):
            l_diversity_metric_invalid.calculate(self.data)

if __name__ == '__main__':
    unittest.main()