import unittest
import pandas as pd

# from core.metrics.privacy.k_anonymity import KAnonymity

class TestKAnonymity(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'SEX': ['F', 'F', 'M', 'M', 'F', 'M'],
            'AGE': [25, 25, 30, 30, 25, 30],
            'CITY': ['Toronto', 'Toronto', 'Toronto', 'Toronto', 'Toronto', 'Toronto'],
            'Income': [50000, 60000, 70000, 80000, 50000, 70000]
        })
        self.quasi_identifiers = ['SEX', 'AGE', 'CITY']
        self.k_anonymity_metric = KAnonymity(self.quasi_identifiers)

    def test_k_anonymity(self):
        result = self.k_anonymity_metric.calculate(self.data)
        print(result)
        self.assertIn('k_value', result)
        self.assertEqual(result['k_value'], 3)  # Expecting k-value to be 2

    def test_invalid_quasi_identifiers(self):
        invalid_quasi_identifiers = ['INVALID_COLUMN']
        k_anonymity_metric_invalid = KAnonymity(invalid_quasi_identifiers)
        with self.assertRaises(ValueError):
            k_anonymity_metric_invalid.calculate(self.data)

if __name__ == '__main__':
    unittest.main()