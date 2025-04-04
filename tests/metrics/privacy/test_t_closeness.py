import unittest
import pandas as pd

from core.metrics.privacy.t_closeness import TCloseness

class TestTCloseness(unittest.TestCase):

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
        self.t_value = 0.1
        self.t_closeness_metric = TCloseness(self.quasi_identifiers, self.sensitive_column, self.t_value)

    def test_t_closeness(self):
        result = self.t_closeness_metric.calculate(self.data)
        # print(result)
        self.assertIn('max_t_value', result)
        self.assertIn('is_t_close', result)

    def test_invalid_quasi_identifiers(self):
        invalid_quasi_identifiers = ['INVALID_COLUMN']
        t_closeness_metric_invalid = TCloseness(invalid_quasi_identifiers, self.sensitive_column, self.t_value)
        with self.assertRaises(ValueError):
            t_closeness_metric_invalid.calculate(self.data)

if __name__ == '__main__':
    unittest.main()