import unittest
import pandas as pd

from metrics.privacy.differential_privacy import DifferentialPrivacy

class TestDifferentialPrivacy(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'Income': [50000, 60000, 70000, 80000, 50000, 70000],
            'Age': [25, 30, 35, 40, 25, 30]
        })
        self.epsilon = 0.1
        self.sensitivity = 10000
        self.dp_metric = DifferentialPrivacy(self.epsilon, self.sensitivity, mechanism="laplace")

    def test_differential_privacy(self):
        result = self.dp_metric.calculate(self.data)
        self.assertIn('original_means', result)
        self.assertIn('dp_means', result)
        self.assertIn('dp_data', result)
        self.assertEqual(len(result['dp_data']['Income']), len(self.data['Income']))  # Ensure dp_data has the same length as input
        self.assertEqual(len(result['dp_data']['Age']), len(self.data['Age']))  # Ensure dp_data has the same length as input

    def test_invalid_data_type(self):
        with self.assertRaises(ValueError):
            self.dp_metric.calculate("invalid_data")  # Should raise an error

    def test_invalid_mechanism(self):
        with self.assertRaises(ValueError):
            DifferentialPrivacy(self.epsilon, self.sensitivity, mechanism="invalid_mechanism")  # Should raise an error

if __name__ == '__main__':
    unittest.main()