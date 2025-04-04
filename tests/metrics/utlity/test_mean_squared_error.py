import unittest
import pandas as pd

from core.metrics.utility.mean_squared_error import MeanSquaredError

class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.real_data = pd.DataFrame({
            'Income': [50000, 60000, 70000, 80000],
            'Age': [25, 30, 35, 40]
        })
        self.synthetic_data = pd.DataFrame({
            'Income': [52000, 58000, 71000, 79000],
            'Age': [26, 29, 34, 41]
        })
        self.mse_metric = MeanSquaredError()

    def test_mean_squared_error(self):
        result = self.mse_metric.calculate(self.real_data, self.synthetic_data, columns=['Income', 'Age'])
        self.assertIn('mse', result)
        self.assertIn('overall_mse', result)
        self.assertEqual(len(result['mse']), 2)  # Expecting MSE for both columns
        self.assertIsNotNone(result['overall_mse'])  # Overall MSE should not be None

    def test_missing_column(self):
        synthetic_data_missing = pd.DataFrame({
            'Income': [52000, 58000, 71000, 79000]
            # 'Age' column is missing
        })
        result = self.mse_metric.calculate(self.real_data, synthetic_data_missing, columns=['Income', 'Age'])
        self.assertIn('mse', result)
        self.assertIsNone(result['mse']['Age'])  # MSE for Age should be None

if __name__ == '__main__':
    unittest.main()