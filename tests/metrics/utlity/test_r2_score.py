import os
import pandas as pd
import unittest

from core.metrics.utility.r2_score import R2Score


class TestR2Score(unittest.TestCase):

    def setUp(self):
        # Define the path to the CSV file
        self.file_path = '../../data_test/RandomPeople-3-clean 3.csv'

        # Check if the file exists
        if not os.path.isfile(self.file_path):
            self.skipTest(f"File not found: {self.file_path}")

        # Load sample data for testing from CSV
        self.real_data = pd.read_csv(self.file_path)

        # Ensure the target column is numeric
        if 'Value' not in self.real_data.columns:
            raise ValueError("The target column 'Value' does not exist in the real data.")

        # Create a synthetic dataset by modifying the real data
        self.synthetic_data = self.real_data.copy()
        self.synthetic_data['Value'] = self.synthetic_data['Value'] * 0.9  # Slightly modify the values

        # Initialize the R2Score metric
        self.r2_metric = R2Score()


    def test_r2_score(self):
        result = self.r2_metric.calculate(self.real_data, self.synthetic_data, target_column='Value')
        self.assertIn('r2_score', result)
        self.assertIsInstance(result['r2_score'], float)  # Ensure the result is a float
        self.assertGreaterEqual(result['r2_score'], -1)  # R² score should be >= -1

    def test_missing_column(self):
        synthetic_data_missing = pd.DataFrame({
            'Other_Column': [1, 2, 3, 4, 5]
            # 'Value' column is missing
        })
        with self.assertRaises(ValueError):
            self.r2_metric.calculate(self.real_data, synthetic_data_missing, target_column='Value')

    def test_invalid_data_type(self):
        with self.assertRaises(ValueError):
            self.r2_metric.calculate("invalid_data", self.synthetic_data, target_column='Value')  # Should raise an error

    def test_non_numeric_values(self):
        synthetic_data_non_numeric = pd.DataFrame({
            'Value': ['a', 'b', 'c', 'd', 'e']
        })
        with self.assertRaises(ValueError):
            self.r2_metric.calculate(self.real_data, synthetic_data_non_numeric, target_column='Value')

if __name__ == '__main__':
    unittest.main()