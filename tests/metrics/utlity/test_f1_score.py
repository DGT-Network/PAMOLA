import os
import unittest
import pandas as pd

from core.metrics.utility.f1_score import F1Score

class TestF1Score(unittest.TestCase):

    def setUp(self):
        # Define the path to the CSV file
        self.file_path = '../../data_test/RandomPeople-3-clean 3.csv'

        # Check if the file exists
        if not os.path.isfile(self.file_path):
            self.skipTest(f"File not found: {self.file_path}")

        # Load sample data for testing from CSV
        self.real_data = pd.read_csv(self.file_path)

        # Create a synthetic dataset by modifying the real data
        self.synthetic_data = self.real_data.copy()
        self.synthetic_data['Income'] = self.synthetic_data['Income'].apply(lambda x: 1 if x == 0 else 0)  # Example modification

        # Initialize the F1Score metric
        self.f1_metric = F1Score()

    def test_f1_score(self):
        result = self.f1_metric.calculate(self.real_data, self.synthetic_data, target_column='Income')
        print(result)
        self.assertIn('f1_score', result)
        self.assertIsInstance(result['f1_score'], float)  # Ensure the result is a float

    def test_missing_column(self):
        synthetic_data_missing = pd.DataFrame({
            'Other_Column': [0, 1, 0, 0, 1, 1]
            # 'Label' column is missing
        })
        with self.assertRaises(ValueError):
            self.f1_metric.calculate(self.real_data, synthetic_data_missing, target_column='Income')

    def test_invalid_data_type(self):
        with self.assertRaises(ValueError):
            self.f1_metric.calculate("invalid_data", self.synthetic_data, target_column='Income')  # Should raise an error

if __name__ == '__main__':
    unittest.main()