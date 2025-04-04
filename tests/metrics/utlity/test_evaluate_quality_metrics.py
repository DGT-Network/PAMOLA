import os
import unittest
import pandas as pd

from core.metrics.utility.evaluate_quality_metrics import EvaluateQualityMetrics

class TestEvaluateQualityMetrics(unittest.TestCase):

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
        self.synthetic_data['Income'] = self.synthetic_data['Income'] * 0.9  # Slightly modify income
        self.synthetic_data['credit_score'] = self.synthetic_data['credit_score'] + 10  # Slightly modify credit score
        
        self.evaluator = EvaluateQualityMetrics()

    def test_kolmogorov_smirnov_test(self):
        results = self.evaluator.kolmogorov_smirnov_test(self.real_data, self.synthetic_data, target_columns=['Income', 'credit_score'])
        print("KS-Test Results:", results)
        self.assertIn('Income_ks_stat', results)
        self.assertIn('Income_ks_pval', results)

    def test_pearson_correlation(self):
        results = self.evaluator.pearson_correlation(self.real_data, self.synthetic_data, target_columns=['Income'])
        print("KS-Test Results:", results)
        self.assertIn('Income', results)
        self.assertIsInstance(results['Income'], float)  # Should return a float value

    def test_kullback_leibler_divergence(self):
        results = self.evaluator.kullback_leibler_divergence(self.real_data, self.synthetic_data, target_columns=['Income'])
        self.assertIn('Income', results)
        self.assertIsInstance(results['Income'], float)  # Should return a float value

    def test_wasserstein_distance(self):
        results = self.evaluator.wasserstein_distance(self.real_data, self.synthetic_data, target_columns=['Income'])
        self.assertIn('Income', results)
        self.assertIsInstance(results['Income'], float)  # Should return a float value

    def test_invalid_column(self):
        results = self.evaluator.kolmogorov_smirnov_test(self.real_data, self.synthetic_data, target_columns=['invalid_column'])
        self.assertEqual(results, {})  # Should return an empty dictionary for invalid columns

if __name__ == '__main__':
    unittest.main()