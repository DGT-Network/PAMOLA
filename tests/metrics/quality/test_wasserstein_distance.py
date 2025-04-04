import unittest
from unittest.mock import patch
import pandas as pd
from pamola.pamola_core.metrics.quality.wasserstein_distance import WassersteinDistance

class TestWassersteinDistance(unittest.TestCase):

    def setUp(self):
        self.wd_test = WassersteinDistance()
        # Load sample data from CSV file
        self.real_data = pd.read_csv(r'e:\dtx\pamola\pamola\tests\data_test\sample_credit_data.csv')
        self.synthetic_data = pd.read_csv(r'e:\dtx\pamola\pamola\tests\data_test\sample_credit_data.csv')  # Use the same data for simplicity

    @patch('pamola.pamola_core.metrics.quality.wasserstein_distance.logger')
    def test_calculate_with_matching_columns(self, mock_logger):
        result = self.wd_test.calculate(self.real_data, self.synthetic_data)
        for col in self.real_data.select_dtypes(include=["number"]).columns:
            self.assertIn(col, result)
        mock_logger.info.assert_called_with("Wasserstein Distance calculation completed successfully")

    @patch('pamola.pamola_core.metrics.quality.wasserstein_distance.logger')
    def test_calculate_with_non_matching_columns(self, mock_logger):
        synthetic_data = pd.DataFrame({
            'D': [1, 2, 3, 4, 5]
        })
        result = self.wd_test.calculate(self.real_data, synthetic_data)
        self.assertEqual(result, {})
        mock_logger.info.assert_called_with("Wasserstein Distance calculation completed successfully")

    @patch('pamola.pamola_core.metrics.quality.wasserstein_distance.logger')
    def test_calculate_with_exception(self, mock_logger):
        with self.assertRaises(Exception):
            self.wd_test.calculate(None, self.synthetic_data)
        mock_logger.error.assert_called()

if __name__ == '__main__':
    unittest.main()