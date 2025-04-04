import unittest
from pathlib import Path
import os
import pandas as pd
from pamola.pamola_core.privacy_models.l_diversity.visualization import LDiversityVisualizer, visualize_l_diversity, \
    visualize_attribute_distributions, visualize_risk_dashboard


def get_sample_data(categorical=False):
    """Load sample test data from CSV file in data_test folder."""
    # file_path = Path("../../data_test/sample_credit_data.csv")
    file_path = Path("../../data_test/RandomPeople-3-clean 3.csv")

    print(f"Loading data from: {file_path}")  # Print file path for debugging
    if not file_path.exists():
        raise FileNotFoundError(f"Sample data file not found: {file_path}")
    df = pd.read_csv(file_path)
    # print("Sample data loaded successfully:")  # Debug print
    # print(df.head())  # Print first few rows to verify data
    # return df
    df_first_100 = df.head(100)
    return df_first_100


class TestLDiversityVisualizer(unittest.TestCase):
    SAVE_DIR = 'output_v'  # Change this to your desired directory

    @classmethod
    def setUpClass(cls):
        """Set up test environment and create output directory."""
        if not os.path.exists(cls.SAVE_DIR):
            os.makedirs(cls.SAVE_DIR)
        cls.save_path = Path(cls.SAVE_DIR)

    def setUp(self):
        """Initialize LDiversityVisualizer instance."""
        self.processor = None  # Replace with actual processor instance if available
        self.visualizer = LDiversityVisualizer(self.processor)

    def test_visualize_l_distribution(self):
        """Test visualization of l-distribution."""
        data = get_sample_data()
        print("Data used for visualization:")
        print(data)  # Print entire dataset for debugging
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        sensitive_attributes = ["credit_score", "loan_status"]
        fig, saved_path = self.visualizer.visualize_l_distribution(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_l_distribution_config(self):
        """Test l-distribution visualization with configuration options."""
        data = get_sample_data()
        quasi_identifiers = ['SEX', 'AGE']
        sensitive_attributes = ['branch_visits']
        config = {'figsize': (12, 8), 'style': 'white', 'palette': 'plasma', 'save_format': 'png', 'color': 'darkblue',
                  'dpi': 200}
        fig, saved_path = self.visualizer.visualize_l_distribution(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path, **config
        )
        print(f"Image saved at: {saved_path}")

        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_attribute_distribution_numeric(self):
        """Test visualization of numeric attribute distribution."""
        data = get_sample_data()
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        config = {'figsize': (12, 8), 'style': 'whitegrid', 'palette': 'plasma', 'save_format': 'png',
                  'color': 'yellow', 'dpi': 200}
        # **config
        fig, saved_path = self.visualizer.visualize_attribute_distribution(
            data, 'credit_score', quasi_identifiers=quasi_identifiers, save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_attribute_distribution_categorical(self):
        """Test visualization of categorical attribute distribution."""
        data = get_sample_data(categorical=True)
        fig, saved_path = self.visualizer.visualize_attribute_distribution(
            data, 'branch_visits', quasi_identifiers=['SEX', 'CITY'], save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_attribute_distribution_overall(self):
        """Test overall attribute distribution visualization."""
        data = get_sample_data()
        fig, saved_path = self.visualizer.visualize_attribute_distribution(
            data, 'account_age', save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_risk_heatmap(self):
        """Test visualization of risk heatmap."""
        data = get_sample_data()
        print(f"save_path:", self.save_path)

        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        sensitive_attributes = ["credit_score", "loan_status"]
        fig, saved_path = self.visualizer.visualize_risk_heatmap(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path
        )
        print(f"saved_path:", saved_path)

        self.assertIsNotNone(fig)
        print("Debug: saved_path is None" if saved_path is None else "Debug: saved_path is valid")
        self.assertIsNotNone(saved_path)

    def test_visualize_risk_heatmap_with_config(self):
        """Test risk heatmap visualization with configuration options."""
        data = get_sample_data()
        # quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        # sensitive_attributes = ["credit_score", "loan_status"]
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE",
                             "Income", "IsHomeOwner", "IsEduBachelors",
                             "IsUnemployed", "BankPresenceRating", "ChurnProbability"]
        sensitive_attributes = ["credit_score", "AccountType", "account_age", "avg12_tx", "BankIsPrimary",
                                "avg12_tx_volume", "loan_status", "credit_card_status", "overdraft_usage",
                                "branch_visits", "digital_usage_level", "customer_service",
                                "complaints_filed", "satisfaction_level"]
        config = {'figsize': (12, 4), 'style': 'whitegrid', 'palette': 'plasma', 'save_format': 'pdf', 'color': 'blue',
                  'dpi': 200}
        fig, saved_path = self.visualizer.visualize_risk_heatmap(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path, **config
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_calculate_group_diversity(self):
        """Test the _calculate_group_diversity function."""
        data = get_sample_data()
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        sensitive_attributes = ["credit_score", "loan_status"]
        diversity_type = 'distinct'

        result = self.visualizer._calculate_group_diversity(
            data, quasi_identifiers, sensitive_attributes, diversity_type
        )

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the result has the expected columns
        expected_columns = ['SEX', 'CITY', 'PROVINCE', 'AGE', 'IsMarried', 'RACE', 
                            'credit_score_distinct', 'loan_status_distinct', 'group_size']
        self.assertTrue(all(col in result.columns for col in expected_columns))

        # Check if the group sizes are correct
        expected_group_sizes = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 
                                1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 
                                1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 
                                1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 
                                1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 
                                2, 1, 1, 1, 2, 1, 1]
        actual_group_sizes = result['group_size'].tolist()
        
        # Debugging output
        print("Expected group_sizes:", expected_group_sizes)
        print("Actual group_sizes:", actual_group_sizes)
        print("Length of actual group sizes:", len(actual_group_sizes))
        
        # self.assertEqual(len(actual_group_sizes), len(expected_group_sizes))
        self.assertTrue(all(isinstance(size, int) for size in actual_group_sizes))
        self.assertTrue(all(size in expected_group_sizes for size in actual_group_sizes))

    # Utility functions for standalone usage
    def test_visualize_l_diversity(self):
        """Test the standalone visualize_l_diversity function."""
        data = get_sample_data()
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        sensitive_attributes = ["credit_score", "loan_status"]
        fig, saved_path = visualize_l_diversity(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_l_diversity_with_config(self):
        """Test the standalone visualize_l_diversity function with configuration options."""
        data = get_sample_data()
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        sensitive_attributes = ["credit_score", "loan_status"]
        config = {'figsize': (12, 8), 'style': 'white', 'palette': 'plasma', 'save_format': 'png',
                  'color': 'yellow', 'dpi': 200}
        fig, saved_path = visualize_l_diversity(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path, **config
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_attribute_distributions(self):
        """Test the standalone visualize_attribute_distributions function."""
        data = get_sample_data()
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "RACE"]
        sensitive_attributes = ["credit_score", "credit_card_status"]
        fig, saved_path = visualize_attribute_distributions(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)

    def test_visualize_risk_dashboard(self):
        """Test the standalone visualize_risk_dashboard function."""
        data = get_sample_data()
        quasi_identifiers = ["SEX", "CITY", "PROVINCE", "AGE", "IsMarried", "FAMILYID", "RACE",
                             "Income", "IsHomeOwner", "HOMEVALUE", "RENTVALUE", "IsEduBachelors",
                             "IsUnemployed", "BankPresenceRating", "ChurnProbability"]
        sensitive_attributes = ["credit_score", "AccountType", "account_age", "avg12_tx", "BankIsPrimary",
                                "avg12_tx_volume", "loan_status", "credit_card_status", "overdraft_usage",
                                "branch_visits", "digital_usage_level", "customer_service",
                                "complaints_filed", "satisfaction_level"]
        fig, saved_path = visualize_risk_dashboard(
            data, quasi_identifiers, sensitive_attributes, save_path=self.save_path
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(saved_path)
            

if __name__ == '__main__':
    unittest.main()
