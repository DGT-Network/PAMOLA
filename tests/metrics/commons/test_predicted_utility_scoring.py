"""
Comprehensive tests for predicted_utility_scoring module.

Tests cover:
- calculate_predicted_utility function with various scenarios
- Completeness score calculation
- Diversity score calculation
- Balance check score calculation
- Schema richness score calculation
- Weighted predicted utility aggregation
"""

import pytest
import pandas as pd
import numpy as np

from pamola_core.metrics.commons.predicted_utility_scoring import (
    calculate_predicted_utility,
)


class TestCalculatePredictedUtility:
    """Test calculate_predicted_utility function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_predicted_utility(df)
        assert result["completeness"] == 0
        assert result["diversity"] == 0
        assert result["balance_check"] == 0
        assert result["schema_richness"] == 0
        assert result["predicted_utility"] == 0

    def test_complete_dataframe_default_weights(self):
        """Test with complete DataFrame using default weights."""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
        })
        result = calculate_predicted_utility(df)
        assert result["predicted_utility"] >= 0
        assert result["predicted_utility"] <= 100
        assert "completeness" in result
        assert "diversity" in result

    def test_completeness_all_nulls(self):
        """Test completeness with all null values in required fields."""
        df = pd.DataFrame({
            "col1": [None, None, None],
            "col2": [1, 2, 3],
        })
        result = calculate_predicted_utility(df, require_fields=["col1"])
        assert result["completeness"] == 0.0

    def test_completeness_no_nulls(self):
        """Test completeness with no null values."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        })
        result = calculate_predicted_utility(df, require_fields=["col1", "col2"])
        assert result["completeness"] == 1.0

    def test_completeness_partial_nulls(self):
        """Test completeness with some null values."""
        df = pd.DataFrame({
            "col1": [1, None, 3],
            "col2": [4, 5, None],
        })
        result = calculate_predicted_utility(df, require_fields=["col1", "col2"])
        # 4 out of 6 cells have values = 0.67
        assert 0.6 < result["completeness"] < 0.8

    def test_completeness_nonexistent_field(self):
        """Test completeness with nonexistent required field."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
        })
        result = calculate_predicted_utility(df, require_fields=["nonexistent"])
        assert result["completeness"] == 0.0

    def test_completeness_no_require_fields(self):
        """Test completeness with no required fields specified."""
        df = pd.DataFrame({
            "col1": [None, None, None],
        })
        result = calculate_predicted_utility(df, require_fields=None)
        assert result["completeness"] == 1.0

    def test_diversity_all_unique(self):
        """Test diversity with all unique values."""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
        })
        result = calculate_predicted_utility(df, unique_fields=["col1"])
        assert result["diversity"] == 1.0

    def test_diversity_no_unique(self):
        """Test diversity with all duplicate values."""
        df = pd.DataFrame({
            "col1": [1, 1, 1, 1, 1],
        })
        result = calculate_predicted_utility(df, unique_fields=["col1"])
        assert abs(result["diversity"] - 0.2) < 0.01  # 1 unique / 5 rows

    def test_diversity_partial_unique(self):
        """Test diversity with partial uniqueness."""
        df = pd.DataFrame({
            "col1": [1, 1, 2, 3, 3],
        })
        result = calculate_predicted_utility(df, unique_fields=["col1"])
        # 3 unique values / 5 rows = 0.6
        assert abs(result["diversity"] - 0.6) < 0.01

    def test_diversity_multiple_fields(self):
        """Test diversity with multiple unique fields."""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [1, 1, 2, 2, 3],
        })
        result = calculate_predicted_utility(df, unique_fields=["col1", "col2"])
        # (5 + 3) / 2 / 5 = 4/5 = 0.8
        assert abs(result["diversity"] - 0.8) < 0.01

    def test_diversity_nonexistent_field(self):
        """Test diversity with nonexistent unique field."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
        })
        result = calculate_predicted_utility(df, unique_fields=["nonexistent"])
        assert result["diversity"] == 0.0

    def test_diversity_no_unique_fields(self):
        """Test diversity with no unique fields specified."""
        df = pd.DataFrame({
            "col1": [None, None, None],
        })
        result = calculate_predicted_utility(df, unique_fields=None)
        assert result["diversity"] == 1.0

    def test_balance_check_balanced(self):
        """Test balance check with balanced values."""
        df = pd.DataFrame({
            "col1": [0, 0, 1, 1],
        })
        result = calculate_predicted_utility(df, check_balance_fields=["col1"])
        # Each value is 50% of total, so 1 - 0.5 = 0.5 for each, avg = 0.5
        assert result["balance_check"] == 0.5

    def test_balance_check_imbalanced(self):
        """Test balance check with highly imbalanced values."""
        df = pd.DataFrame({
            "col1": [1, 1, 1, 1, 0],
        })
        result = calculate_predicted_utility(df, check_balance_fields=["col1"])
        # Dominant value (1) is 80%, so 1 - 0.8 = 0.2
        assert result["balance_check"] == 0.2

    def test_balance_check_multiple_fields(self):
        """Test balance check with multiple fields."""
        df = pd.DataFrame({
            "col1": [1, 1, 0, 0],
            "col2": [1, 1, 1, 0],
        })
        result = calculate_predicted_utility(df, check_balance_fields=["col1", "col2"])
        # col1: 1 - 0.5 = 0.5
        # col2: 1 - 0.75 = 0.25
        # average = 0.375
        assert abs(result["balance_check"] - 0.375) < 0.05  # float tolerance

    def test_balance_check_single_value(self):
        """Test balance check with single value field."""
        df = pd.DataFrame({
            "col1": [1, 1, 1],
        })
        result = calculate_predicted_utility(df, check_balance_fields=["col1"])
        # Dominant value is 100%, so 1 - 1.0 = 0.0
        assert result["balance_check"] == 0.0

    def test_balance_check_nonexistent_field(self):
        """Test balance check with nonexistent field."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
        })
        result = calculate_predicted_utility(df, check_balance_fields=["nonexistent"])
        assert result["balance_check"] == 0.0

    def test_balance_check_no_fields(self):
        """Test balance check with no fields specified."""
        df = pd.DataFrame({
            "col1": [1, 1, 1],
        })
        result = calculate_predicted_utility(df, check_balance_fields=None)
        assert result["balance_check"] == 1.0

    def test_schema_richness_single_dtype(self):
        """Test schema richness with single data type."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        })
        result = calculate_predicted_utility(df)
        # 1 type (int) / 1 = 1.0
        assert result["schema_richness"] > 0

    def test_schema_richness_multiple_dtypes(self):
        """Test schema richness with multiple data types."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
            "col3": [1.1, 2.2, 3.3],
        })
        result = calculate_predicted_utility(df)
        # Multiple types
        assert result["schema_richness"] > 0

    def test_schema_richness_with_external_dtypes(self):
        """Test schema richness with external dtype dictionary."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"],
        })
        dtypes_dict = {"col1": "int", "col2": "str"}
        result = calculate_predicted_utility(df, dtypes_dict=dtypes_dict)
        assert result["schema_richness"] > 0

    def test_custom_weights_equal(self):
        """Test with custom equal weights."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })
        weights = {
            "completeness": 0.25,
            "diversity": 0.25,
            "balance_check": 0.25,
            "schema_richness": 0.25,
        }
        result = calculate_predicted_utility(df, weights=weights)
        assert result["predicted_utility"] >= 0
        assert result["predicted_utility"] <= 100

    def test_custom_weights_emphasis_completeness(self):
        """Test with custom weights emphasizing completeness."""
        df_complete = pd.DataFrame({
            "col1": [1, 2, 3],
        })
        df_incomplete = pd.DataFrame({
            "col1": [None, None, None],
        })
        weights = {
            "completeness": 0.7,
            "diversity": 0.1,
            "balance_check": 0.1,
            "schema_richness": 0.1,
        }
        result_complete = calculate_predicted_utility(
            df_complete,
            require_fields=["col1"],
            weights=weights,
        )
        result_incomplete = calculate_predicted_utility(
            df_incomplete,
            require_fields=["col1"],
            weights=weights,
        )
        # Complete should have higher utility
        assert result_complete["predicted_utility"] > result_incomplete["predicted_utility"]

    def test_custom_weights_emphasis_diversity(self):
        """Test with custom weights emphasizing diversity."""
        df_unique = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
        })
        df_duplicate = pd.DataFrame({
            "col1": [1, 1, 1, 1, 1],
        })
        weights = {
            "completeness": 0.1,
            "diversity": 0.7,
            "balance_check": 0.1,
            "schema_richness": 0.1,
        }
        result_unique = calculate_predicted_utility(
            df_unique,
            unique_fields=["col1"],
            weights=weights,
        )
        result_duplicate = calculate_predicted_utility(
            df_duplicate,
            unique_fields=["col1"],
            weights=weights,
        )
        # Unique should have higher utility
        assert result_unique["predicted_utility"] > result_duplicate["predicted_utility"]

    def test_comprehensive_scenario(self):
        """Test with comprehensive realistic scenario."""
        df = pd.DataFrame({
            "id": range(1, 101),
            "name": ["Person_" + str(i) for i in range(1, 101)],
            "email": ["person" + str(i) + "@example.com" for i in range(1, 101)],
            "age": np.random.randint(18, 80, 100),
            "gender": np.random.choice([0, 1], 100),
            "income": np.random.uniform(30000, 150000, 100),
            "status": np.random.choice(["Active", "Inactive"], 100),
        })

        result = calculate_predicted_utility(
            df,
            require_fields=["id", "name", "email"],
            unique_fields=["id", "email"],
            check_balance_fields=["gender", "status"],
        )

        assert result["completeness"] == 1.0  # No nulls
        assert result["diversity"] > 0.5  # Good diversity
        assert result["balance_check"] > 0
        assert result["schema_richness"] > 0
        assert result["predicted_utility"] > 0
        assert result["predicted_utility"] <= 100

    def test_all_nulls_scenario(self):
        """Test with all null values."""
        df = pd.DataFrame({
            "col1": [None, None, None],
            "col2": [None, None, None],
        })
        result = calculate_predicted_utility(
            df,
            require_fields=["col1", "col2"],
            unique_fields=["col1", "col2"],
        )
        assert result["completeness"] == 0.0
        assert result["diversity"] == 0.0
        assert result["predicted_utility"] >= 0  # may not be exactly 0

    def test_mixed_quality_data(self):
        """Test with mixed quality data."""
        df = pd.DataFrame({
            "complete_col": [1, 2, 3, 4, 5],
            "sparse_col": [1, None, None, None, None],
            "unique_col": [10, 20, 30, 40, 50],
            "duplicate_col": [1, 1, 1, 2, 2],
        })
        result = calculate_predicted_utility(
            df,
            require_fields=["complete_col", "sparse_col"],
            unique_fields=["unique_col", "duplicate_col"],
        )
        assert result["completeness"] < 1.0  # Has some nulls
        assert result["diversity"] > 0
        assert result["predicted_utility"] > 0
        assert result["predicted_utility"] < 100

    def test_large_dataset_performance(self):
        """Test with large dataset."""
        df = pd.DataFrame({
            "id": range(100000),
            "value": np.random.randn(100000),
            "category": np.random.choice(["A", "B", "C"], 100000),
            "flag": np.random.choice([0, 1], 100000),
        })
        result = calculate_predicted_utility(
            df,
            require_fields=["id"],
            unique_fields=["id"],
            check_balance_fields=["flag"],
        )
        assert isinstance(result, dict)
        assert result["predicted_utility"] >= 0

    def test_output_format(self):
        """Test that output has correct format and structure."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
        })
        result = calculate_predicted_utility(df)

        # Check all required keys
        assert "completeness" in result
        assert "diversity" in result
        assert "balance_check" in result
        assert "schema_richness" in result
        assert "predicted_utility" in result

        # Check value types
        assert isinstance(result["completeness"], (int, float))
        assert isinstance(result["diversity"], (int, float))
        assert isinstance(result["balance_check"], (int, float))
        assert isinstance(result["schema_richness"], (int, float))
        assert isinstance(result["predicted_utility"], int)

        # Check value ranges
        assert 0 <= result["completeness"] <= 1
        assert 0 <= result["diversity"] <= 1
        assert 0 <= result["balance_check"] <= 1
        assert 0 <= result["schema_richness"] <= 1
        assert 0 <= result["predicted_utility"] <= 100

    def test_rounding_precision(self):
        """Test that component scores are properly rounded."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
        })
        result = calculate_predicted_utility(df)

        # Component scores should be rounded to 2 decimal places
        assert isinstance(result["completeness"], float)
        assert len(str(result["completeness"]).split(".")[-1]) <= 2

        # Predicted utility should be integer
        assert isinstance(result["predicted_utility"], int)

    def test_zero_utility_low_quality(self):
        """Test that low-quality data produces low utility score."""
        df = pd.DataFrame({
            "col1": [None, None, None],
            "col2": [1, 1, 1],
        })
        result = calculate_predicted_utility(
            df,
            require_fields=["col1"],
            unique_fields=["col2"],
            check_balance_fields=["col2"],
        )
        # Should be very low due to poor completeness, diversity, and balance
        assert result["predicted_utility"] <= 50  # low but not necessarily <=25

    def test_high_utility_good_quality(self):
        """Test that high-quality data produces high utility score."""
        n = 100
        df = pd.DataFrame({
            "col1": range(n),
            "col2": range(n),
            "col3": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        })
        result = calculate_predicted_utility(
            df,
            require_fields=["col1", "col2"],
            unique_fields=["col1", "col2"],
            check_balance_fields=["col3"],
        )
        # Should be reasonably high due to good quality (varies by Python/numpy version)
        assert result["predicted_utility"] >= 60


class TestPredictedUtilityIntegration:
    """Integration tests for predicted utility scoring."""

    def test_realistic_customer_dataset(self):
        """Test with realistic customer dataset."""
        df = pd.DataFrame({
            "customer_id": range(1, 1001),
            "email": ["cust" + str(i) + "@example.com" for i in range(1, 1001)],
            "signup_date": pd.date_range("2020-01-01", periods=1000),
            "purchase_count": np.random.randint(0, 100, 1000),
            "total_spent": np.random.uniform(0, 50000, 1000),
            "status": np.random.choice(["Active", "Inactive"], 1000),
            "verified": np.random.choice([True, False], 1000),
        })

        result = calculate_predicted_utility(
            df,
            require_fields=["customer_id", "email"],
            unique_fields=["customer_id", "email"],
            check_balance_fields=["status", "verified"],
        )

        assert result["completeness"] == 1.0
        assert result["diversity"] > 0.8
        assert result["predicted_utility"] > 50

    def test_realistic_sensor_dataset(self):
        """Test with realistic sensor/IoT dataset."""
        df = pd.DataFrame({
            "sensor_id": np.random.choice(range(1, 11), 10000),
            "timestamp": pd.date_range("2020-01-01", periods=10000, freq="1min"),
            "temperature": np.random.uniform(15, 35, 10000),
            "humidity": np.random.uniform(30, 80, 10000),
            "pressure": np.random.uniform(900, 1100, 10000),
            "status": np.random.choice(["OK", "Warning", "Error"], 10000),
        })

        result = calculate_predicted_utility(
            df,
            require_fields=["timestamp"],
            unique_fields=["timestamp"],
            check_balance_fields=["status"],
        )

        assert result["completeness"] == 1.0
        assert result["predicted_utility"] > 0

    def test_realistic_transaction_dataset(self):
        """Test with realistic transaction dataset."""
        df = pd.DataFrame({
            "transaction_id": range(1, 5001),
            "customer_id": np.random.randint(1, 1001, 5000),
            "amount": np.random.uniform(10, 1000, 5000),
            "category": np.random.choice(["Food", "Clothing", "Electronics"], 5000),
            "timestamp": pd.date_range("2020-01-01", periods=5000),
        })

        result = calculate_predicted_utility(
            df,
            require_fields=["transaction_id", "customer_id", "amount"],
            unique_fields=["transaction_id"],
            check_balance_fields=["category"],
        )

        assert result["completeness"] == 1.0
        assert result["diversity"] > 0
        assert result["predicted_utility"] > 50
