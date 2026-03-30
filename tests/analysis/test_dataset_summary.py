"""
File: test_dataset_summary.py
Test Target: pamola_core.analysis.dataset_summary
Coverage Target: >=90%

Comprehensive test suite for analyze_dataset_summary() function.
Tests dataset analysis including type detection, missing values, and outliers.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pamola_core.analysis.dataset_summary import analyze_dataset_summary, DatasetAnalyzer


# ======================== FIXTURES ========================

@pytest.fixture
def sample_df():
    """Standard test DataFrame with mixed types."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "active": [True, False, True, False, True],
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values."""
    return pd.DataFrame({
        "col1": [1, 2, None, 4, 5],
        "col2": ["a", None, "c", "d", "e"],
        "col3": [10.5, 20.5, 30.5, None, 50.5],
    })


@pytest.fixture
def df_with_numeric_strings():
    """DataFrame with numeric strings in object columns."""
    return pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "price": ["100.5", "200.5", "300.5", "400.5", "500.5"],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    })


@pytest.fixture
def df_single_row():
    """Single-row DataFrame for edge case testing."""
    return pd.DataFrame({
        "col1": [42],
        "col2": ["single"],
        "col3": [3.14],
    })


@pytest.fixture
def df_single_column():
    """Single-column DataFrame for edge case testing."""
    return pd.DataFrame({"value": [1, 2, 3, 4, 5]})


@pytest.fixture
def df_all_nulls():
    """DataFrame with all null values in some columns."""
    return pd.DataFrame({
        "col1": [None, None, None],
        "col2": [1, 2, 3],
        "col3": [None, None, None],
    })


@pytest.fixture
def df_with_outliers():
    """DataFrame with outliers for outlier detection testing."""
    return pd.DataFrame({
        "values": [1, 2, 3, 4, 5, 100, 150],  # 100, 150 are outliers
    })


# ======================== SUCCESS PATH TESTS ========================

class TestAnalyzeDatesetSummarySuccess:
    """Test successful analysis paths."""

    def test_basic_analysis(self, sample_df):
        """Test basic analysis with valid DataFrame."""
        result = analyze_dataset_summary(sample_df)

        assert isinstance(result, dict)
        assert result["rows"] == 5
        assert result["columns"] == 5
        assert result["total_cells"] == 25
        assert "missing_values" in result
        assert "numeric_fields" in result
        assert "categorical_fields" in result
        assert "outliers" in result

    def test_result_structure(self, sample_df):
        """Test result has all required keys and correct types."""
        result = analyze_dataset_summary(sample_df)

        assert isinstance(result["rows"], int)
        assert isinstance(result["columns"], int)
        assert isinstance(result["total_cells"], int)
        assert isinstance(result["missing_values"], dict)
        assert "value" in result["missing_values"]
        assert "fields_with_missing" in result["missing_values"]
        assert isinstance(result["numeric_fields"], dict)
        assert "count" in result["numeric_fields"]
        assert "percentage" in result["numeric_fields"]
        assert isinstance(result["categorical_fields"], dict)
        assert "count" in result["categorical_fields"]
        assert "percentage" in result["categorical_fields"]
        assert isinstance(result["outliers"], dict)
        assert "count" in result["outliers"]
        assert "affected_fields" in result["outliers"]

    def test_numeric_detection(self, df_with_numeric_strings):
        """Test automatic numeric-like detection in object columns."""
        result = analyze_dataset_summary(df_with_numeric_strings)

        # "id" and "price" should be detected as numeric
        assert result["numeric_fields"]["count"] >= 2

    def test_missing_values_detection(self, df_with_missing):
        """Test detection of missing values."""
        result = analyze_dataset_summary(df_with_missing)

        assert result["missing_values"]["value"] > 0
        assert result["missing_values"]["fields_with_missing"] == 3

    def test_no_missing_values(self, sample_df):
        """Test when DataFrame has no missing values."""
        result = analyze_dataset_summary(sample_df)

        assert result["missing_values"]["value"] == 0
        assert result["missing_values"]["fields_with_missing"] == 0

    def test_percentage_calculation(self):
        """Test that percentages are calculated correctly with numeric and categorical cols."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        })
        result = analyze_dataset_summary(df)

        numeric_pct = result["numeric_fields"]["percentage"]
        categorical_pct = result["categorical_fields"]["percentage"]

        # 3 numeric (id, age, salary) + 1 categorical (name) = 4 cols, sum = 1.0
        assert abs(numeric_pct + categorical_pct - 1.0) < 0.01


# ======================== EDGE CASE TESTS ========================

class TestAnalyzeDatesetSummaryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_df(self, df_single_row):
        """Test analysis of single-row DataFrame."""
        result = analyze_dataset_summary(df_single_row)

        assert result["rows"] == 1
        assert result["columns"] == 3

    def test_single_column_df(self, df_single_column):
        """Test analysis of single-column DataFrame."""
        result = analyze_dataset_summary(df_single_column)

        assert result["columns"] == 1
        assert result["rows"] == 5

    def test_all_nulls(self, df_all_nulls):
        """Test DataFrame where some columns are entirely null."""
        result = analyze_dataset_summary(df_all_nulls)

        assert result["missing_values"]["value"] == 6
        assert result["missing_values"]["fields_with_missing"] == 2

    def test_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = analyze_dataset_summary(df)

        assert result["rows"] == 0
        assert result["columns"] == 0
        assert result["total_cells"] == 0

    def test_df_numeric_threshold(self):
        """Test numeric detection with threshold boundaries."""
        # Create DataFrame where conversion is at threshold (75%)
        df = pd.DataFrame({
            "mixed_col": ["1", "2", "3", "4", "not_numeric"],
        })
        result = analyze_dataset_summary(df)
        # 4 out of 5 = 80% > 75%, should be detected as numeric
        assert result["numeric_fields"]["count"] >= 1

    def test_all_categorical(self):
        """Test DataFrame with all categorical columns."""
        df = pd.DataFrame({
            "col1": ["a", "b", "c"],
            "col2": ["x", "y", "z"],
            "col3": ["p", "q", "r"],
        })
        result = analyze_dataset_summary(df)

        assert result["numeric_fields"]["count"] == 0
        assert result["categorical_fields"]["count"] == 3

    def test_all_numeric(self):
        """Test DataFrame with all numeric columns."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4.5, 5.5, 6.5],
            "col3": [7, 8, 9],
        })
        result = analyze_dataset_summary(df)

        assert result["numeric_fields"]["count"] == 3
        assert result["categorical_fields"]["count"] == 0

    def test_large_df(self):
        """Test with large DataFrame."""
        df = pd.DataFrame({
            "col1": range(10000),
            "col2": ["val"] * 10000,
            "col3": np.random.rand(10000),
        })
        result = analyze_dataset_summary(df)

        assert result["rows"] == 10000
        assert result["columns"] == 3
        assert result["total_cells"] == 30000


# ======================== ERROR HANDLING TESTS ========================

class TestAnalyzeDatesetSummaryErrors:
    """Test error handling and exception scenarios."""

    def test_none_input(self):
        """Test with None input raises appropriate error."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            analyze_dataset_summary(None)

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            analyze_dataset_summary("not a dataframe")

    def test_invalid_input_list(self):
        """Test with list instead of DataFrame."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            analyze_dataset_summary([1, 2, 3])


# ======================== ANALYZER CLASS TESTS ========================

class TestDatasetAnalyzerClass:
    """Test DatasetAnalyzer class directly."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = DatasetAnalyzer(numeric_threshold=0.8)
        assert analyzer.numeric_threshold == 0.8

    def test_analyzer_threshold_clamping(self):
        """Test that thresholds are clamped to [0, 1]."""
        analyzer1 = DatasetAnalyzer(numeric_threshold=-0.5)
        assert analyzer1.numeric_threshold == 0.0

        analyzer2 = DatasetAnalyzer(numeric_threshold=1.5)
        assert analyzer2.numeric_threshold == 1.0

    def test_analyzer_method(self, sample_df):
        """Test calling analyzer.analyze_dataset_summary()."""
        analyzer = DatasetAnalyzer()
        result = analyzer.analyze_dataset_summary(sample_df)

        assert isinstance(result, dict)
        assert "rows" in result

    def test_analyzer_with_custom_threshold(self):
        """Test analyzer with custom numeric threshold."""
        df = pd.DataFrame({
            "col": ["1", "2", "3", "not_num"],
        })

        # With 75% threshold, 75% (3/4) converts = detected
        analyzer_strict = DatasetAnalyzer(numeric_threshold=0.9)
        result_strict = analyzer_strict.analyze_dataset_summary(df)
        # 75% < 90%, should NOT be detected
        assert result_strict["numeric_fields"]["count"] == 0

    def test_analyzer_logger_setup(self):
        """Test that analyzer has logger configured."""
        analyzer = DatasetAnalyzer()
        assert analyzer.logger is not None


# ======================== OUTLIER DETECTION MOCKING ========================

class TestOutlierDetection:
    """Test outlier detection integration."""

    @patch("pamola_core.profiling.commons.statistical_analysis.detect_outliers_iqr")
    def test_outlier_detection_called(self, mock_detect, sample_df):
        """Test that outlier detection is invoked."""
        mock_detect.return_value = {"count": 2}

        result = analyze_dataset_summary(sample_df)

        # Should have called outlier detection for numeric columns
        assert mock_detect.called

    @patch("pamola_core.profiling.commons.statistical_analysis.detect_outliers_iqr")
    def test_outliers_reported(self, mock_detect, df_with_outliers):
        """Test outliers are reported in results."""
        mock_detect.return_value = {"count": 2}

        result = analyze_dataset_summary(df_with_outliers)

        assert isinstance(result["outliers"], dict)
        assert "count" in result["outliers"]
        assert "affected_fields" in result["outliers"]

    @patch("pamola_core.profiling.commons.statistical_analysis.detect_outliers_iqr")
    def test_outlier_detection_error_handling(self, mock_detect, sample_df):
        """Test graceful handling when outlier detection fails."""
        mock_detect.side_effect = Exception("Detection failed")

        # Should not raise, should handle gracefully
        result = analyze_dataset_summary(sample_df)
        assert isinstance(result, dict)
        assert "outliers" in result


# ======================== NUMERIC CONVERSION TESTS ========================

class TestNumericConversion:
    """Test numeric conversion in object columns."""

    def test_full_numeric_conversion(self):
        """Test column that converts fully to numeric."""
        df = pd.DataFrame({
            "numeric_str": ["1", "2", "3", "4", "5"],
        })
        result = analyze_dataset_summary(df)

        # Should be detected as numeric (100% conversion)
        assert result["numeric_fields"]["count"] == 1

    def test_partial_numeric_conversion(self):
        """Test column with partial numeric conversion."""
        df = pd.DataFrame({
            "mixed": ["1", "2", "3", "text", "text"],
        })
        result = analyze_dataset_summary(df)

        # 3/5 = 60% < 75% default threshold, should NOT be numeric
        assert result["numeric_fields"]["count"] == 0

    def test_no_numeric_conversion(self):
        """Test column that cannot convert to numeric."""
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "e"],
        })
        result = analyze_dataset_summary(df)

        assert result["numeric_fields"]["count"] == 0
        assert result["categorical_fields"]["count"] == 1

    def test_conversion_with_nans(self):
        """Test numeric conversion with NaN values."""
        df = pd.DataFrame({
            "with_nan": ["1", "2", None, "4", "5"],
        })
        result = analyze_dataset_summary(df)

        # 4/4 non-null values convert = 100%, should be numeric
        assert result["numeric_fields"]["count"] >= 1


# ======================== PERCENTAGE CALCULATION TESTS ========================

class TestPercentageCalculations:
    """Test percentage calculations for field types."""

    def test_percentage_sum_equals_one(self):
        """Test that numeric + categorical percentages sum to 1 (no bool cols)."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        })
        result = analyze_dataset_summary(df)

        total_pct = (
            result["numeric_fields"]["percentage"]
            + result["categorical_fields"]["percentage"]
        )
        assert abs(total_pct - 1.0) < 0.01

    def test_zero_columns(self):
        """Test percentage calculation with zero columns."""
        df = pd.DataFrame()
        result = analyze_dataset_summary(df)

        # Should handle division by zero gracefully
        assert result["numeric_fields"]["percentage"] == 0.0
        assert result["categorical_fields"]["percentage"] == 0.0

    def test_percentage_values_range(self, sample_df):
        """Test that percentages are in [0, 1] range."""
        result = analyze_dataset_summary(sample_df)

        assert 0.0 <= result["numeric_fields"]["percentage"] <= 1.0
        assert 0.0 <= result["categorical_fields"]["percentage"] <= 1.0


# ======================== INTEGRATION TESTS ========================

class TestDatasetSummaryIntegration:
    """Integration tests combining multiple features."""

    def test_complete_workflow(self, sample_df):
        """Test complete analysis workflow."""
        result = analyze_dataset_summary(sample_df)

        # Verify all major components executed
        assert result["rows"] > 0
        assert result["columns"] > 0
        assert "missing_values" in result
        assert "numeric_fields" in result
        assert "categorical_fields" in result
        assert "outliers" in result

    def test_multiple_calls_consistency(self, sample_df):
        """Test that multiple calls on same data are consistent."""
        result1 = analyze_dataset_summary(sample_df)
        result2 = analyze_dataset_summary(sample_df)

        assert result1["rows"] == result2["rows"]
        assert result1["columns"] == result2["columns"]
        assert result1["numeric_fields"]["count"] == result2["numeric_fields"]["count"]

    def test_modified_df_different_results(self, sample_df):
        """Test that modifying DataFrame changes results."""
        result1 = analyze_dataset_summary(sample_df)

        # Add null to change missing value count
        sample_df.loc[0, "name"] = None
        result2 = analyze_dataset_summary(sample_df)

        assert result2["missing_values"]["value"] > result1["missing_values"]["value"]
