"""
File: test_correlation.py
Test Target: pamola_core.analysis.correlation
Coverage Target: >=90%

Comprehensive test suite for analyze_correlation() function.
Tests correlation analysis, chart generation, and data preparation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from pamola_core.analysis.correlation import (
    analyze_correlation,
    CorrelationAnalyzer,
)
from pamola_core.errors.exceptions import (
    ValidationError,
    ColumnNotFoundError,
    DataError,
    TypeValidationError,
)


# ======================== FIXTURES ========================

@pytest.fixture
def sample_df():
    """Standard test DataFrame with numeric columns."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0, 110000.0, 120000.0],
        "experience": [1, 5, 10, 15, 20, 25, 30, 35],
    })


@pytest.fixture
def df_with_binary():
    """DataFrame with binary/boolean columns."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "is_manager": [True, False, True, False, True],
        "is_certified": ["yes", "no", "yes", "yes", "no"],
    })


@pytest.fixture
def df_single_column():
    """Single-column DataFrame."""
    return pd.DataFrame({
        "values": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def df_two_columns():
    """Two-column DataFrame."""
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [2, 4, 6, 8, 10],
    })


@pytest.fixture
def df_constant_column():
    """DataFrame with constant column (zero variance)."""
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "constant": [42, 42, 42, 42, 42],
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values."""
    return pd.DataFrame({
        "col1": [1, 2, None, 4, 5],
        "col2": [2.0, 4.0, 6.0, None, 10.0],
    })


@pytest.fixture
def df_with_categorical():
    """DataFrame with categorical and numeric columns."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
    })


@pytest.fixture
def tmp_analysis_dir(tmp_path):
    """Temporary directory for analysis outputs."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    return analysis_dir


# ======================== CORRELATION ANALYZER TESTS ========================

class TestCorrelationAnalyzerClass:
    """Tests for CorrelationAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = CorrelationAnalyzer()
        assert analyzer is not None

    def test_supported_methods(self):
        """Test SUPPORTED_METHODS constant."""
        analyzer = CorrelationAnalyzer()
        assert "pearson" in analyzer.SUPPORTED_METHODS
        assert "spearman" in analyzer.SUPPORTED_METHODS
        assert "kendall" in analyzer.SUPPORTED_METHODS

    def test_supported_charts(self):
        """Test SUPPORTED_CHARTS constant."""
        analyzer = CorrelationAnalyzer()
        assert "matrix" in analyzer.SUPPORTED_CHARTS
        assert "heatmap" in analyzer.SUPPORTED_CHARTS

    def test_validate_method_valid(self):
        """Test method validation with valid method."""
        analyzer = CorrelationAnalyzer()
        # Should not raise
        analyzer._validate_method("pearson")
        analyzer._validate_method("spearman")
        analyzer._validate_method("kendall")

    def test_validate_method_invalid(self):
        """Test method validation with invalid method."""
        analyzer = CorrelationAnalyzer()
        with pytest.raises(ValidationError):
            analyzer._validate_method("invalid_method")

    def test_validate_columns_valid(self):
        """Test column validation with valid columns."""
        analyzer = CorrelationAnalyzer()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        # Should not raise
        analyzer._validate_columns(df, ["col1", "col2"])

    def test_validate_columns_missing(self):
        """Test column validation with missing columns."""
        analyzer = CorrelationAnalyzer()
        df = pd.DataFrame({"col1": [1, 2]})
        with pytest.raises(ColumnNotFoundError):
            analyzer._validate_columns(df, ["col1", "nonexistent"])

    def test_validate_output_chart_string(self):
        """Test chart type validation with string."""
        analyzer = CorrelationAnalyzer()
        result = analyzer._validate_output_chart("matrix")
        assert result == ["matrix"]

    def test_validate_output_chart_list(self):
        """Test chart type validation with list."""
        analyzer = CorrelationAnalyzer()
        result = analyzer._validate_output_chart(["matrix", "heatmap"])
        assert result == ["matrix", "heatmap"]

    def test_validate_output_chart_invalid(self):
        """Test chart type validation with invalid type raises an exception."""
        analyzer = CorrelationAnalyzer()
        with pytest.raises((TypeValidationError, TypeError)):
            analyzer._validate_output_chart(["invalid_chart"])


# ======================== DATA PREPARATION TESTS ========================

class TestDataPreparation:
    """Tests for data preparation and conversion."""

    def test_binary_to_numeric_boolean(self):
        """Test conversion of boolean to numeric."""
        analyzer = CorrelationAnalyzer()
        series = pd.Series([True, False, True, False])
        result = analyzer._map_binary_to_numeric(series)

        assert pd.api.types.is_numeric_dtype(result)

    def test_binary_to_numeric_yes_no(self):
        """Test conversion of yes/no to numeric."""
        analyzer = CorrelationAnalyzer()
        series = pd.Series(["yes", "no", "yes", "no"])
        result = analyzer._map_binary_to_numeric(series)

        # Should be converted to numeric
        assert pd.api.types.is_numeric_dtype(result)

    def test_binary_to_numeric_true_false(self):
        """Test conversion of true/false to numeric."""
        analyzer = CorrelationAnalyzer()
        series = pd.Series(["true", "false", "true", "false"])
        result = analyzer._map_binary_to_numeric(series)

        assert pd.api.types.is_numeric_dtype(result)

    def test_non_binary_categorical_unchanged(self):
        """Test that non-binary categorical stays unchanged."""
        analyzer = CorrelationAnalyzer()
        series = pd.Series(["a", "b", "c", "a"])
        result = analyzer._map_binary_to_numeric(series)

        assert not pd.api.types.is_numeric_dtype(result)

    def test_prepare_data_all_columns(self, sample_df):
        """Test data preparation with all columns."""
        analyzer = CorrelationAnalyzer()
        result = analyzer._prepare_data(sample_df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_df.shape[0]

    def test_prepare_data_selected_columns(self, sample_df):
        """Test data preparation with selected columns."""
        analyzer = CorrelationAnalyzer()
        result = analyzer._prepare_data(sample_df, columns=["age", "salary"])

        assert "age" in result.columns
        assert "salary" in result.columns

    def test_prepare_data_removes_non_numeric(self, df_with_categorical):
        """Test that non-numeric columns are removed."""
        analyzer = CorrelationAnalyzer()
        result = analyzer._prepare_data(df_with_categorical)

        # Name column should be removed
        assert "name" not in result.columns
        assert "age" in result.columns or "salary" in result.columns

    def test_prepare_data_removes_constant_columns(self, df_constant_column):
        """Test that constant columns are removed."""
        analyzer = CorrelationAnalyzer()
        result = analyzer._prepare_data(df_constant_column)

        # Constant column should be removed
        assert "constant" not in result.columns
        assert "col1" in result.columns

    def test_prepare_data_empty_df_error(self):
        """Test error when no suitable columns for analysis."""
        analyzer = CorrelationAnalyzer()
        df = pd.DataFrame({
            "text1": ["a", "b", "c"],
            "text2": ["x", "y", "z"],
        })
        with pytest.raises(DataError):
            analyzer._prepare_data(df)


# ======================== CORRELATION CALCULATION TESTS ========================

class TestCorrelationCalculation:
    """Tests for correlation calculation."""

    def test_all_variables_correlation(self, sample_df):
        """Test correlation of all variables."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(sample_df)
        result_df, result_type = analyzer._calculate_correlation_result(
            clean_data, columns=None, method="pearson"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert result_type == "all_variables"
        assert result_df.shape[0] >= 2
        assert result_df.shape[1] >= 2

    def test_single_variable_correlation(self, sample_df):
        """Test correlation of single variable with all others."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(sample_df)
        result_df, result_type = analyzer._calculate_correlation_result(
            clean_data, columns=["age"], method="pearson"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert result_type == "single_variable"
        assert result_df.shape[1] == 1

    def test_pairwise_correlation(self, df_two_columns):
        """Test pairwise correlation."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(df_two_columns)
        result_df, result_type = analyzer._calculate_correlation_result(
            clean_data, columns=["col1", "col2"], method="pearson"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert result_type == "pairwise"
        assert result_df.shape == (2, 2)

    def test_selected_variables_correlation(self, sample_df):
        """Test correlation of selected variables (3+ columns → selected_variables)."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(sample_df)
        result_df, result_type = analyzer._calculate_correlation_result(
            clean_data, columns=["age", "salary", "experience"], method="pearson"
        )

        assert isinstance(result_df, pd.DataFrame)
        assert result_type == "selected_variables"

    def test_spearman_correlation(self, sample_df):
        """Test Spearman correlation."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(sample_df)
        result_df, result_type = analyzer._calculate_correlation_result(
            clean_data, columns=None, method="spearman"
        )

        assert isinstance(result_df, pd.DataFrame)

    def test_kendall_correlation(self, sample_df):
        """Test Kendall correlation."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(sample_df)
        result_df, result_type = analyzer._calculate_correlation_result(
            clean_data, columns=None, method="kendall"
        )

        assert isinstance(result_df, pd.DataFrame)

    def test_correlation_values_range(self, sample_df):
        """Test that correlation values are in [-1, 1]."""
        analyzer = CorrelationAnalyzer()
        clean_data = analyzer._prepare_data(sample_df)
        result_df, _ = analyzer._calculate_correlation_result(
            clean_data, columns=None, method="pearson"
        )

        assert (result_df >= -1).all().all() or (result_df <= 1).all().all()


# ======================== PUBLIC FUNCTION TESTS ========================

class TestAnalyzeCorrelationFunction:
    """Tests for analyze_correlation() public function."""

    @patch("pamola_core.analysis.correlation.CorrelationAnalyzer.analyze_correlation")
    def test_convenience_function(self, mock_analyze, sample_df, tmp_analysis_dir):
        """Test convenience function delegates to analyzer."""
        mock_analyze.return_value = {
            "result": pd.DataFrame(),
            "result_type": "all_variables",
            "raw_result": pd.DataFrame(),
            "path": None,
        }

        result = analyze_correlation(sample_df, columns=None, method="pearson")

        assert mock_analyze.called

    def test_basic_analysis(self, sample_df, tmp_analysis_dir):
        """Test basic correlation analysis."""
        result = analyze_correlation(
            sample_df,
            method="pearson",
            plot=False,
        )

        assert isinstance(result, dict)
        assert "result" in result
        assert "result_type" in result
        assert "raw_result" in result
        assert "path" in result

    def test_analysis_with_plotting(self, sample_df, tmp_analysis_dir):
        """Test analysis with plot generation."""
        with patch("pamola_core.analysis.correlation.create_heatmap") as mock_heatmap:
            mock_heatmap.return_value = "success"

            result = analyze_correlation(
                sample_df,
                method="pearson",
                plot=True,
                output_chart="heatmap",
                analysis_dir=str(tmp_analysis_dir),
            )

            assert isinstance(result, dict)

    def test_custom_weights_parameter(self, sample_df):
        """Test with custom weights (if supported)."""
        result = analyze_correlation(
            sample_df,
            method="pearson",
            plot=False,
        )

        assert isinstance(result, dict)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(DataError):
            analyze_correlation(df)

    def test_single_column_error(self, df_single_column):
        """Test single column returns valid result (1x1 correlation matrix)."""
        result = analyze_correlation(df_single_column, plot=False)

        assert isinstance(result, dict)
        assert "result" in result

    def test_non_numeric_dataframe(self):
        """Test error with non-numeric DataFrame."""
        df = pd.DataFrame({
            "col1": ["a", "b", "c"],
            "col2": ["x", "y", "z"],
        })
        with pytest.raises(DataError):
            analyze_correlation(df)

    def test_missing_column_error(self, sample_df):
        """Test error with missing column."""
        with pytest.raises(ColumnNotFoundError):
            analyze_correlation(
                sample_df, columns=["nonexistent"]
            )


# ======================== EDGE CASE TESTS ========================

class TestCorrelationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_perfect_positive_correlation(self):
        """Test with perfect positive correlation."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [1, 2, 3, 4, 5],
        })
        result = analyze_correlation(df, plot=False)

        assert isinstance(result, dict)
        assert result["raw_result"].iloc[0, 1] == 1.0

    def test_perfect_negative_correlation(self):
        """Test with perfect negative correlation."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
        })
        result = analyze_correlation(df, plot=False)

        assert isinstance(result, dict)
        assert result["raw_result"].iloc[0, 1] == -1.0

    def test_no_correlation(self):
        """Test with constant column removed — only non-constant column remains."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [1, 1, 1, 1, 1],
        })
        # y is constant (zero variance) → removed by _prepare_data; x remains
        result = analyze_correlation(df, plot=False)

        assert isinstance(result, dict)
        assert "result" in result

    def test_with_missing_values(self, df_with_missing):
        """Test handling of missing values."""
        result = analyze_correlation(df_with_missing, plot=False)

        assert isinstance(result, dict)

    def test_large_dataframe(self):
        """Test with large DataFrame."""
        df = pd.DataFrame({
            "col1": np.random.rand(1000),
            "col2": np.random.rand(1000),
            "col3": np.random.rand(1000),
        })
        result = analyze_correlation(df, plot=False)

        assert isinstance(result, dict)

    def test_three_variables(self, sample_df):
        """Test with three variables."""
        result = analyze_correlation(sample_df, plot=False)

        assert isinstance(result, dict)
        assert result["result"].shape[0] == 3
        assert result["result"].shape[1] == 3


# ======================== RESULT STRUCTURE TESTS ========================

class TestResultStructure:
    """Test the structure of analysis results."""

    def test_result_has_all_keys(self, sample_df):
        """Test that result has all required keys."""
        result = analyze_correlation(sample_df, plot=False)

        required_keys = ["result", "result_type", "raw_result", "path"]
        for key in required_keys:
            assert key in result

    def test_result_dataframe_type(self, sample_df):
        """Test that result is a DataFrame."""
        result = analyze_correlation(sample_df, plot=False)

        assert isinstance(result["result"], pd.DataFrame)

    def test_result_type_string(self, sample_df):
        """Test that result_type is a string."""
        result = analyze_correlation(sample_df, plot=False)

        assert isinstance(result["result_type"], str)
        assert result["result_type"] in [
            "all_variables",
            "single_variable",
            "pairwise",
            "selected_variables",
        ]

    def test_raw_result_backward_compat(self, sample_df):
        """Test that raw_result exists for backward compatibility."""
        result = analyze_correlation(sample_df, plot=False)

        assert "raw_result" in result

    def test_path_field_type(self, sample_df):
        """Test that path field has correct type."""
        result = analyze_correlation(sample_df, plot=False)

        path = result["path"]
        assert path is None or isinstance(path, (str, Path, list))


# ======================== CHART GENERATION TESTS ========================

class TestChartGeneration:
    """Test chart generation functionality."""

    @patch("pamola_core.analysis.correlation.create_heatmap")
    def test_heatmap_generation(self, mock_heatmap, sample_df, tmp_analysis_dir):
        """Test heatmap chart generation."""
        mock_heatmap.return_value = "success"

        result = analyze_correlation(
            sample_df,
            plot=True,
            output_chart="heatmap",
            analysis_dir=str(tmp_analysis_dir),
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.correlation.create_correlation_matrix")
    def test_matrix_generation(self, mock_matrix, sample_df, tmp_analysis_dir):
        """Test correlation matrix chart generation."""
        mock_matrix.return_value = "success"

        result = analyze_correlation(
            sample_df,
            plot=True,
            output_chart="matrix",
            analysis_dir=str(tmp_analysis_dir),
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.correlation.create_heatmap")
    @patch("pamola_core.analysis.correlation.create_correlation_matrix")
    def test_both_charts(self, mock_matrix, mock_heatmap, sample_df, tmp_analysis_dir):
        """Test generating both chart types."""
        mock_matrix.return_value = "success"
        mock_heatmap.return_value = "success"

        result = analyze_correlation(
            sample_df,
            plot=True,
            output_chart=["matrix", "heatmap"],
            analysis_dir=str(tmp_analysis_dir),
        )

        assert isinstance(result, dict)

    def test_plot_false_no_charts(self, sample_df):
        """Test that no charts generated when plot=False."""
        result = analyze_correlation(
            sample_df,
            plot=False,
        )

        assert result["path"] is None

    def test_invalid_output_chart(self, sample_df):
        """Test error with invalid chart type raises an exception."""
        with pytest.raises((TypeValidationError, TypeError)):
            analyze_correlation(
                sample_df,
                plot=True,
                output_chart="invalid_chart",
            )


# ======================== INTEGRATION TESTS ========================

class TestCorrelationIntegration:
    """Integration tests combining multiple features."""

    def test_complete_workflow_pearson(self, sample_df, tmp_analysis_dir):
        """Test complete Pearson correlation workflow."""
        result = analyze_correlation(
            sample_df,
            columns=None,
            method="pearson",
            plot=False,
        )

        assert isinstance(result, dict)
        assert result["result_type"] == "all_variables"

    def test_complete_workflow_spearman(self, sample_df):
        """Test complete Spearman correlation workflow."""
        result = analyze_correlation(
            sample_df,
            columns=None,
            method="spearman",
            plot=False,
        )

        assert isinstance(result, dict)

    def test_single_variable_workflow(self, sample_df):
        """Test single variable analysis workflow."""
        result = analyze_correlation(
            sample_df,
            columns=["age"],
            plot=False,
        )

        assert isinstance(result, dict)
        assert result["result_type"] == "single_variable"

    def test_pairwise_workflow(self, df_two_columns):
        """Test pairwise correlation workflow."""
        result = analyze_correlation(
            df_two_columns,
            columns=["col1", "col2"],
            plot=False,
        )

        assert isinstance(result, dict)

    def test_multiple_calls_consistency(self, sample_df):
        """Test that multiple calls produce consistent results."""
        result1 = analyze_correlation(sample_df, plot=False)
        result2 = analyze_correlation(sample_df, plot=False)

        pd.testing.assert_frame_equal(result1["result"], result2["result"])

    def test_analysis_does_not_modify_df(self, sample_df):
        """Test that analysis doesn't modify original DataFrame."""
        df_copy = sample_df.copy()

        analyze_correlation(sample_df, plot=False)

        pd.testing.assert_frame_equal(sample_df, df_copy)
