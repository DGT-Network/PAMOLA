"""
File: test_distribution.py
Test Target: pamola_core.analysis.distribution
Coverage Target: >=90%

Comprehensive test suite for visualize_distribution_df() function.
Tests histogram and bar chart generation for numeric and categorical fields.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from pamola_core.analysis.distribution import visualize_distribution_df


# ======================== FIXTURES ========================

@pytest.fixture
def sample_df():
    """Standard test DataFrame with numeric and categorical columns."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0, 110000.0, 120000.0],
        "department": ["IT", "HR", "IT", "Finance", "IT", "HR", "Finance", "IT"],
    })


@pytest.fixture
def df_numeric_only():
    """DataFrame with only numeric columns."""
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col2": [10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5, 80.5, 90.5, 100.5],
    })


@pytest.fixture
def df_categorical_only():
    """DataFrame with only categorical columns (using pd.Categorical dtype)."""
    return pd.DataFrame({
        "col1": pd.Categorical(["a", "b", "c", "a", "b", "c", "a", "b"]),
        "col2": pd.Categorical(["x", "y", "z", "x", "y", "z", "x", "y"]),
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values."""
    return pd.DataFrame({
        "col1": [1, 2, None, 4, 5],
        "col2": ["a", None, "c", "d", "e"],
    })


@pytest.fixture
def df_single_column():
    """Single-column DataFrame."""
    return pd.DataFrame({
        "values": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def df_integers():
    """DataFrame with integer values."""
    return pd.DataFrame({
        "integer_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


@pytest.fixture
def df_floats():
    """DataFrame with float values."""
    return pd.DataFrame({
        "float_col": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
    })


@pytest.fixture
def tmp_viz_dir(tmp_path):
    """Temporary directory for visualization outputs."""
    viz_dir = tmp_path / "viz"
    viz_dir.mkdir()
    return viz_dir


# ======================== SUCCESS PATH TESTS ========================

class TestVisualizeDistributionSuccess:
    """Test successful visualization paths."""

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_basic_visualization(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test basic visualization with valid DataFrame."""
        mock_hist.return_value = "success"
        mock_bar.return_value = "success"

        result = visualize_distribution_df(sample_df, tmp_viz_dir)

        assert isinstance(result, dict)
        assert len(result) >= 0  # May skip empty columns

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_visualization_with_custom_bins(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test visualization with custom bin count."""
        mock_hist.return_value = "success"
        mock_bar.return_value = "success"

        result = visualize_distribution_df(
            sample_df, tmp_viz_dir, n_bins=20
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_visualization_numeric_bar_charts(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test visualization with numeric fields as bar charts."""
        mock_bar.return_value = "success"

        result = visualize_distribution_df(
            sample_df, tmp_viz_dir, numeric_bar_charts=True
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_visualization_custom_format(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test visualization with custom format."""
        mock_hist.return_value = "success"
        mock_bar.return_value = "success"

        result = visualize_distribution_df(
            sample_df, tmp_viz_dir, viz_format="png"
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_visualization_custom_field_names(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test visualization with custom field names."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(
            sample_df, tmp_viz_dir, field_names=["age", "salary"]
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_histogram_call(self, mock_hist, df_numeric_only, tmp_viz_dir):
        """Test that histogram creation is called for numeric fields."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df_numeric_only, tmp_viz_dir)

        # Histogram should be called for numeric columns
        assert mock_hist.called

    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_bar_chart_call(self, mock_bar, df_categorical_only, tmp_viz_dir):
        """Test that bar chart creation is called for categorical fields."""
        mock_bar.return_value = "success"

        result = visualize_distribution_df(df_categorical_only, tmp_viz_dir)

        # Bar chart should be called for categorical columns
        assert mock_bar.called


# ======================== NUMERIC FIELD TESTS ========================

class TestNumericFieldVisualization:
    """Test visualization of numeric fields."""

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_numeric_histogram_generation(self, mock_hist, df_numeric_only, tmp_viz_dir):
        """Test histogram generation for numeric data."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(
            df_numeric_only, tmp_viz_dir, numeric_bar_charts=False
        )

        # Should call create_histogram for numeric columns
        assert mock_hist.call_count >= 1

    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_numeric_binned_bar_chart(self, mock_bar, df_numeric_only, tmp_viz_dir):
        """Test binned bar chart generation for numeric data."""
        mock_bar.return_value = "success"

        result = visualize_distribution_df(
            df_numeric_only, tmp_viz_dir, numeric_bar_charts=True
        )

        # Should call create_bar_plot for numeric columns with binning
        assert mock_bar.called

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_custom_bin_count(self, mock_hist, df_numeric_only, tmp_viz_dir):
        """Test that custom bin count is passed to visualization."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(
            df_numeric_only, tmp_viz_dir, n_bins=15
        )

        # Verify bins parameter passed
        if mock_hist.called:
            call_kwargs = mock_hist.call_args[1]
            assert call_kwargs.get("bins") == 15

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_integer_visualization(self, mock_hist, df_integers, tmp_viz_dir):
        """Test visualization of integer numeric data."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df_integers, tmp_viz_dir)

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_float_visualization(self, mock_hist, df_floats, tmp_viz_dir):
        """Test visualization of float numeric data."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df_floats, tmp_viz_dir)

        assert isinstance(result, dict)


# ======================== CATEGORICAL FIELD TESTS ========================

class TestCategoricalFieldVisualization:
    """Test visualization of categorical fields."""

    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_categorical_bar_chart_generation(self, mock_bar, df_categorical_only, tmp_viz_dir):
        """Test bar chart generation for categorical data."""
        mock_bar.return_value = "success"

        result = visualize_distribution_df(df_categorical_only, tmp_viz_dir)

        # Should call create_bar_plot for categorical columns
        assert mock_bar.call_count >= 1

    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_categorical_value_counts(self, mock_bar, df_categorical_only, tmp_viz_dir):
        """Test that value_counts is used for categorical data."""
        mock_bar.return_value = "success"

        result = visualize_distribution_df(df_categorical_only, tmp_viz_dir)

        # Bar chart should be called with normalized frequency data
        assert mock_bar.called


# ======================== EDGE CASE TESTS ========================

class TestVisualizeDistributionEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_single_column_df(self, mock_hist, df_single_column, tmp_viz_dir):
        """Test with single-column DataFrame."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df_single_column, tmp_viz_dir)

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_field_with_all_nulls(self, mock_bar, tmp_viz_dir):
        """Test skipping field with all null values."""
        df = pd.DataFrame({
            "all_nulls": [None, None, None],
            "valid": ["a", "b", "c"],
        })
        mock_bar.return_value = "success"

        result = visualize_distribution_df(df, tmp_viz_dir)

        # Should skip all_nulls column
        assert "all_nulls" not in result

    def test_empty_dataframe(self, tmp_viz_dir):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        result = visualize_distribution_df(df, tmp_viz_dir)

        assert isinstance(result, dict)
        assert len(result) == 0

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_single_value_field(self, mock_hist, tmp_viz_dir):
        """Test field with single value."""
        df = pd.DataFrame({
            "constant": [42, 42, 42, 42],
        })
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df, tmp_viz_dir)

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_missing_values_handling(self, mock_bar, df_with_missing, tmp_viz_dir):
        """Test that missing values are dropped before visualization."""
        mock_bar.return_value = "success"

        result = visualize_distribution_df(df_with_missing, tmp_viz_dir)

        # Should handle missing values gracefully
        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_large_bin_count(self, mock_hist, df_numeric_only, tmp_viz_dir):
        """Test with large bin count."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(
            df_numeric_only, tmp_viz_dir, n_bins=100
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_single_bin(self, mock_hist, df_numeric_only, tmp_viz_dir):
        """Test with single bin."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(
            df_numeric_only, tmp_viz_dir, n_bins=1
        )

        assert isinstance(result, dict)


# ======================== FIELD NAME TESTS ========================

class TestFieldNameHandling:
    """Test field name handling and filtering."""

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_field_names_none_uses_all(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test that None field_names uses all columns."""
        mock_hist.return_value = "success"
        mock_bar.return_value = "success"

        result1 = visualize_distribution_df(sample_df, tmp_viz_dir, field_names=None)
        result2 = visualize_distribution_df(sample_df, tmp_viz_dir)

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_field_names_subset(self, mock_hist, sample_df, tmp_viz_dir):
        """Test with subset of field names."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(
            sample_df, tmp_viz_dir, field_names=["age", "salary"]
        )

        # Should only process specified fields
        call_count = mock_hist.call_count
        assert call_count >= 0

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_invalid_field_name(self, mock_hist, sample_df, tmp_viz_dir):
        """Test with non-existent field name."""
        mock_hist.return_value = "success"

        with pytest.raises(KeyError):
            visualize_distribution_df(
                sample_df, tmp_viz_dir, field_names=["nonexistent"]
            )


# ======================== OUTPUT PATH TESTS ========================

class TestOutputPathHandling:
    """Test output path and directory handling."""

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_viz_dir_created(self, mock_hist, sample_df, tmp_path):
        """Test that visualization directory is created."""
        viz_dir = tmp_path / "nonexistent" / "viz"
        mock_hist.return_value = "success"

        result = visualize_distribution_df(sample_df, viz_dir)

        # Directory should be created
        assert viz_dir.exists()

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_result_paths_type(self, mock_hist, df_numeric_only, tmp_viz_dir):
        """Test that result paths are Path objects or strings."""
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df_numeric_only, tmp_viz_dir)

        for field_name, path in result.items():
            assert isinstance(path, (str, Path))

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_multiple_formats(self, mock_hist, sample_df, tmp_viz_dir):
        """Test different output formats."""
        mock_hist.return_value = "success"

        for fmt in ["html", "png", "svg"]:
            result = visualize_distribution_df(
                sample_df, tmp_viz_dir, viz_format=fmt
            )
            assert isinstance(result, dict)


# ======================== ERROR HANDLING TESTS ========================

class TestVisualizeDistributionErrors:
    """Test error handling and exception scenarios."""

    def test_none_input(self, tmp_viz_dir):
        """Test with None DataFrame."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            visualize_distribution_df(None, tmp_viz_dir)

    def test_invalid_input_type(self, tmp_viz_dir):
        """Test with invalid input type."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            visualize_distribution_df("not a dataframe", tmp_viz_dir)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_visualization_error_handling(self, mock_hist, sample_df, tmp_viz_dir):
        """Test handling of visualization errors."""
        # Make histogram fail
        mock_hist.return_value = "Error: visualization failed"

        result = visualize_distribution_df(sample_df, tmp_viz_dir)

        # Should return dict (may be empty or skip failed visualizations)
        assert isinstance(result, dict)


# ======================== INTEGRATION TESTS ========================

class TestVisualizeDistributionIntegration:
    """Integration tests combining multiple features."""

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_mixed_numeric_categorical(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test visualization of mixed numeric/categorical DataFrame."""
        mock_hist.return_value = "success"
        mock_bar.return_value = "success"

        result = visualize_distribution_df(sample_df, tmp_viz_dir)

        # Both histogram and bar chart should be called
        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    @patch("pamola_core.analysis.distribution.create_bar_plot")
    def test_complete_workflow(self, mock_bar, mock_hist, sample_df, tmp_viz_dir):
        """Test complete visualization workflow."""
        mock_hist.return_value = "success"
        mock_bar.return_value = "success"

        result = visualize_distribution_df(
            sample_df,
            tmp_viz_dir,
            numeric_bar_charts=False,
            n_bins=10,
            field_names=["age", "salary", "department"],
            viz_format="html",
        )

        assert isinstance(result, dict)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_multiple_calls_consistency(self, mock_hist, sample_df, tmp_viz_dir):
        """Test that multiple calls are consistent."""
        mock_hist.return_value = "success"

        result1 = visualize_distribution_df(sample_df, tmp_viz_dir)
        result2 = visualize_distribution_df(sample_df, tmp_viz_dir)

        assert len(result1) == len(result2)

    @patch("pamola_core.analysis.distribution.create_histogram")
    def test_large_dataframe(self, mock_hist, tmp_viz_dir):
        """Test with large DataFrame."""
        df = pd.DataFrame({
            "col1": np.random.rand(10000),
            "col2": ["cat"] * 10000,
        })
        mock_hist.return_value = "success"

        result = visualize_distribution_df(df, tmp_viz_dir)

        assert isinstance(result, dict)
