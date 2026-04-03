"""Edge-case tests for correlation.py — targets validation and branch gaps."""
import pytest
import pandas as pd
import numpy as np
from pamola_core.analysis.correlation import CorrelationAnalyzer
from pamola_core.errors.exceptions import (
    ColumnNotFoundError,
    ValidationError,
    TypeValidationError,
)


@pytest.fixture
def analyzer():
    return CorrelationAnalyzer()


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    return pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "c": np.random.randn(50),
        "d": np.random.randn(50),
    })


class TestValidation:
    def test_invalid_viz_format(self, analyzer):
        """Line 88: unsupported viz format."""
        with pytest.raises(ValidationError):
            analyzer._validate_viz_format("bmp")

    def test_invalid_output_chart_type(self, analyzer):
        """Line 99: output_chart not str/list — source has init bug, raises TypeError."""
        with pytest.raises(TypeError):
            analyzer._validate_output_chart(123)

    def test_bool_conversion(self, analyzer):
        """Line 139: boolean series conversion (not in categorical mapping)."""
        s = pd.Series([True, False, True, False])
        result = analyzer._map_binary_to_numeric(s)
        assert result.dtype in [np.int32, np.int64]


class TestComputeCorrelation:
    def test_single_column(self, analyzer, numeric_df):
        """Lines 204-224: single column correlation."""
        result, rtype = analyzer._calculate_correlation_result(
            numeric_df, columns=["a"], method="pearson",
        )
        assert rtype == "single_variable"
        assert result.shape[0] == 3  # b, c, d (excludes a)

    def test_single_column_not_found(self, analyzer, numeric_df):
        """Line 208: missing single column."""
        with pytest.raises(ColumnNotFoundError):
            analyzer._calculate_correlation_result(
                numeric_df, columns=["nonexist"], method="pearson",
            )

    def test_two_columns_not_found(self, analyzer, numeric_df):
        """Line 232: missing one of two columns."""
        with pytest.raises(ColumnNotFoundError):
            analyzer._calculate_correlation_result(
                numeric_df, columns=["a", "nonexist"], method="pearson",
            )

    def test_multi_columns_none_available(self, analyzer, numeric_df):
        """Line 246: none of specified cols exist in clean_data."""
        # After _prepare_data, only numeric cols remain — supply non-existent names
        clean = numeric_df[["a", "b"]]
        with pytest.raises(ValidationError):
            analyzer._calculate_correlation_result(
                clean, columns=["x", "y", "z"], method="pearson",
            )

    def test_multi_columns_partial_missing(self, analyzer, numeric_df):
        """Lines 251-252: some columns missing."""
        with pytest.raises(ColumnNotFoundError):
            analyzer._calculate_correlation_result(
                numeric_df, columns=["a", "b", "ghost"], method="pearson",
            )
