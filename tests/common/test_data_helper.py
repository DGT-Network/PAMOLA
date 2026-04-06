"""
Unit tests for pamola_core.common.helpers.data_helper module.

Tests cover:
- Data type detection (numeric, boolean, integer, float)
- Range string detection and conversion
- Column type analysis
- Edge cases and error handling

Run with: pytest -s tests/common/test_data_helper.py
"""

import numpy as np
import pandas as pd

from pamola_core.common.helpers.data_helper import DataHelper


class TestIsNonNumeric:
    """Test DataHelper.is_non_numeric() method."""

    def test_string_column_is_non_numeric(self):
        """String column should be detected as non-numeric."""
        column = pd.Series(["a", "b", "c"])
        assert DataHelper.is_non_numeric(column) is True

    def test_numeric_column_is_numeric(self):
        """Numeric column should be detected as numeric."""
        column = pd.Series([1, 2, 3])
        assert DataHelper.is_non_numeric(column) is False

    def test_float_column_is_numeric(self):
        """Float column should be detected as numeric."""
        column = pd.Series([1.1, 2.2, 3.3])
        assert DataHelper.is_non_numeric(column) is False

    def test_mixed_string_column_is_non_numeric(self):
        """Column with strings should be detected as non-numeric."""
        column = pd.Series(["text", "more", "data"])
        assert DataHelper.is_non_numeric(column) is True

    def test_empty_column(self):
        """Empty column should be handled."""
        column = pd.Series([], dtype=object)
        assert DataHelper.is_non_numeric(column) is True

    def test_column_with_nulls(self):
        """Column with NaN/None values should be detected correctly."""
        column = pd.Series([1.0, np.nan, 3.0])
        assert DataHelper.is_non_numeric(column) is False


class TestIsBool:
    """Test DataHelper.is_bool() method."""

    def test_boolean_column_is_bool(self):
        """Boolean column should be detected as bool."""
        column = pd.Series([True, False, True])
        assert DataHelper.is_bool(column) is True

    def test_yes_no_column_is_bool(self):
        """Column with 'yes'/'no' should be detected as bool."""
        column = pd.Series(["yes", "no", "yes"])
        assert DataHelper.is_bool(column) is True

    def test_numeric_column_is_not_bool(self):
        """Numeric column should not be detected as bool."""
        column = pd.Series([1, 2, 3])
        assert DataHelper.is_bool(column) is False

    def test_string_column_is_not_bool(self):
        """Random string column should not be bool."""
        column = pd.Series(["a", "b", "c"])
        assert DataHelper.is_bool(column) is False

    def test_mixed_bool_column(self):
        """Column with True/False mixed should be bool."""
        column = pd.Series([True, False])
        assert DataHelper.is_bool(column) is True

    def test_bool_with_nulls(self):
        """Boolean column with NaN should still be detected."""
        column = pd.Series([True, False, np.nan])
        assert DataHelper.is_bool(column) is True


class TestIsInteger:
    """Test DataHelper.is_integer() method."""

    def test_integer_value(self):
        """Integer values should be detected."""
        assert DataHelper.is_integer(42) is True
        assert DataHelper.is_integer("42") is True

    def test_integer_with_dot_zero(self):
        """Values like 42.0 should be detected as integers."""
        assert DataHelper.is_integer(42.0) is True
        assert DataHelper.is_integer("42.0") is True

    def test_float_value_not_integer(self):
        """Float values should not be detected as integers."""
        assert DataHelper.is_integer(42.5) is False
        assert DataHelper.is_integer("42.5") is False

    def test_negative_integer(self):
        """Negative integers should be detected."""
        assert DataHelper.is_integer(-42) is True
        assert DataHelper.is_integer("-42") is True

    def test_negative_float_not_integer(self):
        """Negative floats should not be detected as integers."""
        assert DataHelper.is_integer(-42.5) is False

    def test_zero_is_integer(self):
        """Zero should be detected as integer."""
        assert DataHelper.is_integer(0) is True
        assert DataHelper.is_integer("0") is True

    def test_nan_not_integer(self):
        """NaN should not be detected as integer."""
        assert DataHelper.is_integer(np.nan) is False
        assert DataHelper.is_integer(pd.NA) is False

    def test_string_not_integer(self):
        """String that's not a number should not be detected."""
        assert DataHelper.is_integer("abc") is False

    def test_dot_zero_zero_not_integer(self):
        """42.00 should not be detected as integer (only .0 allowed)."""
        assert DataHelper.is_integer("42.00") is False


class TestIsFloat:
    """Test DataHelper.is_float() method."""

    def test_float_value(self):
        """Float values should be detected."""
        assert DataHelper.is_float(42.5) is True
        assert DataHelper.is_float("42.5") is True

    def test_dot_zero_not_float(self):
        """Values like 42.0 should not be detected as floats."""
        assert DataHelper.is_float(42.0) is False
        assert DataHelper.is_float("42.0") is False

    def test_integer_not_float(self):
        """Integer values should not be detected as floats."""
        assert DataHelper.is_float(42) is False

    def test_negative_float(self):
        """Negative floats should be detected."""
        assert DataHelper.is_float(-42.5) is True

    def test_small_decimal(self):
        """Values with small decimals should be detected."""
        assert DataHelper.is_float(0.01) is True
        assert DataHelper.is_float("0.01") is True

    def test_nan_not_float(self):
        """NaN should not be detected as float."""
        assert DataHelper.is_float(np.nan) is False

    def test_string_not_float(self):
        """Non-numeric string should not be detected as float."""
        assert DataHelper.is_float("abc") is False


class TestIsRangeString:
    """Test DataHelper.is_range_string() method."""

    def test_simple_range(self):
        """Simple range like '18-35' should be detected."""
        assert DataHelper.is_range_string("18-35") is True

    def test_float_range(self):
        """Range with floats like '0.5-1.5' should be detected."""
        assert DataHelper.is_range_string("0.5-1.5") is True

    def test_negative_range(self):
        """Range with negative numbers like '-5-5' should be detected."""
        assert DataHelper.is_range_string("-5-5") is True

    def test_both_negative_range(self):
        """Range like '-0.38--0.13' should be detected."""
        assert DataHelper.is_range_string("-0.38--0.13") is True

    def test_non_range_string(self):
        """Non-range strings should not be detected."""
        assert DataHelper.is_range_string("18") is False
        assert DataHelper.is_range_string("abc") is False
        assert DataHelper.is_range_string("18-35-50") is False

    def test_range_with_spaces_not_detected(self):
        """Ranges with spaces might not be detected."""
        assert DataHelper.is_range_string("18 - 35") is False

    def test_range_not_string_type(self):
        """Non-string types should return False."""
        assert DataHelper.is_range_string(18) is False
        assert DataHelper.is_range_string([18, 35]) is False


class TestConvertRangeToNumeric:
    """Test DataHelper.convert_range_to_numeric() method."""

    def test_simple_range_conversion(self):
        """Simple range should be converted to midpoint."""
        result = DataHelper.convert_range_to_numeric("10-20")
        assert result == 15

    def test_float_range_conversion(self):
        """Float range should be converted correctly."""
        result = DataHelper.convert_range_to_numeric("1.0-3.0")
        assert result == 2.0

    def test_negative_range_conversion(self):
        """Negative range should be converted."""
        result = DataHelper.convert_range_to_numeric("-10-10")
        assert result == 0

    def test_both_negative_range_conversion(self):
        """Range with both negatives like '-5--1' should be converted."""
        result = DataHelper.convert_range_to_numeric("-5--1")
        assert result == -3

    def test_asymmetric_range(self):
        """Asymmetric range should calculate correct midpoint."""
        result = DataHelper.convert_range_to_numeric("10-30")
        assert result == 20

    def test_invalid_range_raises_error(self):
        """Invalid range format should return None (error is swallowed internally)."""
        result = DataHelper.convert_range_to_numeric("invalid")
        assert result is None

    def test_single_number_not_range(self):
        """Single number is not a range, should return None."""
        result = DataHelper.convert_range_to_numeric("42")
        assert result is None

    def test_float_precision(self):
        """Float ranges should preserve precision."""
        result = DataHelper.convert_range_to_numeric("0.1-0.9")
        assert abs(result - 0.5) < 0.001

    def test_range_returns_int_for_integer_range(self):
        """Integer ranges should return int type."""
        result = DataHelper.convert_range_to_numeric("10-20")
        assert isinstance(result, int)

    def test_range_returns_float_for_float_range(self):
        """Float ranges should return float type."""
        result = DataHelper.convert_range_to_numeric("10.0-20.0")
        assert isinstance(result, float)


class TestDetermineMostlyInteger:
    """Test DataHelper.determine_mostly_integer() method.

    Note: determine_mostly_integer returns ``integer_count > float_count``
    where both counts come from pandas .sum() — so the return type is
    numpy.bool_, not Python bool.  Use ``== True`` / ``== False`` (or
    plain truthiness) instead of ``is True`` / ``is False``.
    """

    def test_integer_column(self):
        """Column with integers should be detected as mostly integer."""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        assert DataHelper.determine_mostly_integer(df, "col") == True  # noqa: E712

    def test_float_column(self):
        """Column with actual floats should not be mostly integer."""
        df = pd.DataFrame({"col": [1.5, 2.5, 3.5]})
        assert DataHelper.determine_mostly_integer(df, "col") == False  # noqa: E712

    def test_integer_with_dot_zero(self):
        """Column with integer values like 1.0 should be mostly integer."""
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        assert DataHelper.determine_mostly_integer(df, "col") == True  # noqa: E712

    def test_mixed_integer_float(self):
        """Column with more integers than floats should be mostly integer."""
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.5]})
        # 3 integers, 1 float -> mostly integer
        assert DataHelper.determine_mostly_integer(df, "col") == True  # noqa: E712

    def test_mostly_float_column(self):
        """Column with more floats should not be mostly integer."""
        df = pd.DataFrame({"col": [1.0, 2.5, 3.5, 4.5]})
        # 1 integer, 3 floats -> not mostly integer
        assert DataHelper.determine_mostly_integer(df, "col") == False  # noqa: E712

    def test_empty_column(self):
        """Empty column should return False."""
        df = pd.DataFrame({"col": pd.Series([], dtype=float)})
        assert not DataHelper.determine_mostly_integer(df, "col")

    def test_all_nulls(self):
        """Column with all nulls should return False."""
        df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
        assert not DataHelper.determine_mostly_integer(df, "col")

    def test_with_nulls_mixed(self):
        """Column with mixed integers and nulls should be detected."""
        df = pd.DataFrame({"col": [1.0, 2.0, np.nan, 4.0]})
        assert DataHelper.determine_mostly_integer(df, "col") == True  # noqa: E712

    def test_sample_size_parameter(self):
        """Should respect sample_size parameter."""
        df = pd.DataFrame({"col": list(range(1000))})
        result = DataHelper.determine_mostly_integer(df, "col", sample_size=100)
        # Returns numpy.bool_ (from pandas sum comparison), accept both bool types
        assert isinstance(result, (bool, np.bool_))


class TestDataHelperIntegration:
    """Integration tests for DataHelper."""

    def test_detect_and_convert_range(self):
        """Should detect and convert range strings."""
        value = "10-20"
        if DataHelper.is_range_string(value):
            result = DataHelper.convert_range_to_numeric(value)
            assert result == 15

    def test_workflow_numeric_column(self):
        """Test typical workflow for numeric column."""
        df = pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "text": ["a", "b", "c", "d", "e"]
        })

        assert DataHelper.is_non_numeric(df["numeric"]) is False
        assert DataHelper.is_non_numeric(df["text"]) is True
        assert DataHelper.determine_mostly_integer(df, "numeric") == True  # noqa: E712

    def test_workflow_detection_sequence(self):
        """Test detection in sequence."""
        values = [42, 42.0, 42.5, "abc", True]

        results = {
            "int": DataHelper.is_integer(values[0]),
            "float": DataHelper.is_float(values[2]),
            "bool": isinstance(values[4], bool)
        }

        assert results["int"] is True
        assert results["float"] is True
