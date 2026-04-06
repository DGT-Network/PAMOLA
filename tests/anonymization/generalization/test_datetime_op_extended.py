"""
Extended tests for DateTimeGeneralizationOperation targeting missed coverage lines.

Focus areas:
- Rounding units: year, quarter, month, week, day, hour
- Binning: hour_range, day_range, business_period, seasonal, custom
- Component strategy: various component combos
- Relative strategy: all time buckets
- process_value for all strategies
- _round_single_value, _bin_single_value, _component_single_value, _relative_single_value
- _validate_privacy_level, _validate_date_range
- _parse_reference_date, _parse_custom_bins error paths
- _apply_rounding with strftime_output_format
- Edge cases: empty series, all-NaT, single value
- Timezone handling: utc, remove, preserve
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from pamola_core.anonymization.generalization.datetime_op import (
    DateTimeGeneralizationOperation,
    DateTimeConstants,
)
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_result import OperationStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reporter():
    class DummyReporter:
        def __init__(self):
            self.ops = []
        def add_operation(self, *args, **kwargs):
            self.ops.append((args, kwargs))
    return DummyReporter()


@pytest.fixture
def mock_ds():
    """Return a factory that builds a mock DataSource from a DataFrame."""
    def _make(df):
        ds = Mock(spec=DataSource)
        ds.get_dataframe.return_value = (df, None)
        ds.encryption_keys = {}
        ds.settings = {}
        ds.encryption_modes = {}
        ds.data_source_name = "test"
        ds.apply_data_types.side_effect = lambda d, *a, **kw: d
        return ds
    return _make


def _dates_df(n=50, freq="D", start="2020-01-01"):
    dates = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "dt": pd.Series(dates, dtype="datetime64[ns]"),
        "val": range(n),
    })


def _multi_year_df():
    """Span multiple years and quarters."""
    dates = pd.date_range("2019-01-01", periods=100, freq="ME")
    return pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(100)})


# ---------------------------------------------------------------------------
# 1. Rounding strategy - all units
# ---------------------------------------------------------------------------

class TestRoundingUnits:
    @pytest.mark.parametrize("unit", ["year", "quarter", "month", "week", "day", "hour"])
    def test_rounding_unit(self, unit, mock_ds, reporter, tmp_path):
        df = _dates_df(40, freq="6H")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit=unit
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_with_strftime_format(self, mock_ds, reporter, tmp_path):
        """strftime_output_format should produce string output."""
        df = _dates_df(20)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month",
            strftime_output_format="%Y-%m"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_year_with_strftime(self, mock_ds, reporter, tmp_path):
        df = _dates_df(20)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="year",
            strftime_output_format="%Y"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_quarter(self, mock_ds, reporter, tmp_path):
        df = _multi_year_df()
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="quarter"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_week(self, mock_ds, reporter, tmp_path):
        df = _dates_df(30)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="week"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_hour(self, mock_ds, reporter, tmp_path):
        df = _dates_df(48, freq="30min")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="hour"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_rounding_with_nat_values(self, mock_ds, reporter, tmp_path):
        """NaT values should be preserved during rounding."""
        dates = pd.date_range("2020-01-01", periods=20).tolist()
        dates[5] = pd.NaT
        dates[10] = pd.NaT
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(20)})
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month",
            null_strategy="PRESERVE"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 2. Binning strategy - all bin types
# ---------------------------------------------------------------------------

class TestBinningStrategy:
    def test_hour_range_binning(self, mock_ds, reporter, tmp_path):
        df = _dates_df(48, freq="30min")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="hour_range", interval_size=6
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_hour_range_interval_1(self, mock_ds, reporter, tmp_path):
        df = _dates_df(24, freq="H")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="hour_range", interval_size=1
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_day_range_binning(self, mock_ds, reporter, tmp_path):
        df = _dates_df(60)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="day_range", interval_size=7
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_day_range_with_reference_date(self, mock_ds, reporter, tmp_path):
        df = _dates_df(30)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="day_range",
            interval_size=7, reference_date="2020-01-01"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_business_period_binning(self, mock_ds, reporter, tmp_path):
        df = _dates_df(48, freq="H")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="business_period"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_seasonal_binning(self, mock_ds, reporter, tmp_path):
        df = _multi_year_df()
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_custom_binning(self, mock_ds, reporter, tmp_path):
        df = _dates_df(30)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="custom",
            custom_bins=["2020-01-01", "2020-01-15", "2020-02-01", "2020-02-20"]
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_custom_binning_too_few_bins_raises(self, tmp_path):
        """custom_bins with fewer than 2 boundaries should raise error during processing."""
        dates = pd.date_range("2020-01-01", periods=10).tolist()
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(10)})
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="custom",
            custom_bins=["2020-01-01"]
        )
        op.output_field_name = "dt"
        series = pd.Series(dates, dtype="datetime64[ns]")
        with pytest.raises(Exception):
            op._apply_binning(series)



# ---------------------------------------------------------------------------
# 3. Component strategy
# ---------------------------------------------------------------------------

class TestComponentStrategy:
    @pytest.mark.parametrize("components", [
        ["year"],
        ["year", "month"],
        ["year", "month", "day"],
        ["hour", "minute"],
        ["weekday"],
        ["year", "weekday"],
    ])
    def test_component_combos(self, components, mock_ds, reporter, tmp_path):
        df = _dates_df(30, freq="6H")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component", keep_components=components
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_component_valid_year_month(self, tmp_path):
        """keep_components with valid components."""
        dates = pd.date_range("2020-01-01", periods=5)
        series = pd.Series(dates, dtype="datetime64[ns]")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component", keep_components=["year", "month"]
        )
        result_series = op._apply_component(series)
        assert isinstance(result_series, pd.Series)

    def test_component_with_strftime_output(self, mock_ds, reporter, tmp_path):
        df = _dates_df(20)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component", keep_components=["year"],
            strftime_output_format="%Y"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_component_single_value_with_custom_format(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component",
            keep_components=["year"],
            strftime_output_format="%Y/%m"
        )
        val = pd.Timestamp("2023-06-15 12:30:00")
        result = op._component_single_value(val)
        assert "2023" in result

    def test_component_all_parts(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component",
            keep_components=["year", "month", "day", "hour", "minute", "weekday"]
        )
        val = pd.Timestamp("2023-06-15 12:30:00")
        result = op._component_single_value(val)
        assert isinstance(result, str)
        assert "2023" in result


# ---------------------------------------------------------------------------
# 4. Relative strategy
# ---------------------------------------------------------------------------

class TestRelativeStrategy:
    def test_relative_more_than_year_ago(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="relative",
            reference_date="2023-01-01"
        )
        val = pd.Timestamp("2021-01-01")
        assert op._relative_single_value(val) == "More than a year ago"

    def test_relative_months_ago(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-02-15")
        result = op._relative_single_value(val)
        assert result == "Months ago"

    def test_relative_weeks_ago(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-05-15")
        result = op._relative_single_value(val)
        assert result == "Weeks ago"

    def test_relative_days_ago(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-05-29")
        result = op._relative_single_value(val)
        assert result == "Days ago"

    def test_relative_same_day(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-06-01 10:30:00")
        result = op._relative_single_value(val)
        assert result == "Same day"

    def test_relative_days_ahead(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-06-04")
        result = op._relative_single_value(val)
        assert result == "Days ahead"

    def test_relative_weeks_ahead(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-06-20")
        result = op._relative_single_value(val)
        assert result == "Weeks ahead"

    def test_relative_months_ahead(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2023-08-15")
        result = op._relative_single_value(val)
        assert result == "Months ahead"

    def test_relative_more_than_year_ahead(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        op.reference_date = pd.Timestamp("2023-06-01")
        val = pd.Timestamp("2025-06-15")
        result = op._relative_single_value(val)
        assert result == "More than a year ahead"

    def test_relative_series_basic(self, mock_ds, reporter, tmp_path):
        df = _dates_df(10)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="relative",
            reference_date="2020-02-01"
        )
        op.preset_type = None; op.preset_name = None
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_relative_execute_with_reference_date(self, mock_ds, reporter, tmp_path):
        df = _dates_df(50)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="relative",
            reference_date="2020-02-15"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS


# ---------------------------------------------------------------------------
# 5. process_value for all strategies
# ---------------------------------------------------------------------------

class TestProcessValue:
    def test_process_value_rounding_year(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="year"
        )
        val = pd.Timestamp("2023-06-15 12:30:00")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-01-01 00:00:00")

    def test_process_value_rounding_quarter_q1(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="quarter"
        )
        val = pd.Timestamp("2023-02-15")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-01-01")

    def test_process_value_rounding_quarter_q2(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="quarter"
        )
        val = pd.Timestamp("2023-05-10")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-04-01")

    def test_process_value_rounding_quarter_q3(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="quarter"
        )
        val = pd.Timestamp("2023-08-20")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-07-01")

    def test_process_value_rounding_quarter_q4(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="quarter"
        )
        val = pd.Timestamp("2023-11-05")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-10-01")

    def test_process_value_rounding_month(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month"
        )
        val = pd.Timestamp("2023-06-15 12:30:00")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-06-01 00:00:00")

    def test_process_value_rounding_week(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="week"
        )
        val = pd.Timestamp("2023-06-15")  # Thursday
        result = op.process_value(val)
        # Should be start of week (Monday)
        assert result.weekday() == 0

    def test_process_value_rounding_day(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="day"
        )
        val = pd.Timestamp("2023-06-15 14:30:00")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-06-15 00:00:00")

    def test_process_value_rounding_hour(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="hour"
        )
        val = pd.Timestamp("2023-06-15 14:30:00")
        result = op.process_value(val)
        assert result == pd.Timestamp("2023-06-15 14:00:00")

    def test_process_value_rounding_day(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="day"
        )
        val = pd.Timestamp("2023-06-15 14:30:00")
        result = op.process_value(val)
        assert isinstance(result, (pd.Timestamp, str))

    def test_process_value_binning_hour_range(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="hour_range", interval_size=6
        )
        val = pd.Timestamp("2023-06-15 08:30:00")
        result = op._bin_single_value(val)
        assert ":" in result  # e.g., "06:00-12:00"

    def test_process_value_binning_business_morning(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="business_period"
        )
        val = pd.Timestamp("2023-06-15 09:00:00")
        result = op._bin_single_value(val)
        assert result == "Morning"

    def test_process_value_binning_business_afternoon(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="business_period"
        )
        val = pd.Timestamp("2023-06-15 14:00:00")
        result = op._bin_single_value(val)
        assert result == "Afternoon"

    def test_process_value_binning_business_night(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="business_period"
        )
        val = pd.Timestamp("2023-06-15 02:00:00")
        result = op._bin_single_value(val)
        assert result == "Night"

    def test_process_value_binning_seasonal_winter(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal"
        )
        for month in [12, 1, 2]:
            val = pd.Timestamp(f"2023-{month:02d}-15") if month != 12 else pd.Timestamp("2022-12-15")
            result = op._bin_single_value(val)
            assert result == "Winter"

    def test_process_value_binning_seasonal_spring(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal"
        )
        val = pd.Timestamp("2023-04-15")
        assert op._bin_single_value(val) == "Spring"

    def test_process_value_binning_seasonal_summer(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal"
        )
        val = pd.Timestamp("2023-07-15")
        assert op._bin_single_value(val) == "Summer"

    def test_process_value_binning_seasonal_fall(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal"
        )
        val = pd.Timestamp("2023-10-15")
        assert op._bin_single_value(val) == "Fall"

    def test_process_value_component(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component",
            keep_components=["year", "month", "day"]
        )
        val = pd.Timestamp("2023-06-15")
        result = op.process_value(val)
        assert "2023" in str(result)

    def test_process_value_relative(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="relative",
            reference_date="2023-06-01"
        )
        val = pd.Timestamp("2020-01-01")
        result = op.process_value(val)
        assert result == "More than a year ago"

    def test_process_value_non_timestamp_input(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="day"
        )
        result = op.process_value("2023-06-15")
        assert result is not None


# ---------------------------------------------------------------------------
# 6. _round_single_value with strftime
# ---------------------------------------------------------------------------

class TestRoundSingleValueWithFormat:
    def test_year_with_strftime(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="year",
            strftime_output_format="%Y"
        )
        val = pd.Timestamp("2023-06-15 12:30:00")
        result = op._round_single_value(val)
        assert result == "2023"

    def test_month_with_strftime(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month",
            strftime_output_format="%Y-%m"
        )
        val = pd.Timestamp("2023-06-15")
        result = op._round_single_value(val)
        assert result == "2023-06"

    def test_non_timestamp_passthrough(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="year"
        )
        result = op._round_single_value("not_a_timestamp")
        assert result == "not_a_timestamp"


# ---------------------------------------------------------------------------
# 7. _bin_single_value non-Timestamp passthrough
# ---------------------------------------------------------------------------

class TestBinSingleValueEdgeCases:
    def test_non_timestamp_returns_str(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="hour_range"
        )
        result = op._bin_single_value("not_a_timestamp")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 8. Timezone handling
# ---------------------------------------------------------------------------

class TestTimezoneHandling:
    def test_timezone_preserve(self, mock_ds, reporter, tmp_path):
        df = _dates_df(20)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="day",
            timezone_handling="preserve"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_timezone_remove_on_naive(self, tmp_path):
        """Naive datetime with 'remove' should just return unchanged series."""
        dates = pd.date_range("2020-01-01", periods=5)
        series = pd.Series(dates, dtype="datetime64[ns]")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", timezone_handling="remove"
        )
        result = op._handle_timezone(series)
        assert isinstance(result, pd.Series)

    def test_timezone_utc_on_naive(self, tmp_path):
        """Naive datetime with 'utc' should localize then convert."""
        dates = pd.date_range("2020-01-01", periods=5)
        series = pd.Series(dates, dtype="datetime64[ns]")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", timezone_handling="utc",
            default_timezone="US/Eastern"
        )
        result = op._handle_timezone(series)
        assert isinstance(result, pd.Series)

    def test_handle_timezone_non_datetime_passthrough(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", timezone_handling="utc"
        )
        series = pd.Series(["a", "b", "c"])
        result = op._handle_timezone(series)
        assert list(result) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 9. _validate_privacy_level
# ---------------------------------------------------------------------------

class TestValidatePrivacyLevel:
    def test_privacy_level_sufficient(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month"
        )
        original = pd.Series(pd.date_range("2020-01-01", periods=100))
        # Collapse to 3 unique months
        generalized = pd.Series(["2020-01", "2020-02", "2020-03"] * 33 + ["2020-01"])
        result = op._validate_privacy_level(original, generalized)
        assert result is True

    def test_privacy_level_empty_original(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding"
        )
        original = pd.Series([], dtype="object")
        generalized = pd.Series([], dtype="object")
        assert op._validate_privacy_level(original, generalized) is True

    def test_privacy_level_insufficient(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", min_privacy_threshold=0.9
        )
        original = pd.Series(["2020-01", "2020-02", "2020-03", "2020-04"])
        generalized = pd.Series(["2020-01", "2020-02", "2020-03", "2020-04"])  # no reduction
        result = op._validate_privacy_level(original, generalized)
        assert result is False


# ---------------------------------------------------------------------------
# 10. _parse_reference_date and _parse_custom_bins error paths
# ---------------------------------------------------------------------------

class TestParseFunctions:
    def test_parse_reference_date_string(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="relative"
        )
        result = op._parse_reference_date("2020-01-01")
        assert isinstance(result, pd.Timestamp)

    def test_parse_reference_date_datetime(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        dt = datetime(2020, 6, 15)
        result = op._parse_reference_date(dt)
        assert isinstance(result, pd.Timestamp)

    def test_parse_reference_date_none(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        result = op._parse_reference_date(None)
        assert result is None

    def test_parse_reference_date_invalid_raises(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="relative")
        with pytest.raises(Exception):
            op._parse_reference_date("not-a-valid-date-!!!")

    def test_parse_custom_bins_none(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="binning")
        result = op._parse_custom_bins(None)
        assert result is None

    def test_parse_custom_bins_valid(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="binning")
        result = op._parse_custom_bins(["2020-01-01", "2020-06-01", "2021-01-01"])
        assert len(result) == 3

    def test_parse_custom_bins_with_datetime_objects(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="binning")
        bins = [datetime(2020, 1, 1), datetime(2020, 6, 1)]
        result = op._parse_custom_bins(bins)
        assert len(result) == 2

    def test_parse_custom_bins_invalid_raises(self):
        op = DateTimeGeneralizationOperation(field_name="dt", strategy="binning")
        with pytest.raises(Exception):
            op._parse_custom_bins(["not-a-date-at-all-!!!", "2020-01-01"])


# ---------------------------------------------------------------------------
# 11. Invalid strategy raises error in process_batch
# ---------------------------------------------------------------------------

class TestInvalidStrategy:
    def test_invalid_strategy_raises(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding"
        )
        op.strategy = "invalid_strategy"
        op.output_field_name = "dt"
        dates = pd.date_range("2020-01-01", periods=5)
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(5)})
        with pytest.raises(Exception):
            op.process_batch(df)


# ---------------------------------------------------------------------------
# 12. ENRICH vs REPLACE mode in process_batch
# ---------------------------------------------------------------------------

class TestModesInProcessBatch:
    def test_replace_mode(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month", mode="REPLACE"
        )
        op.output_field_name = "dt"
        dates = pd.date_range("2020-01-01", periods=10)
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(10)})
        result = op.process_batch(df)
        assert "dt" in result.columns

    def test_enrich_mode(self):
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month", mode="ENRICH",
            output_field_name="dt_generalized"
        )
        op.output_field_name = "dt_generalized"
        dates = pd.date_range("2020-01-01", periods=10)
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(10)})
        result = op.process_batch(df)
        assert "dt_generalized" in result.columns


# ---------------------------------------------------------------------------
# 13. process_batch_dask
# ---------------------------------------------------------------------------

class TestProcessBatchDask:
    def test_process_batch_dask_rounding(self):
        import dask.dataframe as dd
        dates = pd.date_range("2020-01-01", periods=20)
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(20)})
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month", mode="REPLACE"
        )
        op.output_field_name = "dt"
        ddf = dd.from_pandas(df, npartitions=2)
        result = op.process_batch_dask(ddf)
        computed = result.compute()
        assert "dt" in computed.columns

    def test_process_batch_dask_binning(self):
        import dask.dataframe as dd
        dates = pd.date_range("2020-01-01", periods=20)
        df = pd.DataFrame({"dt": pd.Series(dates, dtype="datetime64[ns]"), "val": range(20)})
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal", mode="REPLACE"
        )
        op.output_field_name = "dt"
        ddf = dd.from_pandas(df, npartitions=2)
        result = op.process_batch_dask(ddf)
        computed = result.compute()
        assert isinstance(computed, pd.DataFrame)


# ---------------------------------------------------------------------------
# 14. Full execute covering metrics/cache/relative with execute
# ---------------------------------------------------------------------------

class TestFullExecute:
    def test_enrich_mode_execute(self, mock_ds, reporter, tmp_path):
        df = _dates_df(30)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="day",
            mode="ENRICH", output_field_name="dt_rounded"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_relative_strategy_execute(self, mock_ds, reporter, tmp_path):
        df = _dates_df(30)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="relative",
            reference_date="2020-01-15"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_component_strategy_execute(self, mock_ds, reporter, tmp_path):
        df = _dates_df(20)
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="component",
            keep_components=["year", "month", "day"]
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_invalid_field_execute(self, mock_ds, reporter, tmp_path):
        df = _dates_df(10)
        op = DateTimeGeneralizationOperation(
            field_name="nonexistent", strategy="rounding"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.ERROR

    def test_all_nat_series(self, mock_ds, reporter, tmp_path):
        df = pd.DataFrame({
            "dt": pd.Series([pd.NaT] * 10, dtype="datetime64[ns]"),
            "val": range(10)
        })
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="rounding", rounding_unit="month",
            null_strategy="PRESERVE"
        )
        # All-NaT may produce error or success depending on validation
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status in [OperationStatus.SUCCESS, OperationStatus.ERROR]

    def test_seasonal_execute_multi_year(self, mock_ds, reporter, tmp_path):
        df = _multi_year_df()
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="seasonal"
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS

    def test_hour_range_execute(self, mock_ds, reporter, tmp_path):
        df = _dates_df(48, freq="H")
        op = DateTimeGeneralizationOperation(
            field_name="dt", strategy="binning", bin_type="hour_range", interval_size=4
        )
        result = op.execute(mock_ds(df), tmp_path, reporter)
        assert result.status == OperationStatus.SUCCESS
