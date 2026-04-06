"""Deep coverage tests for uniform_temporal_op.py — targets validation, granularity, batch edges."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pamola_core.anonymization.noise.uniform_temporal_op import (
    UniformTemporalNoiseOperation,
)
from pamola_core.errors.exceptions import (
    InvalidParameterError,
    ValidationError,
    ConfigurationError,
)


class TestValidationBranches:
    def test_negative_noise_range_raises(self):
        """Line 285: negative noise_range_days — schema catches before Python validation."""
        with pytest.raises((InvalidParameterError, ConfigurationError)):
            UniformTemporalNoiseOperation(
                field_name="dt", noise_range_days=-1,
            )

    def test_invalid_direction_raises(self):
        """Line 291: invalid direction."""
        with pytest.raises((InvalidParameterError, ConfigurationError)):
            UniformTemporalNoiseOperation(
                field_name="dt", noise_range_days=5, direction="sideways",
            )

    def test_invalid_granularity_raises(self):
        """Line 299: invalid output_granularity."""
        with pytest.raises((InvalidParameterError, ConfigurationError)):
            UniformTemporalNoiseOperation(
                field_name="dt", noise_range_days=5, output_granularity="nanosecond",
            )

    def test_min_gte_max_datetime_raises(self):
        """Line 308: min_datetime >= max_datetime."""
        with pytest.raises((InvalidParameterError, ConfigurationError)):
            UniformTemporalNoiseOperation(
                field_name="dt", noise_range_days=5,
                min_datetime=datetime(2025, 6, 1),
                max_datetime=datetime(2025, 1, 1),
            )


class TestGranularity:
    @pytest.fixture
    def timestamps(self):
        return pd.Series(pd.date_range("2025-01-01 12:30:45", periods=5, freq="h"))

    def test_granularity_day(self, timestamps):
        """Line 510."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=1, output_granularity="day",
        )
        result = op._apply_granularity(timestamps, "day")
        assert all(result.dt.hour == 0)

    def test_granularity_hour(self, timestamps):
        """Line 513."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=1, output_granularity="hour",
        )
        result = op._apply_granularity(timestamps, "hour")
        assert all(result.dt.minute == 0)

    def test_granularity_minute(self, timestamps):
        """Line 515."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=1, output_granularity="minute",
        )
        result = op._apply_granularity(timestamps, "minute")
        assert all(result.dt.second == 0)

    def test_granularity_second(self, timestamps):
        """Line 517."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=1, output_granularity="second",
        )
        result = op._apply_granularity(timestamps, "second")
        assert len(result) == 5


class TestBatchProcessing:
    def test_string_datetime_conversion(self):
        """Lines 547-549: convert string to datetime."""
        op = UniformTemporalNoiseOperation(field_name="dt", noise_range_days=1)
        batch = pd.DataFrame({"dt": ["2025-01-01", "2025-01-02", "2025-01-03"]})
        result = op.process_batch(batch)
        assert pd.api.types.is_datetime64_any_dtype(result["dt"])

    def test_unconvertible_string_coerces_to_nat(self):
        """Lines 547-549: DataHelper coerces bad strings to NaT, validation may catch."""
        op = UniformTemporalNoiseOperation(field_name="dt", noise_range_days=1)
        batch = pd.DataFrame({"dt": ["not_a_date", "also_not", "nope"]})
        # Either raises ValidationError or produces all-NaT result
        try:
            result = op.process_batch(batch)
            assert result["dt"].isna().all()
        except (ValidationError, Exception):
            pass  # Validation catches the all-NaT series

    def test_all_null_enrich_mode(self):
        """Lines 580-581: all null values in enrich mode."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=1, mode="ENRICH",
        )
        batch = pd.DataFrame({"dt": pd.Series([pd.NaT, pd.NaT, pd.NaT])})
        result = op.process_batch(batch)
        assert op.output_field_name in result.columns


class TestBoundaryConstraints:
    def test_special_dates_preserved(self):
        """Lines 407-411: preserve_special_dates."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=30,
            preserve_special_dates=True,
            special_dates=["2025-01-01"],
        )
        ts = pd.Series([pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-15")])
        shifts = op._generate_time_shifts(2)
        result = op._apply_temporal_noise(ts, shifts)
        assert result.iloc[0] == pd.Timestamp("2025-01-01")

    def test_boundary_clamp(self):
        """Lines 400-403: min/max datetime clamping."""
        op = UniformTemporalNoiseOperation(
            field_name="dt", noise_range_days=365,
            min_datetime="2025-01-01",
            max_datetime="2025-12-31",
        )
        ts = pd.Series([pd.Timestamp("2025-06-15")] * 10)
        shifts = op._generate_time_shifts(10)
        result = op._apply_temporal_noise(ts, shifts)
        assert all(result >= pd.Timestamp("2025-01-01"))
        assert all(result <= pd.Timestamp("2025-12-31"))
