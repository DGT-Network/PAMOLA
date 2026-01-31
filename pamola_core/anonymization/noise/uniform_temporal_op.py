"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise Operation
Package:       pamola_core.anonymization.noise
Version:       1.0.0
Status:        development
Author:        PAMOLA Core Team
Created:       2025-01-20
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
   This module implements the UniformTemporalNoiseOperation class for adding
   uniformly distributed random time shifts to datetime fields. It supports
   flexible time unit specifications, directional control, and various
   preservation constraints for maintaining temporal patterns.

   The operation integrates with the PAMOLA.CORE framework, providing secure
   random generation, comprehensive metrics collection, and support for both
   pandas and Dask processing engines.

Key Features:
   - Uniform time shifts with configurable ranges in multiple units
   - Directional control (forward, backward, or both)
   - Boundary datetime enforcement
   - Special date and pattern preservation
   - Weekend and time-of-day preservation
   - Output granularity control
   - Cryptographically secure or reproducible random generation
   - Full Dask support for distributed processing
   - Comprehensive temporal metrics
   - Integration with framework utilities

Framework Integration:
   - Inherits from AnonymizationOperation base class
   - Uses SecureRandomGenerator from noise_utils
   - Integrates with validation_utils for parameter validation
   - Uses metric_utils and statistical_utils for metrics
   - Supports DataWriter for output and metrics
   - Compatible with ProgressTracker for monitoring

Changelog:
   1.0.0 (2025-06-15):
      - Initial implementation following noise SRS specifications
      - Full integration with commons utilities
      - Dask support for large-scale processing
      - Comprehensive metrics and validation
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Import base class
from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation

# Import noise utilities
from pamola_core.anonymization.commons.noise_utils import (
    SecureRandomGenerator,
    SECONDS_PER_MINUTE,
    SECONDS_PER_HOUR,
    SECONDS_PER_DAY,
)

# Import statistical utilities
from pamola_core.anonymization.commons.statistical_utils import (
    analyze_temporal_noise_impact,
)

# Import validation utilities
from pamola_core.anonymization.commons.validation import (
    InvalidParameterError,
    DateTimeFieldValidator,
)
from pamola_core.anonymization.schemas.uniform_temporal_op_core_schema import (
    UniformTemporalNoiseConfig,
)
from pamola_core.common.helpers.data_helper import DataHelper

# Import framework utilities
from pamola_core.utils.ops.op_registry import register

# Constants
VALID_DIRECTIONS = ["both", "forward", "backward"]
VALID_GRANULARITIES = ["day", "hour", "minute", "second", None]
BUSINESS_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday


@register(version="1.0.0")
class UniformTemporalNoiseOperation(AnonymizationOperation):
    """
    Operation for adding uniform random time shifts to datetime fields.

    Implements REQ-TEMPORAL-001 through REQ-TEMPORAL-007 from the
    PAMOLA.CORE Noise Operations Sub-Specification.

    This operation adds uniformly distributed random time shifts to datetime
    values, supporting flexible units, directionality, preservation of special
    dates, and temporal granularity control.
    """

    def __init__(
        self,
        field_name: str,
        # Temporal noise parameters
        noise_range_days: Optional[float] = None,
        noise_range_hours: Optional[float] = None,
        noise_range_minutes: Optional[float] = None,
        noise_range_seconds: Optional[float] = None,
        noise_range: Optional[Dict[str, Optional[float]]] = None,
        # Direction control
        direction: str = "both",
        # Boundary constraints
        min_datetime: Optional[Union[str, pd.Timestamp]] = None,
        max_datetime: Optional[Union[str, pd.Timestamp]] = None,
        # Special date handling
        preserve_special_dates: bool = False,
        special_dates: Optional[List[Union[str, pd.Timestamp]]] = None,
        preserve_weekends: bool = False,
        preserve_time_of_day: bool = False,
        # Granularity
        output_granularity: Optional[str] = None,
        # Reproducibility
        random_seed: Optional[int] = None,
        use_secure_random: bool = True,
        **kwargs,
    ):
        """
        Initialize the UniformTemporalNoiseOperation.

        Parameters
        ----------
        field_name : str
            Field to which temporal noise will be applied.
        noise_range_days : float, optional
            Range of time shift in days.
        noise_range_hours : float, optional
            Range of time shift in hours.
        noise_range_minutes : float, optional
            Range of time shift in minutes.
        noise_range_seconds : float, optional
            Range of time shift in seconds.
        noise_range : dict, optional
            Dictionary specifying one or more of the above noise ranges.
        direction : str, default="both"
            Direction of shift ('both', 'forward', or 'backward').
        min_datetime : str or pd.Timestamp, optional
            Minimum datetime allowed after applying noise.
        max_datetime : str or pd.Timestamp, optional
            Maximum datetime allowed after applying noise.
        preserve_special_dates : bool, default=False
            Whether to preserve certain special dates unchanged.
        special_dates : list of str or pd.Timestamp, optional
            Specific dates to be preserved (no shift applied).
        preserve_weekends : bool, default=False
            Maintain weekend/weekday status after noise.
        preserve_time_of_day : bool, default=False
            Preserve time-of-day component; only shift the date.
        output_granularity : str, optional
            Granularity to round the output to ('day', 'hour', 'minute', 'second').
        random_seed : int, optional
            Random seed for reproducibility (ignored if use_secure_random=True).
        use_secure_random : bool, default=True
            Whether to use cryptographically secure randomness.
        **kwargs : dict
            Additional keyword arguments passed to the base AnonymizationOperation.
        """
        # Ensure description fallback
        kwargs.setdefault(
            "description",
            f"Uniform temporal noise for field '{field_name}'",
        )

        if noise_range is not None:
            noise_range_days = noise_range.get("noise_range_days", noise_range_days)
            noise_range_hours = noise_range.get("noise_range_hours", noise_range_hours)
            noise_range_minutes = noise_range.get(
                "noise_range_minutes", noise_range_minutes
            )
            noise_range_seconds = noise_range.get(
                "noise_range_seconds", noise_range_seconds
            )

        # Build config
        config = UniformTemporalNoiseConfig(
            field_name=field_name,
            noise_range_days=noise_range_days,
            noise_range_hours=noise_range_hours,
            noise_range_minutes=noise_range_minutes,
            noise_range_seconds=noise_range_seconds,
            direction=direction,
            min_datetime=min_datetime,
            max_datetime=max_datetime,
            preserve_special_dates=preserve_special_dates,
            special_dates=special_dates,
            preserve_weekends=preserve_weekends,
            preserve_time_of_day=preserve_time_of_day,
            output_granularity=output_granularity,
            random_seed=random_seed,
            use_secure_random=use_secure_random,
            **kwargs,
        )

        # Pass config into kwargs for parent
        kwargs["config"] = config

        # Initialize parent class
        super().__init__(field_name=field_name, **kwargs)

        # Save config attributes to self
        for k, v in config.to_dict().items():
            setattr(self, k, v)

        self.direction = direction.lower()

        # Convert datetime boundaries
        self.min_datetime = pd.Timestamp(min_datetime) if min_datetime else None
        self.max_datetime = pd.Timestamp(max_datetime) if max_datetime else None

        # Validate and compute ranges
        self._validate_temporal_parameters(
            noise_range_days,
            noise_range_hours,
            noise_range_minutes,
            noise_range_seconds,
            direction,
            output_granularity,
            min_datetime,
            max_datetime,
        )

        self.special_dates = pd.to_datetime(special_dates) if special_dates else None
        self._generator: Optional[SecureRandomGenerator] = None

        # Calculate total shift range in seconds for efficiency
        self._total_shift_seconds = self._calculate_total_shift_seconds()

        # Operation metadata
        self.operation_name = self.__class__.__name__

    def _validate_temporal_parameters(
        self,
        days: Optional[float],
        hours: Optional[float],
        minutes: Optional[float],
        seconds: Optional[float],
        direction: str,
        granularity: Optional[str],
        min_datetime: Optional[Union[str, pd.Timestamp]] = None,
        max_datetime: Optional[Union[str, pd.Timestamp]] = None,
    ) -> None:
        """
        Validate temporal noise parameters.

        Args:
            days: Days component
            hours: Hours component
            minutes: Minutes component
            seconds: Seconds component
            direction: Shift direction
            granularity: Output granularity

        Raises:
            InvalidParameterError: If parameters are invalid
        """
        # At least one time component must be specified
        if all(x is None for x in [days, hours, minutes, seconds]):
            raise InvalidParameterError(
                param_name="noise_range",
                param_value=None,
                reason="at least one time component must be specified",
            )

        # Validate positive values
        for param_name, value in [
            ("noise_range_days", days),
            ("noise_range_hours", hours),
            ("noise_range_minutes", minutes),
            ("noise_range_seconds", seconds),
        ]:
            if value is not None and value <= 0:
                raise InvalidParameterError(
                    param_name=param_name, param_value=value, reason="must be positive"
                )

        # Validate direction
        if direction.lower() not in VALID_DIRECTIONS:
            raise InvalidParameterError(
                param_name="direction",
                param_value=direction,
                reason=f"must be one of {VALID_DIRECTIONS}",
            )

        # Validate granularity
        if granularity is not None and granularity not in VALID_GRANULARITIES:
            raise InvalidParameterError(
                param_name="output_granularity",
                param_value=granularity,
                reason=f"must be one of {VALID_GRANULARITIES}",
            )

        # Validate datetime boundaries
        if min_datetime and max_datetime:
            if min_datetime >= max_datetime:
                raise InvalidParameterError(
                    param_name="min_datetime",
                    param_value=min_datetime,
                    reason=f"must be less than max_datetime ({max_datetime})",
                )

    def _calculate_total_shift_seconds(self) -> float:
        """
        Calculate total shift range in seconds.

        Returns:
            Total shift range in seconds
        """
        total = 0.0
        if self.noise_range_days:
            total += self.noise_range_days * SECONDS_PER_DAY
        if self.noise_range_hours:
            total += self.noise_range_hours * SECONDS_PER_HOUR
        if self.noise_range_minutes:
            total += self.noise_range_minutes * SECONDS_PER_MINUTE
        if self.noise_range_seconds:
            total += self.noise_range_seconds
        return total

    def _generate_time_shifts(self, size: int) -> pd.TimedeltaIndex:
        """
        Generate random time shifts.

        Args:
            size: Number of shifts to generate

        Returns:
            TimedeltaIndex with random shifts
        """
        # Get parameters from self
        direction = getattr(self, "direction", "both")
        use_secure_random = getattr(self, "use_secure_random", True)
        random_seed = getattr(self, "random_seed", None)
        _generator = getattr(self, "_generator", None)
        _total_shift_seconds = getattr(self, "_total_shift_seconds", 0.0)

        if _generator is None:
            _generator = SecureRandomGenerator(
                use_secure=use_secure_random, seed=random_seed
            )
        # Generate shifts based on direction
        if direction == "both":
            # Shifts in both directions
            shift_seconds = _generator.uniform(
                -_total_shift_seconds, _total_shift_seconds, size
            )
        elif direction == "forward":
            # Only positive shifts (future)
            shift_seconds = _generator.uniform(0, _total_shift_seconds, size)
        else:  # backward
            # Only negative shifts (past)
            shift_seconds = _generator.uniform(-_total_shift_seconds, 0, size)

        # Convert to TimedeltaIndex
        return pd.to_timedelta(shift_seconds, unit="s")

    def _apply_temporal_noise(
        self, timestamps: pd.Series, shifts: pd.TimedeltaIndex
    ) -> pd.Series:
        """
        Apply noise with temporal constraints.

        Args:
            timestamps: Original timestamps
            shifts: Time shifts to apply

        Returns:
            Series with shifted timestamps
        """
        # Get parameters from self
        min_datetime = getattr(self, "min_datetime", None)
        max_datetime = getattr(self, "max_datetime", None)
        preserve_special_dates = getattr(self, "preserve_special_dates", False)
        special_dates = getattr(self, "special_dates", None)
        preserve_weekends = getattr(self, "preserve_weekends", False)
        preserve_time_of_day = getattr(self, "preserve_time_of_day", False)
        output_granularity = getattr(self, "output_granularity", None)

        # Apply shifts
        noisy_timestamps = timestamps + shifts

        # Apply boundary constraints
        if min_datetime:
            noisy_timestamps = noisy_timestamps.clip(lower=min_datetime)
        if max_datetime:
            noisy_timestamps = noisy_timestamps.clip(upper=max_datetime)

        # Preserve special dates
        if preserve_special_dates and special_dates is not None:
            for special_date in special_dates:
                # Check if any original dates match special dates
                mask = timestamps.dt.date == special_date.date()
                if mask.any():
                    noisy_timestamps.loc[mask] = timestamps.loc[mask]

        # Preserve weekends
        if preserve_weekends:
            noisy_timestamps = self._adjust_for_weekends(
                timestamps, noisy_timestamps, min_datetime, max_datetime
            )

        # Preserve time of day
        if preserve_time_of_day:
            # Keep original time, only shift date
            noisy_timestamps = pd.to_datetime(
                noisy_timestamps.dt.date.astype(str)
                + " "
                + timestamps.dt.time.astype(str)
            )

        # Apply granularity
        if output_granularity:
            noisy_timestamps = self._apply_granularity(
                noisy_timestamps, output_granularity
            )

        return noisy_timestamps

    def _adjust_for_weekends(
        self,
        original: pd.Series,
        noisy: pd.Series,
        min_datetime: Optional[Union[str, pd.Timestamp]] = None,
        max_datetime: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.Series:
        """
        Adjust shifts to preserve weekend/weekday status.

        Args:
            original: Original timestamps
            noisy: Shifted timestamps
            min_datetime: Minimum allowed datetime
            max_datetime: Maximum allowed datetime

        Returns:
            Adjusted timestamps preserving weekend status
        """
        # Get day of week (0=Monday, 6=Sunday)
        original_dow = original.dt.dayofweek
        noisy_dow = noisy.dt.dayofweek

        # Identify mismatches
        original_weekend = original_dow.isin(WEEKEND_DAYS)
        noisy_weekend = noisy_dow.isin(WEEKEND_DAYS)
        mismatch = original_weekend != noisy_weekend

        if not mismatch.any():
            return noisy

        # Adjust mismatched dates
        adjusted = noisy.copy(deep=True)

        # For each mismatched date, find nearest matching day type
        for idx in noisy[mismatch].index:
            orig_is_weekend = original_weekend.loc[idx]
            current_date = noisy.loc[idx]

            # Search for nearest matching day type (max 3 days)
            for offset in [1, -1, 2, -2, 3, -3]:
                test_date = current_date + pd.Timedelta(days=offset)
                test_is_weekend = test_date.dayofweek in WEEKEND_DAYS

                if test_is_weekend == orig_is_weekend:
                    # Check if within bounds
                    if min_datetime and test_date < min_datetime:
                        continue
                    if max_datetime and test_date > max_datetime:
                        continue

                    adjusted.loc[idx] = test_date
                    break

        return adjusted

    def _apply_granularity(
        self, timestamps: pd.Series, output_granularity: str
    ) -> pd.Series:
        """
        Round timestamps to specified granularity.

        Args:
            timestamps: Timestamps to round
            output_granularity: Granularity to apply ('day', 'hour', 'minute', 'second')

        Returns:
            Rounded timestamps
        """
        if output_granularity == "day":
            return timestamps.dt.floor("D")
        elif output_granularity == "hour":
            # FutureWarning: 'H' is deprecated and will be removed in a future version, please use 'h' instead
            return timestamps.dt.floor("h")
        elif output_granularity == "minute":
            return timestamps.dt.floor("T")
        elif output_granularity == "second":
            return timestamps.dt.floor("S")
        else:
            return timestamps

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data by adding temporal noise.

        Parameters:
        -----------
            batch : pd.DataFrame
                DataFrame batch to process

        Returns:
        --------
        pd.DataFrame
            Processed DataFrame with generalized datetimes
        """
        # Extract parameters from self
        field_name = self.field_name
        output_field_name = self.output_field_name
        mode = self.mode
        null_strategy = self.null_strategy

        result = batch.copy(deep=True)

        # Validate datetime field
        validator = DateTimeFieldValidator(allow_null=(null_strategy != "ERROR"))
        if not pd.api.types.is_datetime64_any_dtype(batch[field_name]):
            # Try to convert to datetime
            try:
                datetime_series = DataHelper.convert_to_datetime(batch[field_name])
                result[field_name] = datetime_series
            except Exception as e:
                raise ValueError(
                    f"Field '{field_name}' cannot be converted to datetime: {e}"
                )
        else:
            datetime_series = batch[field_name]

        # Validate the datetime series
        validation_result = validator.validate(datetime_series, field_name=field_name)
        if not validation_result.is_valid:
            raise ValueError(f"Datetime validation failed: {validation_result.errors}")

        # Handle nulls
        non_null_mask = datetime_series.notna()
        non_null_values = datetime_series[non_null_mask]
        if len(non_null_values) > 0:
            # Generate and apply shifts
            shifts = self._generate_time_shifts(len(non_null_values))
            noisy_values = self._apply_temporal_noise(non_null_values, shifts)

            # Update result
            if mode == "REPLACE":
                result.loc[non_null_mask, field_name] = noisy_values
            else:  # ENRICH
                result[output_field_name] = datetime_series.copy(deep=True)
                result.loc[non_null_mask, output_field_name] = noisy_values
        else:
            # No non-null values to process
            if mode == "ENRICH":
                result[output_field_name] = datetime_series.copy(deep=True)

        return result

    def _collect_specific_metrics(
        self, original_data: pd.Series, anonymized_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Collect temporal noise specific metrics.

        Args:
            original_data: Original datetime series
            anonymized_data: Datetime series after noise addition

        Returns:
            Dictionary of temporal-specific metrics
        """
        # Ensure datetime types
        if not pd.api.types.is_datetime64_any_dtype(original_data):
            original_data = DataHelper.convert_to_datetime(original_data)
        if not pd.api.types.is_datetime64_any_dtype(anonymized_data):
            anonymized_data = DataHelper.convert_to_datetime(anonymized_data)

        # Basic configuration metrics
        metrics = {
            "noise_range_config": {
                "days": self.noise_range_days,
                "hours": self.noise_range_hours,
                "minutes": self.noise_range_minutes,
                "seconds": self.noise_range_seconds,
                "total_seconds": self._total_shift_seconds,
            },
            "direction": self.direction,
            "constraints_applied": {
                "min_datetime": str(self.min_datetime) if self.min_datetime else None,
                "max_datetime": str(self.max_datetime) if self.max_datetime else None,
            },
            "preservation": {
                "special_dates_preserved": self.preserve_special_dates,
                "weekends_preserved": self.preserve_weekends,
                "time_of_day_preserved": self.preserve_time_of_day,
            },
            "output_granularity": self.output_granularity,
            "secure_random": self.use_secure_random,
        }

        # Calculate actual shifts
        shifts = (anonymized_data - original_data).dt.total_seconds()
        shifts_clean = shifts.dropna()

        if len(shifts_clean) > 0:
            # Shift statistics
            metrics["actual_shifts"] = {
                "mean_seconds": float(shifts_clean.mean()),
                "std_seconds": float(shifts_clean.std()),
                "min_seconds": float(shifts_clean.min()),
                "max_seconds": float(shifts_clean.max()),
                "mean_days": float(shifts_clean.mean() / SECONDS_PER_DAY),
                "max_days": float(shifts_clean.abs().max() / SECONDS_PER_DAY),
            }

            # Direction analysis
            metrics["shift_direction"] = {
                "forward_count": int((shifts_clean > 0).sum()),
                "backward_count": int((shifts_clean < 0).sum()),
                "zero_count": int((shifts_clean == 0).sum()),
            }

            # Boundary violations
            if self.min_datetime or self.max_datetime:
                metrics["constraints_applied"]["values_at_min"] = (
                    int((anonymized_data == self.min_datetime).sum())
                    if self.min_datetime
                    else 0
                )
                metrics["constraints_applied"]["values_at_max"] = (
                    int((anonymized_data == self.max_datetime).sum())
                    if self.max_datetime
                    else 0
                )

        # Use statistical utils for detailed temporal analysis
        temporal_impact = analyze_temporal_noise_impact(original_data, anonymized_data)
        metrics["temporal_impact"] = temporal_impact

        return metrics

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        Returns:
            Dictionary of parameters affecting the operation output
        """

        # Add temporal-specific parameters
        return {
            "noise_range_days": self.noise_range_days,
            "noise_range_hours": self.noise_range_hours,
            "noise_range_minutes": self.noise_range_minutes,
            "noise_range_seconds": self.noise_range_seconds,
            "direction": self.direction,
            "min_datetime": str(self.min_datetime) if self.min_datetime else None,
            "max_datetime": str(self.max_datetime) if self.max_datetime else None,
            "preserve_special_dates": self.preserve_special_dates,
            "special_dates": (
                [str(d) for d in self.special_dates]
                if self.special_dates is not None
                else None
            ),
            "preserve_weekends": self.preserve_weekends,
            "preserve_time_of_day": self.preserve_time_of_day,
            "output_granularity": self.output_granularity,
            "random_seed": self.random_seed if not self.use_secure_random else None,
            "use_secure_random": self.use_secure_random,
        }

    def __repr__(self) -> str:
        """String representation of the operation."""
        # Build range string
        range_parts = []
        if self.noise_range_days:
            range_parts.append(f"{self.noise_range_days}d")
        if self.noise_range_hours:
            range_parts.append(f"{self.noise_range_hours}h")
        if self.noise_range_minutes:
            range_parts.append(f"{self.noise_range_minutes}m")
        if self.noise_range_seconds:
            range_parts.append(f"{self.noise_range_seconds}s")
        range_str = " ".join(range_parts)

        return (
            f"UniformTemporalNoiseOperation("
            f"field='{self.field_name}', "
            f"range='{range_str}', "
            f"direction='{self.direction}', "
            f"secure={self.use_secure_random})"
        )
