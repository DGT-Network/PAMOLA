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
from pamola_core.utils.ops.op_config import OperationConfig

# Import framework utilities
from pamola_core.utils.ops.op_field_utils import generate_output_field_name
from pamola_core.utils.ops.op_registry import register_operation

# Constants
VALID_DIRECTIONS = ["both", "forward", "backward"]
VALID_GRANULARITIES = ["day", "hour", "minute", "second", None]
BUSINESS_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday


class UniformTemporalNoiseConfig(OperationConfig):
    """Configuration for UniformTemporalNoiseOperation."""

    schema = {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},
            # Temporal noise parameters
            "noise_range_days": {"type": ["number", "null"]},
            "noise_range_hours": {"type": ["number", "null"]},
            "noise_range_minutes": {"type": ["number", "null"]},
            "noise_range_seconds": {"type": ["number", "null"]},
            # Direction control
            "direction": {"type": "string"},
            # Boundary constraints
            "min_datetime": {"type": ["string", "null"]},
            "max_datetime": {"type": ["string", "null"]},
            # Special date handling
            "preserve_special_dates": {"type": "boolean"},
            "special_dates": {"type": ["array", "null"]},
            "preserve_weekends": {"type": "boolean"},
            "preserve_time_of_day": {"type": "boolean"},
            # Granularity
            "output_granularity": {"type": ["string", "null"]},
            # Reproducibility
            "random_seed": {"type": ["integer", "null"]},
            "use_secure_random": {"type": "boolean"},
            # Conditional processing parameters
            "condition_field": {"type": ["string", "null"]},
            "condition_values": {"type": ["array", "null"]},
            "condition_operator": {"type": "string"},
            # Multi-field conditions
            "multi_conditions": {"type": ["array", "null"]},
            "condition_logic": {"type": "string"},
            # K-anonymity integration
            "ka_risk_field": {"type": ["string", "null"]},
            "risk_threshold": {"type": "number"},
            "vulnerable_record_strategy": {"type": "string"},
            # Memory optimization
            "optimize_memory": {"type": "boolean"},
            "adaptive_chunk_size": {"type": "boolean"},
            # Standard parameters
            "description": {"type": "string"},
            "mode": {"type": "string", "enum": ["REPLACE", "ENRICH"]},
            "column_prefix": {"type": "string"},
            "output_field_name": {"type": ["string", "null"]},
            "null_strategy": {
                "type": "string",
                "enum": ["PRESERVE", "EXCLUDE", "ERROR", "ANONYMIZE"],
            },
            "chunk_size": {"type": "integer"},
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": ["integer", "null"]},
            "dask_partition_size": {"type": ["string", "null"]},
            "use_vectorization": {"type": "boolean"},
            "parallel_processes": {"type": ["integer", "null"]},
            "use_cache": {"type": "boolean"},
            "output_format": {
                "type": "string",
                "enum": ["csv", "parquet", "arrow"],
                "default": "csv",
            },
            "visualization_theme": {"type": ["string", "null"]},
            "visualization_backend": {"type": ["string", "null"]},
            "visualization_strict": {"type": "boolean"},
            "visualization_timeout": {"type": "integer"},
            "use_encryption": {"type": "boolean"},
            "encryption_mode": {"type": ["string", "null"]},
            "encryption_key": {"type": ["string", "null"]},
            "noise_range": {
                "type": ["object", "null"],
                "properties": {
                    "noise_range_days": {"type": ["number", "null"]},
                    "noise_range_hours": {"type": ["number", "null"]},
                    "noise_range_minutes": {"type": ["number", "null"]},
                    "noise_range_seconds": {"type": ["number", "null"]},
                },
            },
        },
        "required": ["field_name"],
    }


class UniformTemporalNoiseOperation(AnonymizationOperation):
    """
    Operation for adding uniform random time shifts to datetime fields.

    This operation implements REQ-TEMPORAL-001 through REQ-TEMPORAL-007 from the
    PAMOLA.CORE Noise Operations Sub-Specification.

    The operation adds uniformly distributed random time shifts to datetime values,
    supporting various time units, preservation constraints, and output controls
    for maintaining temporal patterns while providing privacy protection.

    Attributes:
        noise_range_days: Range in days for time shifts
        noise_range_hours: Range in hours for time shifts
        noise_range_minutes: Range in minutes for time shifts
        noise_range_seconds: Range in seconds for time shifts
        direction: Direction of shifts ('both', 'forward', 'backward')
        min_datetime: Minimum allowed datetime after shift
        max_datetime: Maximum allowed datetime after shift
        preserve_special_dates: Whether to preserve special dates unchanged
        special_dates: List of dates to preserve
        preserve_weekends: Whether to maintain weekend/weekday status
        preserve_time_of_day: Whether to keep time unchanged (shift date only)
        output_granularity: Rounding granularity for output
        random_seed: Seed for reproducibility (ignored if use_secure_random=True)
        use_secure_random: Use cryptographically secure random
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
        # Conditional processing parameters
        condition_field: Optional[str] = None,
        condition_values: Optional[List] = None,
        condition_operator: str = "in",
        # Multi-field conditions
        multi_conditions: Optional[List[Dict[str, Any]]] = None,
        condition_logic: str = "AND",
        # K-anonymity integration
        ka_risk_field: Optional[str] = None,
        risk_threshold: float = 5.0,
        vulnerable_record_strategy: str = "suppress",
        # Memory optimization
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,
        # Standard parameters
        description: str = "",
        mode: str = "REPLACE",
        column_prefix: str = "_",
        output_field_name: Optional[str] = None,
        null_strategy: str = "PRESERVE",
        chunk_size: int = 10000,
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        use_cache: bool = False,
        output_format: str = "csv",
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,
        use_encryption: bool = False,
        encryption_mode: Optional[str] = None,
        encryption_key: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the uniform temporal noise operation.

        Args:
            field_name: Field to add temporal noise to
            noise_range_days: Time shift range in days
            noise_range_hours: Time shift range in hours
            noise_range_minutes: Time shift range in minutes
            noise_range_seconds: Time shift range in seconds
            noise_range: Dictionary containing noise range components
            direction: Direction of shifts ('both', 'forward', 'backward')
            min_datetime: Minimum allowed datetime after shift
            max_datetime: Maximum allowed datetime after shift
            preserve_special_dates: Don't shift special dates
            special_dates: List of dates to preserve unchanged
            preserve_weekends: Maintain weekend/weekday status
            preserve_time_of_day: Keep time unchanged (shift date only)
            output_granularity: Round output ('day', 'hour', 'minute', 'second')
            random_seed: Seed for reproducibility (ignored if use_secure_random=True)
            use_secure_random: Use cryptographically secure random
            condition_field: Field name for conditional processing
            condition_values: Values to match for conditional processing
            condition_operator: Operator for condition matching ('in', 'not_in', 'eq', 'ne', 'gt', 'lt', 'ge', 'le')
            multi_conditions: List of multiple field conditions for complex filtering
            condition_logic: Logic operator for conditions ('AND', 'OR')
            ka_risk_field: Field containing k-anonymity risk scores
            risk_threshold: Threshold for identifying vulnerable records
            vulnerable_record_strategy: Strategy for handling vulnerable records ('suppress', 'preserve', 'enhance')
            optimize_memory: Enable memory optimization techniques
            adaptive_chunk_size: Automatically adjust chunk size based on available memory
            description: Human-readable description of the operation
            mode: Processing mode ('REPLACE' to modify original field, 'ENRICH' to add new field)
            column_prefix: Prefix for new column names in ENRICH mode
            output_field_name: Custom name for output field (overrides automatic naming)
            null_strategy: How to handle null values ('PRESERVE', 'EXCLUDE', 'ERROR', 'ANONYMIZE')
            chunk_size: Number of rows to process in each chunk
            use_dask: Enable Dask for distributed processing
            npartitions: Number of partitions for Dask processing
            dask_partition_size: Target size for Dask partitions (e.g., '100MB')
            use_vectorization: Enable vectorized operations where possible
            parallel_processes: Number of parallel processes for multi-processing
            use_cache: Enable caching of operation results
            output_format: Format for output files ('csv', 'parquet', 'arrow')
            visualization_theme: Theme for generated visualizations
            visualization_backend: Backend for visualization ('matplotlib', 'plotly')
            visualization_strict: Strict mode for visualization generation
            visualization_timeout: Timeout in seconds for visualization generation
            use_encryption: Enable encryption for sensitive outputs
            encryption_mode: Encryption algorithm to use
            encryption_key: Path to encryption key file or key string
            **: Additional parameters for base class
        """
        if noise_range is not None:
            noise_range_days = noise_range.get("noise_range_days", noise_range_days)
            noise_range_hours = noise_range.get("noise_range_hours", noise_range_hours)
            noise_range_minutes = noise_range.get(
                "noise_range_minutes", noise_range_minutes
            )
            noise_range_seconds = noise_range.get(
                "noise_range_seconds", noise_range_seconds
            )

        config = UniformTemporalNoiseConfig(
            field_name=field_name,
            # Temporal noise parameters
            noise_range_days=noise_range_days,
            noise_range_hours=noise_range_hours,
            noise_range_minutes=noise_range_minutes,
            noise_range_seconds=noise_range_seconds,
            # Direction control
            direction=direction,
            # Boundary constraints
            min_datetime=min_datetime,
            max_datetime=max_datetime,
            # Special date handling
            preserve_special_dates=preserve_special_dates,
            special_dates=special_dates,
            preserve_weekends=preserve_weekends,
            preserve_time_of_day=preserve_time_of_day,
            # Granularity
            output_granularity=output_granularity,
            # Reproducibility
            random_seed=random_seed,
            use_secure_random=use_secure_random,
            # Conditional processing parameters
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
            # Multi-field conditions
            multi_conditions=multi_conditions,
            condition_logic=condition_logic,
            # K-anonymity integration
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
            # Memory optimization
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            # Standard parameters
            description=description,
            mode=mode,
            column_prefix=column_prefix,
            output_field_name=output_field_name,
            null_strategy=null_strategy,
            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_cache=use_cache,
            output_format=output_format,
            visualization_theme=visualization_theme,
            visualization_backend=visualization_backend,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
            use_encryption=use_encryption,
            encryption_mode=encryption_mode,
            encryption_key=encryption_key,
        )

        # Initialize base class
        super().__init__(
            field_name=field_name,
            # Conditional processing parameters
            condition_field=condition_field,
            condition_values=condition_values,
            condition_operator=condition_operator,
            # Multi-field conditions
            multi_conditions=multi_conditions,
            condition_logic=condition_logic,
            # K-anonymity integration
            ka_risk_field=ka_risk_field,
            risk_threshold=risk_threshold,
            vulnerable_record_strategy=vulnerable_record_strategy,
            # Memory optimization
            optimize_memory=optimize_memory,
            adaptive_chunk_size=adaptive_chunk_size,
            # Standard parameters
            description=f"Uniform temporal noise for field '{field_name}'",
            mode=mode,
            column_prefix=column_prefix,
            output_field_name=output_field_name,
            null_strategy=null_strategy,
            chunk_size=chunk_size,
            use_dask=use_dask,
            npartitions=npartitions,
            dask_partition_size=dask_partition_size,
            use_vectorization=use_vectorization,
            parallel_processes=parallel_processes,
            use_cache=use_cache,
            output_format=output_format,
            visualization_theme=visualization_theme,
            visualization_backend=visualization_backend,
            visualization_strict=visualization_strict,
            visualization_timeout=visualization_timeout,
            use_encryption=use_encryption,
            encryption_mode=encryption_mode,
            encryption_key=encryption_key,
        )

        self.config = config

        # Validate and store temporal parameters
        self._validate_temporal_parameters(
            noise_range_days,
            noise_range_hours,
            noise_range_minutes,
            noise_range_seconds,
            direction,
            output_granularity,
        )

        self.noise_range_days = noise_range_days
        self.noise_range_hours = noise_range_hours
        self.noise_range_minutes = noise_range_minutes
        self.noise_range_seconds = noise_range_seconds
        self.direction = direction.lower()

        # Convert and store datetime boundaries
        self.min_datetime = pd.Timestamp(min_datetime) if min_datetime else None
        self.max_datetime = pd.Timestamp(max_datetime) if max_datetime else None

        # Validate datetime boundaries
        if self.min_datetime and self.max_datetime:
            if self.min_datetime >= self.max_datetime:
                raise InvalidParameterError(
                    param_name="min_datetime",
                    param_value=min_datetime,
                    reason=f"must be less than max_datetime ({max_datetime})",
                )

        # Store preservation settings
        self.preserve_special_dates = preserve_special_dates
        self.special_dates = pd.to_datetime(special_dates) if special_dates else None
        self.preserve_weekends = preserve_weekends
        self.preserve_time_of_day = preserve_time_of_day

        # Store other settings
        self.output_granularity = output_granularity
        self.random_seed = random_seed
        self.use_secure_random = use_secure_random

        # Initialize generator (will be created per execution)
        self._generator: Optional[SecureRandomGenerator] = None

        # Calculate total shift range in seconds for efficiency
        self._total_shift_seconds = self._calculate_total_shift_seconds()

        # Version
        self.version = "1.0.0"

    def _validate_temporal_parameters(
        self,
        days: Optional[float],
        hours: Optional[float],
        minutes: Optional[float],
        seconds: Optional[float],
        direction: str,
        granularity: Optional[str],
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

    def _initialize_generator(self) -> None:
        """Initialize the random number generator for this execution."""
        self._generator = SecureRandomGenerator(
            use_secure=self.use_secure_random, seed=self.random_seed
        )
        self.logger.debug(
            f"Initialized {'secure' if self.use_secure_random else 'standard'} "
            f"random generator for temporal noise"
        )

    def _generate_time_shifts(self, size: int) -> pd.TimedeltaIndex:
        """
        Generate random time shifts.

        Args:
            size: Number of shifts to generate

        Returns:
            TimedeltaIndex with random shifts
        """
        if self._generator is None:
            self._initialize_generator()

        # Generate shifts based on direction
        if self.direction == "both":
            # Shifts in both directions
            shift_seconds = self._generator.uniform(
                -self._total_shift_seconds, self._total_shift_seconds, size
            )
        elif self.direction == "forward":
            # Only positive shifts (future)
            shift_seconds = self._generator.uniform(0, self._total_shift_seconds, size)
        else:  # backward
            # Only negative shifts (past)
            shift_seconds = self._generator.uniform(-self._total_shift_seconds, 0, size)

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
        # Apply shifts
        noisy_timestamps = timestamps + shifts

        # Apply boundary constraints
        if self.min_datetime:
            noisy_timestamps = noisy_timestamps.clip(lower=self.min_datetime)
        if self.max_datetime:
            noisy_timestamps = noisy_timestamps.clip(upper=self.max_datetime)

        # Preserve special dates
        if self.preserve_special_dates and self.special_dates is not None:
            for special_date in self.special_dates:
                # Check if any original dates match special dates
                mask = timestamps.dt.date == special_date.date()
                if mask.any():
                    noisy_timestamps.loc[mask] = timestamps.loc[mask]

        # Preserve weekends
        if self.preserve_weekends:
            noisy_timestamps = self._adjust_for_weekends(timestamps, noisy_timestamps)

        # Preserve time of day
        if self.preserve_time_of_day:
            # Keep original time, only shift date
            noisy_timestamps = pd.to_datetime(
                noisy_timestamps.dt.date.astype(str)
                + " "
                + timestamps.dt.time.astype(str)
            )

        # Apply granularity
        if self.output_granularity:
            noisy_timestamps = self._apply_granularity(noisy_timestamps)

        return noisy_timestamps

    def _adjust_for_weekends(self, original: pd.Series, noisy: pd.Series) -> pd.Series:
        """
        Adjust shifts to preserve weekend/weekday status.

        Args:
            original: Original timestamps
            noisy: Shifted timestamps

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
        adjusted = noisy.copy()

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
                    if self.min_datetime and test_date < self.min_datetime:
                        continue
                    if self.max_datetime and test_date > self.max_datetime:
                        continue

                    adjusted.loc[idx] = test_date
                    break

        return adjusted

    def _apply_granularity(self, timestamps: pd.Series) -> pd.Series:
        """
        Round timestamps to specified granularity.

        Args:
            timestamps: Timestamps to round

        Returns:
            Rounded timestamps
        """
        if self.output_granularity == "day":
            return timestamps.dt.floor("D")
        elif self.output_granularity == "hour":
            # FutureWarning: 'H' is deprecated and will be removed in a future version, please use 'h' instead
            return timestamps.dt.floor("h")
        elif self.output_granularity == "minute":
            return timestamps.dt.floor("T")
        elif self.output_granularity == "second":
            return timestamps.dt.floor("S")
        else:
            return timestamps

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of data by adding temporal noise.

        Args:
            batch: DataFrame batch to process

        Returns:
            Processed DataFrame with temporal noise added
        """
        result = batch.copy()

        # Validate datetime field
        validator = DateTimeFieldValidator(allow_null=(self.null_strategy != "ERROR"))
        if not pd.api.types.is_datetime64_any_dtype(batch[self.field_name]):
            # Try to convert to datetime
            try:
                datetime_series = pd.to_datetime(batch[self.field_name])
                result[self.field_name] = datetime_series
            except Exception as e:
                raise ValueError(
                    f"Field '{self.field_name}' cannot be converted to datetime: {e}"
                )
        else:
            datetime_series = batch[self.field_name]

        # Validate the datetime series
        validation_result = validator.validate(
            datetime_series, field_name=self.field_name
        )
        if not validation_result.is_valid:
            raise ValueError(f"Datetime validation failed: {validation_result.errors}")

        # Use framework utility for output field naming
        output_col = generate_output_field_name(
            self.field_name,
            self.mode,
            self.output_field_name,
            operation_suffix="shifted",
            column_prefix=self.column_prefix,
        )

        # Handle nulls
        non_null_mask = datetime_series.notna()
        non_null_values = datetime_series[non_null_mask]
        if len(non_null_values) > 0:
            # Generate and apply shifts
            shifts = self._generate_time_shifts(len(non_null_values))
            noisy_values = self._apply_temporal_noise(non_null_values, shifts)

            # Update result
            if self.mode == "REPLACE":
                result.loc[non_null_mask, self.field_name] = noisy_values
            else:  # ENRICH
                result[output_col] = datetime_series.copy()
                result.loc[non_null_mask, output_col] = noisy_values
        else:
            # No non-null values to process
            if self.mode == "ENRICH":
                result[output_col] = datetime_series.copy()

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
            original_data = pd.to_datetime(original_data)
        if not pd.api.types.is_datetime64_any_dtype(anonymized_data):
            anonymized_data = pd.to_datetime(anonymized_data)

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
        params = super()._get_cache_parameters()

        # Add temporal-specific parameters
        params.update(
            {
                "field_name": self.field_name,
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
                "condition_field": self.condition_field,
                "condition_values": self.condition_values,
                "condition_operator": self.condition_operator,
                "multi_conditions": self.multi_conditions,
                "condition_logic": self.condition_logic,
                "ka_risk_field": self.ka_risk_field,
                "risk_threshold": self.risk_threshold,
                "vulnerable_record_strategy": self.vulnerable_record_strategy,
                "optimize_memory": self.optimize_memory,
                "adaptive_chunk_size": self.adaptive_chunk_size,
                "mode": self.mode,
                "column_prefix": self.column_prefix,
                "output_field_name": self.output_field_name,
                "null_strategy": self.null_strategy,
                "chunk_size": self.chunk_size,
                "use_dask": self.use_dask,
                "npartitions": self.npartitions,
                "dask_partition_size": self.dask_partition_size,
                "use_vectorization": self.use_vectorization,
                "parallel_processes": self.parallel_processes,
                "use_cache": self.use_cache,
                "output_format": self.output_format,
                "visualization_theme": self.visualization_theme,
                "visualization_backend": self.visualization_backend,
                "visualization_strict": self.visualization_strict,
                "visualization_timeout": self.visualization_timeout,
                "use_encryption": self.use_encryption,
                "encryption_mode": self.encryption_mode,
                "encryption_key": self.encryption_key,
            }
        )

        return params

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


# Register the operation with the framework
register_operation(UniformTemporalNoiseOperation)
