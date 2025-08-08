# PAMOLA.CORE Noise Operations Software Requirements Sub-Specification

**Document Version:** 1.0.0  
**Parent Document:** PAMOLA.CORE Anonymization Package SRS v4.1.0  
**Last Updated:** 2025-01-20  
**Status:** Draft

## 1. Introduction

### 1.1 Purpose

This Software Requirements Sub-Specification (Sub-SRS) defines the detailed requirements for noise addition operations within the PAMOLA.CORE anonymization package. The MVP focuses on uniform noise addition for numeric and datetime fields, providing a foundation for future differential privacy implementations.

### 1.2 Scope

This document covers uniform noise addition operations for MVP:
- **Numeric Noise Addition**: Add uniform random noise to numeric fields (integers and floats)
- **Datetime Noise Addition**: Add temporal noise to datetime fields (days, hours, minutes)

Future versions will extend to differential privacy mechanisms (Laplace, Gaussian).

### 1.3 Document Conventions

- **REQ-NOISE-XXX**: General noise operation requirements
- **REQ-UNIFORM-XXX**: Uniform noise specific requirements
- **REQ-TEMPORAL-XXX**: Temporal noise specific requirements

### 1.4 Architecture Overview

```
pamola_core/anonymization/noise/
├── __init__.py
├── uniform_numeric_op.py     # Uniform noise for numeric fields
└── uniform_temporal_op.py    # Uniform noise for datetime fields

pamola_core/anonymization/commons/   # Shared utilities (existing)
├── noise_utils.py           # Noise-specific utilities
└── statistical_utils.py     # Statistical analysis helpers
```

## 2. Common Noise Requirements

### 2.1 Base Class Inheritance

**REQ-NOISE-001 [MUST]** All noise operations SHALL inherit from `AnonymizationOperation` and follow the standard operation contract defined in the parent SRS.

### 2.2 Noise Generation Standards

**REQ-NOISE-002 [MUST]** Noise operations SHALL use cryptographically secure random number generation:
- Python's `secrets` module for security-critical applications
- NumPy's random generator with proper seeding for reproducibility when required
- Thread-safe random generation for concurrent processing

### 2.3 Data Integrity

**REQ-NOISE-003 [MUST]** Noise operations SHALL maintain data type integrity:
- Integer fields remain integers after noise addition
- Float precision is preserved
- Datetime fields maintain valid datetime formats
- Null values are handled according to null_strategy

### 2.4 Framework Integration

**REQ-NOISE-004 [MUST]** Operations SHALL integrate with existing framework utilities:
- Use `op_field_utils.generate_output_field_name()` for output field naming
- Use `op_data_processing.get_dataframe_chunks()` for batch processing
- Use `op_data_processing.optimize_dataframe_dtypes()` for memory optimization
- Use `DataWriter` for all file operations
- Use commons validation framework

## 3. Uniform Numeric Noise Operation

### 3.1 Overview

**REQ-UNIFORM-001 [MUST]** The `UniformNumericNoiseOperation` adds uniformly distributed random noise to numeric fields within specified bounds.

### 3.2 Constructor Interface

**REQ-UNIFORM-002 [MUST]** Constructor signature:

```python
class UniformNumericNoiseOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field to add noise to
                 # Noise parameters
                 noise_range: Union[float, Tuple[float, float]],  # Symmetric or asymmetric range
                 noise_type: str = "additive",       # additive, multiplicative
                 # Bounds and constraints
                 output_min: Optional[float] = None, # Minimum output value
                 output_max: Optional[float] = None, # Maximum output value
                 preserve_zero: bool = False,        # Don't add noise to zero values
                 # Integer handling
                 round_to_integer: bool = None,      # Auto-detect from field type
                 # Statistical parameters
                 scale_by_std: bool = False,         # Scale noise by field std dev
                 scale_factor: float = 1.0,          # Additional scaling factor
                 # Reproducibility
                 random_seed: Optional[int] = None,  # For reproducible noise
                 use_secure_random: bool = True,     # Use cryptographically secure random
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = False,            # Caching not applicable
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 3.3 Noise Generation

**REQ-UNIFORM-003 [MUST]** Support flexible noise generation:

```python
def _generate_noise(self, size: int) -> np.ndarray:
    """Generate uniform noise values."""
    if self.use_secure_random:
        # Use secrets for cryptographic security
        if isinstance(self.noise_range, tuple):
            min_noise, max_noise = self.noise_range
        else:
            min_noise, max_noise = -self.noise_range, self.noise_range
        
        # Generate secure random values
        noise = np.array([
            secrets.SystemRandom().uniform(min_noise, max_noise) 
            for _ in range(size)
        ])
    else:
        # Use NumPy for performance with optional seed
        rng = np.random.default_rng(self.random_seed)
        if isinstance(self.noise_range, tuple):
            noise = rng.uniform(self.noise_range[0], self.noise_range[1], size)
        else:
            noise = rng.uniform(-self.noise_range, self.noise_range, size)
    
    return noise * self.scale_factor
```

### 3.4 Noise Application

**REQ-UNIFORM-004 [MUST]** Apply noise with proper constraints:

```python
def _apply_noise(self, values: pd.Series, noise: np.ndarray) -> pd.Series:
    """Apply noise to values with constraints."""
    # Handle noise type
    if self.noise_type == "additive":
        noisy_values = values + noise
    elif self.noise_type == "multiplicative":
        noisy_values = values * (1 + noise)
    else:
        raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    # Apply bounds if specified
    if self.output_min is not None:
        noisy_values = np.maximum(noisy_values, self.output_min)
    if self.output_max is not None:
        noisy_values = np.minimum(noisy_values, self.output_max)
    
    # Handle integer fields
    if self.round_to_integer:
        noisy_values = np.round(noisy_values).astype(values.dtype)
    
    # Preserve zeros if requested
    if self.preserve_zero:
        zero_mask = values == 0
        noisy_values[zero_mask] = 0
    
    return pd.Series(noisy_values, index=values.index)
```

### 3.5 Statistical Scaling

**REQ-UNIFORM-005 [SHOULD]** Support noise scaling based on data statistics:

```python
def _calculate_scale_factor(self, series: pd.Series) -> float:
    """Calculate noise scale factor based on data statistics."""
    if not self.scale_by_std:
        return self.scale_factor
    
    # Calculate standard deviation
    std = series.std()
    if std > 0:
        return self.scale_factor * std
    else:
        self.logger.warning("Standard deviation is zero, using base scale factor")
        return self.scale_factor
```

### 3.6 Batch Processing

**REQ-UNIFORM-006 [MUST]** Process batches efficiently:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process batch with noise addition."""
    result = batch.copy()
    
    # Use op_field_utils for output field naming
    output_col = generate_output_field_name(
        self.field_name, self.mode, self.output_field_name,
        operation_suffix="noisy", column_prefix="_"
    )
    
    # Get numeric values
    values = batch[self.field_name]
    
    # Auto-detect integer type if not specified
    if self.round_to_integer is None:
        self.round_to_integer = pd.api.types.is_integer_dtype(values)
    
    # Calculate scale factor if needed
    if self.scale_by_std and not hasattr(self, '_scale_factor_calculated'):
        self._scale_factor_calculated = self._calculate_scale_factor(values)
    
    # Handle nulls
    non_null_mask = values.notna()
    non_null_values = values[non_null_mask]
    
    if len(non_null_values) > 0:
        # Generate and apply noise
        noise = self._generate_noise(len(non_null_values))
        noisy_values = self._apply_noise(non_null_values, noise)
        
        # Update result
        if self.mode == "REPLACE":
            result.loc[non_null_mask, self.field_name] = noisy_values
        else:
            result[output_col] = values.copy()
            result.loc[non_null_mask, output_col] = noisy_values
    
    return result
```

### 3.7 Metrics Collection

**REQ-UNIFORM-007 [MUST]** Collect noise-specific metrics:

```python
def _collect_specific_metrics(self, original_data: pd.Series,
                             noisy_data: pd.Series) -> Dict[str, Any]:
    """Collect uniform noise metrics."""
    # Calculate actual noise added
    actual_noise = noisy_data - original_data
    
    metrics = {
        "noise_type": self.noise_type,
        "noise_range": self.noise_range,
        "actual_noise_mean": float(actual_noise.mean()),
        "actual_noise_std": float(actual_noise.std()),
        "actual_noise_min": float(actual_noise.min()),
        "actual_noise_max": float(actual_noise.max()),
        "signal_to_noise_ratio": float(original_data.std() / actual_noise.std()) 
                                if actual_noise.std() > 0 else float('inf'),
        "values_at_bounds": {
            "at_min": int((noisy_data == self.output_min).sum()) if self.output_min else 0,
            "at_max": int((noisy_data == self.output_max).sum()) if self.output_max else 0
        },
        "preserved_zeros": int((original_data == 0).sum()) if self.preserve_zero else None,
        "secure_random": self.use_secure_random
    }
    
    return metrics
```

## 4. Uniform Temporal Noise Operation

### 4.1 Overview

**REQ-TEMPORAL-001 [MUST]** The `UniformTemporalNoiseOperation` adds uniform random time shifts to datetime fields.

### 4.2 Constructor Interface

**REQ-TEMPORAL-002 [MUST]** Constructor signature:

```python
class UniformTemporalNoiseOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Datetime field to add noise to
                 # Temporal noise parameters
                 noise_range_days: Optional[float] = None,    # Days component
                 noise_range_hours: Optional[float] = None,   # Hours component  
                 noise_range_minutes: Optional[float] = None, # Minutes component
                 noise_range_seconds: Optional[float] = None, # Seconds component
                 # Direction control
                 direction: str = "both",            # both, forward, backward
                 # Boundary constraints
                 min_datetime: Optional[Union[str, pd.Timestamp]] = None,
                 max_datetime: Optional[Union[str, pd.Timestamp]] = None,
                 # Special date handling
                 preserve_special_dates: bool = False,
                 special_dates: Optional[List[Union[str, pd.Timestamp]]] = None,
                 preserve_weekends: bool = False,
                 preserve_time_of_day: bool = False, # Keep hour/minute/second unchanged
                 # Granularity
                 output_granularity: Optional[str] = None,  # day, hour, minute, second
                 # Reproducibility
                 random_seed: Optional[int] = None,
                 use_secure_random: bool = True,
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = False,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 4.3 Temporal Noise Generation

**REQ-TEMPORAL-003 [MUST]** Generate time shifts in specified units:

```python
def _generate_time_shifts(self, size: int) -> pd.TimedeltaIndex:
    """Generate random time shifts."""
    # Calculate total shift range in seconds
    total_seconds = 0
    if self.noise_range_days:
        total_seconds += self.noise_range_days * 86400
    if self.noise_range_hours:
        total_seconds += self.noise_range_hours * 3600
    if self.noise_range_minutes:
        total_seconds += self.noise_range_minutes * 60
    if self.noise_range_seconds:
        total_seconds += self.noise_range_seconds
    
    # Generate random shifts
    if self.use_secure_random:
        if self.direction == "both":
            shifts = [secrets.SystemRandom().uniform(-total_seconds, total_seconds) 
                     for _ in range(size)]
        elif self.direction == "forward":
            shifts = [secrets.SystemRandom().uniform(0, total_seconds) 
                     for _ in range(size)]
        else:  # backward
            shifts = [secrets.SystemRandom().uniform(-total_seconds, 0) 
                     for _ in range(size)]
    else:
        rng = np.random.default_rng(self.random_seed)
        if self.direction == "both":
            shifts = rng.uniform(-total_seconds, total_seconds, size)
        elif self.direction == "forward":
            shifts = rng.uniform(0, total_seconds, size)
        else:  # backward
            shifts = rng.uniform(-total_seconds, 0, size)
    
    # Convert to TimedeltaIndex
    return pd.to_timedelta(shifts, unit='s')
```

### 4.4 Temporal Constraints

**REQ-TEMPORAL-004 [MUST]** Apply temporal constraints and preservation rules:

```python
def _apply_temporal_noise(self, timestamps: pd.Series, 
                         shifts: pd.TimedeltaIndex) -> pd.Series:
    """Apply noise with temporal constraints."""
    # Apply shifts
    noisy_timestamps = timestamps + shifts
    
    # Apply boundary constraints
    if self.min_datetime:
        min_ts = pd.Timestamp(self.min_datetime)
        noisy_timestamps = noisy_timestamps.clip(lower=min_ts)
    if self.max_datetime:
        max_ts = pd.Timestamp(self.max_datetime)
        noisy_timestamps = noisy_timestamps.clip(upper=max_ts)
    
    # Preserve special dates
    if self.preserve_special_dates and self.special_dates:
        special_ts = pd.to_datetime(self.special_dates)
        for special_date in special_ts:
            mask = timestamps.dt.date == special_date.date()
            noisy_timestamps[mask] = timestamps[mask]
    
    # Preserve weekends
    if self.preserve_weekends:
        weekend_mask = timestamps.dt.dayofweek.isin([5, 6])
        # Adjust shifts to keep weekends as weekends
        noisy_timestamps = self._adjust_for_weekends(
            timestamps, noisy_timestamps, weekend_mask
        )
    
    # Preserve time of day
    if self.preserve_time_of_day:
        noisy_timestamps = pd.to_datetime(
            noisy_timestamps.dt.date.astype(str) + ' ' + 
            timestamps.dt.time.astype(str)
        )
    
    # Apply granularity
    if self.output_granularity:
        noisy_timestamps = self._apply_granularity(noisy_timestamps)
    
    return noisy_timestamps
```

### 4.5 Granularity Handling

**REQ-TEMPORAL-005 [SHOULD]** Support output granularity control:

```python
def _apply_granularity(self, timestamps: pd.Series) -> pd.Series:
    """Round timestamps to specified granularity."""
    if self.output_granularity == "day":
        return timestamps.dt.floor('D')
    elif self.output_granularity == "hour":
        return timestamps.dt.floor('H')
    elif self.output_granularity == "minute":
        return timestamps.dt.floor('T')
    elif self.output_granularity == "second":
        return timestamps.dt.floor('S')
    else:
        return timestamps
```

### 4.6 Batch Processing

**REQ-TEMPORAL-006 [MUST]** Process datetime batches:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process batch with temporal noise."""
    result = batch.copy()
    
    # Validate datetime field
    if not pd.api.types.is_datetime64_any_dtype(batch[self.field_name]):
        # Try to convert
        try:
            datetime_series = pd.to_datetime(batch[self.field_name])
        except:
            raise ValueError(f"Field {self.field_name} cannot be converted to datetime")
    else:
        datetime_series = batch[self.field_name]
    
    # Use op_field_utils for output field
    output_col = generate_output_field_name(
        self.field_name, self.mode, self.output_field_name,
        operation_suffix="shifted", column_prefix="_"
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
        else:
            result[output_col] = datetime_series.copy()
            result.loc[non_null_mask, output_col] = noisy_values
    
    return result
```

### 4.7 Metrics Collection

**REQ-TEMPORAL-007 [MUST]** Collect temporal noise metrics:

```python
def _collect_specific_metrics(self, original_data: pd.Series,
                             noisy_data: pd.Series) -> Dict[str, Any]:
    """Collect temporal noise metrics."""
    # Calculate actual shifts
    shifts = (noisy_data - original_data).dt.total_seconds()
    
    metrics = {
        "noise_range_config": {
            "days": self.noise_range_days,
            "hours": self.noise_range_hours,
            "minutes": self.noise_range_minutes,
            "seconds": self.noise_range_seconds
        },
        "actual_shifts": {
            "mean_seconds": float(shifts.mean()),
            "std_seconds": float(shifts.std()),
            "min_seconds": float(shifts.min()),
            "max_seconds": float(shifts.max()),
            "mean_days": float(shifts.mean() / 86400),
            "max_days": float(shifts.abs().max() / 86400)
        },
        "direction": self.direction,
        "constraints_applied": {
            "min_datetime": str(self.min_datetime) if self.min_datetime else None,
            "max_datetime": str(self.max_datetime) if self.max_datetime else None,
            "values_at_min": int((noisy_data == self.min_datetime).sum()) if self.min_datetime else 0,
            "values_at_max": int((noisy_data == self.max_datetime).sum()) if self.max_datetime else 0
        },
        "preservation": {
            "special_dates_preserved": self.preserve_special_dates,
            "weekends_preserved": self.preserve_weekends,
            "time_of_day_preserved": self.preserve_time_of_day
        },
        "output_granularity": self.output_granularity,
        "secure_random": self.use_secure_random
    }
    
    return metrics
```

## 5. Common Utilities

### 5.1 Noise Utilities (commons/noise_utils.py)

**REQ-NOISE-005 [MUST]** Provide common noise utilities:

```python
"""Noise operation utilities for PAMOLA.CORE."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

def calculate_noise_impact(original: pd.Series, 
                          noisy: pd.Series) -> Dict[str, float]:
    """
    Calculate the impact of noise on data utility.
    
    Returns:
        Dict with rmse, mae, correlation, etc.
    """
    # Remove nulls for comparison
    mask = original.notna() & noisy.notna()
    orig_clean = original[mask]
    noisy_clean = noisy[mask]
    
    if len(orig_clean) == 0:
        return {}
    
    return {
        "rmse": float(np.sqrt(np.mean((orig_clean - noisy_clean) ** 2))),
        "mae": float(np.mean(np.abs(orig_clean - noisy_clean))),
        "correlation": float(orig_clean.corr(noisy_clean)),
        "relative_error": float(np.mean(np.abs(orig_clean - noisy_clean) / (np.abs(orig_clean) + 1e-10))),
        "max_absolute_error": float(np.max(np.abs(orig_clean - noisy_clean)))
    }

def calculate_distribution_preservation(original: pd.Series,
                                      noisy: pd.Series,
                                      n_bins: int = 20) -> Dict[str, Any]:
    """
    Analyze how well noise preserves data distribution.
    
    Returns:
        Dict with KS statistic, histogram distance, etc.
    """
    # Clean data
    orig_clean = original.dropna()
    noisy_clean = noisy.dropna()
    
    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(orig_clean, noisy_clean)
    
    # Histogram comparison
    bins = np.histogram_bin_edges(np.concatenate([orig_clean, noisy_clean]), bins=n_bins)
    hist_orig, _ = np.histogram(orig_clean, bins=bins, density=True)
    hist_noisy, _ = np.histogram(noisy_clean, bins=bins, density=True)
    
    # Earth mover's distance approximation
    hist_distance = np.sum(np.abs(hist_orig - hist_noisy)) * (bins[1] - bins[0])
    
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "histogram_distance": float(hist_distance),
        "mean_shift": float(noisy_clean.mean() - orig_clean.mean()),
        "std_ratio": float(noisy_clean.std() / orig_clean.std()) if orig_clean.std() > 0 else float('inf'),
        "percentile_shifts": {
            "p25": float(noisy_clean.quantile(0.25) - orig_clean.quantile(0.25)),
            "p50": float(noisy_clean.quantile(0.50) - orig_clean.quantile(0.50)),
            "p75": float(noisy_clean.quantile(0.75) - orig_clean.quantile(0.75))
        }
    }

def suggest_noise_range(series: pd.Series,
                       target_snr: float = 10.0,
                       noise_type: str = "additive") -> float:
    """
    Suggest appropriate noise range based on target signal-to-noise ratio.
    
    Args:
        series: Original data
        target_snr: Desired signal-to-noise ratio
        noise_type: Type of noise (additive/multiplicative)
        
    Returns:
        Suggested noise range
    """
    std = series.std()
    
    if noise_type == "additive":
        # For uniform noise: std = range / sqrt(12)
        # SNR = signal_std / noise_std
        noise_std = std / target_snr
        suggested_range = noise_std * np.sqrt(12)
    else:  # multiplicative
        # For multiplicative: noise affects proportionally
        suggested_range = 1.0 / target_snr
    
    return float(suggested_range)

def validate_noise_bounds(series: pd.Series,
                         noise_range: Union[float, Tuple[float, float]],
                         output_min: Optional[float] = None,
                         output_max: Optional[float] = None) -> Dict[str, Any]:
    """Validate that noise bounds are reasonable."""
    data_min = series.min()
    data_max = series.max()
    data_range = data_max - data_min
    
    # Calculate noise magnitude
    if isinstance(noise_range, tuple):
        noise_magnitude = max(abs(noise_range[0]), abs(noise_range[1]))
    else:
        noise_magnitude = abs(noise_range)
    
    warnings = []
    
    # Check if noise is too large
    if noise_magnitude > data_range:
        warnings.append("Noise range exceeds data range")
    
    # Check output bounds
    if output_min is not None and output_min > data_min:
        warnings.append(f"Output minimum ({output_min}) exceeds data minimum ({data_min})")
    
    if output_max is not None and output_max < data_max:
        warnings.append(f"Output maximum ({output_max}) is less than data maximum ({data_max})")
    
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "noise_to_data_ratio": noise_magnitude / data_range if data_range > 0 else float('inf'),
        "data_bounds": {"min": float(data_min), "max": float(data_max)},
        "noise_magnitude": noise_magnitude
    }
```

### 5.2 Statistical Utilities (commons/statistical_utils.py)

**REQ-NOISE-006 [SHOULD]** Provide statistical analysis utilities:

```python
"""Statistical utilities for noise operations."""

def calculate_utility_metrics(original: pd.Series,
                            transformed: pd.Series,
                            metric_set: str = "standard") -> Dict[str, float]:
    """
    Calculate utility preservation metrics.
    
    Metric sets:
        - "standard": Basic statistics preservation
        - "detailed": Include distribution metrics
        - "minimal": Only essential metrics
    """
    metrics = {}
    
    # Basic metrics (always included)
    metrics["mean_preservation"] = 1 - abs(original.mean() - transformed.mean()) / (abs(original.mean()) + 1e-10)
    metrics["std_preservation"] = min(transformed.std() / original.std(), original.std() / transformed.std()) if original.std() > 0 else 0
    
    if metric_set in ["standard", "detailed"]:
        # Correlation
        metrics["correlation"] = original.corr(transformed)
        
        # Rank correlation
        metrics["rank_correlation"] = original.corr(transformed, method='spearman')
    
    if metric_set == "detailed":
        # Distribution metrics
        metrics["skewness_diff"] = abs(original.skew() - transformed.skew())
        metrics["kurtosis_diff"] = abs(original.kurtosis() - transformed.kurtosis())
        
        # Percentile preservation
        for p in [10, 25, 50, 75, 90]:
            orig_p = original.quantile(p/100)
            trans_p = transformed.quantile(p/100)
            metrics[f"p{p}_preservation"] = 1 - abs(orig_p - trans_p) / (abs(orig_p) + 1e-10)
    
    return metrics

def analyze_noise_distribution(noise_values: np.ndarray) -> Dict[str, Any]:
    """Analyze the actual distribution of noise values."""
    return {
        "mean": float(np.mean(noise_values)),
        "std": float(np.std(noise_values)),
        "min": float(np.min(noise_values)),
        "max": float(np.max(noise_values)),
        "skewness": float(stats.skew(noise_values)),
        "kurtosis": float(stats.kurtosis(noise_values)),
        "uniformity_test": {
            "statistic": float(stats.kstest(noise_values, 'uniform')[0]),
            "pvalue": float(stats.kstest(noise_values, 'uniform')[1])
        }
    }
```

## 6. Integration Requirements

### 6.1 Validation Integration

**REQ-NOISE-007 [MUST]** Use commons validation framework:

```python
from pamola_core.anonymization.commons.validation import (
    NumericFieldValidator,
    DateTimeFieldValidator,
    validate_range
)

def validate_configuration(self) -> None:
    """Validate noise operation configuration."""
    if isinstance(self, UniformNumericNoiseOperation):
        # Validate numeric field
        validator = NumericFieldValidator()
        result = validator.validate(self.df[self.field_name])
        if not result.is_valid:
            raise ValidationError(result.errors[0])
        
        # Validate noise range
        if isinstance(self.noise_range, tuple):
            if self.noise_range[0] >= self.noise_range[1]:
                raise ValueError("Invalid noise range: min >= max")
    
    elif isinstance(self, UniformTemporalNoiseOperation):
        # Validate datetime field
        validator = DateTimeFieldValidator()
        result = validator.validate(self.df[self.field_name])
        if not result.is_valid:
            raise ValidationError(result.errors[0])
```

### 6.2 Metrics Integration

**REQ-NOISE-008 [MUST]** Use standard metrics collection:

```python
from pamola_core.anonymization.commons.metric_utils import (
    collect_operation_metrics,
    save_process_metrics
)

def _collect_all_metrics(self, original_data: pd.Series,
                        noisy_data: pd.Series) -> Dict[str, Any]:
    """Collect comprehensive metrics."""
    # Standard operation metrics
    metrics = collect_operation_metrics(
        operation_type="noise_addition",
        original_data=original_data,
        processed_data=noisy_data,
        operation_params=self._get_operation_params(),
        timing_info=self._get_timing_info()
    )
    
    # Add noise-specific metrics
    metrics.update(self._collect_specific_metrics(original_data, noisy_data))
    
    # Add utility metrics
    from pamola_core.anonymization.commons.noise_utils import calculate_noise_impact
    metrics["utility"] = calculate_noise_impact(original_data, noisy_data)
    
    return metrics
```

### 6.3 Visualization Integration

**REQ-NOISE-009 [SHOULD]** Generate noise visualizations:

```python
from pamola_core.anonymization.commons.visualization_utils import (
    create_comparison_visualization,
    create_metric_visualization
)

def _generate_visualizations(self, original_data: pd.Series,
                           noisy_data: pd.Series,
                           task_dir: Path,
                           writer: DataWriter) -> Dict[str, Path]:
    """Generate noise-specific visualizations."""
    vis_paths = {}
    
    # Standard before/after comparison
    comparison_path = create_comparison_visualization(
        original_data, noisy_data, task_dir,
        self.field_name, "noise_addition"
    )
    if comparison_path:
        vis_paths["comparison"] = comparison_path
    
    # Noise distribution visualization
    if isinstance(self, UniformNumericNoiseOperation):
        noise_values = noisy_data - original_data
        noise_dist_data = {
            "mean": float(noise_values.mean()),
            "std": float(noise_values.std()),
            "min": float(noise_values.min()),
            "max": float(noise_values.max())
        }
        
        dist_path = create_metric_visualization(
            "noise_distribution", noise_dist_data, task_dir,
            self.field_name, "uniform_noise"
        )
        if dist_path:
            vis_paths["noise_distribution"] = dist_path
    
    return vis_paths
```

## 7. Package Initialization

### 7.1 Package __init__.py

```python
"""
PAMOLA.CORE Noise Operations Package

This package provides noise addition operations for privacy preservation:
- UniformNumericNoiseOperation: Add uniform noise to numeric fields
- UniformTemporalNoiseOperation: Add temporal shifts to datetime fields

Future versions will include differential privacy mechanisms.
"""

from .uniform_numeric_op import UniformNumericNoiseOperation
from .uniform_temporal_op import UniformTemporalNoiseOperation

__all__ = [
    'UniformNumericNoiseOperation',
    'UniformTemporalNoiseOperation'
]

__version__ = '1.0.0'
```

## 8. Error Handling

### 8.1 Common Errors

**REQ-NOISE-010 [MUST]** Handle these error conditions:

1. **Invalid noise ranges**: Negative ranges, min > max
2. **Type mismatches**: Non-numeric fields for numeric noise
3. **Boundary violations**: Output exceeding specified bounds
4. **Random generation failures**: Entropy exhaustion
5. **Memory errors**: Large batch processing

### 8.2 Error Recovery

**REQ-NOISE-011 [MUST]** Implement graceful error handling:

```python
def _handle_error(self, error: Exception, batch_index: int) -> None:
    """Handle errors during noise addition."""
    if isinstance(error, ValueError) and "cannot convert" in str(error):
        self.logger.error(f"Data type error in batch {batch_index}: {error}")
        if self.continue_on_error:
            return self._skip_batch(batch_index)
        raise
    
    # For random generation errors, retry with different seed
    if isinstance(error, SystemError) and "random" in str(error):
        self.logger.warning(f"Random generation error, retrying with new seed")
        self.random_seed = None  # Reset to use system entropy
        return self._retry_batch(batch_index)
    
    super()._handle_error(error, batch_index)
```

## 9. Testing Requirements

### 9.1 Unit Tests

**REQ-NOISE-012 [MUST]** Test coverage must include:

1. **Numeric Noise Tests**:
   - Uniform distribution validation
   - Boundary constraint enforcement
   - Integer preservation
   - Zero preservation
   - Statistical scaling

2. **Temporal Noise Tests**:
   - Time shift generation
   - Boundary constraints
   - Special date preservation
   - Weekend preservation
   - Granularity rounding

3. **Common Tests**:
   - Null value handling
   - Batch processing
   - Error handling
   - Metrics calculation

### 9.2 Statistical Tests

**REQ-NOISE-013 [SHOULD]** Include statistical validation:

```python
def test_uniform_distribution():
    """Verify noise follows uniform distribution."""
    op = UniformNumericNoiseOperation(
        field_name="value",
        noise_range=10.0,
        random_seed=42
    )
    
    # Generate large sample
    original = pd.Series(np.zeros(10000))
    noisy = op.process_batch(pd.DataFrame({"value": original}))["value"]
    noise = noisy - original
    
    # Kolmogorov-Smirnov test for uniformity
    ks_stat, p_value = stats.kstest(
        noise, 'uniform', args=(-10, 20)  # loc=-10, scale=20
    )
    assert p_value > 0.05  # Fail to reject uniform hypothesis

def test_utility_preservation():
    """Verify utility metrics are within acceptable bounds."""
    # Test that correlation remains high
    # Test that distribution shape is preserved
    # Test that key statistics are maintained
```

## 10. Performance Considerations

### 10.1 Optimization Strategies

**REQ-NOISE-014 [SHOULD]** Optimize for performance:

- Use vectorized NumPy operations
- Pre-allocate arrays for large batches
- Cache random generators
- Use parallel processing for independent batches

### 10.2 Memory Management

**REQ-NOISE-015 [SHOULD]** Manage memory efficiently:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process with memory optimization."""
    # For large numeric arrays, use in-place operations
    if len(batch) > 50000 and self.mode == "REPLACE":
        # Modify in place to save memory
        noise = self._generate_noise(len(batch))
        batch[self.field_name] += noise  # In-place addition
        return batch
    else:
        # Standard processing for smaller batches
        return super().process_batch(batch)
```

## 11. Future Extensions

### 11.1 Differential Privacy (Post-MVP)

Future versions will add:
- Laplace mechanism for ε-differential privacy
- Gaussian mechanism for (ε,δ)-differential privacy
- Privacy budget tracking
- Sensitivity calculation

### 11.2 Advanced Noise Types

- Exponential noise
- Geometric noise (for discrete data)
- Truncated normal noise
- Custom distribution support

## 12. Example Usage

### 12.1 Numeric Noise

```python
# Basic uniform noise
operation = UniformNumericNoiseOperation(
    field_name="salary",
    noise_range=5000,  # ±5000
    output_min=0,      # No negative salaries
    round_to_integer=True
)

# Scaled noise based on data variance
operation = UniformNumericNoiseOperation(
    field_name="age",
    noise_range=0.1,   # Base range
    scale_by_std=True, # Scale by standard deviation
    output_min=0,
    output_max=120
)

# Multiplicative noise
operation = UniformNumericNoiseOperation(
    field_name="revenue",
    noise_range=0.05,  # ±5% variation
    noise_type="multiplicative",
    preserve_zero=True
)
```

### 12.2 Temporal Noise

```python
# Date shifting within 30 days
operation = UniformTemporalNoiseOperation(
    field_name="birth_date",
    noise_range_days=30,
    direction="both",
    output_granularity="day"
)

# Time-preserving date shift
operation = UniformTemporalNoiseOperation(
    field_name="appointment_time",
    noise_range_days=7,
    preserve_time_of_day=True,
    preserve_weekends=True
)

# Bounded temporal noise
operation = UniformTemporalNoiseOperation(
    field_name="transaction_date",
    noise_range_hours=24,
    min_datetime="2024-01-01",
    max_datetime="2024-12-31"
)
```

## 13. Summary

The noise operations package provides essential privacy-preserving transformations through controlled randomization:

- **Uniform Numeric Noise**: Flexible noise addition for all numeric types with comprehensive constraints
- **Uniform Temporal Noise**: Sophisticated datetime shifting with preservation options

Key design principles:
- Integration with existing PAMOLA.CORE infrastructure
- Comprehensive validation and metrics
- Production-ready error handling
- Statistical rigor with utility preservation
- Foundation for future differential privacy extensions