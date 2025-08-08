# UniformTemporalNoiseOperation Documentation

## Overview

The `UniformTemporalNoiseOperation` is a privacy-preserving transformation that adds uniformly distributed random time shifts to datetime fields in datasets. This operation is part of the PAMOLA.CORE anonymization framework and provides sophisticated temporal perturbation while maintaining important date/time patterns.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Import](#installation--import)
3. [Basic Usage](#basic-usage)
4. [Parameters](#parameters)
5. [Advanced Features](#advanced-features)
6. [Pattern Preservation](#pattern-preservation)
7. [Metrics & Analysis](#metrics--analysis)
8. [Performance Considerations](#performance-considerations)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

## Introduction

### Purpose

The UniformTemporalNoiseOperation adds controlled random time shifts to datetime data to prevent temporal analysis attacks while preserving important temporal patterns. It's particularly useful for:

- Protecting sensitive dates (birth dates, medical appointments)
- Anonymizing transaction timestamps
- Perturbing event timelines
- Obscuring activity patterns
- Meeting privacy regulations for temporal data

### Key Features

- **Flexible Time Units**: Specify shifts in days, hours, minutes, and/or seconds
- **Directional Control**: Shift forward, backward, or both directions
- **Pattern Preservation**: Maintain weekends, special dates, time-of-day
- **Boundary Enforcement**: Ensure dates stay within valid ranges
- **Granularity Control**: Round output to specific time units
- **Secure Generation**: Cryptographically secure random shifts
- **Large-scale Support**: Integrated Dask support for big data

## Installation & Import

```python
from pamola_core.anonymization.noise import UniformTemporalNoiseOperation

# Or import from the package root
from pamola_core.anonymization import UniformTemporalNoiseOperation
```

## Basic Usage

### Simple Example

```python
import pandas as pd
from pamola_core.anonymization.noise import UniformTemporalNoiseOperation

# Create sample data
df = pd.DataFrame({
    'user_id': range(1000),
    'birth_date': pd.date_range('1950-01-01', periods=1000, freq='D'),
    'last_login': pd.date_range('2024-01-01', periods=1000, freq='H')
})

# Add ±30 days noise to birth dates
birth_date_noise = UniformTemporalNoiseOperation(
    field_name='birth_date',
    noise_range_days=30,
    direction='both'
)

# Execute the operation
result = birth_date_noise.execute(
    data_source=data_source,
    task_dir=Path('./output'),
    reporter=reporter
)
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_name` | str | The name of the datetime field to add noise to |

### Time Range Parameters

At least one of these must be specified:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_range_days` | float | None | Time shift range in days |
| `noise_range_hours` | float | None | Time shift range in hours |
| `noise_range_minutes` | float | None | Time shift range in minutes |
| `noise_range_seconds` | float | None | Time shift range in seconds |

### Shift Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | str | "both" | Direction: "both", "forward", or "backward" |
| `min_datetime` | datetime/str | None | Minimum allowed datetime after shift |
| `max_datetime` | datetime/str | None | Maximum allowed datetime after shift |
| `output_granularity` | str | None | Round to: "day", "hour", "minute", "second" |

### Preservation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preserve_special_dates` | bool | False | Keep special dates unchanged |
| `special_dates` | list | None | List of dates to preserve |
| `preserve_weekends` | bool | False | Maintain weekend/weekday status |
| `preserve_time_of_day` | bool | False | Keep time unchanged (shift date only) |

### Security Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_secure_random` | bool | True | Use cryptographically secure random |
| `random_seed` | int | None | Seed for reproducible noise |

### Standard Operation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "REPLACE" | "REPLACE" or "ENRICH" |
| `output_field_name` | str | None | Output field name (ENRICH mode) |
| `null_strategy` | str | "PRESERVE" | How to handle nulls |
| `batch_size` | int | 10000 | Batch size for processing |
| `engine` | str | "auto" | Processing engine: "pandas", "dask", "auto" |

## Advanced Features

### 1. Multi-Unit Time Specification

Combine multiple time units for precise control:

```python
# Add shifts of up to 7 days, 12 hours, and 30 minutes
operation = UniformTemporalNoiseOperation(
    field_name='appointment_time',
    noise_range_days=7,
    noise_range_hours=12,
    noise_range_minutes=30
)
```

### 2. Directional Shifting

Control the direction of time shifts:

```python
# Only shift dates to the future (forward)
future_shift = UniformTemporalNoiseOperation(
    field_name='release_date',
    noise_range_days=14,
    direction='forward'
)

# Only shift dates to the past (backward)
past_shift = UniformTemporalNoiseOperation(
    field_name='expiry_date',
    noise_range_days=30,
    direction='backward'
)
```

### 3. Boundary Constraints

Ensure dates stay within valid ranges:

```python
# Keep dates within a specific year
operation = UniformTemporalNoiseOperation(
    field_name='transaction_date',
    noise_range_days=30,
    min_datetime='2024-01-01',
    max_datetime='2024-12-31'
)
```

### 4. Output Granularity

Round shifted dates to specific units:

```python
# Round to nearest day (remove time component)
operation = UniformTemporalNoiseOperation(
    field_name='birth_date',
    noise_range_days=15,
    output_granularity='day'
)

# Round to nearest hour
operation = UniformTemporalNoiseOperation(
    field_name='log_timestamp',
    noise_range_hours=24,
    output_granularity='hour'
)
```

## Pattern Preservation

### 1. Weekend Preservation

Maintain the weekend/weekday status of dates:

```python
# Shift dates but keep weekends as weekends
operation = UniformTemporalNoiseOperation(
    field_name='work_date',
    noise_range_days=5,
    preserve_weekends=True
)
```

The algorithm intelligently adjusts shifts to ensure:
- Weekdays remain weekdays (Monday-Friday)
- Weekends remain weekends (Saturday-Sunday)
- Shifts are minimally adjusted to maintain patterns

### 2. Special Date Preservation

Keep important dates unchanged:

```python
# Preserve holidays and special events
holidays = ['2024-01-01', '2024-07-04', '2024-12-25']

operation = UniformTemporalNoiseOperation(
    field_name='event_date',
    noise_range_days=7,
    preserve_special_dates=True,
    special_dates=holidays
)
```

### 3. Time-of-Day Preservation

Shift dates while keeping times unchanged:

```python
# Change dates but preserve exact times
operation = UniformTemporalNoiseOperation(
    field_name='appointment_datetime',
    noise_range_days=14,
    preserve_time_of_day=True
)
# Example: 2024-03-15 14:30:00 → 2024-03-22 14:30:00
```

## Metrics & Analysis

The operation collects comprehensive temporal metrics:

### Shift Statistics
- Mean, standard deviation, min/max shifts (in seconds and days)
- Direction distribution (forward/backward/unchanged)
- Boundary constraint violations

### Pattern Preservation Metrics
- Weekday preservation rate
- Hour-of-day preservation rate
- Weekend status preservation
- Special date preservation count

### Temporal Ordering Metrics
- Order preservation (Kendall's tau)
- Inversion count
- Temporal sequence integrity

### Example: Accessing Metrics

```python
# Execute operation
result = operation.execute(data_source, task_dir, reporter)

# Access temporal metrics
metrics = result.metrics
temporal_impact = metrics['temporal_impact']

print(f"Mean shift: {temporal_impact['shift_statistics']['mean_days']:.1f} days")
print(f"Weekend preservation: {temporal_impact['pattern_preservation']['weekend_preserved']:.1%}")
print(f"Order preservation: {temporal_impact['ordering_preservation']['kendall_tau']:.3f}")
```

## Performance Considerations

### Memory Optimization

```python
# Enable adaptive batch sizing for large datasets
operation = UniformTemporalNoiseOperation(
    field_name='timestamp',
    noise_range_hours=48,
    adaptive_batch_size=True
)
```

### Large Dataset Processing

For datasets with millions of datetime values:

```python
# Configure for large-scale processing
operation = UniformTemporalNoiseOperation(
    field_name='event_time',
    noise_range_days=7,
    engine='auto',              # Automatic Dask switching
    max_rows_in_memory=500000,  # Dask threshold
    batch_size=50000           # Larger batches for efficiency
)
```

### Performance Tips

1. **Granularity**: Coarser granularity (day vs. second) improves performance
2. **Pattern Preservation**: Weekend preservation requires additional computation
3. **Boundary Checks**: Minimize boundary constraints for better performance
4. **Secure Random**: Consider disabling for non-sensitive data

## Examples

### Example 1: Birth Date Anonymization

```python
# Protect birth dates with ±60 days noise, preserving age brackets
birth_date_noise = UniformTemporalNoiseOperation(
    field_name='date_of_birth',
    noise_range_days=60,
    direction='both',
    output_granularity='day',  # Remove time component
    min_datetime='1900-01-01', # Reasonable bounds
    max_datetime='2010-01-01'
)
```

### Example 2: Transaction Timestamp Protection

```python
# Anonymize transaction times within business constraints
transaction_noise = UniformTemporalNoiseOperation(
    field_name='transaction_timestamp',
    noise_range_hours=4,
    noise_range_minutes=30,
    preserve_weekends=True,    # Keep business day pattern
    min_datetime='2024-01-01',
    max_datetime='2024-12-31'
)
```

### Example 3: Medical Appointment Shifting

```python
# Shift appointments while preserving scheduling patterns
appointment_noise = UniformTemporalNoiseOperation(
    field_name='appointment_datetime',
    noise_range_days=7,
    preserve_time_of_day=True,  # Keep appointment times
    preserve_weekends=True,     # Maintain weekday appointments
    direction='both',
    special_dates=['2024-12-25', '2024-01-01']  # Preserve holidays
)
```

### Example 4: Log File Anonymization

```python
# Anonymize log timestamps for privacy
log_noise = UniformTemporalNoiseOperation(
    field_name='log_timestamp',
    noise_range_hours=2,
    noise_range_minutes=30,
    output_granularity='minute',  # Round to minutes
    mode='ENRICH',                # Keep original for debugging
    output_field_name='anon_timestamp'
)
```

### Example 5: Event Timeline Perturbation

```python
# Perturb event timeline while maintaining sequence
event_noise = UniformTemporalNoiseOperation(
    field_name='event_date',
    noise_range_days=14,
    direction='both',
    preserve_special_dates=True,
    special_dates=pd.date_range('2024-01-01', '2024-12-31', freq='MS'),  # Month starts
    output_granularity='day'
)

# Check if temporal ordering is preserved
result = event_noise.execute(data_source, task_dir)
ordering_preserved = result.metrics['temporal_impact']['ordering_preservation']['order_fully_preserved']
```

## Troubleshooting

### Common Issues

#### 1. Invalid Datetime Format
```python
# Error: Field cannot be converted to datetime
# Solution: Ensure proper datetime format
df['date_field'] = pd.to_datetime(df['date_field'], format='%Y-%m-%d')
```

#### 2. Boundary Violations
```python
# Warning: Many values clipped to boundaries
# Solution: Adjust boundaries or reduce noise range
operation = UniformTemporalNoiseOperation(
    field_name='date',
    noise_range_days=30,
    min_datetime=df['date'].min() - pd.Timedelta(days=30),
    max_datetime=df['date'].max() + pd.Timedelta(days=30)
)
```

#### 3. Weekend Preservation Conflicts
```python
# Issue: Weekend preservation causing excessive adjustments
# Solution: Reduce noise range or relax preservation
operation = UniformTemporalNoiseOperation(
    field_name='work_date',
    noise_range_days=3,  # Smaller range for better weekend preservation
    preserve_weekends=True
)
```

#### 4. Pattern Preservation Performance
```python
# Issue: Slow processing with pattern preservation
# Solution: Process in smaller batches or disable preservation
operation = UniformTemporalNoiseOperation(
    field_name='timestamp',
    noise_range_days=7,
    preserve_weekends=False,  # Disable for performance
    batch_size=25000         # Smaller batches
)
```

### Validation Helpers

```python
# Validate temporal data before processing
from pamola_core.anonymization.commons.validation import DateTimeFieldValidator

validator = DateTimeFieldValidator()
result = validator.validate(df['date_field'])
if not result.is_valid:
    print("Validation errors:", result.errors)

# Check shift effectiveness
from pamola_core.anonymization.commons.statistical_utils import analyze_temporal_noise_impact

impact = analyze_temporal_noise_impact(
    original_timestamps=df['original_date'],
    noisy_timestamps=df['noisy_date']
)
```

## Best Practices

### 1. Privacy vs. Utility Balance

```python
# Start with smaller shifts and increase as needed
# Good practice: Test on sample data first
sample_df = df.sample(1000)
test_shifts = [7, 14, 30, 60]  # Days

for shift_days in test_shifts:
    op = UniformTemporalNoiseOperation(
        field_name='date',
        noise_range_days=shift_days
    )
    # Evaluate utility metrics
```

### 2. Pattern Preservation Strategy

```python
# Prioritize patterns based on use case
# Example: Business analytics
business_noise = UniformTemporalNoiseOperation(
    field_name='transaction_date',
    noise_range_days=5,
    preserve_weekends=True,      # Critical for business patterns
    preserve_time_of_day=False,  # Less important
    output_granularity='day'     # Aggregate by day anyway
)
```

### 3. Secure Configuration

```python
# Production configuration for sensitive data
secure_noise = UniformTemporalNoiseOperation(
    field_name='sensitive_date',
    noise_range_days=30,
    use_secure_random=True,      # Always for sensitive data
    random_seed=None,            # Never set seed in production
    min_datetime='2020-01-01',   # Reasonable bounds
    max_datetime='2025-12-31'
)
```

### 4. Documentation and Audit

```python
# Document noise parameters for reproducibility
config = {
    'operation': 'UniformTemporalNoiseOperation',
    'parameters': {
        'noise_range_days': 30,
        'direction': 'both',
        'preserve_weekends': True,
        'applied_date': datetime.now().isoformat()
    }
}

# Save configuration with results
import json
with open(task_dir / 'noise_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

## API Reference

### Class: UniformTemporalNoiseOperation

```python
class UniformTemporalNoiseOperation(AnonymizationOperation):
    """Operation for adding uniform random time shifts to datetime fields."""
    
    def __init__(self, field_name: str, **kwargs)
    def execute(self, data_source: DataSource, task_dir: Path, ...) -> OperationResult
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame
```

### Key Methods

- `execute()`: Main execution method for full dataset processing
- `process_batch()`: Process a single batch of datetime data
- `_generate_time_shifts()`: Generate random time deltas
- `_apply_temporal_noise()`: Apply shifts with constraints
- `_adjust_for_weekends()`: Intelligent weekend preservation
- `_apply_granularity()`: Round to specified time unit

### Integration Points

- Inherits from `AnonymizationOperation`
- Uses `SecureRandomGenerator` for shift generation
- Integrates with `DateTimeFieldValidator`
- Compatible with `ProgressTracker`
- Supports `OperationResult` for metrics

## Related Operations

- **UniformNumericNoiseOperation**: For adding noise to numeric fields
- **DateGeneralizationOperation**: For date generalization (binning)
- **Future**: Temporal pattern mining protection

## References

- [PAMOLA.CORE Noise Operations Sub-Specification](./specs/noise_srs.md)
- [Temporal Privacy Patterns](./guides/temporal_privacy.md)
- [Datetime Anonymization Best Practices](./guides/datetime_anon.md)