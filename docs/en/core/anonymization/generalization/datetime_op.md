# DateTime Generalization Operation

## Overview

The DateTime Generalization Operation is a privacy-preserving transformation that reduces the precision of datetime fields while maintaining temporal utility. This operation is part of the PAMOLA.CORE anonymization framework and implements multiple strategies for generalizing temporal data.

**Module:** `pamola_core.anonymization.generalization.datetime_op`  
**Version:** 2.0.1  
**Status:** Stable  
**Last Updated:** 2025-06-15

## Features

- **Multiple Generalization Strategies**
  - Rounding: Reduce precision to year, quarter, month, week, day, or hour
  - Binning: Group into time intervals (hour ranges, day ranges, business periods, seasons)
  - Component-based: Keep only specific datetime components
  - Relative: Express dates relative to a reference point

- **Robust Data Handling**
  - Support for various datetime formats
  - Timezone-aware processing
  - Handling of incomplete dates and null values
  - Validation of pandas datetime bounds

- **Privacy Protection**
  - Integration with k-anonymity risk assessment
  - Configurable privacy thresholds
  - Validation of anonymization effectiveness

- **Performance Optimization**
  - Vectorized operations for batch processing
  - Caching support for repeated operations
  - Memory-efficient processing

## Installation

The DateTime Generalization Operation is included in the PAMOLA.CORE package:

```bash
pip install pamola-core
```

## Basic Usage

### Simple Example

```python
from pamola_core.anonymization.generalization.datetime_op import DateTimeGeneralizationOperation

# Create operation to round dates to month
operation = DateTimeGeneralizationOperation(
    field_name="birth_date",
    strategy="rounding",
    rounding_unit="month"
)

# Process a DataFrame
anonymized_df = operation.process(df)
```

### Strategy Examples

#### 1. Rounding Strategy

```python
# Round to year
year_op = DateTimeGeneralizationOperation(
    field_name="registration_date",
    strategy="rounding",
    rounding_unit="year",
    output_format="%Y"
)

# Round to week
week_op = DateTimeGeneralizationOperation(
    field_name="event_timestamp",
    strategy="rounding",
    rounding_unit="week"
)
```

#### 2. Binning Strategy

```python
# Group into hour ranges
hour_bin_op = DateTimeGeneralizationOperation(
    field_name="access_time",
    strategy="binning",
    bin_type="hour_range",
    interval_size=4  # 4-hour bins
)

# Business periods
business_op = DateTimeGeneralizationOperation(
    field_name="transaction_time",
    strategy="binning",
    bin_type="business_period"  # Morning/Afternoon/Night
)

# Seasonal binning
season_op = DateTimeGeneralizationOperation(
    field_name="order_date",
    strategy="binning",
    bin_type="seasonal"  # Winter/Spring/Summer/Fall
)
```

#### 3. Component Strategy

```python
# Keep only year and month
component_op = DateTimeGeneralizationOperation(
    field_name="visit_date",
    strategy="component",
    keep_components=["year", "month"],
    output_format="%Y-%m"
)
```

#### 4. Relative Strategy

```python
# Express relative to current date
relative_op = DateTimeGeneralizationOperation(
    field_name="last_login",
    strategy="relative"
    # Uses current date as reference by default
)

# Express relative to specific date
relative_custom_op = DateTimeGeneralizationOperation(
    field_name="project_deadline",
    strategy="relative",
    reference_date="2025-01-01"
)
```

## Advanced Usage

### Timezone Handling

```python
# Convert to UTC
utc_op = DateTimeGeneralizationOperation(
    field_name="timestamp",
    strategy="rounding",
    rounding_unit="hour",
    timezone_handling="utc",
    default_timezone="America/New_York"  # For naive datetimes
)

# Remove timezone information
no_tz_op = DateTimeGeneralizationOperation(
    field_name="timestamp",
    strategy="rounding",
    rounding_unit="day",
    timezone_handling="remove"
)
```

### Custom Binning

```python
# Define custom bin boundaries
custom_bins = [
    "2025-01-01",
    "2025-04-01",
    "2025-07-01",
    "2025-10-01",
    "2026-01-01"
]

custom_bin_op = DateTimeGeneralizationOperation(
    field_name="date",
    strategy="binning",
    bin_type="custom",
    custom_bins=custom_bins
)
```

### Privacy Validation

```python
# Ensure at least 50% reduction in unique values
privacy_op = DateTimeGeneralizationOperation(
    field_name="sensitive_date",
    strategy="rounding",
    rounding_unit="month",
    min_privacy_threshold=0.5
)
```

### K-Anonymity Integration

```python
# Integrate with k-anonymity assessment
k_anon_op = DateTimeGeneralizationOperation(
    field_name="birth_date",
    strategy="rounding",
    rounding_unit="year",
    quasi_identifiers=["birth_date", "zip_code", "gender"],
    ka_risk_field="k_anonymity_risk",
    risk_threshold=5.0,
    vulnerable_record_strategy="suppress"
)
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Field to generalize |
| `strategy` | str | Required | Generalization strategy: "rounding", "binning", "component", "relative" |
| `mode` | str | "REPLACE" | Operation mode: "REPLACE" or "ENRICH" |
| `output_field_name` | str | None | Custom name for output field (ENRICH mode) |
| `null_strategy` | str | "PRESERVE" | How to handle nulls: "PRESERVE", "EXCLUDE", "ERROR", "ANONYMIZE" |

### Strategy-Specific Parameters

#### Rounding Strategy
| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `rounding_unit` | str | "year", "quarter", "month", "week", "day", "hour" | Unit to round to |

#### Binning Strategy
| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `bin_type` | str | "hour_range", "day_range", "business_period", "seasonal", "custom" | Type of binning |
| `interval_size` | int | > 0 | Size of intervals |
| `interval_unit` | str | "hours", "days", "weeks", "months" | Unit for intervals |
| `custom_bins` | list | datetime strings/objects | Custom bin boundaries |

#### Component Strategy
| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `keep_components` | list | ["year", "month", "day", "hour", "minute", "weekday"] | Components to keep |

#### Relative Strategy
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_date` | str/datetime | Current date | Reference point for relative dates |

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | str | None | Custom strftime format for output |
| `timezone_handling` | str | "preserve" | "preserve", "utc", "remove" |
| `default_timezone` | str | "UTC" | Default timezone for naive datetimes |
| `input_formats` | list | Common formats | List of formats to try when parsing |
| `min_privacy_threshold` | float | 0.3 | Minimum reduction in unique values (0-1) |
| `batch_size` | int | 10000 | Records per batch |
| `use_cache` | bool | True | Enable result caching |

## Error Handling

The operation defines custom exceptions for specific error cases:

```python
from pamola_core.anonymization.generalization.datetime_op import (
    DateTimeParsingError,
    DateTimeGeneralizationError,
    InsufficientPrivacyError
)

try:
    operation = DateTimeGeneralizationOperation(
        field_name="date_field",
        strategy="binning",
        bin_type="custom",
        custom_bins=["invalid", "dates"]
    )
except DateTimeParsingError as e:
    print(f"Failed to parse dates: {e}")

try:
    result = operation.process(df)
except InsufficientPrivacyError as e:
    print(f"Privacy requirements not met: {e}")
```

## Metrics and Reporting

The operation collects comprehensive metrics about the generalization:

```python
# Process data
result = operation.process(df)

# Access metrics
metrics = result.get_metrics()
print(f"Unique values before: {metrics['unique_patterns_before']}")
print(f"Unique values after: {metrics['unique_patterns_after']}")
print(f"Privacy reduction: {metrics['privacy_reduction_ratio']:.2%}")
print(f"Average temporal loss: {metrics.get('avg_temporal_loss_hours', 'N/A')} hours")
```

## Best Practices

### 1. Strategy Selection

- **Rounding**: Best for maintaining temporal ordering while reducing precision
- **Binning**: Ideal for creating categorical groups from continuous time
- **Component**: Useful when only certain date parts are needed
- **Relative**: Good for hiding exact dates while preserving recency information

### 2. Privacy Considerations

```python
# Start with coarser generalization
initial_op = DateTimeGeneralizationOperation(
    field_name="sensitive_date",
    strategy="rounding",
    rounding_unit="year"
)

# Refine if privacy allows
if initial_op.process(df).get_metrics()['privacy_reduction_ratio'] > 0.7:
    refined_op = DateTimeGeneralizationOperation(
        field_name="sensitive_date",
        strategy="rounding",
        rounding_unit="quarter"
    )
```

### 3. Format Consistency

```python
# Ensure consistent output format
operation = DateTimeGeneralizationOperation(
    field_name="mixed_dates",
    strategy="rounding",
    rounding_unit="month",
    input_formats=[
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m-%d-%Y"
    ],
    output_format="%Y-%m"
)
```

### 4. Performance Optimization

```python
# Use appropriate batch size for large datasets
large_data_op = DateTimeGeneralizationOperation(
    field_name="timestamp",
    strategy="binning",
    bin_type="hour_range",
    interval_size=6,
    batch_size=50000,  # Larger batches for better performance
    use_cache=True     # Cache results for repeated values
)
```

## Integration with PAMOLA Framework

### Pipeline Integration

```python
from pamola_core.pipeline import AnonymizationPipeline

pipeline = AnonymizationPipeline()

# Add datetime generalization
pipeline.add_operation(
    DateTimeGeneralizationOperation(
        field_name="birth_date",
        strategy="rounding",
        rounding_unit="year"
    )
)

# Add other operations
pipeline.add_operation(...)

# Process data
result = pipeline.process(df)
```

### Risk Assessment Integration

```python
from pamola_core.risk import RiskAssessment

# Configure with risk assessment
operation = DateTimeGeneralizationOperation(
    field_name="admission_date",
    strategy="rounding",
    rounding_unit="month",
    quasi_identifiers=["admission_date", "diagnosis", "age_group"],
    ka_risk_field="k_risk"
)

# Assess risk after anonymization
risk_assessment = RiskAssessment(quasi_identifiers=operation.quasi_identifiers)
risk_metrics = risk_assessment.assess(result.data)
```

## Troubleshooting

### Common Issues

1. **DateTimeParsingError**
   - Check input date formats
   - Verify data types
   - Handle mixed format columns

2. **InsufficientPrivacyError**
   - Use coarser generalization
   - Combine with other anonymization techniques
   - Adjust privacy threshold

3. **Timezone Issues**
   - Specify default_timezone for naive datetimes
   - Use consistent timezone handling across pipeline
   - Consider removing timezone info if not needed

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger('pamola_core.anonymization.generalization.datetime_op').setLevel(logging.DEBUG)

# Operation will now log detailed information
operation = DateTimeGeneralizationOperation(
    field_name="debug_date",
    strategy="binning",
    bin_type="day_range"
)
```

## Performance Considerations

- **Vectorized Operations**: The operation uses pandas vectorized operations for optimal performance
- **Memory Usage**: Batch processing controls memory usage for large datasets
- **Caching**: Enable caching for datasets with many repeated values
- **Format Conversion**: Pre-convert to datetime dtype to avoid repeated parsing

## Version History

- **2.0.1** (2025-06-15): Critical bug fixes, improved error handling, privacy validation
- **2.0.0** (2025-01-15): Complete rewrite with new features and performance improvements
- **1.1.0** (2024-06-16): Bug fixes and metric improvements
- **1.0.0** (2024-06-15): Initial implementation

## License

BSD 3-Clause License

## Support

For issues and questions:
- GitHub: [PAMOLA.CORE Issues](https://github.com/pamola/pamola-pamola_core/issues)
- Documentation: [PAMOLA Documentation](https://docs.pamola.ai)
- Email: support@pamola.ai