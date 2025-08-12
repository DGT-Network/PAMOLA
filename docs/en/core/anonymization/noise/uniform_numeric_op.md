# UniformNumericNoiseOperation Documentation

## Overview

The `UniformNumericNoiseOperation` is a privacy-preserving transformation that adds uniformly distributed random noise to numeric fields in datasets. This operation is part of the PAMOLA.CORE anonymization framework and provides a balance between privacy protection and data utility preservation.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Import](#installation--import)
3. [Basic Usage](#basic-usage)
4. [Parameters](#parameters)
5. [Advanced Features](#advanced-features)
6. [Metrics & Analysis](#metrics--analysis)
7. [Performance Considerations](#performance-considerations)
8. [Security Notes](#security-notes)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Introduction

### Purpose

The UniformNumericNoiseOperation adds controlled random noise to numeric data fields to prevent exact value disclosure while maintaining statistical properties. It's particularly useful for:

- Protecting sensitive numeric values (salaries, ages, scores)
- Preventing re-identification through exact matching
- Maintaining data utility for analysis
- Meeting privacy compliance requirements

### Key Features

- **Uniform Distribution**: Noise follows a uniform distribution within specified bounds
- **Flexible Noise Types**: Supports both additive and multiplicative noise
- **Constraint Enforcement**: Ensures output values stay within valid ranges
- **Type Preservation**: Maintains integer types when appropriate
- **Statistical Scaling**: Can scale noise based on data variance
- **Secure Generation**: Offers cryptographically secure random generation
- **Large-scale Support**: Integrated Dask support for big data

## Installation & Import

```python
from pamola_core.anonymization.noise import UniformNumericNoiseOperation

# Or import from the package root
from pamola_core.anonymization import UniformNumericNoiseOperation
```

## Basic Usage

### Simple Example

```python
import pandas as pd
from pamola_core.anonymization.noise import UniformNumericNoiseOperation

# Create sample data
df = pd.DataFrame({
    'employee_id': range(1000),
    'salary': [50000 + i * 100 for i in range(1000)],
    'age': [25 + i % 40 for i in range(1000)]
})

# Add ±5000 noise to salary
salary_noise_op = UniformNumericNoiseOperation(
    field_name='salary',
    noise_range=5000,  # Symmetric range: -5000 to +5000
    output_min=0       # Ensure no negative salaries
)

# Execute the operation
result = salary_noise_op.execute(
    data_source=data_source,
    task_dir=Path('./output'),
    reporter=reporter
)
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `field_name` | str | The name of the numeric field to add noise to |
| `noise_range` | float or tuple | Symmetric range (float) or asymmetric range (min, max) |

### Noise Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_type` | str | "additive" | Type of noise: "additive" or "multiplicative" |
| `output_min` | float | None | Minimum allowed output value |
| `output_max` | float | None | Maximum allowed output value |
| `preserve_zero` | bool | False | Keep zero values unchanged |
| `round_to_integer` | bool | None | Round to integers (auto-detected if None) |

### Statistical Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale_by_std` | bool | False | Scale noise by field standard deviation |
| `scale_factor` | float | 1.0 | Additional scaling factor for noise |

### Security Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_secure_random` | bool | True | Use cryptographically secure random generation |
| `random_seed` | int | None | Seed for reproducible noise (ignored if secure=True) |

### Standard Operation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "REPLACE" | "REPLACE" to modify in-place, "ENRICH" to create new field |
| `output_field_name` | str | None | Name for output field (ENRICH mode) |
| `null_strategy` | str | "PRESERVE" | How to handle nulls: "PRESERVE", "EXCLUDE", "ERROR" |
| `batch_size` | int | 10000 | Batch size for processing |
| `use_cache` | bool | False | Enable operation caching |
| `use_encryption` | bool | False | Encrypt output files |
| `engine` | str | "auto" | Processing engine: "pandas", "dask", or "auto" |
| `max_rows_in_memory` | int | 1000000 | Threshold for automatic Dask switching |

## Advanced Features

### 1. Asymmetric Noise Range

Add different amounts of positive and negative noise:

```python
# Add noise from -1000 to +5000
operation = UniformNumericNoiseOperation(
    field_name='bonus',
    noise_range=(-1000, 5000)
)
```

### 2. Multiplicative Noise

Add noise as a percentage of the original value:

```python
# Add ±10% variation
operation = UniformNumericNoiseOperation(
    field_name='revenue',
    noise_range=0.1,
    noise_type='multiplicative'
)
```

### 3. Statistical Scaling

Scale noise based on data variance for adaptive privacy:

```python
# Scale noise by standard deviation
operation = UniformNumericNoiseOperation(
    field_name='test_score',
    noise_range=0.5,      # Base range (multiplied by std dev)
    scale_by_std=True,    # Enable statistical scaling
    scale_factor=2.0      # Additional multiplier
)
```

### 4. Zero Preservation

Preserve special values like zero:

```python
# Don't add noise to zero values
operation = UniformNumericNoiseOperation(
    field_name='debt_amount',
    noise_range=1000,
    preserve_zero=True,
    output_min=0
)
```

### 5. Integer Type Preservation

Automatically maintain integer types:

```python
# Auto-detect and preserve integer type
operation = UniformNumericNoiseOperation(
    field_name='age',
    noise_range=5,
    round_to_integer=None  # Auto-detect from data
)
```

### 6. Reproducible Noise

Generate the same noise for testing:

```python
# Reproducible noise for testing
operation = UniformNumericNoiseOperation(
    field_name='score',
    noise_range=10,
    use_secure_random=False,
    random_seed=42
)
```

## Metrics & Analysis

The operation collects comprehensive metrics about the noise addition process:

### Noise Statistics
- Mean, standard deviation, min/max of actual noise added
- Signal-to-noise ratio (SNR)
- Values hitting output bounds
- Number of preserved zeros

### Impact Metrics
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Correlation preservation
- Relative error statistics

### Distribution Metrics
- Kolmogorov-Smirnov test results
- Wasserstein distance
- Histogram comparison
- Percentile shifts

### Uniformity Analysis
- Chi-square test for uniform distribution
- Actual vs. expected noise range
- Distribution shape metrics

### Example: Accessing Metrics

```python
# Execute operation
result = operation.execute(data_source, task_dir, reporter)

# Access metrics from result
metrics = result.metrics
print(f"SNR: {metrics['noise_effectiveness']['actual_value']:.2f} dB")
print(f"Correlation preserved: {metrics['noise_impact']['correlation']:.3f}")
print(f"Distribution p-value: {metrics['distribution_preservation']['ks_pvalue']:.4f}")
```

## Performance Considerations

### Memory Optimization

The operation automatically optimizes memory usage:

```python
# Enable adaptive batch sizing
operation = UniformNumericNoiseOperation(
    field_name='value',
    noise_range=100,
    adaptive_batch_size=True  # Adjusts batch size based on memory
)
```

### Large Dataset Processing

For datasets exceeding memory limits, the operation automatically switches to Dask:

```python
# Configure Dask parameters
operation = UniformNumericNoiseOperation(
    field_name='amount',
    noise_range=1000,
    engine='auto',              # Automatic engine selection
    max_rows_in_memory=500000,  # Switch to Dask above this
    dask_chunk_size='50MB'      # Partition size for Dask
)
```

### Performance Tips

1. **Batch Size**: Larger batches are more efficient but use more memory
2. **Secure Random**: Disable for non-sensitive data to improve speed
3. **Statistical Scaling**: Pre-calculate if applying to multiple datasets
4. **Bounds Checking**: Minimize bound constraints for better performance

## Security Notes

### Cryptographically Secure Random

By default, the operation uses cryptographically secure random generation:

```python
# Default secure generation (recommended for production)
operation = UniformNumericNoiseOperation(
    field_name='sensitive_value',
    noise_range=1000,
    use_secure_random=True  # Default
)
```

### Privacy Considerations

1. **Noise Range**: Larger ranges provide more privacy but less utility
2. **Bounds**: Tight bounds may leak information about original values
3. **Zero Preservation**: May reduce privacy for sparse data
4. **Reproducibility**: Never use seeds in production for sensitive data

### Recommended Practices

```python
# Production-ready configuration
operation = UniformNumericNoiseOperation(
    field_name='salary',
    noise_range=5000,
    noise_type='additive',
    output_min=0,
    output_max=None,        # No upper bound
    preserve_zero=False,    # Don't preserve special values
    use_secure_random=True, # Secure generation
    random_seed=None        # No seed for security
)
```

## Examples

### Example 1: Age Anonymization

```python
# Add ±3 years to age with bounds
age_noise = UniformNumericNoiseOperation(
    field_name='age',
    noise_range=3,
    output_min=0,
    output_max=120,
    round_to_integer=True
)

result = age_noise.execute(data_source, task_dir)
```

### Example 2: Financial Data Protection

```python
# Add percentage-based noise to transaction amounts
transaction_noise = UniformNumericNoiseOperation(
    field_name='transaction_amount',
    noise_range=0.05,  # ±5%
    noise_type='multiplicative',
    output_min=0.01,   # Minimum transaction
    preserve_zero=True # Don't modify zero transactions
)
```

### Example 3: Test Score Anonymization

```python
# Add noise scaled by score variance
score_noise = UniformNumericNoiseOperation(
    field_name='test_score',
    noise_range=0.1,    # 10% of standard deviation
    scale_by_std=True,
    output_min=0,
    output_max=100,
    round_to_integer=True
)
```

### Example 4: Salary Band Protection

```python
# Add asymmetric noise to hide exact salaries
salary_noise = UniformNumericNoiseOperation(
    field_name='annual_salary',
    noise_range=(-2000, 8000),  # More positive noise
    output_min=15000,           # Minimum wage
    round_to_integer=True,
    mode='ENRICH',              # Keep original for comparison
    output_field_name='noisy_salary'
)
```

### Example 5: Batch Processing with Progress

```python
from pamola.pamola_core.utils.progress import ProgressTracker

# Create progress tracker
progress = ProgressTracker(total=100, description="Adding noise")

# Configure operation
operation = UniformNumericNoiseOperation(
    field_name='sensor_reading',
    noise_range=0.1,
    noise_type='multiplicative',
    batch_size=50000
)

# Execute with progress tracking
result = operation.execute(
    data_source=data_source,
    task_dir=Path('./output'),
    progress_tracker=progress
)
```

## Troubleshooting

### Common Issues

#### 1. Non-numeric Field Error
```python
# Error: Field 'category' is not numeric
# Solution: Ensure field contains numeric data
df['numeric_category'] = pd.to_numeric(df['category'], errors='coerce')
```

#### 2. Values Outside Bounds
```python
# Warning: Many values hitting bounds
# Solution: Adjust bounds or noise range
operation = UniformNumericNoiseOperation(
    field_name='value',
    noise_range=100,
    output_min=df['value'].min() - 100,  # Dynamic bounds
    output_max=df['value'].max() + 100
)
```

#### 3. Memory Issues
```python
# Error: MemoryError during processing
# Solution: Enable Dask or reduce batch size
operation = UniformNumericNoiseOperation(
    field_name='large_field',
    noise_range=1000,
    engine='dask',           # Force Dask usage
    batch_size=5000,         # Smaller batches
    dask_chunk_size='25MB'   # Smaller partitions
)
```

#### 4. Poor Noise Distribution
```python
# Warning: Noise not uniformly distributed
# Solution: Check scale factors and bounds
operation = UniformNumericNoiseOperation(
    field_name='value',
    noise_range=1000,
    scale_by_std=False,  # Disable auto-scaling
    scale_factor=1.0     # No additional scaling
)
```

### Validation Helpers

```python
# Validate noise effectiveness
from pamola_core.anonymization.commons.noise_utils import (
    suggest_noise_range,
    validate_noise_bounds
)

# Get recommended noise range
suggested_range = suggest_noise_range(
    df['salary'],
    target_snr=10.0  # Target signal-to-noise ratio
)

# Validate bounds
validation = validate_noise_bounds(
    df['salary'],
    noise_range=5000,
    output_min=0,
    output_max=200000
)
if not validation['valid']:
    print("Warnings:", validation['warnings'])
```

## Best Practices

1. **Start Conservative**: Begin with smaller noise ranges and increase as needed
2. **Monitor Metrics**: Check SNR and utility preservation metrics
3. **Test Thoroughly**: Validate on sample data before full processing
4. **Document Choices**: Record noise parameters and rationale
5. **Consider Context**: Different fields may need different noise strategies

## API Reference

### Class: UniformNumericNoiseOperation

```python
class UniformNumericNoiseOperation(AnonymizationOperation):
    """Operation for adding uniform random noise to numeric fields."""
    
    def __init__(self, field_name: str, noise_range: Union[float, Tuple[float, float]], ...)
    def execute(self, data_source: DataSource, task_dir: Path, ...) -> OperationResult
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame
```

### Key Methods

- `execute()`: Main execution method that processes the entire dataset
- `process_batch()`: Processes a single batch of data
- `_generate_noise()`: Generates uniform noise values
- `_apply_noise()`: Applies noise with constraints
- `_collect_specific_metrics()`: Collects noise-specific metrics

### Integration Points

- Inherits from `AnonymizationOperation`
- Uses `SecureRandomGenerator` for noise generation
- Integrates with `DataWriter` for output
- Compatible with `ProgressTracker`
- Supports `OperationResult` for metrics and artifacts

## Related Operations

- **NumericGeneralizationOperation**: For binning/rounding instead of noise
- **UniformTemporalNoiseOperation**: For adding noise to datetime fields
- **Future**: Laplace/Gaussian noise for differential privacy

## References

- [PAMOLA.CORE Noise Operations Sub-Specification](./specs/noise_srs.md)
- [Anonymization Best Practices Guide](./guides/anonymization.md)
- [Privacy-Utility Tradeoff Analysis](./guides/privacy_utility.md)