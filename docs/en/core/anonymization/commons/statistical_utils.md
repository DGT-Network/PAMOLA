# Statistical Utilities for Noise Operations

## Overview

The `statistical_utils.py` module provides specialized statistical functions for analyzing and validating noise operations in the PAMOLA anonymization framework. It extends the general statistical metrics with noise-specific analysis capabilities.

**Package:** `pamola_core.anonymization.commons`  
**Version:** 1.0.1  
**Status:** Stable  
**License:** BSD 3-Clause

## Purpose

This module serves as the statistical foundation for noise operations, providing:

- **Noise Quality Assessment**: Measure how well generated noise matches expected distributions
- **Signal Preservation Analysis**: Quantify the impact of noise on data utility
- **Temporal Pattern Analysis**: Evaluate noise effects on datetime fields
- **Multi-field Correlation**: Ensure noise independence across fields

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Statistical tests
- `pamola_core.utils.statistical_metrics`: Base statistical functions

## Exception Classes

### `StatisticalUtilsError`
Base exception for all statistical utility errors.

```python
class StatisticalUtilsError(Exception):
    """Base exception for statistical utilities."""
```

### `InvalidParameterError`
Raised when invalid parameters are provided to functions.

```python
class InvalidParameterError(StatisticalUtilsError):
    """Raised when invalid parameters are provided."""
```

## Core Functions

### Signal-to-Noise Ratio Calculation

#### `calculate_signal_to_noise_ratio()`

Calculates the Signal-to-Noise Ratio (SNR) to measure signal preservation quality after noise addition.

```python
def calculate_signal_to_noise_ratio(
    original: Union[pd.Series, np.ndarray],
    noisy: Union[pd.Series, np.ndarray],
    method: str = "standard"
) -> float
```

**Parameters:**
- `original`: Original signal values
- `noisy`: Signal values after noise addition
- `method`: Calculation method
  - `"standard"`: 20 * log10(signal_std / noise_std) in dB
  - `"power"`: 10 * log10(signal_power / noise_power) in dB
  - `"ratio"`: Simple ratio signal_std / noise_std

**Returns:**
- `float`: SNR value (in dB for standard/power methods, ratio for ratio method)

**Raises:**
- `InvalidParameterError`: If method is unknown

**Example:**
```python
original = pd.Series([100, 102, 98, 101, 99])
noisy = pd.Series([101, 100, 97, 103, 98])
snr_db = calculate_signal_to_noise_ratio(original, noisy, method="standard")
print(f"SNR: {snr_db:.2f} dB")  # SNR: 24.44 dB
```

### Noise Distribution Analysis

#### `analyze_noise_uniformity()`

Analyzes whether noise follows the expected uniform distribution using statistical tests.

```python
def analyze_noise_uniformity(
    noise_values: Union[pd.Series, np.ndarray],
    expected_min: float,
    expected_max: float,
    n_bins: int = 20
) -> Dict[str, Any]
```

**Parameters:**
- `noise_values`: Actual noise values (difference between noisy and original)
- `expected_min`: Expected minimum noise value
- `expected_max`: Expected maximum noise value
- `n_bins`: Number of bins for chi-square test (default: 20)

**Returns:**
Dictionary containing:
- `uniformity_test`: Statistical test results
  - `chi2_statistic`, `chi2_p_value`: Chi-square test results
  - `ks_statistic`, `ks_p_value`: Kolmogorov-Smirnov test results
  - `is_uniform_chi2`, `is_uniform_ks`: Boolean indicators (p > 0.05)
- `actual_range`: Actual vs expected range information
  - `min`, `max`: Actual minimum and maximum
  - `expected_min`, `expected_max`: Expected bounds
  - `within_bounds`: Whether actual range is within expected
- `distribution_metrics`: Distribution characteristics
  - `mean`, `std`: Actual mean and standard deviation
  - `skewness`, `kurtosis`: Distribution shape metrics
  - `expected_mean`, `expected_std`: Theoretical values for uniform distribution

**Raises:**
- `InvalidParameterError`: If expected_min >= expected_max or no valid values

**Example:**
```python
noise = np.random.uniform(-5, 5, 1000)
analysis = analyze_noise_uniformity(noise, -5, 5)
if analysis['uniformity_test']['is_uniform_chi2']:
    print("Noise follows uniform distribution (Chi-square test)")
```

### Utility Preservation Metrics

#### `calculate_utility_preservation()`

Measures how well statistical properties are preserved after noise addition.

```python
def calculate_utility_preservation(
    original: pd.Series,
    noisy: pd.Series,
    metrics: List[str] = None
) -> Dict[str, float]
```

**Parameters:**
- `original`: Original data series
- `noisy`: Data after noise addition
- `metrics`: Specific metrics to calculate (default: all available)
  - `"mean"`: Mean preservation
  - `"std"`: Standard deviation preservation
  - `"median"`: Median preservation
  - `"iqr"`: Interquartile range preservation
  - `"correlation"`: Pearson correlation
  - `"rank_correlation"`: Spearman rank correlation

**Returns:**
Dictionary with preservation metrics (values closer to 1 indicate better preservation)

**Example:**
```python
original = pd.Series([1, 2, 3, 4, 5])
noisy = pd.Series([1.1, 1.9, 3.2, 3.8, 5.1])
preservation = calculate_utility_preservation(original, noisy)
print(f"Mean preservation: {preservation['mean_preservation']:.4f}")  # 0.9600
print(f"Correlation: {preservation['correlation']:.4f}")  # 0.9934
```

### Temporal Noise Analysis

#### `analyze_temporal_noise_impact()`

Analyzes the impact of noise on temporal data patterns.

```python
def analyze_temporal_noise_impact(
    original_timestamps: pd.Series,
    noisy_timestamps: pd.Series
) -> Dict[str, Any]
```

**Parameters:**
- `original_timestamps`: Original datetime values
- `noisy_timestamps`: Datetime values after noise addition

**Returns:**
Dictionary containing:
- `shift_statistics`: Time shift metrics
  - Mean, std, min, max, median shifts in hours
  - Count of zero, forward, and backward shifts
- `pattern_preservation`: Temporal pattern preservation
  - `weekday_preserved`: Fraction maintaining same day of week
  - `hour_preserved`: Fraction maintaining same hour
  - `weekend_preserved`: Fraction maintaining weekend status
- `ordering_preservation`: Temporal order metrics
  - `order_fully_preserved`: Boolean indicator
  - `kendall_tau`: Kendall's tau correlation
  - `inversions`: Number of order inversions

**Example:**
```python
original = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03'])
noisy = pd.to_datetime(['2025-01-01 12:00', '2025-01-02 08:00', '2025-01-02 20:00'])
impact = analyze_temporal_noise_impact(pd.Series(original), pd.Series(noisy))
print(f"Mean shift: {impact['shift_statistics']['mean_shift_hours']:.2f} hours")
```

### Distribution Fitting

#### `calculate_noise_distribution_fit()`

Tests how well noise fits an expected theoretical distribution.

```python
def calculate_noise_distribution_fit(
    noise_values: Union[pd.Series, np.ndarray],
    distribution: str = "uniform",
    params: Dict[str, float] = None
) -> Dict[str, Any]
```

**Parameters:**
- `noise_values`: Actual noise values
- `distribution`: Expected distribution ("uniform", "normal", "laplace")
- `params`: Distribution parameters (auto-estimated if None)

**Returns:**
Dictionary with goodness-of-fit test results

### Multi-field Analysis

#### `calculate_multifield_noise_correlation()`

Calculates correlation between noise added to different fields to ensure independence.

```python
def calculate_multifield_noise_correlation(
    noise_dict: Dict[str, Union[pd.Series, np.ndarray]]
) -> pd.DataFrame
```

**Parameters:**
- `noise_dict`: Dictionary mapping field names to noise values

**Returns:**
- `pd.DataFrame`: Correlation matrix between noise fields

**Example:**
```python
noise_dict = {
    "age": np.random.uniform(-5, 5, 100),
    "salary": np.random.uniform(-1000, 1000, 100),
    "score": np.random.uniform(-0.1, 0.1, 100)
}
corr_matrix = calculate_multifield_noise_correlation(noise_dict)
# Ideally, off-diagonal elements should be close to 0
```

### Comprehensive Summary

#### `get_noise_quality_summary()`

Generates a comprehensive noise quality report combining multiple metrics.

```python
def get_noise_quality_summary(
    original: pd.Series,
    noisy: pd.Series,
    expected_noise_range: Tuple[float, float] = None,
    noise_type: str = "uniform"
) -> Dict[str, Any]
```

**Parameters:**
- `original`: Original data
- `noisy`: Data after noise addition
- `expected_noise_range`: Expected (min, max) for uniform noise
- `noise_type`: Type of noise ("uniform", "normal", "laplace")

**Returns:**
Comprehensive dictionary containing:
- Basic metrics (record counts)
- SNR analysis
- Utility preservation metrics
- Distribution analysis (if applicable)
- Distribution fit tests
- Noise distribution summary

## Planned Functions (TODO)

The following functions are planned for future implementation as per REQ-UNIFORM-007:

### `calculate_utility_metrics()`
Comprehensive utility metrics calculation with configurable metric sets.

### `calculate_correlation_preservation()`
Multi-field correlation preservation analysis.

### `estimate_information_loss()`
Information-theoretic metrics for privacy-utility tradeoff.

## Usage Examples

### Basic Noise Quality Assessment

```python
from pamola_core.anonymization.commons.statistical_utils import (
    calculate_signal_to_noise_ratio,
    analyze_noise_uniformity,
    get_noise_quality_summary
)

# Generate some test data
original = pd.Series(np.random.normal(100, 10, 1000))
noise = np.random.uniform(-5, 5, 1000)
noisy = original + noise

# Quick quality check
snr = calculate_signal_to_noise_ratio(original, noisy)
print(f"Signal-to-Noise Ratio: {snr:.2f} dB")

# Detailed uniformity analysis
uniformity = analyze_noise_uniformity(noise, -5, 5)
if uniformity['uniformity_test']['is_uniform_ks']:
    print("✓ Noise is uniformly distributed")
else:
    print("✗ Noise deviates from uniform distribution")

# Comprehensive summary
summary = get_noise_quality_summary(original, noisy, (-5, 5), "uniform")
```

### Temporal Noise Validation

```python
# Original timestamps
dates = pd.date_range('2025-01-01', periods=100, freq='D')
original_ts = pd.Series(dates)

# Add random shifts of ±12 hours
shifts = np.random.uniform(-12, 12, 100)
noisy_ts = original_ts + pd.to_timedelta(shifts, unit='hours')

# Analyze temporal impact
temporal_impact = analyze_temporal_noise_impact(original_ts, noisy_ts)

print(f"Weekday preservation: {temporal_impact['pattern_preservation']['weekday_preserved']:.1%}")
print(f"Order preservation: {temporal_impact['ordering_preservation']['kendall_tau']:.3f}")
```

### Multi-field Independence Check

```python
# Generate noise for multiple fields
fields = ['age', 'income', 'score']
noise_data = {}

for field in fields:
    if field == 'age':
        noise_data[field] = np.random.uniform(-5, 5, 1000)
    elif field == 'income':
        noise_data[field] = np.random.uniform(-1000, 1000, 1000)
    else:
        noise_data[field] = np.random.uniform(-0.1, 0.1, 1000)

# Check independence
corr_matrix = calculate_multifield_noise_correlation(noise_data)
max_correlation = corr_matrix.abs().values[~np.eye(len(fields), dtype=bool)].max()

if max_correlation < 0.05:
    print("✓ Noise is independent across fields")
else:
    print(f"✗ Correlation detected: {max_correlation:.3f}")
```

## Integration with Noise Operations

This module is designed to integrate seamlessly with noise operations:

```python
class UniformNumericNoiseOperation(AnonymizationOperation):
    def _collect_specific_metrics(self, original_data, noisy_data):
        # Use statistical utils for comprehensive metrics
        metrics = get_noise_quality_summary(
            original_data,
            noisy_data,
            expected_noise_range=(self.noise_min, self.noise_max),
            noise_type="uniform"
        )
        
        # Extract key metrics for reporting
        return {
            "snr_db": metrics["signal_to_noise"]["snr_db"],
            "uniformity_test_passed": metrics["uniformity_analysis"]["uniformity_test"]["is_uniform_ks"],
            "mean_preservation": metrics["utility_preservation"]["mean_preservation"],
            "correlation": metrics["utility_preservation"]["correlation"]
        }
```

## Best Practices

1. **Always validate noise parameters** before analysis:
   ```python
   if expected_min >= expected_max:
       raise InvalidParameterError("Invalid noise range")
   ```

2. **Handle edge cases** gracefully:
   ```python
   if signal.size == 0:
       logger.warning("No valid values for analysis")
       return default_metrics()
   ```

3. **Use appropriate test significance levels**:
   - Default α = 0.05 for hypothesis tests
   - Consider multiple testing corrections for many fields

4. **Monitor both statistical and practical significance**:
   - High p-values indicate statistical uniformity
   - But also check if noise magnitude is appropriate for privacy

5. **Document expected vs actual behavior**:
   - Log warnings when noise deviates from expectations
   - Include diagnostic information in metrics

## Performance Considerations

- Functions are optimized for vectorized operations
- Use `float()` conversions to ensure type compatibility
- Large datasets (>1M records) may benefit from chunked processing
- Chi-square test requires sufficient samples per bin (typically >5)

## Version History

- **1.0.1** (2025-06-16): Fixed type annotations, added parameter validation
- **1.0.0** (2025-06-16): Initial implementation

## See Also

- [`pamola_core.utils.statistical_metrics`](../../../utils/statistical_metrics.md): Base statistical functions
- [`noise_utils.py`](./noise_utils.md): Noise generation and application utilities
- [Noise Operations Sub-SRS](../../../docs/specs/noise_operations.md): Requirements specification