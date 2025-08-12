# noise_utils.py Documentation

## Module Overview

**Module:** `pamola_core.anonymization.commons.noise_utils`  
**Version:** 1.0.0  
**Status:** Development  
**Package:** PAMOLA.CORE Privacy-Preserving AI Data Processors

### Description

Comprehensive utility functions and classes for noise addition operations in PAMOLA.CORE. This module provides secure random generation, noise impact analysis, distribution preservation metrics, and practical noise parameter recommendations. It supports both cryptographically secure and standard random generation, making it suitable for privacy-critical applications and performance-optimized scenarios.

### Key Features

- Thread-safe secure random number generation with multiple distributions
- Comprehensive noise impact and utility metrics calculation
- Distribution preservation analysis with statistical tests
- Intelligent noise range recommendations based on signal-to-noise ratio
- Validation utilities for noise bounds and parameters
- Support for numeric and temporal noise generation

## Constants

### Distribution Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `NOISE_DISTRIBUTIONS` | List[str] | `['uniform', 'normal', 'laplace', 'exponential']` | Supported noise distribution types |
| `TEMPORAL_UNITS` | List[str] | `['seconds', 'minutes', 'hours', 'days', 'weeks']` | Supported temporal units for time-based noise |
| `DEFAULT_HISTOGRAM_BINS` | int | `20` | Default number of bins for histogram analysis |
| `MAX_SAMPLE_SIZE` | int | `10000` | Maximum sample size for statistical tests to ensure performance |

### Time Conversion Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `SECONDS_PER_MINUTE` | int | `60` | Number of seconds in a minute |
| `SECONDS_PER_HOUR` | int | `3600` | Number of seconds in an hour |
| `SECONDS_PER_DAY` | int | `86400` | Number of seconds in a day |
| `SECONDS_PER_WEEK` | int | `604800` | Number of seconds in a week |

### Mathematical Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `EPSILON` | float | `1e-10` | Small value to avoid division by zero |
| `PI` | float | `np.pi` | Mathematical constant π |
| `SQRT_2` | float | `np.sqrt(2)` | Square root of 2, used in Laplace distribution |
| `SQRT_12` | float | `np.sqrt(12)` | Square root of 12, used for uniform distribution variance |

### Box-Muller Transform Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `BOX_MULLER_CONST` | float | `-2.0` | Constant for Box-Muller transform |
| `BOX_MULLER_ANGLE` | float | `2.0` | Angle multiplier for Box-Muller transform |

### Distribution Default Parameters

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `UNIFORM_RANGE_DEFAULT` | float | `1.0` | Default range for uniform distribution |
| `NORMAL_LOC_DEFAULT` | float | `0.0` | Default mean for normal distribution |
| `NORMAL_SCALE_DEFAULT` | float | `1.0` | Default standard deviation for normal distribution |
| `LAPLACE_LOC_DEFAULT` | float | `0.0` | Default location for Laplace distribution |
| `LAPLACE_SCALE_DEFAULT` | float | `1.0` | Default scale for Laplace distribution |
| `EXPONENTIAL_SCALE_DEFAULT` | float | `1.0` | Default scale for exponential distribution |

### Percentile Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `DEFAULT_PERCENTILES` | List[int] | `[10, 25, 50, 75, 90]` | Default percentiles for distribution analysis |
| `PERCENTILE_CHECK_DEFAULT` | float | `99.0` | Default percentile for bound violation checks |
| `PERCENTILE_LOWER_CHECK` | float | `0.01` | Lower percentile for range validation |
| `PERCENTILE_UPPER_CHECK` | float | `0.99` | Upper percentile for range validation |

### Threshold Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `SKEWNESS_THRESHOLD` | float | `1.0` | Threshold for significant skewness |
| `VIOLATION_THRESHOLD_PCT` | float | `5.0` | Percentage threshold for bound violations |
| `NOISE_RATIO_WARNING` | float | `0.5` | Noise-to-data ratio threshold for warnings |
| `NOISE_RATIO_EXCESSIVE` | float | `1.0` | Noise-to-data ratio threshold for excessive noise |
| `UTILITY_RATIO_THRESHOLD` | float | `0.3` | Recommended noise-to-data ratio for utility |
| `CORRELATION_LOW_THRESHOLD` | float | `0.7` | Threshold for low correlation warning |
| `DISTRIBUTION_PVALUE_THRESHOLD` | float | `0.05` | P-value threshold for distribution tests |
| `UTILITY_SCORE_LOW` | float | `0.5` | Threshold for low utility score |

### Signal-to-Noise Ratio Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `DB_CONVERSION_FACTOR` | float | `10.0` | Factor for converting to/from decibels |
| `DB_LOG_BASE` | int | `10` | Logarithm base for decibel calculations |
| `DEFAULT_TARGET_SNR` | float | `10.0` | Default target signal-to-noise ratio |
| `MULTIPLICATIVE_DEFAULT_RANGE` | float | `0.1` | Default range for multiplicative noise (10%) |
| `NOISE_SCALE_DIVISOR` | float | `3.0` | Divisor for scaling non-uniform distributions |

## Classes

### SecureRandomGenerator

Thread-safe random number generator supporting both secure and fast generation.

```python
class SecureRandomGenerator(use_secure: bool = True, seed: Optional[int] = None)
```

#### Parameters
- **use_secure** (bool): Whether to use cryptographically secure generation
- **seed** (Optional[int]): Random seed for reproducibility (ignored if use_secure=True)

#### Methods

##### uniform(low, high, size)
Generate uniform random values in [low, high).

```python
def uniform(low: float = 0.0, high: float = 1.0, 
            size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]
```

##### normal(loc, scale, size)
Generate normal (Gaussian) random values.

```python
def normal(loc: float = 0.0, scale: float = 1.0,
           size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]
```

##### laplace(loc, scale, size)
Generate Laplace random values (for differential privacy).

```python
def laplace(loc: float = 0.0, scale: float = 1.0,
            size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]
```

##### choice(a, size, replace, p)
Generate random sample from given array or range.

```python
def choice(a: Union[int, np.ndarray], size: Optional[int] = None,
           replace: bool = True, p: Optional[np.ndarray] = None) -> Union[Any, np.ndarray]
```

## Functions

### calculate_noise_impact

Calculate comprehensive metrics measuring the impact of noise on data utility.

```python
def calculate_noise_impact(original: pd.Series, 
                          noisy: pd.Series,
                          sample_size: Optional[int] = None) -> Dict[str, float]
```

#### Parameters
- **original** (pd.Series): Original data series
- **noisy** (pd.Series): Data series after noise addition
- **sample_size** (Optional[int]): If provided, sample data for faster computation

#### Returns
Dictionary containing:
- **rmse**: Root Mean Square Error
- **mae**: Mean Absolute Error
- **mape**: Mean Absolute Percentage Error
- **correlation**: Pearson correlation coefficient
- **rank_correlation**: Spearman rank correlation
- **relative_error**: Mean relative error
- **max_absolute_error**: Maximum absolute difference
- **snr**: Signal-to-Noise Ratio (in dB)

### calculate_distribution_preservation

Analyze how well noise addition preserves the data distribution.

```python
def calculate_distribution_preservation(original: pd.Series,
                                      noisy: pd.Series,
                                      n_bins: int = DEFAULT_HISTOGRAM_BINS,
                                      percentiles: Optional[List[int]] = None) -> Dict[str, Any]
```

#### Parameters
- **original** (pd.Series): Original data series
- **noisy** (pd.Series): Data series after noise addition
- **n_bins** (int): Number of bins for histogram comparison
- **percentiles** (Optional[List[int]]): List of percentiles to compare

#### Returns
Dictionary containing:
- **ks_statistic**: Kolmogorov-Smirnov test statistic
- **ks_pvalue**: p-value for KS test
- **wasserstein_distance**: Earth mover's distance between distributions
- **histogram_distance**: L1 distance between normalized histograms
- **mean_shift**: Difference in means
- **std_ratio**: Ratio of standard deviations
- **skewness_diff**: Difference in skewness
- **kurtosis_diff**: Difference in kurtosis
- **percentile_shifts**: Dict of percentile differences

### suggest_noise_range

Suggest appropriate noise range based on target signal-to-noise ratio.

```python
def suggest_noise_range(series: pd.Series,
                       target_snr: float = DEFAULT_TARGET_SNR,
                       noise_type: str = "additive",
                       distribution: str = "uniform") -> float
```

#### Parameters
- **series** (pd.Series): Original data series
- **target_snr** (float): Desired signal-to-noise ratio (linear scale, not dB)
- **noise_type** (str): Type of noise application ('additive' or 'multiplicative')
- **distribution** (str): Noise distribution type

#### Returns
- **float**: Suggested noise range/parameter value

### validate_noise_bounds

Validate that noise bounds are reasonable for the data.

```python
def validate_noise_bounds(series: pd.Series,
                         noise_range: Union[float, Tuple[float, float]],
                         output_min: Optional[float] = None,
                         output_max: Optional[float] = None,
                         percentile_check: float = PERCENTILE_CHECK_DEFAULT) -> Dict[str, Any]
```

#### Parameters
- **series** (pd.Series): Original data series
- **noise_range** (Union[float, Tuple[float, float]]): Noise range
- **output_min** (Optional[float]): Minimum allowed output value
- **output_max** (Optional[float]): Maximum allowed output value
- **percentile_check** (float): Percentile to check for bound violations

#### Returns
Dictionary containing:
- **valid**: Whether bounds are reasonable
- **warnings**: List of warning messages
- **noise_to_data_ratio**: Ratio of noise range to data range
- **expected_violations**: Estimated percentage of values hitting bounds
- **recommendations**: Suggested adjustments

### generate_numeric_noise

Generate numeric noise values with specified distribution.

```python
def generate_numeric_noise(size: int,
                          distribution: str = "uniform",
                          params: Optional[Dict[str, float]] = None,
                          secure: bool = True,
                          seed: Optional[int] = None) -> np.ndarray
```

#### Parameters
- **size** (int): Number of noise values to generate
- **distribution** (str): Distribution type from NOISE_DISTRIBUTIONS
- **params** (Optional[Dict[str, float]]): Distribution parameters
- **secure** (bool): Whether to use cryptographically secure generation
- **seed** (Optional[int]): Random seed (only used if secure=False)

#### Returns
- **np.ndarray**: Array of noise values

### generate_temporal_noise

Generate temporal noise values as time deltas.

```python
def generate_temporal_noise(size: int,
                           range_value: float,
                           unit: str = "days",
                           distribution: str = "uniform",
                           secure: bool = True,
                           seed: Optional[int] = None) -> pd.TimedeltaIndex
```

#### Parameters
- **size** (int): Number of time deltas to generate
- **range_value** (float): Maximum shift in specified unit
- **unit** (str): Time unit from TEMPORAL_UNITS
- **distribution** (str): Distribution type
- **secure** (bool): Whether to use cryptographically secure generation
- **seed** (Optional[int]): Random seed

#### Returns
- **pd.TimedeltaIndex**: Time delta values

### analyze_noise_effectiveness

Analyze the effectiveness of noise addition for privacy protection.

```python
def analyze_noise_effectiveness(original: pd.Series,
                               noisy: pd.Series,
                               privacy_metric: str = "snr",
                               target_value: Optional[float] = None) -> Dict[str, Any]
```

#### Parameters
- **original** (pd.Series): Original data series
- **noisy** (pd.Series): Data series after noise addition
- **privacy_metric** (str): Metric to use ('snr', 'rmse', 'correlation')
- **target_value** (Optional[float]): Target value for the privacy metric

#### Returns
Dictionary containing:
- **effective**: Whether noise meets privacy requirements
- **actual_value**: Actual metric value achieved
- **target_value**: Target value (if provided)
- **utility_score**: Overall utility preservation score (0-1)
- **recommendations**: List of suggestions for improvement

### create_noise_report

Create comprehensive noise analysis report.

```python
def create_noise_report(original: pd.Series,
                       noisy: pd.Series,
                       noise_params: Dict[str, Any]) -> Dict[str, Any]
```

#### Parameters
- **original** (pd.Series): Original data series
- **noisy** (pd.Series): Data series after noise addition
- **noise_params** (Dict[str, Any]): Dictionary of noise parameters used

#### Returns
- **Dict[str, Any]**: Comprehensive report with configuration, metrics, and analysis

## Usage Examples

### Basic Usage

```python
from pamola_core.anonymization.commons.noise_utils import (
    SecureRandomGenerator, calculate_noise_impact, suggest_noise_range
)

# Create secure random generator
rng = SecureRandomGenerator(use_secure=True)
noise = rng.uniform(-10, 10, size=1000)

# Analyze noise impact
original = pd.Series(np.random.randn(1000))
noisy = original + noise
impact = calculate_noise_impact(original, noisy)
print(f"RMSE: {impact['rmse']:.3f}")

# Get noise range recommendation
suggested_range = suggest_noise_range(original, target_snr=10.0)
print(f"Suggested range: ±{suggested_range:.3f}")
```

### Advanced Usage

```python
# Validate noise bounds
validation = validate_noise_bounds(
    data, noise_range=50, output_min=0, output_max=200
)
if not validation['valid']:
    print("Warnings:", validation['warnings'])

# Generate temporal noise
time_shifts = generate_temporal_noise(
    size=1000, range_value=7, unit='days'
)

# Create comprehensive report
report = create_noise_report(
    original, noisy,
    {'type': 'uniform', 'range': 10.0}
)
```

## Performance Considerations

- SecureRandomGenerator using secrets module is slower than NumPy
- For large-scale operations, consider use_secure=False with proper seeding
- Distribution calculations cache results where possible
- Statistical tests use sampling for very large datasets (>MAX_SAMPLE_SIZE)

## Security Notes

- Use `use_secure=True` for privacy-critical applications
- Standard NumPy random is suitable for non-sensitive applications
- Always validate noise bounds to prevent information leakage
- Consider differential privacy extensions for formal guarantees

## Dependencies

- **numpy**: Numerical operations and random generation
- **pandas**: Series and DataFrame operations
- **scipy**: Statistical tests and distributions
- **secrets**: Cryptographically secure random generation
- **threading**: Thread safety for random generators
- **typing**: Type hints and annotations

## Future Extensions

- Differential privacy mechanisms (Laplace, Gaussian)
- Adaptive noise based on local sensitivity
- Multi-dimensional correlated noise
- Query-specific noise calibration