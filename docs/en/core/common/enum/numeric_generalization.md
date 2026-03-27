# NumericMethod Enumeration

**Module:** `pamola_core.common.enum.numeric_generalization`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

NumericMethod defines the generalization techniques available for protecting privacy in numeric data. It provides type-safe selection of strategies for transforming numeric values while preserving statistical properties and utility.

## Members

| Member | Value | Description |
|--------|-------|-------------|
| `BINNING` | `"binning"` | Group numeric values into ranges or bins |
| `ROUNDING` | `"rounding"` | Round values to nearest discrete step |
| `SCALING` | `"scaling"` | Scale values to normalized range [0, 1] |

## Usage

### Basic Enumeration Access

```python
from pamola_core.common.enum.numeric_generalization import NumericMethod

# Access members
method = NumericMethod.BINNING
print(method.value)  # Output: "binning"
print(method.name)   # Output: "BINNING"
```

### Strategy Selection

```python
from pamola_core.common.enum.numeric_generalization import NumericMethod
import pandas as pd

def apply_generalization(series: pd.Series, method: NumericMethod) -> pd.Series:
    """Apply numeric generalization based on method."""
    if method == NumericMethod.BINNING:
        return pd.cut(series, bins=5, labels=False)
    elif method == NumericMethod.ROUNDING:
        return (series / 10).round() * 10
    elif method == NumericMethod.SCALING:
        return (series - series.min()) / (series.max() - series.min())

# Usage
df = pd.DataFrame({"age": [23, 45, 67, 34, 56, 78]})
df["age_binned"] = apply_generalization(df["age"], NumericMethod.BINNING)
```

### Listing Available Methods

```python
from pamola_core.common.enum.numeric_generalization import NumericMethod

methods = [m.value for m in NumericMethod]
print(methods)  # ["binning", "rounding", "scaling"]
```

## Member Descriptions

### BINNING
**Value:** `"binning"`

Groups numeric values into discrete ranges or "bins", replacing exact values with interval labels. Privacy protection increases with bin width at the cost of reduced utility.

**Privacy Mechanism:**
- Generalizes exact values to range membership
- Hides individual values within bin boundaries
- Number of bins controls privacy-utility tradeoff

**Example:**
```
Original: [23, 25, 28, 45, 67, 78]
Binned:   [20-30, 20-30, 20-30, 40-50, 60-70, 70-80]
```

**Use cases:**
- Age anonymization (e.g., 5-year age groups)
- Salary bands for compensation privacy
- Income brackets for financial analysis
- Temperature ranges for sensor data

**Utility Impact:** Moderate to High (depends on bin width)

**Parameters:**
- `num_bins`: Number of equal-width bins
- `bin_edges`: Custom bin boundaries for non-uniform binning

### ROUNDING
**Value:** `"rounding"`

Rounds numeric values to the nearest multiple of a specified step (e.g., nearest 10, nearest 5). Simple privacy approach with good utility retention.

**Privacy Mechanism:**
- Generalizes to nearest discrete value
- Hides precision of original measurements
- Step size controls privacy-utility tradeoff

**Example:**
```
Original: [23, 25, 28, 45, 67, 78]
Rounded:  [20, 30, 30, 50, 70, 80]  (step=10)
```

**Use cases:**
- Income rounding to nearest thousand
- Temperature rounding to nearest degree
- Weight rounding to nearest 5 kg
- Timestamp rounding to nearest hour

**Utility Impact:** High (minimal information loss)

**Parameters:**
- `step`: Rounding interval (e.g., 10, 5, 1)

### SCALING
**Value:** `"scaling"`

Scales numeric values to a normalized range [0, 1], transforming the data while preserving relative relationships and distributions.

**Privacy Mechanism:**
- Removes original scale information
- Preserves relative ordering
- Useful for machine learning pipelines

**Example:**
```
Original: [23, 45, 67, 34, 56, 78]
Scaled:   [0.0, 0.37, 0.73, 0.18, 0.55, 1.0]
```

**Use cases:**
- Feature normalization for machine learning
- Distribution preservation with scale obfuscation
- Multi-scale data alignment
- Preparing data for neural networks

**Utility Impact:** High (relationships preserved)

**Parameters:**
- `min_val`, `max_val`: Custom scaling range bounds
- `method`: Standard scaling, min-max, z-score normalization

## Privacy-Utility Tradeoff

| Method | Privacy | Utility | Speed | Best For |
|--------|---------|---------|-------|----------|
| Binning | High | Moderate | Fast | Categorical-like protection |
| Rounding | Moderate | High | Fast | Precision obfuscation |
| Scaling | Low-Moderate | High | Fast | ML preprocessing |

## Selection Guide

### By Data Type

**Continuous Data:**
- Use `BINNING` for strong privacy (age, salary, measurements)
- Use `ROUNDING` for moderate privacy (precise values)

**Skewed Distributions:**
- Use `BINNING` with custom edges for frequency-based privacy

**Machine Learning:**
- Use `SCALING` for model input normalization

### By Privacy Requirements

**Strong Privacy (High Protection)**
- BINNING with wide bins (e.g., 5-year age groups)

**Moderate Privacy (Balanced)**
- ROUNDING with reasonable step size
- Fine-grained BINNING

**Weak Privacy (Maximum Utility)**
- SCALING for relative relationships
- Fine rounding with small steps

## Related Components

- **DateTime Generalization:** Similar patterns for temporal data (`DatetimeMethod`)
- **Utility Metrics:** Evaluate information loss with `UtilityMetricsType`
- **Privacy Metrics:** Assess privacy preservation with `PrivacyMetricsType`

## Common Patterns

### Configurable Generalization

```python
from pamola_core.common.enum.numeric_generalization import NumericMethod

config = {
    NumericMethod.BINNING: {"num_bins": 5},
    NumericMethod.ROUNDING: {"step": 10},
    NumericMethod.SCALING: {"min_val": 0, "max_val": 1}
}

def apply_with_config(series, method: NumericMethod):
    params = config.get(method, {})
    if method == NumericMethod.BINNING:
        return pd.cut(series, bins=params["num_bins"])
    # ... handle other methods
```

### Privacy-Aware Selection

```python
from pamola_core.common.enum.numeric_generalization import NumericMethod

def select_method(privacy_level: str) -> NumericMethod:
    """Choose generalization based on privacy requirements."""
    if privacy_level == "high":
        return NumericMethod.BINNING
    elif privacy_level == "moderate":
        return NumericMethod.ROUNDING
    else:
        return NumericMethod.SCALING
```

## Best Practices

1. **Use Enum for Type Safety**
   ```python
   # Good
   method = NumericMethod.BINNING

   # Avoid
   method = "binning"  # String is error-prone
   ```

2. **Match Method to Data Sensitivity**
   ```python
   # Highly sensitive (medical data)
   method = NumericMethod.BINNING

   # Less sensitive (derived features)
   method = NumericMethod.SCALING
   ```

3. **Test Privacy-Utility Tradeoff**
   ```python
   for method in NumericMethod:
       anonymized = apply_generalization(df, method)
       privacy_score = evaluate_privacy(anonymized)
       utility_score = evaluate_utility(anonymized, original)
       print(f"{method.value}: Privacy={privacy_score}, Utility={utility_score}")
   ```

4. **Document Generalization Parameters**
   ```python
   # Good - explains privacy decision
   method = NumericMethod.BINNING  # 5-year bins for age
   ```

## Examples

### Age Generalization

```python
import pandas as pd
from pamola_core.common.enum.numeric_generalization import NumericMethod

df = pd.DataFrame({"age": [5, 15, 25, 35, 45, 55, 65, 75]})

# BINNING: 10-year age groups
df["age_binned"] = pd.cut(df["age"], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80],
                          labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"])

# ROUNDING: to nearest 5 years
df["age_rounded"] = (df["age"] / 5).round() * 5

# SCALING: to [0, 1]
df["age_scaled"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
```

### Salary Anonymization

```python
import pandas as pd
from pamola_core.common.enum.numeric_generalization import NumericMethod

salaries = pd.Series([45000, 52000, 68000, 75000, 82000, 95000])

# BINNING: salary bands
bands = pd.cut(salaries, bins=[0, 50000, 70000, 100000],
               labels=["Entry", "Mid", "Senior"])

# ROUNDING: to nearest 1000
rounded = (salaries / 1000).round() * 1000
```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [DateTime Generalization](./datetime_generalization.md)
- [Utility Metrics Type](./utility_metrics_type.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- Three numeric generalization methods
- Privacy-utility analysis and selection guide
