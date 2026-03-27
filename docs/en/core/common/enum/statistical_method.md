# StatisticalMethod Enumeration

**Module:** `pamola_core.common.enum.statistical_method`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

StatisticalMethod defines the aggregation and summary statistics functions available for data analysis in PAMOLA.CORE. It provides type-safe selection of statistical measures for grouping, summarization, and privacy-preserving analysis operations.

## Members

| Member | Value | Metric | Formula/Description |
|--------|-------|--------|---------------------|
| `MEAN` | `"mean"` | Arithmetic Mean | Sum / Count (average) |
| `MEDIAN` | `"median"` | Median | Middle value when sorted |
| `MODE` | `"mode"` | Mode | Most frequently occurring value |

## Usage

### Basic Enumeration Access

```python
from pamola_core.common.enum.statistical_method import StatisticalMethod

# Access members
method = StatisticalMethod.MEAN
print(method.value)  # Output: "mean"
print(method.name)   # Output: "MEAN"
```

### Conditional Statistics Calculation

```python
from pamola_core.common.enum.statistical_method import StatisticalMethod
import pandas as pd
import numpy as np
from scipy import stats

def calculate_statistic(series: pd.Series,
                       method: StatisticalMethod):
    """Calculate statistic based on method."""
    if method == StatisticalMethod.MEAN:
        return series.mean()
    elif method == StatisticalMethod.MEDIAN:
        return series.median()
    elif method == StatisticalMethod.MODE:
        return stats.mode(series, keepdims=True).mode[0]

# Usage
data = pd.Series([10, 20, 20, 30, 40])
mean_val = calculate_statistic(data, StatisticalMethod.MEAN)      # 24
median_val = calculate_statistic(data, StatisticalMethod.MEDIAN)  # 20
mode_val = calculate_statistic(data, StatisticalMethod.MODE)      # 20
```

### Listing Available Methods

```python
from pamola_core.common.enum.statistical_method import StatisticalMethod

methods = [m.value for m in StatisticalMethod]
print(methods)  # ["mean", "median", "mode"]
```

## Member Descriptions

### MEAN
**Value:** `"mean"`

The arithmetic mean (average) - sum of all values divided by count. Most common summary statistic.

**Formula:** `mean = Σx / n`

**Characteristics:**
- Affected by all data values
- Sensitive to outliers
- Frequently used and well-understood
- Single representative value

**Use cases:**
- Average income across population
- Mean response time in systems
- Average temperature over period
- Central tendency for continuous data

**Privacy Implications:**
- Reveals aggregate information
- Individual outliers can be inferred with context
- Should not be published without k-anonymity protection
- Combine with standard deviation for variance awareness

**Example:**
```
Values: [10, 20, 30, 40, 50]
Mean: (10+20+30+40+50)/5 = 30
```

### MEDIAN
**Value:** `"median"`

The middle value when data is sorted. Divides ordered dataset into two equal halves - 50% below, 50% above.

**Characteristics:**
- Robust to outliers
- Better than mean for skewed distributions
- Requires sorting
- May not be a value in the dataset

**Use cases:**
- Income distribution analysis (less sensitive than mean)
- House prices (skewed data)
- Test scores with outliers
- Robust central tendency measure

**Privacy Implications:**
- More robust to outlier inference
- Less sensitive to extreme values
- Still requires protection in privacy-aware contexts
- Better for skewed sensitive data

**Example:**
```
Sorted values: [10, 20, 30, 40, 50]
Median: 30 (middle value)

Sorted values: [10, 20, 30, 40, 50, 60]
Median: (30+40)/2 = 35 (average of middle two)
```

### MODE
**Value:** `"mode"`

The most frequently occurring value in a dataset. May be multiple modes (multimodal) or no mode if all values unique.

**Characteristics:**
- Useful for categorical data
- Can have multiple modes
- Not affected by outliers
- Less common for continuous data

**Use cases:**
- Most popular category or choice
- Most common job title in company
- Most frequent age group in demographic
- Categorical data summarization

**Privacy Implications:**
- Reveals most frequent class membership
- Lower privacy risk than mean/median for small groups
- Still requires context protection
- Good for qualitative summaries

**Example:**
```
Values: ["A", "B", "B", "C", "C", "C", "D"]
Mode: "C" (occurs 3 times)

Numeric values: [10, 20, 20, 20, 30]
Mode: 20 (occurs 3 times)
```

## Comparison Guide

| Aspect | MEAN | MEDIAN | MODE |
|--------|------|--------|------|
| **Data Type** | Numeric | Numeric | Any |
| **Robustness** | Sensitive to outliers | Robust | Very robust |
| **Skewed Data** | Poor choice | Good choice | Good choice |
| **Interpretation** | Most intuitive | Intuitive | Most obvious |
| **Computation** | Fast | Requires sort | Frequency count |
| **Privacy Risk** | High | Moderate | Low |
| **Multiple Values** | Never | Sometimes | Often |

## Selection Guide

### By Data Distribution

**Normal Distribution**
- Use `MEAN` - most representative

**Skewed Distribution**
- Use `MEDIAN` - resistant to outliers
- Use `MODE` for categorical data

**Bimodal/Multimodal**
- Use `MODE` - identifies clusters
- Use `MEDIAN` as compromise

### By Data Type

**Continuous Numeric**
- Prefer `MEAN` or `MEDIAN`
- Avoid `MODE` unless very discrete

**Categorical**
- Use `MODE` - only applicable option
- Not meaningful with MEAN or MEDIAN

**Ordinal (Ranked)**
- Use `MEDIAN` - respects ordering
- `MODE` acceptable for categories

### By Privacy Requirements

**High Privacy Sensitivity**
- `MODE` has lowest identifiability risk
- `MEDIAN` more robust than MEAN

**Standard Privacy**
- `MEDIAN` good balance
- Requires aggregation/k-anonymity

**Lower Sensitivity**
- `MEAN` acceptable with protection

## Related Components

- **Aggregation Operations:** Uses statistical methods for group-by operations
- **Privacy Metrics:** Methods affect privacy assessment
- **Data Profiling:** Statistical summary of datasets

## Common Patterns

### Multi-Method Analysis

```python
from pamola_core.common.enum.statistical_method import StatisticalMethod
import pandas as pd

def analyze_column(series: pd.Series) -> dict:
    """Compute all statistics for a column."""
    stats = {}
    for method in StatisticalMethod:
        if method == StatisticalMethod.MEAN:
            stats[method.value] = series.mean()
        elif method == StatisticalMethod.MEDIAN:
            stats[method.value] = series.median()
        elif method == StatisticalMethod.MODE:
            stats[method.value] = series.mode().iloc[0] if not series.mode().empty else None

    return stats

# Usage
data = pd.Series([10, 20, 20, 30, 40])
results = analyze_column(data)
# {"mean": 24, "median": 20, "mode": 20}
```

### Privacy-Aware Aggregation

```python
from pamola_core.common.enum.statistical_method import StatisticalMethod

def aggregate_with_privacy(group: pd.DataFrame,
                          method: StatisticalMethod,
                          k_anon: int) -> dict:
    """Aggregate with privacy check."""
    if len(group) < k_anon:
        return None  # Suppress if below k-anonymity

    if method == StatisticalMethod.MEAN:
        return {"value": group.values.mean(), "count": len(group)}
    elif method == StatisticalMethod.MEDIAN:
        return {"value": group.values.median(), "count": len(group)}
    elif method == StatisticalMethod.MODE:
        return {"value": group.values.mode()[0], "count": len(group)}
```

### Configurable Statistical Operations

```python
from pamola_core.common.enum.statistical_method import StatisticalMethod

operation_config = {
    "salary": {
        "method": StatisticalMethod.MEDIAN,  # Robust for outliers
        "min_group_size": 5
    },
    "job_title": {
        "method": StatisticalMethod.MODE,  # Most common title
        "min_group_size": 3
    },
    "age": {
        "method": StatisticalMethod.MEAN,  # Average age
        "min_group_size": 10
    }
}
```

## Best Practices

1. **Use Enum for Type Safety**
   ```python
   # Good
   method = StatisticalMethod.MEDIAN

   # Avoid
   method = "median"  # String is error-prone
   ```

2. **Match Method to Distribution**
   ```python
   # Check distribution first
   if data.skew() > 1:
       method = StatisticalMethod.MEDIAN  # Skewed data
   else:
       method = StatisticalMethod.MEAN    # Normal data
   ```

3. **Document Statistical Choice**
   ```python
   # Good - explains reasoning
   # Use MEDIAN for salary (resistant to extreme outliers)
   stat_method = StatisticalMethod.MEDIAN
   ```

4. **Handle Edge Cases**
   ```python
   def safe_statistics(series: pd.Series,
                      method: StatisticalMethod):
       if series.empty:
           return None
       if method == StatisticalMethod.MODE:
           modes = series.mode()
           return modes.iloc[0] if not modes.empty else None
       # ... handle other methods
   ```

## Examples

### Income Analysis

```python
import pandas as pd
from pamola_core.common.enum.statistical_method import StatisticalMethod

income_data = pd.Series([30000, 35000, 40000, 45000, 200000])  # Has outlier

mean_income = income_data.mean()      # 70000 (inflated by outlier)
median_income = income_data.median()  # 40000 (unaffected by outlier)

# For privacy reporting: use MEDIAN to be robust to outlier identification
print(f"Median Income: ${median_income}")  # More representative
```

### Job Title Distribution

```python
import pandas as pd
from pamola_core.common.enum.statistical_method import StatisticalMethod

job_titles = pd.Series(["Engineer", "Engineer", "Engineer", "Manager", "Analyst"])

most_common = job_titles.mode()[0]  # "Engineer"
# Mode shows most common role
```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- Three statistical methods: MEAN, MEDIAN, MODE
- Distribution-aware selection guide
- Privacy considerations for each method
