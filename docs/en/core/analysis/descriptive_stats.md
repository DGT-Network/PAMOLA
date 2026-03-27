# Descriptive Statistics Analysis

**Module:** `pamola_core.analysis.descriptive_stats`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

The `descriptive_stats` module computes normalized descriptive statistics for pandas DataFrames. It extends the standard `DataFrame.describe()` with configurable extra statistics (median, mode, unique counts) and returns results as consistent dictionary structures suitable for downstream processing, reporting, and visualization.

Unlike raw pandas output, this module normalizes statistics per-field and handles numeric vs. non-numeric columns appropriately, ensuring consistent APIs across heterogeneous datasets.

## Key Features

- **Normalized Output**: Returns consistent dictionaries instead of mixed pandas objects
- **Configurable Statistics**: Choose which stats to include (mean, std, median, mode, unique, etc.)
- **Type-Aware Handling**: Applies appropriate statistics to numeric vs. categorical fields
- **Missing Count Tracking**: Automatically calculates missing values per field
- **Extra Statistics**: Adds median and mode on top of pandas.describe() defaults
- **Production-Ready**: Safe defaults and graceful handling of edge cases

## Core Function

### `analyze_descriptive_stats()`

Compute and normalize descriptive statistics for DataFrame fields.

**Signature:**
```python
def analyze_descriptive_stats(
    df: pd.DataFrame,
    field_names: Optional[List[str]] = None,
    describe_order: Optional[List[str]] = None,
    extra_statistics: Optional[List[str]] = None,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | — | Input dataset |
| `field_names` | List[str] \| None | All columns | Specific columns to analyze; if None, analyzes all |
| `describe_order` | List[str] \| None | `["count", "unique", "top", "freq", "mean", "std", "min", "max"]` | Order and selection of pandas describe() stats to include |
| `extra_statistics` | List[str] \| None | `["unique", "median", "mode"]` | Additional stats to compute: `median`, `mode`, `unique` |

**Returns:**
```python
{
    "field_name_1": {
        "count": float,         # Non-null count
        "missing": int,         # Missing count (always added)
        "mean": float,          # Numeric: average value
        "std": float,           # Numeric: standard deviation
        "min": float,           # Numeric: minimum
        "max": float,           # Numeric: maximum
        "median": float,        # Numeric: if in extra_statistics
        "mode": float|str,      # Most frequent value; if in extra_statistics
        "unique": int,          # Count of distinct values; if in extra_statistics
        "top": str,             # Categorical: most common value
        "freq": int             # Categorical: frequency of top value
        # (only relevant fields present per data type)
    },
    "field_name_2": {...},
    ...
}
```

**Notes:**
- Each field dict contains only applicable statistics (e.g., numeric fields don't have "top"/"freq")
- NaN values are dropped before computing each statistic
- `missing` is always included and equals `len(df) - count`

### Examples

#### Basic Usage
```python
import pandas as pd
from pamola_core.analysis import analyze_descriptive_stats

df = pd.DataFrame({
    'age': [25, 30, None, 45, 50],
    'salary': [50000.0, 60000.0, 55000.0, 75000.0, 80000.0],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT']
})

stats = analyze_descriptive_stats(df)

print(stats['age'])
# Output:
# {
#     'count': 4.0,
#     'missing': 1,
#     'mean': 37.5,
#     'std': 10.31,
#     'min': 25.0,
#     'max': 50.0,
#     'median': 37.5,
#     'mode': None or most frequent,
#     'unique': 4
# }

print(stats['department'])
# Output:
# {
#     'count': 5.0,
#     'missing': 0,
#     'unique': 3,
#     'top': 'Sales',
#     'freq': 2,
#     'mode': 'Sales'
# }
```

#### Selective Fields
```python
# Analyze only numeric columns
stats = analyze_descriptive_stats(
    df,
    field_names=['age', 'salary']
)

# Result includes only 'age' and 'salary'
assert 'department' not in stats
```

#### Custom Statistics Selection
```python
# Minimal stats: only count, missing, mean, std
stats = analyze_descriptive_stats(
    df,
    describe_order=['count', 'mean', 'std'],
    extra_statistics=[]  # No extras
)

# Result will have only: count, missing, mean, std per field
```

#### Include All Defaults
```python
# Use all available statistics
stats = analyze_descriptive_stats(
    df,
    extra_statistics=['unique', 'median', 'mode']
)

# Numeric fields: count, unique, mean, std, min, max, median, mode, missing
# Categorical: count, unique, top, freq, mode, missing
```

## Field Type Handling

### Numeric Fields (int64, float64)
```python
# Include statistics:
# count, missing, mean, std, min, max
# + extra_statistics: median, mode, unique
```

### Categorical Fields (object, category)
```python
# Include statistics:
# count, missing, unique, top, freq
# + extra_statistics: mode
```

### Mixed/String Fields
All available describe() statistics are computed and returned as-is (typically strings).

## Statistical Computation

### Basic Stats (pandas.describe())
Provided by `DataFrame[fields].describe(include='all')` with configurable order.

### Extra Statistics

| Statistic | Numeric | Categorical | Computation |
|-----------|---------|-------------|------------|
| `median` | ✓ | — | `Series.median()` |
| `mode` | ✓ | ✓ | `Series.mode().iloc[0]` (most frequent) |
| `unique` | ✓ | ✓ | `Series.nunique()` |

### Missing Count
Always computed: `missing = len(df) - count`

## Best Practices

1. **Combine with Dataset Summary**: Use `analyze_dataset_summary()` first for overview, then `analyze_descriptive_stats()` for detailed field analysis.

2. **Handle Missing Values**: Check the `missing` count per field. Decide whether to impute, filter, or document for downstream analysis.

3. **Check Variance**: For numeric fields, compute `std` and verify non-zero variance before using in statistical models.

4. **Validate Mode**: For categorical fields, ensure the `mode` is meaningful (not just due to data entry error).

5. **Review Outliers**: Use min/max to identify potential outliers; cross-reference with `analyze_dataset_summary()` outlier detection.

6. **Document Thresholds**: If field stats inform downstream decisions (e.g., binning, filtering), document the thresholds used.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Field missing from output | Column not in DataFrame or typo in field_names | Check DataFrame columns; verify spelling |
| NaN/inf in stats | All values in field are NaN or inf | Pre-filter or impute before analysis |
| mode is NaN | Tie or no mode computed | Use `top` for categorical instead |
| std is 0 | Constant field (all same value) | Remove zero-variance columns |
| count != len(df) | Missing values present | Check `missing` count; decide on imputation |

## Performance Considerations

- **Single pass**: Function iterates once; O(n) complexity per field
- **Memory**: Loads full DataFrame; suitable for <10GB datasets
- **Optimization**: For very large datasets (>1M rows), consider sampling before analysis

## Related Components

- [`analyze_dataset_summary()`](./dataset_summary.md) - Quick field type and outlier overview
- [`visualize_distribution_df()`](./distribution.md) - Visualize field distributions
- [`analyze_correlation()`](./correlation.md) - Relationships between numeric fields
- [`profiling.analyze_dataset_attributes()`](../profiling/) - Extended field profiling

## Output Schema Example

```python
{
    "age": {
        "count": 100.0,
        "missing": 5,
        "mean": 38.5,
        "std": 12.3,
        "min": 18.0,
        "max": 75.0,
        "median": 37.0,
        "mode": 35.0,
        "unique": 52
    },
    "salary": {
        "count": 100.0,
        "missing": 0,
        "mean": 65000.0,
        "std": 15000.0,
        "min": 30000.0,
        "max": 150000.0,
        "median": 62000.0,
        "mode": 60000.0,
        "unique": 98
    },
    "department": {
        "count": 100.0,
        "missing": 0,
        "unique": 5,
        "top": "Sales",
        "freq": 25,
        "mode": "Sales"
    },
    "email": {
        "count": 100.0,
        "missing": 0,
        "unique": 100,
        "top": "john@example.com",
        "freq": 1,
        "mode": "john@example.com"
    }
}
```

## Summary Analysis

**Purpose**: Compute normalized descriptive statistics per field for reporting, validation, and downstream analysis.

**Typical Workflow**:
1. Run `analyze_dataset_summary()` for overview
2. Run `analyze_descriptive_stats()` for detailed field statistics
3. Use results to guide data cleaning, feature engineering, and model selection

**Strengths**:
- Normalized output across all data types
- Configurable stat selection
- Automatic missing count tracking
- Combines pandas describe() with extra stats (median, mode, unique)

**Limitations**:
- Doesn't handle time series or hierarchical patterns
- No percentile customization beyond defaults
- Mode computation assumes single clear mode (ties handled arbitrarily)

**Integration**: Output format (nested dicts) integrates cleanly with JSON serialization, reporting systems, and BI dashboards.
