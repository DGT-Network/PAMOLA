# Dataset Summary Analysis

**Module:** `pamola_core.analysis.dataset_summary`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

The `dataset_summary` module provides rapid overview analysis of pandas DataFrames. It detects field types automatically, counts missing values, identifies outliers using IQR (Interquartile Range), and computes basic statistics for all columns in a single pass.

This is the entry point for understanding dataset structure before deeper analysis. The module handles numeric coercion intelligently—columns stored as strings but containing numeric data (e.g., "123.45") are detected and classified appropriately.

## Key Features

- **Automatic field type detection** with configurable numeric-like threshold (default 75%)
- **Missing value analysis** including per-field counts and affected field tracking
- **Outlier detection** using IQR method on numeric columns
- **Numeric coercion** for object/categorical columns with numeric content
- **Robust error handling** with detailed logging and safe fallbacks
- **Fast single-pass analysis** suitable for large datasets

## Architecture

```
DatasetAnalyzer (class)
├── __init__(numeric_threshold, logger)
├── _analyze_field_types(df) → (numeric_cols, categorical_cols, coerced_dict)
├── _detect_outliers(df, numeric_cols, coerced) → (count, affected_fields)
└── analyze_dataset_summary(df) → Dict

analyze_dataset_summary() (function wrapper for backward compatibility)
```

## Core Classes/Methods

### `DatasetAnalyzer`

**Constructor:**
```python
DatasetAnalyzer(numeric_threshold: float = 0.75, logger: Optional[logging.Logger] = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_threshold` | float | 0.75 | Minimum ratio of successfully converted values (0–1) to classify column as numeric-like |
| `logger` | Logger \| None | None | Custom logger instance; auto-creates if not provided |

**Methods:**

### `analyze_dataset_summary()`

Main analysis function. Returns comprehensive overview dictionary.

**Parameters:**
```python
def analyze_dataset_summary(df: pd.DataFrame) -> Dict
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | Input dataset to analyze |

**Returns:**
```python
{
    "rows": int,                                # Number of records
    "columns": int,                             # Number of fields
    "total_cells": int,                         # rows × columns
    "missing_values": {
        "value": int,                           # Total missing cells
        "fields_with_missing": int              # Count of columns with ≥1 missing value
    },
    "numeric_fields": {
        "count": int,                           # Native + numeric-like columns
        "percentage": float                     # As fraction of total columns (0–1)
    },
    "categorical_fields": {
        "count": int,                           # Object/category columns (non-numeric)
        "percentage": float                     # As fraction of total columns (0–1)
    },
    "outliers": {
        "count": int,                           # Total outliers across all numeric fields
        "affected_fields": List[str]            # Column names where outliers detected
    }
}
```

**Raises:**
| Exception | Condition |
|-----------|-----------|
| `ValueError` | DataFrame is None or invalid |

**Examples:**

### Basic Usage
```python
import pandas as pd
from pamola_core.analysis import analyze_dataset_summary

df = pd.read_csv("customer_data.csv")
summary = analyze_dataset_summary(df)

print(f"Dataset shape: {summary['rows']} rows × {summary['columns']} columns")
print(f"Completeness: {100 - (summary['missing_values']['value'] / summary['total_cells'] * 100):.1f}%")
print(f"Data types: {summary['numeric_fields']['count']} numeric, {summary['categorical_fields']['count']} categorical")
```

### Numeric Coercion Example
```python
# DataFrame with mixed types
df = pd.DataFrame({
    'age': ['25', '30', 'unknown', '45'],      # String but mostly numeric
    'city': ['NYC', 'LA', 'NYC', 'CHI'],       # Pure categorical
    'salary': [50000.5, 60000.0, 75000.0, 80000.0]  # Native numeric
})

summary = analyze_dataset_summary(df)
# With default threshold 0.75, 'age' is detected as numeric (3/4 = 0.75)
assert summary['numeric_fields']['count'] == 2  # 'age' + 'salary'
assert summary['categorical_fields']['count'] == 1  # 'city'
```

### Outlier Detection
```python
df = pd.DataFrame({
    'normal_dist': [1, 2, 3, 4, 5] * 10,       # Normal values
    'with_outliers': list(range(1, 51)) + [500, 600]  # Has extreme values
})

summary = analyze_dataset_summary(df)
print(f"Outliers found: {summary['outliers']['count']}")
print(f"Affected fields: {summary['outliers']['affected_fields']}")
```

### Custom Logger
```python
import logging
from pamola_core.analysis.dataset_summary import DatasetAnalyzer

custom_logger = logging.getLogger("my_app.analysis")
analyzer = DatasetAnalyzer(numeric_threshold=0.90, logger=custom_logger)

summary = analyzer.analyze_dataset_summary(df)
```

## Numeric Detection Logic

The module uses a two-stage approach to classify numeric fields:

### Stage 1: Native Types
Columns with dtype in `int64`, `float64`, etc., are immediately classified as numeric.

### Stage 2: Numeric-Like Coercion
For object/category columns, the analyzer:
1. Attempts `pd.to_numeric(..., errors='coerce')` conversion
2. Calculates conversion ratio: `(non-null converted) / (original non-null)`
3. If ratio ≥ `numeric_threshold`, column is marked numeric-like

**Example:**
```python
# Column: ['1', '2', 'unknown', '4', '5'] (5 total, 4 numeric)
# Conversion ratio: 4/5 = 0.80
# With threshold 0.75, classified as numeric ✓
```

## Missing Value Analysis

Missing values are detected via `pd.isna()` and counted:
- **Total**: All NaN/None cells across entire DataFrame
- **Per-field**: Number of columns with ≥1 missing value

The module applies safe calculations with try-catch to handle edge cases.

## Outlier Detection

Uses IQR (Interquartile Range) method via `detect_outliers_iqr()` from `pamola_core.profiling.commons.statistical_analysis`:

1. For each numeric column (native + coerced):
   - Drop NaN values
   - Skip if <3 data points (insufficient for IQR)
   - Call `detect_outliers_iqr(series)` → `{"count": N, ...}`
2. Aggregate counts and field names

**IQR method**: Outliers are values outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

## Best Practices

1. **Review Numeric Coercion**: Check `numeric_fields['count']` against expected types. If too many or too few fields are detected, adjust `numeric_threshold`.

2. **Handle Outliers Downstream**: Outlier detection is informational; decide separately whether to remove, cap, or transform them.

3. **Monitor Missing Values**: Use `missing_values['fields_with_missing']` to prioritize imputation or filtering.

4. **Pre-Process Large Datasets**: For DataFrames >1M rows, consider sampling before analysis, as outlier detection can be compute-intensive.

5. **Combine with Profiling**: Use this summary as a starting point, then dive deeper with `analyze_descriptive_stats()` or `profiling.analyze_dataset_attributes()`.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Too many numeric fields detected | Low `numeric_threshold` | Increase threshold (e.g., 0.90) |
| Too few numeric fields detected | High `numeric_threshold` | Decrease threshold (e.g., 0.50) |
| Zero outliers found | All data normal-distributed | Data is clean; no action needed |
| High missing value count | Sparse dataset | Pre-impute or filter before anonymization |
| Negative outlier count (edge case) | Bug in IQR calculation | Report with sample data |

## Related Components

- [`analyze_descriptive_stats()`](./descriptive_stats.md) - Detailed statistics per field
- [`calculate_full_risk()`](./privacy_risk.md) - Privacy assessment after summary
- [`profiling.analyze_dataset_attributes()`](../profiling/) - Extended field analysis
- [`profiling.commons.statistical_analysis.detect_outliers_iqr()`](../profiling/) - IQR outlier detection

## Summary Analysis

**Purpose**: Rapid overview of dataset structure, completeness, and data quality.

**Input**: Any pandas DataFrame.

**Output**: Structured dictionary with rows, columns, missing values, field types, and outliers.

**Strengths**:
- Single-pass analysis (fast, memory-efficient)
- Automatic numeric detection with configurable threshold
- Handles mixed types gracefully
- Detailed outlier tracking

**Limitations**:
- Numeric coercion is heuristic-based; may misclassify edge cases
- Outlier detection uses IQR only (assumes roughly normal distribution)
- No temporal or hierarchical pattern detection

**Typical Usage**: First step in EDA, data profiling, or pre-anonymization assessment.
