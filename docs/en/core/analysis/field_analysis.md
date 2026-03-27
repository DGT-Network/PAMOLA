# Field-Level Analysis

**Module:** `pamola_core.analysis.field_analysis`
**Version:** 1.0.0
**Status:** Stable
**Last Updated:** 2026-03-23

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Core Functions](#core-functions)
4. [Usage Examples](#usage-examples)
5. [Data Types & Behavior](#data-types--behavior)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)

---

## Overview

The `field_analysis` module provides field-level statistical analysis and visualization for pandas DataFrames. It computes descriptive statistics, distribution visualizations, and privacy-quality indicators (uniqueness, missingness) suitable for integration into larger anonymization and data governance workflows.

**Design Goals:**
- Simple, extensible API for single-field analysis
- Consistent integration with `descriptive_stats` and `distribution` modules
- Safe defaults with comprehensive input validation
- Diagnostic logging for troubleshooting

---

## Key Features

- **Automatic Type Detection**: Distinguishes numeric vs. categorical fields automatically
- **Customized Statistics**: Returns different metrics based on field type
- **Distribution Visualization**: Generates histograms (numeric) or bar charts (categorical)
- **Missingness & Uniqueness**: Computes quality indicators for privacy assessment
- **Project Root Fallback**: Uses `get_project_root()` if no output directory specified
- **Structured Output**: Returns dict with separate analysis and visualization results

---

## Core Functions

### `analyze_field_level()`

Analyze a single field with type-appropriate descriptive stats and visualization.

**Signature:**
```python
def analyze_field_level(
    df: pd.DataFrame,
    field_name: str,
    viz_dir: Optional[Path] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | — | Input dataset containing the field |
| `field_name` | str | — | Name of column to analyze (must exist in df) |
| `viz_dir` | Path \| None | None | Directory to save visualization files; defaults to project root if None |

**Returns:**

```python
{
    "field_level_analysis": Dict[str, Any],    # Descriptive stats result
    "field_level_visualization": Dict[str, Any] # Chart generation result
}
```

**Field-Level Analysis (Numeric):**
- **Order**: count, mean, std, min, max
- **Extra Statistics**: unique, median, mode

**Field-Level Analysis (Categorical):**
- **Order**: count, unique, top (most frequent), freq (frequency of top)
- **Extra Statistics**: mode

**Field-Level Visualization:**
- **Numeric**: Histogram with 10 bins (auto-scaled)
- **Categorical**: Bar chart with bar chart mode enabled

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `KeyError` | field_name not in DataFrame columns |
| `ValueError` | df is empty or field_name is invalid |

---

## Usage Examples

### Basic Field Analysis

```python
import pandas as pd
from pamola_core.analysis import analyze_field_level
from pathlib import Path

# Load data
df = pd.read_csv("customers.csv")

# Analyze age field
result = analyze_field_level(df, field_name="age", viz_dir=Path("output/analysis"))

# Extract results
stats = result["field_level_analysis"]
charts = result["field_level_visualization"]

print(f"Mean age: {stats['mean']:.2f}")
print(f"Unique values: {stats['unique']}")
print(f"Chart saved to: {charts.get('path')}")
```

### Numeric Field Analysis

```python
# Analyze numeric salary field
result = analyze_field_level(df, field_name="salary")

stats = result["field_level_analysis"]
# Output includes: count, mean, std, min, max, unique, median, mode
print(f"Salary range: {stats['min']:.2f} - {stats['max']:.2f}")
print(f"Standard deviation: {stats['std']:.2f}")
print(f"Unique salary values: {stats['unique']}")
```

### Categorical Field Analysis

```python
# Analyze categorical department field
result = analyze_field_level(df, field_name="department")

stats = result["field_level_analysis"]
# Output includes: count, unique, top, freq, mode
print(f"Unique departments: {stats['unique']}")
print(f"Most common: {stats['top']} (count: {stats['freq']})")
print(f"Mode: {stats['mode']}")
```

### With Custom Output Directory

```python
from pathlib import Path

output_dir = Path("./results/field_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

for field in ["age", "salary", "department"]:
    result = analyze_field_level(df, field_name=field, viz_dir=output_dir)
    print(f"✓ Analyzed {field}")
    print(f"  Chart: {result['field_level_visualization'].get('path')}")
```

### Default Project Root Behavior

```python
# If viz_dir is None, uses project root automatically
result = analyze_field_level(df, field_name="email")

# Chart is saved to <project_root>/email_distribution.html
# or similar, depending on distribution module defaults
```

---

## Data Types & Behavior

### Numeric Fields

**Detected as:** `pd.api.types.is_numeric_dtype()`

**Statistics Returned:**
```python
{
    "count": int,           # Non-null count
    "mean": float,          # Average
    "std": float,           # Standard deviation
    "min": float,           # Minimum value
    "max": float,           # Maximum value
    "unique": int,          # Count of unique values
    "median": float,        # 50th percentile
    "mode": float|None      # Most frequent value
}
```

**Visualization:** Histogram with 10 bins, auto-scaled for value range

**Example:**
```python
result = analyze_field_level(df, "age")
# {
#   "field_level_analysis": {
#       "count": 1000, "mean": 42.3, "std": 15.2,
#       "min": 18, "max": 85, "unique": 68, "median": 41, "mode": 40
#   },
#   "field_level_visualization": {...}
# }
```

### Categorical Fields

**Detected as:** Non-numeric dtypes (object, category, string)

**Statistics Returned:**
```python
{
    "count": int,           # Non-null count
    "unique": int,          # Number of distinct values
    "top": str,             # Most frequent category
    "freq": int,            # Frequency of most frequent
    "mode": str|None        # Same as top for categorical
}
```

**Visualization:** Bar chart with all categories (or top N if >50 unique)

**Example:**
```python
result = analyze_field_level(df, "department")
# {
#   "field_level_analysis": {
#       "count": 1000, "unique": 12, "top": "Engineering",
#       "freq": 250, "mode": "Engineering"
#   },
#   "field_level_visualization": {...}
# }
```

### Missing Values

**Behavior:**
- Automatic skip by pandas `describe()` (count = non-null only)
- Missing values NOT included in mode/median calculations
- Uniqueness counts non-null distinct values only

**Example:**
```python
# Field with 1000 rows, 150 null values
result = analyze_field_level(df, "phone")
# "count": 850 (non-null count, not total rows)
# Visualization uses 850 non-null values
```

---

## Best Practices

### 1. Validate Field Existence Before Analysis

```python
field_name = "customer_id"
if field_name not in df.columns:
    raise ValueError(f"Field '{field_name}' not found in DataFrame")

result = analyze_field_level(df, field_name)
```

### 2. Use Explicit Output Directories

```python
from pathlib import Path

# Good: Clear, explicit directory
viz_dir = Path("./output/field_analysis")
viz_dir.mkdir(parents=True, exist_ok=True)

result = analyze_field_level(df, "age", viz_dir=viz_dir)
```

### 3. Integrate into Batch Analysis

```python
analysis_results = {}

for field in df.columns:
    try:
        result = analyze_field_level(df, field, viz_dir=output_dir)
        analysis_results[field] = result
        print(f"✓ {field}")
    except Exception as e:
        print(f"✗ {field}: {e}")

# Aggregate results
for field, result in analysis_results.items():
    stats = result["field_level_analysis"]
    print(f"{field}: unique={stats.get('unique')}, count={stats.get('count')}")
```

### 4. Interpret Statistics by Type

```python
result = analyze_field_level(df, "transaction_amount")
stats = result["field_level_analysis"]

# For numeric fields:
print(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")
print(f"Average: {stats['mean']:.2f} ± {stats['std']:.2f}")
print(f"Central tendency: {stats['median']:.2f}")

# For categorical fields:
result = analyze_field_level(df, "country")
stats = result["field_level_analysis"]
print(f"Diversity: {stats['unique']} unique countries")
print(f"Concentration: {stats['freq']} records in {stats['top']}")
```

### 5. Check Output Before Processing

```python
result = analyze_field_level(df, field_name="email")

# Verify visualization was created
if result["field_level_visualization"]:
    print(f"Chart: {result['field_level_visualization'].get('path')}")
else:
    print("No visualization generated")
```

---

## Related Components

- [`analyze_descriptive_stats()`](./descriptive_stats.md) - Underlying statistics computation (multi-field)
- [`visualize_distribution_df()`](./distribution.md) - Distribution visualization backend
- [`get_project_root()`](../utils/paths.md) - Fallback directory resolution
- [`analyze_dataset_summary()`](./dataset_summary.md) - Multi-field dataset overview

---

## Summary Analysis

**Purpose:** Compute field-level descriptive statistics and visualizations with type-appropriate defaults.

**Inputs:** DataFrame, field name, optional output directory.

**Outputs:** Nested dict with analysis (stats) and visualization (chart) results.

**Strengths:**
- Automatic type detection and adaptive statistics
- Consistent integration with broader analysis modules
- Safe defaults (project root fallback)
- Single function for unified analysis

**Limitations:**
- Single field at a time (use `descriptive_stats` for multi-field)
- Basic visualization (no customization)
- No advanced statistical tests

**Typical Workflow:** Loop through DataFrame columns → Analyze each → Aggregate statistics → Generate privacy report.
