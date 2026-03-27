# Constants Module Documentation

**Module:** `pamola_core.common.constants`
**Class:** `Constants`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Constants Reference](#constants-reference)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Related Components](#related-components)

## Overview

The `Constants` class centralizes global constants used throughout PAMOLA.CORE, eliminating hardcoded values and improving maintainability. This module defines standard operation names, date formats, type mappings, artifact categories, and configuration standards.

**Class Type:** Static Constants Container
**Access Method:** Class attributes (no instantiation required)
**Usage Pattern:** `Constants.CONSTANT_NAME`

## Key Features

- **Operation Names**: Standardized operation identifiers
- **Date Formats**: 40+ common date/time format patterns
- **Type Mappings**: Pandas/NumPy dtype conversions
- **Distribution Metrics**: Standard metric display labels
- **Artifact Categories**: Classification for outputs
- **Safe Evaluation**: Safe globals for expression evaluation
- **I/O Configuration**: CSV, JSON, and data handling constants
- **Analysis Modes**: Standard analysis mode identifiers

## Constants Reference

### Operation Names

```python
from pamola_core.common.constants import Constants

# Supported operation types
Constants.OPERATION_NAMES
# Output: ["generalization", "noise_addition"]
```

**Used By:** Operation identification and logging

**Common Operations:**
- `"generalization"` - Categorical/numeric/datetime generalization
- `"noise_addition"` - Uniform numeric/temporal noise

### Date Format Patterns

```python
Constants.COMMON_DATE_FORMATS
# 40+ date format patterns for parsing and formatting
```

**Complete List of Formats:**

**ISO & International**
- `ISO8601` - ISO 8601 with optional timezone
- `%Y-%m-%d` - YYYY-MM-DD
- `%Y/%m/%d` - YYYY/MM/DD
- `%Y.%m.%d` - YYYY.MM.DD

**English Month Names**
- `%d-%b-%Y` - 01-Jan-2023
- `%d-%B-%Y` - 01-January-2023
- `%d %b %Y` - 01 Jan 2023
- `%B %d, %Y` - January 1, 2023

**DMY Format**
- `%d.%m.%Y` - 01.01.2023
- `%d/%m/%Y` - 01/01/2023
- `%d-%m-%Y` - 01-01-2023

**MDY Format**
- `%m/%d/%Y` - 01/01/2023
- `%m-%d-%Y` - 01-01-2023

**With Time (24-hour)**
- `%Y-%m-%d %H:%M` - 2023-01-01 14:30
- `%Y-%m-%d %H:%M:%S` - 2023-01-01 14:30:45
- `%d.%m.%Y %H:%M` - 01.01.2023 14:30

**With Time (12-hour AM/PM)**
- `%Y-%m-%d %I:%M %p` - 2023-01-01 02:30 PM
- `%B %d, %Y %I:%M %p` - January 1, 2023 02:30 PM

**Non-separated Formats**
- `%Y%m%d` - 20230101
- `%d%m%Y` - 01012023

**Usage:**
```python
from pamola_core.common.constants import Constants
import pandas as pd

# Use in date conversion
df['date_col'] = pd.to_datetime(
    df['date_str'],
    format=Constants.COMMON_DATE_FORMATS[0]
)

# Try multiple formats
for fmt in Constants.COMMON_DATE_FORMATS:
    try:
        result = pd.to_datetime(df['date_col'], format=fmt)
        if result.notna().sum() / len(df) > 0.95:  # >95% parsed
            break
    except:
        continue
```

### Frequency Mapping

```python
Constants.FREQ_MAP
# Output: {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
```

**Pandas Frequency Aliases:**

| Key | Value | Meaning |
|-----|-------|---------|
| `"day"` | `"D"` | Daily |
| `"week"` | `"W"` | Weekly |
| `"month"` | `"MS"` | Month start |
| `"quarter"` | `"QS"` | Quarter start |
| `"year"` | `"YS"` | Year start |

**Usage:**
```python
from pamola_core.common.constants import Constants

# For time-based grouping
df.resample(Constants.FREQ_MAP["month"]).sum()

# For frequency labels
freq_label = "month"
pandas_freq = Constants.FREQ_MAP[freq_label]
```

### Logging Configuration

```python
Constants.LOG_DIR  # "logs"
```

Standard directory for storing operation logs.

### Distribution & Metric Labels

```python
Constants.DISTRIBUTION_LABELS
```

**Dictionary of Metric Display Names:**

| Key | Display Name |
|-----|--------------|
| `"distribution_similarity_score"` | Distribution Similarity Score |
| `"kl_divergence_orig_mid"` | KL Divergence (Original vs Midpoint) |
| `"kl_divergence_gen_mid"` | KL Divergence (Generated vs Midpoint) |
| `"uniqueness_preservation"` | Uniqueness Preservation |
| `"entropy_original"` | Entropy (Original) |
| `"entropy_generated"` | Entropy (Generated) |
| `"gini_original"` | Gini Coefficient (Original) |
| `"gini_generated"` | Gini Coefficient (Generated) |
| `"top_value_overlap@10"` | Top-10 Value Overlap |

**Usage:**
```python
from pamola_core.common.constants import Constants

# For visualization and reporting
metrics_report = {}
for metric_key in results.keys():
    display_name = Constants.DISTRIBUTION_LABELS.get(
        metric_key,
        metric_key  # Fallback to key if not in labels
    )
    metrics_report[display_name] = results[metric_key]
```

### Artifact Categories

```python
Constants.Artifact_Category_Dictionary   # "dictionary"
Constants.Artifact_Category_Output       # "output"
Constants.Artifact_Category_Visualization # "visualization"
Constants.Artifact_Category_Metrics      # "metrics"
Constants.Artifact_Category_Mapping      # "mapping"
```

**Artifact Types:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `Artifact_Category_Dictionary` | `"dictionary"` | Mapping/reference dictionaries |
| `Artifact_Category_Output` | `"output"` | Primary output data |
| `Artifact_Category_Visualization` | `"visualization"` | Plots and charts |
| `Artifact_Category_Metrics` | `"metrics"` | Measurement results |
| `Artifact_Category_Mapping` | `"mapping"` | Field and value mappings |

**Usage:**
```python
from pamola_core.common.constants import Constants

# Categorize operation outputs
output_artifacts = {
    Constants.Artifact_Category_Output: anonymized_data,
    Constants.Artifact_Category_Metrics: privacy_scores,
    Constants.Artifact_Category_Visualization: plots,
    Constants.Artifact_Category_Mapping: field_mappings
}
```

### Pandas Data Type Mapping

```python
Constants.PANDAS_DTYPE_MAP
```

**Type Conversions:**

| String Key | Pandas/NumPy Type | Nullable |
|------------|-------------------|----------|
| `"string"` | `pd.StringDtype()` | Yes |
| `"boolean"` | `pd.BooleanDtype()` | Yes |
| `"int64"` | `pd.Int64Dtype()` | Yes |
| `"float64"` | `pd.Float64Dtype()` | Yes |
| `"datetime"` | `np.dtype("datetime64[ns]")` | No |
| `"datetimeutc"` | `pd.DatetimeTZDtype(tz="UTC")` | No |

**Usage:**
```python
from pamola_core.common.constants import Constants
import pandas as pd

# Convert column dtype using mapping
df['age'] = df['age'].astype(
    Constants.PANDAS_DTYPE_MAP["int64"]
)

# Safe type conversion
def convert_dtype(series, dtype_str):
    if dtype_str in Constants.PANDAS_DTYPE_MAP:
        return series.astype(Constants.PANDAS_DTYPE_MAP[dtype_str])
    return series
```

### Safe Evaluation Globals

```python
Constants.SAFE_GLOBALS
# Output: {"__builtins__": {}, "pd": pd, "np": np}
```

**Purpose:** Secure environment for evaluating user expressions

**Contents:**
- `"__builtins__"`: Empty (prevents access to built-in functions)
- `"pd"`: pandas module (for DataFrame operations)
- `"np"`: numpy module (for numerical operations)

**Usage:**
```python
from pamola_core.common.constants import Constants

# Safely evaluate custom expressions
user_expr = "df['age'].apply(lambda x: x * 2)"
result = eval(user_expr, Constants.SAFE_GLOBALS, {"df": df})

# Prevents dangerous operations
# This would fail safely:
# eval("__import__('os').system('rm -rf /')", Constants.SAFE_GLOBALS)
```

### I/O Configuration Constants

```python
Constants.ORIENT              # "orient"
Constants.ENCODING            # "encoding"
Constants.SEP                 # "sep"
Constants.DELIMITER           # "delimiter"
Constants.QUOTE_CHAR          # "quotechar"
Constants.TRANSFORMED_DATASET_NAME  # "transformed_dataset_name"

Constants.DELIMITER_COMMA     # ","
Constants.DOUBLE_QUOTE        # '"'
Constants.COLUMNS_ORIENT      # "columns"
Constants.UTF_8               # "utf-8"
```

**Standard CSV/JSON Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `DELIMITER_COMMA` | `","` | CSV delimiter |
| `DOUBLE_QUOTE` | `'"'` | Quote character |
| `COLUMNS_ORIENT` | `"columns"` | JSON orientation |
| `UTF_8` | `"utf-8"` | Default encoding |

**Usage:**
```python
from pamola_core.common.constants import Constants
import pandas as pd

# Read CSV with standard delimiter
df = pd.read_csv(
    filepath,
    sep=Constants.DELIMITER_COMMA,
    encoding=Constants.UTF_8,
    quotechar=Constants.DOUBLE_QUOTE
)

# Write JSON with standard orientation
df.to_json(
    outpath,
    orient=Constants.COLUMNS_ORIENT,
    encoding=Constants.UTF_8
)
```

### Analysis Modes

```python
Constants.ANALYSIS_MODE      # "analysis_mode"
Constants.ENRICH             # "ENRICH"
Constants.BOTH               # "BOTH"
```

**Analysis Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `ANALYSIS_MODE` | `"analysis_mode"` | Config key |
| `ENRICH` | `"ENRICH"` | Enrich dataset mode |
| `BOTH` | `"BOTH"` | Analyze and enrich |

## Usage Examples

### Import and Use Constants

```python
from pamola_core.common.constants import Constants

# Access operation names
if operation in Constants.OPERATION_NAMES:
    print(f"Valid operation: {operation}")

# Access date formats for parsing
for fmt in Constants.COMMON_DATE_FORMATS[:5]:
    print(f"Trying format: {fmt}")

# Get artifact categorization
output = {
    "data": anonymized_data,
    "category": Constants.Artifact_Category_Output
}
```

### Date Format Detection

```python
from pamola_core.common.constants import Constants
import pandas as pd

def parse_flexible_dates(date_column):
    """Parse dates with multiple format support."""
    for fmt in Constants.COMMON_DATE_FORMATS:
        try:
            if fmt == "ISO8601":
                result = pd.to_datetime(date_column)
            else:
                result = pd.to_datetime(date_column, format=fmt)

            # If >95% parsed successfully, return
            if result.notna().sum() / len(date_column) > 0.95:
                return result
        except:
            continue

    # Fallback to pandas inference
    return pd.to_datetime(date_column, errors="coerce")
```

### Type Conversion Using Mappings

```python
from pamola_core.common.constants import Constants
import pandas as pd

def standardize_dtypes(df):
    """Convert DataFrame dtypes using standard mapping."""
    dtype_map = {
        "age": Constants.PANDAS_DTYPE_MAP["int64"],
        "name": Constants.PANDAS_DTYPE_MAP["string"],
        "is_active": Constants.PANDAS_DTYPE_MAP["boolean"],
        "salary": Constants.PANDAS_DTYPE_MAP["float64"],
    }

    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df
```

### Metric Reporting

```python
from pamola_core.common.constants import Constants

def generate_report(metrics):
    """Generate human-readable metrics report."""
    report = {}

    for metric_key, metric_value in metrics.items():
        # Get display name from labels
        display_name = Constants.DISTRIBUTION_LABELS.get(
            metric_key,
            metric_key.replace("_", " ").title()
        )
        report[display_name] = metric_value

    return report
```

## Best Practices

1. **Use Constants Instead of Hardcoded Values**
   ```python
   # Good
   df.to_csv(filepath, sep=Constants.DELIMITER_COMMA)

   # Avoid
   df.to_csv(filepath, sep=",")
   ```

2. **Reference Log Directory Centrally**
   ```python
   # Good
   import os
   log_path = os.path.join(Constants.LOG_DIR, "operation.log")

   # Avoid
   log_path = os.path.join("logs", "operation.log")
   ```

3. **Use Safe Globals for Expression Evaluation**
   ```python
   # Good - secure
   result = eval(user_expr, Constants.SAFE_GLOBALS, locals())

   # Avoid - unsafe
   result = eval(user_expr)
   ```

4. **Leverage Frequency Mappings**
   ```python
   # Good
   freq = Constants.FREQ_MAP.get(frequency_str, "D")
   df.resample(freq).sum()

   # Avoid
   freq = {"month": "M", "day": "D"}.get(frequency_str)
   ```

5. **Use Artifact Categories Consistently**
   ```python
   # Good - standardized categorization
   outputs = {
       Constants.Artifact_Category_Output: data,
       Constants.Artifact_Category_Metrics: metrics
   }
   ```

## Related Components

- **DataHelper** (`pamola_core.common.helpers.data_helper`) - Uses COMMON_DATE_FORMATS
- **I/O Operations** (`pamola_core.io`) - Uses CSV/JSON constants
- **Logging** (`pamola_core.common.logging`) - Uses LOG_DIR
- **Type Aliases** (`pamola_core.common.type_aliases`) - Related to PANDAS_DTYPE_MAP
- **Metrics** (`pamola_core.metrics`) - Uses DISTRIBUTION_LABELS

## Constants Summary Table

| Category | Examples | Purpose |
|----------|----------|---------|
| Operations | OPERATION_NAMES | Operation identification |
| Dates | COMMON_DATE_FORMATS | Date parsing/formatting |
| Types | PANDAS_DTYPE_MAP | Data type conversion |
| Metrics | DISTRIBUTION_LABELS | Display names |
| Artifacts | Artifact_Category_* | Output categorization |
| I/O | DELIMITER_COMMA, UTF_8 | Data handling |
| Evaluation | SAFE_GLOBALS | Secure expression evaluation |
| Frequency | FREQ_MAP | Time-based grouping |

## Maintenance Notes

When adding new constants:
1. Group by category
2. Document with comments
3. Update this documentation
4. Consider backward compatibility
5. Use descriptive naming (e.g., `ARTIFACT_CATEGORY_OUTPUT`)

## Implementation Notes

- Constants are defined as class attributes (no instantiation needed)
- All constants are immutable (use tuples, frozen sets where applicable)
- String constants use lowercase with underscores
- Date formats follow Python's strftime conventions
- Type mapping uses pandas nullable dtypes where available
