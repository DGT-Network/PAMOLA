# PAMOLA Cell Suppression Operation - Detailed Documentation

## Overview

The `CellSuppressionOperation` class provides fine-grained control over suppressing (replacing) individual cell values in datasets. Unlike attribute suppression (which removes entire columns) or record suppression (which removes entire rows), cell suppression allows targeted replacement of specific values while preserving the dataset structure.

## Table of Contents

1. [Key Features](#key-features)
2. [Installation & Import](#installation--import)
3. [Constructor Parameters](#constructor-parameters)
4. [Suppression Strategies](#suppression-strategies)
5. [Suppression Conditions](#suppression-conditions)
6. [Usage Examples](#usage-examples)
7. [Metrics & Reporting](#metrics--reporting)
8. [Advanced Features](#advanced-features)
9. [Performance Considerations](#performance-considerations)
10. [Error Handling](#error-handling)

## Key Features

- **Multiple replacement strategies**: null, mean, median, mode, constant, group-based statistics
- **Automatic suppression**: Detect and suppress outliers, rare values, or null values
- **Conditional suppression**: Target specific values based on field conditions
- **Group-based processing**: Apply different replacements per group
- **Type preservation**: Attempts to maintain original data types
- **Memory management**: Efficient handling of large datasets with statistics caching
- **Dask support**: Distributed processing for big data scenarios
- **Thread-safe**: Safe for parallel processing with proper locking

## Installation & Import

```python
from pamola_core.anonymization.suppression.cell_op import CellSuppressionOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pathlib import Path
```

## Constructor Parameters

### Required Parameters

- **`field_name`** (str): The field/column containing cells to suppress

### Suppression Strategy Parameters

- **`suppression_strategy`** (str, default="null"): How to replace suppressed cells
  - `"null"`: Replace with NULL/NaN
  - `"mean"`: Replace with column mean (numeric only)
  - `"median"`: Replace with column median (numeric only)
  - `"mode"`: Replace with most frequent value (any type)
  - `"constant"`: Replace with specified value
  - `"group_mean"`: Replace with group-specific mean
  - `"group_mode"`: Replace with group-specific mode

- **`suppression_value`** (Any, optional): Value for constant strategy

### Mode Parameters

- **`mode`** (str, default="REPLACE"): Operation mode
  - `"REPLACE"`: Modify values in-place
  - `"ENRICH"`: Create new column with suppressed values

- **`output_field_name`** (str, optional): Output field name for ENRICH mode

### Conditional Suppression Parameters

- **`condition_field`** (str, optional): Field to check for conditional suppression
- **`condition_values`** (List, optional): Values to match in condition field
- **`condition_operator`** (str, default="in"): Operator for condition
  - `"in"`, `"not_in"`, `"eq"`, `"ne"`, `"gt"`, `"lt"`, `"ge"`, `"le"`

### Automatic Suppression Parameters

- **`suppress_if`** (str, optional): Automatic suppression trigger
  - `"outlier"`: Suppress statistical outliers
  - `"rare"`: Suppress infrequent values
  - `"null"`: Suppress null values

- **`outlier_method`** (str, default="iqr"): Method for outlier detection
  - `"iqr"`: Interquartile range method
  - `"zscore"`: Z-score method

- **`outlier_threshold`** (float, default=1.5): 
  - For IQR: multiplier for IQR (1.5 = mild outliers, 3.0 = extreme)
  - For Z-score: number of standard deviations

- **`rare_threshold`** (int, default=10): Minimum frequency for non-rare values

### Group-Based Parameters

- **`group_by_field`** (str, optional): Field for grouping (required for group strategies)
- **`min_group_size`** (int, default=5): Minimum group size for statistics calculation

### Processing Parameters

- **`batch_size`** (int, default=10000): Records per batch
- **`use_cache`** (bool, default=True): Enable operation caching
- **`engine`** (str, default="auto"): Processing engine
  - `"pandas"`: Force pandas processing
  - `"dask"`: Force Dask processing
  - `"auto"`: Choose based on data size

## Suppression Strategies

### 1. Null Strategy
Replaces values with NULL/NaN. Simple but changes data type.

```python
op = CellSuppressionOperation(
    field_name="age",
    suppression_strategy="null",
    suppress_if="outlier"
)
```

### 2. Mean Strategy
Replaces with column mean. Numeric fields only.

```python
op = CellSuppressionOperation(
    field_name="salary",
    suppression_strategy="mean",
    suppress_if="outlier",
    outlier_threshold=2.0
)
```

### 3. Median Strategy
Replaces with column median. More robust to outliers than mean.

```python
op = CellSuppressionOperation(
    field_name="income",
    suppression_strategy="median",
    suppress_if="outlier"
)
```

### 4. Mode Strategy
Replaces with most frequent value. Works for any data type.

```python
op = CellSuppressionOperation(
    field_name="category",
    suppression_strategy="mode",
    suppress_if="rare",
    rare_threshold=5
)
```

### 5. Constant Strategy
Replaces with specified value.

```python
op = CellSuppressionOperation(
    field_name="sensitive_code",
    suppression_strategy="constant",
    suppression_value="REDACTED",
    condition_field="consent",
    condition_values=[False]
)
```

### 6. Group Mean/Mode Strategy
Calculates replacement value per group.

```python
op = CellSuppressionOperation(
    field_name="salary",
    suppression_strategy="group_mean",
    group_by_field="department",
    min_group_size=10,
    suppress_if="outlier"
)
```

## Suppression Conditions

### Outlier Detection

#### IQR Method
```python
# Suppress mild outliers (default)
op = CellSuppressionOperation(
    field_name="price",
    suppress_if="outlier",
    outlier_method="iqr",
    outlier_threshold=1.5  # Q1-1.5*IQR, Q3+1.5*IQR
)

# Suppress extreme outliers only
op = CellSuppressionOperation(
    field_name="price",
    suppress_if="outlier",
    outlier_method="iqr",
    outlier_threshold=3.0
)
```

#### Z-Score Method
```python
# Suppress values beyond 2 standard deviations
op = CellSuppressionOperation(
    field_name="temperature",
    suppress_if="outlier",
    outlier_method="zscore",
    outlier_threshold=2.0
)
```

### Rare Value Detection
```python
# Suppress categories appearing less than 10 times
op = CellSuppressionOperation(
    field_name="job_title",
    suppression_strategy="mode",
    suppress_if="rare",
    rare_threshold=10
)
```

### Conditional Suppression
```python
# Suppress specific values
op = CellSuppressionOperation(
    field_name="country",
    suppression_strategy="constant",
    suppression_value="OTHER",
    condition_field="country",
    condition_values=["Unknown", "N/A", "Invalid"],
    condition_operator="in"
)

# Suppress based on another field
op = CellSuppressionOperation(
    field_name="salary",
    suppression_strategy="null",
    condition_field="public_employee",
    condition_values=[True],
    condition_operator="eq"
)
```

## Usage Examples

### Basic Example
```python
from pamola_core.utils.ops.op_data_source import DataSource
from pathlib import Path

# Create operation
op = CellSuppressionOperation(
    field_name="age",
    suppression_strategy="mean",
    suppress_if="outlier"
)

# Execute
data_source = DataSource(file_path="data.csv")
task_dir = Path("./output")
result = op.execute(data_source, task_dir)

# Check results
print(f"Cells suppressed: {result.metrics['cells_suppressed']}")
print(f"Suppression rate: {result.metrics['suppression_rate']:.2f}%")
```

### Advanced Example with Groups
```python
# Department-specific outlier suppression
op = CellSuppressionOperation(
    field_name="salary",
    suppression_strategy="group_mean",
    group_by_field="department",
    min_group_size=20,
    suppress_if="outlier",
    outlier_method="iqr",
    outlier_threshold=2.0,
    mode="ENRICH",
    output_field_name="salary_cleaned"
)

# Execute with progress tracking
from pamola_core.utils.progress import ProgressTracker

tracker = ProgressTracker(total=100, description="Processing")
result = op.execute(
    data_source,
    task_dir,
    progress_tracker=tracker
)
```

### Handling Multiple Conditions
```python
# Complex conditional suppression
op = CellSuppressionOperation(
    field_name="personal_info",
    suppression_strategy="constant",
    suppression_value="[REDACTED]",
    condition_field="age",
    condition_values=[18],
    condition_operator="lt"  # Suppress for minors
)
```

## Metrics & Reporting

### Available Metrics

The operation collects comprehensive metrics:

```python
result = op.execute(data_source, task_dir)

# Basic metrics
cells_suppressed = result.metrics['cells_suppressed']
suppression_rate = result.metrics['suppression_rate']
total_processed = result.metrics['total_cells_processed']

# Strategy-specific metrics
if 'outliers_detected' in result.metrics:
    print(f"Outliers found: {result.metrics['outliers_detected']}")

# Group statistics (if applicable)
if 'group_count' in result.metrics:
    print(f"Groups processed: {result.metrics['group_count']}")
```

### Detailed Statistics Output

The operation saves detailed statistics to JSON:

```json
{
  "suppression_summary": {
    "total_cells_processed": 100000,
    "non_null_cells_processed": 95000,
    "cells_suppressed": 1425,
    "suppression_rate": 1.5,
    "suppression_strategy": "mean",
    "suppress_if": "outlier"
  },
  "suppression_by_reason": {
    "outlier": 1425
  },
  "suppression_by_strategy": {
    "mean": 1425
  },
  "global_statistics": {
    "mean": 45.67
  },
  "parameters": {
    "field_name": "age",
    "outlier_method": "iqr",
    "outlier_threshold": 1.5
  }
}
```

### Visualizations

The operation generates:
- **Comparison visualization**: Shows distribution before/after suppression
- **Distribution histogram**: For numeric fields after suppression

## Advanced Features

### Memory Management

For large numbers of groups, statistics are automatically managed:

```python
# Handles up to 10,000 groups efficiently
op = CellSuppressionOperation(
    field_name="value",
    suppression_strategy="group_mean",
    group_by_field="fine_grained_category",
    min_group_size=5
)
```

### Type Preservation

The operation attempts to preserve original data types:

```python
# Integer field remains integer after mean replacement
df = pd.DataFrame({'score': [1, 2, 3, 100, 5]})  # outlier: 100
op = CellSuppressionOperation(
    field_name="score",
    suppression_strategy="mean",
    suppress_if="outlier"
)
# Result maintains integer type where possible
```

### Dask Support

For large datasets, Dask processing is automatic:

```python
# Automatically switches to Dask for >1M rows
op = CellSuppressionOperation(
    field_name="transaction_amount",
    suppression_strategy="median",
    suppress_if="outlier",
    engine="auto",  # or force with "dask"
    max_rows_in_memory=1_000_000
)
```

## Performance Considerations

### Optimization Tips

1. **Batch Size**: Adjust based on memory
   ```python
   op = CellSuppressionOperation(
       field_name="field",
       batch_size=50000  # Larger batches for simple operations
   )
   ```

2. **Group Statistics**: Pre-filter groups
   ```python
   # Filter data before grouping for better performance
   filtered_df = df[df['group_size'] >= min_size]
   ```

3. **Outlier Detection**: Use IQR for speed, Z-score for accuracy
   ```python
   # IQR is faster for large datasets
   outlier_method="iqr"  # O(n log n) for sorting
   # vs
   outlier_method="zscore"  # O(n) but needs mean/std calculation
   ```

### Memory Usage

- **Pandas mode**: ~2-3x data size in memory
- **Dask mode**: Configurable partition size
- **Group cache**: Limited to 10,000 groups by default

## Error Handling

### Common Errors and Solutions

1. **Type Mismatch**
   ```python
   # Error: Using mean on non-numeric field
   # Solution: Use mode or constant strategy for categorical data
   ```

2. **Missing Group Field**
   ```python
   # Error: group_by_field not found
   # Solution: Verify field name or use global strategy
   ```

3. **Small Groups**
   ```python
   # Warning: Group size < min_group_size
   # Behavior: Falls back to global statistics
   ```

### Validation

The operation includes comprehensive validation:

```python
try:
    result = op.execute(data_source, task_dir)
except FieldNotFoundError as e:
    print(f"Field not found: {e}")
except FieldTypeError as e:
    print(f"Invalid field type: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

## Best Practices

1. **Choose appropriate strategies**:
   - Numeric outliers → mean/median
   - Categorical rare values → mode
   - Sensitive data → null/constant

2. **Set reasonable thresholds**:
   - IQR: 1.5 (mild), 3.0 (extreme)
   - Z-score: 2.0 (95%), 3.0 (99.7%)
   - Rare: Based on data distribution

3. **Monitor performance**:
   - Check suppression rates
   - Verify group statistics
   - Review processing time

4. **Preserve utility**:
   - Use group-based strategies when possible
   - Prefer median over mean for skewed data
   - Document suppression criteria

## Integration with PAMOLA Framework

The operation integrates seamlessly with other PAMOLA components:

```python
# Chain with other operations
from pamola_core.anonymization.generalization import GeneralizationOperation

# First generalize age ranges
gen_op = GeneralizationOperation(
    field_name="age",
    strategy="binning",
    bin_count=10
)

# Then suppress outliers in income within age groups
supp_op = CellSuppressionOperation(
    field_name="income",
    suppression_strategy="group_median",
    group_by_field="age",  # Uses generalized age
    suppress_if="outlier"
)
```

## Summary

The `CellSuppressionOperation` provides powerful, flexible cell-level data suppression with:
- Multiple replacement strategies
- Automatic outlier/rare value detection  
- Group-aware processing
- Type preservation
- Enterprise-scale performance
- Comprehensive metrics and monitoring

It's an essential tool for privacy-preserving data processing in the PAMOLA framework.