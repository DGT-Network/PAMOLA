# PAMOLA Core: Transformation Metric Utilities Module

## Overview

The `metric_utils.py` module provides a comprehensive suite of utilities for collecting, calculating, and saving metrics related to data transformation operations within the PAMOLA Core framework. It is designed to support privacy-preserving AI data processing by enabling detailed analysis and comparison of datasets before and after transformation, as well as field-level statistics and performance metrics.

This module is essential for evaluating the impact of data transformations, ensuring data quality, and supporting compliance and transparency in data processing pipelines.

---

## Key Features

- **Dataset Comparison**: Compare row/column counts, value changes, nulls, and memory usage between original and transformed datasets.
- **Field-Level Statistics**: Compute descriptive statistics, null/unique counts, top values, and pattern detection for fields.
- **Transformation Impact Analysis**: Assess changes in data quality, completeness, distribution, and correlation.
- **Performance Metrics**: Measure elapsed time, throughput, and processing speed for transformation operations.
- **Flexible Metrics Saving**: Save metrics to JSON files, with optional encryption and integration with the `DataWriter` utility.
- **Progress Tracking**: Integrate with hierarchical progress trackers for reporting.

---

## Dependencies

### Standard Library
- `datetime`
- `logging`
- `pathlib`
- `time`

### Third-Party Libraries
- `numpy`
- `pandas`
- `scipy.stats`

### Internal Modules
- `pamola_core.utils.io` (`ensure_directory`, `write_json`)
- `pamola_core.utils.ops.op_data_writer` (`DataWriter`)
- `pamola_core.utils.progress` (`HierarchicalProgressTracker`)

---

## Exception Classes

This module does not define custom exception classes. Instead, it raises standard Python exceptions (e.g., `ValueError`, `Exception`) for error conditions. Below are common exceptions and their handling:

- **ValueError**: Raised when required arguments (such as DataFrames) are missing or invalid.

  **Example:**
  ```python
  try:
      calculate_dataset_comparison(None, df2)
  except ValueError as e:
      print(f"Error: {e}")  # Both DataFrames must be provided
  ```
  *Raised when either input DataFrame is `None`.*

- **Exception**: Used for unexpected errors during metric calculations or file operations. These are logged and re-raised for upstream handling.

  **Example:**
  ```python
  try:
      calculate_field_statistics(df)
  except Exception as e:
      logger.error(f"Unexpected error: {e}")
  ```

---

## Main Functions and Their Usage

### 1. `calculate_dataset_comparison`

```python
def calculate_dataset_comparison(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame
) -> Dict[str, Any]
```
**Parameters:**
- `original_df`: The original DataFrame before transformation.
- `transformed_df`: The transformed DataFrame after processing.

**Returns:**
- Dictionary containing comparison metrics (row/column counts, value/null changes, memory usage).

**Raises:**
- `ValueError` if either DataFrame is `None`.

---

### 2. `calculate_field_statistics`

```python
def calculate_field_statistics(
    df: pd.DataFrame,
    fields: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]
```
**Parameters:**
- `df`: DataFrame to analyze.
- `fields`: List of field names to analyze (all columns if `None`).

**Returns:**
- Dictionary mapping field names to statistics (basic stats, nulls, uniques, top values, patterns).

**Raises:**
- `ValueError` if DataFrame is `None`.

---

### 3. `calculate_transformation_impact`

```python
def calculate_transformation_impact(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame
) -> Dict[str, Any]
```
**Parameters:**
- `original_df`: The original DataFrame before transformation.
- `transformed_df`: The transformed DataFrame after processing.

**Returns:**
- Dictionary with data quality, completeness, distribution, correlation, and field impact metrics.

**Raises:**
- `ValueError` if either DataFrame is `None`.

---

### 4. `calculate_performance_metrics`

```python
def calculate_performance_metrics(
    start_time: float,
    end_time: float,
    input_rows: int,
    output_rows: int
) -> Dict[str, Any]
```
**Parameters:**
- `start_time`: Start time (from `time.time()`).
- `end_time`: End time (from `time.time()`).
- `input_rows`: Number of input rows processed.
- `output_rows`: Number of output rows produced.

**Returns:**
- Dictionary with elapsed time, rows/sec, throughput ratio, and performance rating.

**Raises:**
- `ValueError` if times are invalid.

---

### 5. `save_metrics_json`

```python
def save_metrics_json(
    metrics: Dict[str, Any],
    task_dir: Path,
    operation_name: str,
    field_name: str,
    writer: Optional[DataWriter] = None,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    encrypt_output: bool = False
) -> Path
```
**Parameters:**
- `metrics`: Metrics to save.
- `task_dir`: Directory for saving metrics.
- `operation_name`: Name of the operation.
- `field_name`: Name of the field.
- `writer`: Optional `DataWriter` for advanced saving.
- `progress_tracker`: Optional progress tracker.
- `encrypt_output`: Whether to encrypt the output file.

**Returns:**
- Path to the saved metrics file.

**Raises:**
- Logs and returns path even if save fails.

---

## Dependency Resolution and Completion Validation

This module does not directly manage dependencies, but integrates with `DataWriter` and `HierarchicalProgressTracker` for advanced data and progress management. Ensure that these dependencies are properly configured and available in your pipeline for full functionality.

---

## Usage Examples

### Comparing Datasets and Saving Metrics

```python
import pandas as pd
from pathlib import Path
from pamola_core.transformations.commons.metric_utils import (
    calculate_dataset_comparison, save_metrics_json
)

# Load or create your DataFrames
df_original = pd.read_csv('data/raw/input.csv')
df_transformed = pd.read_csv('data/processed/output.csv')

# Calculate comparison metrics
metrics = calculate_dataset_comparison(df_original, df_transformed)

# Save metrics to a JSON file
metrics_path = save_metrics_json(
    metrics,
    task_dir=Path('metrics/'),
    operation_name='transform',
    field_name='all_fields'
)
print(f"Metrics saved to: {metrics_path}")
```

### Field Statistics and Pattern Detection

```python
from pamola_core.transformations.commons.metric_utils import calculate_field_statistics

stats = calculate_field_statistics(df_original, fields=['email', 'birthdate'])
print(stats['email']['pattern_detection'])  # Detects email pattern
```

### Performance Metrics Example

```python
import time
start = time.time()
# ... perform transformation ...
end = time.time()
perf = calculate_performance_metrics(start, end, input_rows=10000, output_rows=9500)
print(perf['rows_per_second'])
```

### Handling Errors

```python
try:
    calculate_dataset_comparison(None, df_transformed)
except ValueError as e:
    print(f"Input error: {e}")
```

---

## Integration Notes

- **With BaseTask**: Use these utilities in your custom task classes to automatically collect and save transformation metrics as part of your pipeline.
- **With DataWriter**: For advanced output management (e.g., encryption, versioning), pass a `DataWriter` instance to `save_metrics_json`.
- **With Progress Tracking**: Use `HierarchicalProgressTracker` to report progress during long-running metric calculations or saves.

---

## Error Handling and Exception Hierarchy

- All functions validate input and raise `ValueError` for missing or invalid arguments.
- Unexpected errors are logged and re-raised as generic `Exception`.
- File save failures are logged, and the intended path is returned for traceability.

---

## Configuration Requirements

- Ensure that the `task_dir` exists or can be created for saving metrics.
- If using `DataWriter`, configure it with appropriate encryption keys and output settings.
- For encrypted output, set `encrypt_output=True` and manage encryption keys securely.

---

## Security Considerations and Best Practices

- **Do not disable encryption for sensitive metrics unless necessary.**
- **Risks of Disabling Path Security**: Saving metrics to arbitrary or insecure locations can expose sensitive data. Always validate and restrict output paths.

**Security Failure Example:**
```python
# BAD: Saving metrics to a world-writable or public directory
save_metrics_json(metrics, Path('/tmp/public/'), 'transform', 'field')
# This may expose sensitive data to unauthorized users.
```

**Proper Handling:**
```python
# GOOD: Save to a restricted, project-specific directory
save_metrics_json(metrics, Path('metrics/'), 'transform', 'field', encrypt_output=True)
```

---

## Internal vs. External Dependencies

- **Internal**: Use within the PAMOLA pipeline, leveraging `DataWriter` and project directories.
- **External (Absolute Path)**: Only use absolute paths for outputs that must be shared outside the pipeline. Avoid unless required for integration.

---

## Best Practices

1. **Validate All Inputs**: Always check that DataFrames and paths are valid before calling metric functions.
2. **Use Encryption for Sensitive Data**: Enable `encrypt_output` when saving metrics containing personal or confidential information.
3. **Integrate with Progress Tracking**: For long-running operations, use `HierarchicalProgressTracker` to provide user feedback.
4. **Handle Exceptions Gracefully**: Catch and log errors to avoid pipeline crashes.
5. **Prefer Internal Paths**: Use project-relative paths for saving metrics to maintain data security and traceability.
6. **Document Your Metrics**: Save and version metrics for reproducibility and auditability.
