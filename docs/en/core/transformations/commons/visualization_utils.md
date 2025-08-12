# PAMOLA Core: `visualization_utils` Module

---

## Overview

The `visualization_utils` module is a core component of the PAMOLA framework, providing a suite of utilities for visualizing and analyzing the impact of data transformation operations. It is designed to support privacy-preserving AI data processing by enabling before/after comparisons, profiling, and reporting of dataset changes throughout transformation pipelines.

This module is intended for use by developers and data scientists working with the PAMOLA Core framework, especially in scenarios where transparency, auditability, and interpretability of data transformations are required.

---

## Key Features

- **Field and Record Count Comparisons**: Compare the number of fields (columns) and records (rows) before and after transformations.
- **Data Distribution Analysis**: Visualize and summarize changes in data distributions for specific fields.
- **Dataset Profiling**: Generate comprehensive overviews and profiling statistics for datasets.
- **Visualization Generation**: Create bar charts, histograms, and pie charts for both original and transformed data.
- **Support for Aggregated and Grouped Data**: Utilities support both individual and grouped/aggregate visualizations.
- **Standardized Output Filenames**: Consistent naming for generated visualizations, supporting reproducibility and traceability.
- **Sampling for Large Datasets**: Efficiently sample large datasets for visualization without memory overload.

---

## Dependencies

### Standard Library
- `logging`: For logging and debugging.
- `datetime`: For timestamping outputs.
- `pathlib.Path`: For file and directory path management.
- `typing`: For type annotations (`Dict`, `Any`, `Optional`).

### Third-Party
- `pandas`: For data manipulation and analysis.

### Internal Modules
- `pamola_core.utils.visualization`: Provides plotting functions:
    - `create_bar_plot`
    - `create_histogram`
    - `create_pie_chart`

---

## Exception Classes

> **Note:** This module does not define custom exception classes. All errors are raised as standard Python exceptions (e.g., `ValueError`, `TypeError`, `KeyError`).

### Example: Handling Standard Exceptions

```python
try:
    # Attempt to generate a visualization with invalid data
    generate_visualization_filename(None, 'histogram')
except Exception as e:
    # Handle any error (e.g., TypeError if operation_name is None)
    print(f"Visualization generation failed: {e}")
```

**When exceptions are likely to be raised:**
- Passing `None` or invalid types to required parameters.
- Providing a non-existent column name to a function expecting a DataFrame column.
- File system errors when saving visualizations.

---

## Main Functions

### 1. `generate_visualization_filename`

```python
def generate_visualization_filename(
    operation_name: str,
    visualization_type: str,
    extension: str = "png",
    join_filename: Optional[str] = None,
    include_timestamp: Optional[bool] = None,
) -> str:
```
**Parameters:**
- `operation_name`: Name of the operation creating the visualization.
- `visualization_type`: Type of visualization (e.g., "histogram").
- `extension`: File extension (default: "png").
- `join_filename`: Optional string to join to the filename.
- `include_timestamp`: If True, appends a timestamp.

**Returns:**
- Standardized filename as a string.

**Raises:**
- `TypeError` if required arguments are missing or invalid.

---

### 2. `create_field_count_comparison`

```python
def create_field_count_comparison(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `original_df`: DataFrame before transformation.
- `transformed_df`: DataFrame after transformation.
- `operation_name`: Name of the transformation.
- `output_path`: Path for saving outputs (currently unused).

**Returns:**
- Dictionary with field count comparison, added/removed fields, and chart recommendation.

---

### 3. `create_record_count_comparison`

```python
def create_record_count_comparison(
    original_df: pd.DataFrame,
    transformed_dfs: Dict[str, pd.DataFrame],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `original_df`: Original DataFrame.
- `transformed_dfs`: Dict mapping output names to DataFrames.
- `operation_name`: Name of the operation.
- `output_path`: Path for outputs (unused).

**Returns:**
- Dictionary with record counts, changes, and chart recommendations.

---

### 4. `create_data_distribution_comparison`

```python
def create_data_distribution_comparison(
    original_series: pd.Series,
    transformed_series: pd.Series,
    field_name: str,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `original_series`: Series before transformation.
- `transformed_series`: Series after transformation.
- `field_name`: Name of the field.
- `operation_name`: Name of the operation.
- `output_path`: Path for outputs (unused).

**Returns:**
- Dictionary with statistics, plot data, and chart recommendations.

---

### 5. `create_dataset_overview`

```python
def create_dataset_overview(
    df: pd.DataFrame, 
    title: str, 
    output_path: Path
) -> Dict[str, Any]:
```
**Parameters:**
- `df`: DataFrame to profile.
- `title`: Title for the dataset.
- `output_path`: Path for outputs (unused).

**Returns:**
- Dictionary with dataset statistics, profiling, and chart recommendations.

---

### 6. Visualization Generation Functions

- `generate_dataset_overview_vis`
- `generate_data_distribution_comparison_vis`
- `generate_record_count_comparison_vis`
- `generate_field_count_comparison_vis`

Each of these functions generates visualizations (bar charts, histograms, pie charts) and returns a dictionary of output file paths.

---

### 7. `sample_large_dataset`

```python
def sample_large_dataset(
    data: pd.Series, 
    max_samples: int = 10000, 
    random_state: int = 42
) -> pd.Series:
```
**Parameters:**
- `data`: Series to sample.
- `max_samples`: Maximum number of samples.
- `random_state`: Seed for reproducibility.

**Returns:**
- Sampled Series.

---

## Dependency Resolution and Completion Validation

This module does not implement explicit dependency resolution or completion validation logic. Instead, it is designed to be used as a utility within larger transformation or pipeline tasks, where such logic is managed by the pipeline controller or task manager (e.g., `BaseTask`).

- **Dependency Handling**: All data dependencies (e.g., DataFrames, Series) must be provided by the caller. The module does not fetch or resolve dependencies on its own.
- **Completion Validation**: The module assumes that input data is valid and complete. Validation of task completion is the responsibility of the calling context.

---

## Usage Examples

### Example 1: Field Count Comparison

```python
from pamola_core.transformations.commons.visualization_utils import create_field_count_comparison

# Compare columns before and after transformation
result = create_field_count_comparison(
    original_df=original_df,
    transformed_df=transformed_df,
    operation_name="Imputation",
    output_path=Path("./outputs")
)
print(result)
```

### Example 2: Data Distribution Visualization

```python
from pamola_core.transformations.commons.visualization_utils import generate_data_distribution_comparison_vis

# Generate and save a histogram/bar chart for a field
vis_paths = generate_data_distribution_comparison_vis(
    original_data=original_df["age"],
    transformed_data=transformed_df["age"],
    field_label="age",
    operation_name="Anonymization",
    task_dir=Path("./outputs"),
    timestamp="20250523"
)
print(vis_paths)
```

### Example 3: Sampling a Large Dataset

```python
from pamola_core.transformations.commons.visualization_utils import sample_large_dataset

# Sample a large column for visualization
sampled = sample_large_dataset(df["salary"], max_samples=5000)
```

---

## Integration Notes

- **With BaseTask**: These utilities are typically called from within a `BaseTask` subclass or pipeline step, after data has been loaded and transformed.
- **Visualization Output**: All visualization functions require a `Path` for output, and return the path(s) to generated files for further use or reporting.
- **Logging**: The module uses the standard `logging` library. Set the logger level to `DEBUG` for detailed trace output.

---

## Error Handling and Exception Hierarchy

- All errors are standard Python exceptions (e.g., `TypeError`, `ValueError`).
- No custom exception hierarchy is defined in this module.
- Always validate input data types and existence of required columns before calling functions.

---

## Configuration Requirements

- No explicit configuration object is required for this module.
- All configuration (e.g., output paths, operation names) must be provided as function arguments.
- Ensure that output directories exist and are writable before generating visualizations.

---

## Security Considerations and Best Practices

- **File Path Security**: Always use `Path` objects and avoid user-supplied raw strings for file paths to prevent path traversal vulnerabilities.
- **Data Privacy**: Do not log or visualize sensitive data fields unless necessary. Use sampling and aggregation to minimize exposure.
- **Output Directory Permissions**: Ensure that output directories are secure and not world-writable.

### Example: Security Failure and Handling

```python
from pathlib import Path

# BAD: Using user input directly as a path (risk of path traversal)
user_path = "../../etc/passwd"
# This could overwrite or expose sensitive files if not validated

# GOOD: Restrict output to a known safe directory
safe_dir = Path("./outputs")
safe_path = safe_dir / "visualization.png"
```

**Risks of Disabling Path Security:**
- Allowing arbitrary output paths can lead to data leakage, file overwrites, or privilege escalation.
- Always validate and sanitize any user-supplied paths.

---

## Best Practices

1. **Validate Input Data**: Always check that DataFrames and Series contain the expected columns and data types before calling visualization functions.
2. **Use Standardized Filenames**: Use `generate_visualization_filename` to ensure consistent and traceable output files.
3. **Sample Large Datasets**: Use `sample_large_dataset` to avoid memory issues when visualizing very large columns.
4. **Integrate with Task Pipelines**: Call these utilities from within pipeline tasks to automate reporting and visualization.
5. **Handle Exceptions Gracefully**: Wrap calls in try/except blocks and log errors for debugging.
6. **Restrict Output Paths**: Always write visualizations to controlled directories.
7. **Document Data Transformations**: Use the returned dictionaries to document and audit changes in your pipeline reports.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: All data dependencies (DataFrames, Series) are expected to be produced within the pipeline and passed directly to these utilities.
- **External (Absolute Path) Dependencies**: Avoid using absolute paths for input/output unless necessary for integration with external systems. Always prefer relative or controlled paths within your project structure.
