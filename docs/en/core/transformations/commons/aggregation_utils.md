# PAMOLA Core: Aggregation Utilities Module

---

## Overview

The `aggregation_utils.py` module is a core component of the PAMOLA framework, providing robust utilities for preparing, visualizing, and managing aggregate/groupby operations on tabular data. It is designed to support privacy-preserving data processing, enabling both standard and custom aggregation workflows, and generating insightful visualizations for data analysis and reporting.

This module is intended for use within the PAMOLA Core pipeline, but its functions are general enough to be integrated into custom data processing tasks or extended for advanced analytics.

---

## Key Features

- **Aggregate Data Preparation**: Utilities for preparing data for groupby and aggregation operations.
- **Visualization Generation**: Functions to create bar charts and histograms for record counts, aggregation comparisons, and group size distributions.
- **Aggregation Dictionary Builder**: Flexible construction of aggregation dictionaries, supporting both standard and custom aggregation functions.
- **MultiIndex Handling**: Tools to flatten MultiIndex columns for easier downstream processing.
- **Custom Aggregation Support**: Seamless integration of user-defined aggregation functions, including post-processing for Dask-incompatible functions.
- **Dask Compatibility Checks**: Utilities to determine if aggregation functions are compatible with Dask.
- **Comprehensive Logging**: All major steps and data preparation are logged at DEBUG level for traceability.

---

## Dependencies

### Standard Library
- `logging`
- `pathlib.Path`
- `typing` (Any, Callable, Dict, List, Optional, Union)

### Third-Party
- `pandas`

### Internal Modules
- `pamola_core.common.helpers.custom_aggregations_helper`  
  - Provides `CUSTOM_AGG_FUNCTIONS`, `STANDARD_AGGREGATIONS`
- `pamola_core.utils.visualization`  
  - Provides `create_bar_plot`, `create_histogram`

---

## Exception Classes

This module does not define custom exception classes. Instead, it raises standard exceptions (e.g., `ValueError`) for invalid aggregation function names. Example:

### ValueError
- **When Raised**: If an aggregation function name is not found in the allowed registries (`STANDARD_AGGREGATIONS` or `CUSTOM_AGG_FUNCTIONS`).
- **Example Handling:**

```python
try:
    func = _get_aggregation_function('nonexistent_agg')
except ValueError as e:
    print(f"Aggregation error: {e}")
```

---

## Main Functions

### create_record_count_per_group_data

```python
def create_record_count_per_group_data(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `agg_df`: Aggregated DataFrame (result of groupby + agg).
- `group_by_fields`: List of fields used for grouping.
- `operation_name`: Name of the aggregate operation.
- `output_path`: Path for saving outputs.

**Returns:**
- Dict with group labels, record counts, and chart recommendation.

**Raises:**
- None (logs errors internally).

---

### generate_record_count_per_group_vis

```python
def generate_record_count_per_group_vis(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- `agg_df`: Aggregated DataFrame.
- `group_by_fields`: Fields used for grouping.
- `field_label`: Used in filenames and chart titles.
- `operation_name`: Name of the aggregate operation.
- `task_dir`: Directory to save plots.
- `timestamp`: For unique filenames.
- `visualization_paths`: Dict to collect plot paths.

**Returns:**
- Updated visualization_paths dict.

**Raises:**
- None (logs errors internally).

---

### create_aggregation_comparison_data

```python
def create_aggregation_comparison_data(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    agg_fields: List[str],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `agg_df`: Aggregated DataFrame.
- `group_by_fields`: Fields used for grouping.
- `agg_fields`: Fields that were aggregated.
- `operation_name`: Name of the aggregate operation.
- `output_path`: Path for saving outputs.

**Returns:**
- Dict with group labels, aggregation comparison data, and chart recommendation.

**Raises:**
- None (logs errors internally).

---

### generate_aggregation_comparison_vis

```python
def generate_aggregation_comparison_vis(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    agg_fields: List[str],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- See above.

**Returns:**
- Updated visualization_paths dict.

---

### create_group_size_distribution_data

```python
def create_group_size_distribution_data(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- See above.

**Returns:**
- Dict with group size distribution and chart recommendation.

---

### generate_group_size_distribution_vis

```python
def generate_group_size_distribution_vis(
    agg_df: pd.DataFrame,
    group_by_fields: List[str],
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- See above.

**Returns:**
- Updated visualization_paths dict.

---

### build_aggregation_dict

```python
def build_aggregation_dict(
    aggregations: Optional[Dict[str, List[str]]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
) -> Dict[str, List[Callable]]:
```
**Parameters:**
- `aggregations`: Dict mapping fields to standard aggregation names.
- `custom_aggregations`: Dict mapping fields to custom aggregation callables.

**Returns:**
- Aggregation dictionary mapping fields to lists of callables.

**Raises:**
- ValueError if an aggregation function is not found.

---

### flatten_multiindex_columns

```python
def flatten_multiindex_columns(columns) -> List[str]:
```
**Parameters:**
- `columns`: DataFrame column index (can be MultiIndex).

**Returns:**
- List of flattened column names.

---

### _get_aggregation_function

```python
def _get_aggregation_function(agg_name: str) -> Callable:
```
**Parameters:**
- `agg_name`: Name of the aggregation function.

**Returns:**
- Callable aggregation function.

**Raises:**
- ValueError if not found in allowed registries.

---

### is_dask_compatible_function

```python
def is_dask_compatible_function(func: Union[str, Callable]) -> bool:
```
**Parameters:**
- `func`: Function or name to check.

**Returns:**
- True if Dask-compatible, else False.

---

### apply_custom_aggregations_post_dask

```python
def apply_custom_aggregations_post_dask(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    custom_agg_dict: Dict[str, List[Callable]],
    group_by_fields: List[str],
) -> pd.DataFrame:
```
**Parameters:**
- `original_df`: Original DataFrame before aggregation.
- `result_df`: DataFrame after Dask-compatible aggregations.
- `custom_agg_dict`: Dict mapping columns to custom aggregation functions.
- `group_by_fields`: Fields used for grouping.

**Returns:**
- DataFrame with additional columns for custom aggregations.

**Raises:**
- Logs errors for failed custom aggregations.

---

## Dependency Resolution and Completion Validation

- **Aggregation Function Resolution**: The module resolves aggregation functions by name, prioritizing standard aggregations but allowing custom functions. If a function is not found, a `ValueError` is raised.
- **Completion Validation**: When merging custom aggregation results, the module ensures all group-by fields are present and columns are flattened for consistency.

---

## Usage Examples

### Basic Aggregation and Visualization

```python
import pandas as pd
from pathlib import Path
from pamola_core.transformations.commons import aggregation_utils

# Example DataFrame
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'C'],
    'value': [1, 2, 3, 4, 5]
})

# Group and aggregate
group_by_fields = ['group']
agg_df = df.groupby(group_by_fields).agg({'value': 'sum'}).reset_index()

# Prepare record count data
record_count_data = aggregation_utils.create_record_count_per_group_data(
    agg_df, group_by_fields, 'sum', Path('.')
)

# Generate bar chart visualization
aggregation_utils.generate_record_count_per_group_vis(
    agg_df, group_by_fields, 'value', 'sum', Path('./output'), '20250523'
)
```

### Handling Custom Aggregations

```python
def custom_range(series):
    return series.max() - series.min()

custom_agg_dict = {'value': [custom_range]}
result_df = aggregation_utils.apply_custom_aggregations_post_dask(
    original_df=df,
    result_df=agg_df,
    custom_agg_dict=custom_agg_dict,
    group_by_fields=group_by_fields
)
```

### Handling Aggregation Function Errors

```python
try:
    func = aggregation_utils._get_aggregation_function('not_a_real_agg')
except ValueError as e:
    print(f"Error: {e}")
```

---

## Integration Notes

- Designed for use in PAMOLA Core transformation pipelines.
- Can be integrated with custom processors or used in standalone scripts.
- Visualizations are saved to disk and paths are returned for further use.
- Compatible with Dask for scalable aggregation, with post-processing for custom functions.

---

## Error Handling and Exception Hierarchy

- **ValueError**: Raised for invalid aggregation function names.
- All other errors are logged at the DEBUG or ERROR level for traceability.
- No custom exception classes are defined in this module.

---

## Configuration Requirements

- No explicit config object is required for this module.
- Paths for output and visualization must be provided as `Path` objects.
- Custom aggregation functions must be registered in `CUSTOM_AGG_FUNCTIONS` for global use.

---

## Security Considerations and Best Practices

- **Path Security**: Always validate output paths to avoid overwriting important files or writing to unauthorized locations.
- **Custom Aggregations**: Ensure custom functions do not leak sensitive data or perform unsafe operations.
- **Logging**: Avoid logging sensitive data at DEBUG level in production environments.

### Example: Security Failure and Handling

```python
# BAD: Writing to an unvalidated path
output_path = Path('/etc/passwd')  # Dangerous!
aggregation_utils.generate_record_count_per_group_vis(
    agg_df, group_by_fields, 'value', 'sum', output_path, '20250523'
)
# This could overwrite system files if not properly checked.

# GOOD: Always use validated, project-specific directories
safe_output_path = Path('./output')
```

**Risks of Disabling Path Security**: Allowing arbitrary output paths can lead to data leaks, file overwrites, or privilege escalation. Always restrict output to controlled directories.

---

## Internal vs. External Dependencies

- **Internal**: Use logical group-by fields and aggregation functions for data produced within the pipeline.
- **External**: Only use absolute paths for outputs that must be shared outside the pipeline or integrated with external systems.

---

## Best Practices

1. **Use Logical Groupings**: Always group by meaningful fields to ensure interpretable results.
2. **Validate Output Paths**: Restrict output to project directories.
3. **Register Custom Aggregations**: Add custom functions to `CUSTOM_AGG_FUNCTIONS` for consistency.
4. **Check Dask Compatibility**: Use `is_dask_compatible_function` to ensure scalable aggregation.
5. **Log at Appropriate Levels**: Avoid logging sensitive data in production.
6. **Handle Errors Gracefully**: Catch and log exceptions, especially when applying custom aggregations.
