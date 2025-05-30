# PAMOLA Core: Merging Utilities Module

**Module Path:** `pamola_core.transformations.commons.merging_utils`

---

## Overview

The `merging_utils` module provides a suite of utility functions for preparing, generating, and saving visualizations related to data merging and join operations within the PAMOLA Core framework. It is designed to support privacy-preserving AI data processing by offering clear, auditable, and reproducible visual summaries of key merging steps, including record and field overlaps, dataset size comparisons, and join result distributions.

This module is intended for use in data transformation pipelines, especially where transparency and traceability of merging operations are required. All major steps and data preparation are logged at the DEBUG level for traceability.

---

## Key Features

- **Record Overlap Visualization:**
  - Prepares and generates Venn diagrams showing overlap between record keys in two datasets.
- **Field (Column) Overlap Visualization:**
  - Prepares and generates Venn diagrams for field (column) overlap between datasets.
- **Dataset Size Comparison:**
  - Prepares and generates bar charts comparing the sizes of datasets before and after merging.
- **Join Type Distribution:**
  - Prepares and generates pie charts showing the distribution of join results (matched, only left, only right).
- **Comprehensive Logging:**
  - All data preparation and visualization steps are logged for auditability.

---

## Dependencies

### Standard Library
- `logging`
- `pathlib.Path`
- `typing.Any`, `Dict`, `Optional`

### Third-Party
- `pandas`

### Internal Modules
- `pamola_core.utils.visualization`
  - `create_bar_plot`
  - `create_pie_chart`
  - `create_venn_diagram`

---

## Exception Classes

> **Note:** This module does not define custom exception classes. Standard Python exceptions (e.g., `KeyError`, `FileNotFoundError`) may be raised if input DataFrames or paths are invalid. Exception handling should be implemented by the caller as appropriate.

### Example: Handling a KeyError
```python
try:
    overlap_data = create_record_overlap_data(left_df, right_df, 'id', 'id', 'merge', output_path)
except KeyError as e:
    # Handle missing key column in DataFrame
    logger.error(f"Key column missing: {e}")
    # Take corrective action or abort
```
**When Raised:**
- If the specified key column does not exist in one of the DataFrames.

---

## Main Functions

### 1. `create_record_overlap_data`
```python
def create_record_overlap_data(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `left_df`: Left-side DataFrame.
- `right_df`: Right-side DataFrame.
- `left_key`: Key column in left DataFrame.
- `right_key`: Key column in right DataFrame.
- `operation_name`: Label for the operation.
- `output_path`: Path for saving outputs.

**Returns:**
- Dictionary with counts and sets for overlap, only-left, only-right, and chart recommendation.

**Raises:**
- `KeyError` if key columns are missing.

---

### 2. `generate_record_overlap_vis`
```python
def generate_record_overlap_vis(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- `left_df`, `right_df`, `left_key`, `right_key`, `field_label`, `operation_name`, `task_dir`, `timestamp`, `visualization_paths` (see above).

**Returns:**
- Updated `visualization_paths` with the Venn diagram path.

**Raises:**
- Exceptions from `create_venn_diagram` if file cannot be written.

---

### 3. `create_dataset_size_comparison`
```python
def create_dataset_size_comparison(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `left_df`, `right_df`, `merged_df`, `operation_name`, `output_path` (see above).

**Returns:**
- Dictionary with dataset sizes and chart recommendation.

---

### 4. `generate_dataset_size_comparison_vis`
```python
def generate_dataset_size_comparison_vis(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- As above.

**Returns:**
- Updated `visualization_paths` with the bar chart path.

---

### 5. `create_field_overlap_data`
```python
def create_field_overlap_data(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `left_df`, `right_df`, `operation_name`, `output_path` (see above).

**Returns:**
- Dictionary with field overlap counts and sets.

---

### 6. `generate_field_overlap_vis`
```python
def generate_field_overlap_vis(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- As above.

**Returns:**
- Updated `visualization_paths` with the field overlap Venn diagram path.

---

### 7. `create_join_type_distribution_data`
```python
def create_join_type_distribution_data(
    merged_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_type: str,
    operation_name: str,
    output_path: Path,
) -> Dict[str, Any]:
```
**Parameters:**
- `merged_df`, `left_key`, `right_key`, `join_type`, `operation_name`, `output_path` (see above).

**Returns:**
- Dictionary with join result counts and chart recommendation.

---

### 8. `generate_join_type_distribution_vis`
```python
def generate_join_type_distribution_vis(
    merged_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    join_type: str,
    field_label: str,
    operation_name: str,
    task_dir: Path,
    timestamp: str,
    visualization_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```
**Parameters:**
- As above.

**Returns:**
- Updated `visualization_paths` with the pie chart path.

---

## Dependency Resolution and Completion Validation

This module does not implement explicit dependency resolution or completion validation logic. It is designed to be used as a utility within larger data processing pipelines, where such logic is typically managed by pipeline/task orchestration classes (e.g., `BaseTask`).

- **Input Validation:**
  - Functions expect valid pandas DataFrames and correct key/column names. Validation is performed implicitly by pandas and will raise exceptions if inputs are invalid.
- **Output Validation:**
  - Visualization functions return paths to generated files, which can be checked for existence by the caller.

---

## Usage Examples

### Example 1: Generating a Record Overlap Venn Diagram
```python
from pamola_core.transformations.commons.merging_utils import generate_record_overlap_vis

# Prepare DataFrames
left_df = ...  # pd.DataFrame with 'id' column
right_df = ... # pd.DataFrame with 'id' column

# Set parameters
field_label = 'customer_id'
operation_name = 'merge_customers'
task_dir = Path('/tmp/merge_outputs')
timestamp = '20250523'

# Generate Venn diagram
vis_paths = generate_record_overlap_vis(
    left_df, right_df, 'id', 'id', field_label, operation_name, task_dir, timestamp
)
# vis_paths['record_overlap_venn'] contains the path to the saved image
```

### Example 2: Handling Missing Key Columns
```python
try:
    vis_paths = generate_record_overlap_vis(
        left_df, right_df, 'missing_key', 'id', field_label, operation_name, task_dir, timestamp
    )
except KeyError as e:
    logger.error(f"Missing key column: {e}")
    # Handle error or notify user
```

### Example 3: Dataset Size Comparison Bar Chart
```python
from pamola_core.transformations.commons.merging_utils import generate_dataset_size_comparison_vis

# ... prepare left_df, right_df, merged_df ...
vis_paths = generate_dataset_size_comparison_vis(
    left_df, right_df, merged_df, field_label, operation_name, task_dir, timestamp
)
# vis_paths['dataset_size_comparison_bar_chart'] contains the path
```

---

## Integration Notes

- Designed for use within PAMOLA Core data transformation pipelines.
- Can be integrated with task orchestration classes (e.g., `BaseTask`) to automate visualization of merging steps.
- Visualization output paths can be stored in task metadata for reporting or auditing.

---

## Error Handling and Exception Hierarchy

- **Standard Exceptions:**
  - `KeyError`: Raised if a specified key or column is missing in a DataFrame.
  - `FileNotFoundError`: Raised if output directories do not exist (ensure directories are created before calling visualization functions).
  - `TypeError`: Raised if input types are incorrect (e.g., passing a list instead of a DataFrame).

**Best Practice:** Always validate DataFrame columns and output paths before calling utility functions.

---

## Configuration Requirements

- No explicit configuration object is required for this module.
- Output directories (`task_dir`, `output_path`) must exist and be writable.
- DataFrames must contain the specified key columns.

---

## Security Considerations and Best Practices

- **Path Security:**
  - Always validate and sanitize output paths to prevent overwriting critical files.
  - Avoid using user-supplied paths without validation.

**Example: Security Failure**
```python
# BAD: Using untrusted user input for output path
output_path = Path(user_supplied_path)
# This could overwrite system files if not validated!
```

**Mitigation:**
```python
# GOOD: Restrict output to a known safe directory
safe_dir = Path('/tmp/merge_outputs')
output_path = safe_dir / 'output.png'
```

- **Risks of Disabling Path Security:**
  - Disabling path security or using unchecked paths can lead to data loss, privilege escalation, or system compromise.
  - Always restrict output to controlled directories.

---

## Internal vs. External Dependencies

- **Internal Dependencies:**
  - Data produced within the pipeline (e.g., DataFrames from previous tasks).
  - Use logical task IDs and managed paths.
- **External Dependencies:**
  - Data from outside the pipeline (e.g., external CSVs, absolute paths).
  - Use absolute paths with caution and document their use.

---

## Best Practices

1. **Validate DataFrames and Columns:**
   - Ensure all required columns exist before calling utility functions.
2. **Pre-create Output Directories:**
   - Always create output directories before saving visualizations.
3. **Use Logging for Traceability:**
   - Leverage the module's DEBUG logging for audit trails.
4. **Handle Exceptions Gracefully:**
   - Catch and log exceptions to avoid silent failures in pipelines.
5. **Restrict Output Paths:**
   - Never allow untrusted user input to control output file locations.
6. **Document External Data Sources:**
   - Clearly distinguish between internal and external data dependencies in your pipeline configuration.
