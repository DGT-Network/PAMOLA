# SplitFieldsOperation Module

**Module Path:** `pamola_core.transformations.splitting.split_fields_op`

---

## Overview

The `split_fields_op` module provides the `SplitFieldsOperation` class, a flexible transformation operation within the PAMOLA Core framework. Its primary purpose is to split a dataset into multiple subsets based on user-defined field groups, supporting advanced output, caching, metrics, and visualization features. This operation is designed for use in data pipelines where field-level partitioning and modular output are required, such as privacy-preserving analytics, data engineering, and ETL workflows.

---

## Key Features

- **Field-Based Splitting:** Partition a dataset into multiple subsets by grouping fields (columns) as specified.
- **ID Field Inclusion:** Optionally include a unique identifier field in each output subset.
- **Flexible Output Formats:** Supports CSV and JSON output for each subset.
- **Caching:** Intelligent caching based on input data and operation parameters to avoid redundant computation.
- **Metrics Collection:** Collects and saves detailed metrics about the split operation.
- **Visualization:** Generates bar charts and network diagrams to visualize field distribution and schema changes.
- **Progress Tracking:** Integrates with progress trackers for UI/monitoring.
- **Configurable Execution:** Supports dynamic parameter overrides, parallel processing, and batch operations.
- **Error Handling:** Robust error reporting and logging throughout the operation.

---

## Dependencies

### Standard Library
- `json`, `time`, `datetime`, `enum`, `pathlib`, `typing`, `hashlib`

### Third-Party
- `pandas`, `matplotlib`, `seaborn`, `plotly`

### Internal Modules
- `pamola_core.transformations.base_transformation_op`
- `pamola_core.utils.io`
- `pamola_core.utils.logging`
- `pamola_core.utils.ops.op_cache`
- `pamola_core.utils.ops.op_data_source`
- `pamola_core.utils.ops.op_result`
- `pamola_core.utils.progress`
- `pamola_core.utils.ops.op_registry`

---

## Exception Classes

This module does not define custom exception classes. All errors are handled using standard Python exceptions (e.g., `Exception`) and are reported via the `OperationResult` object and logger. Example error handling is shown below:

```python
try:
    result = split_op.execute(data_source, task_dir, reporter)
except Exception as e:
    # Handle unexpected errors
    print(f"SplitFieldsOperation failed: {e}")
```

**When errors are likely to be raised:**
- Invalid field group configuration (missing columns)
- Data loading failures
- Output directory or file write errors
- Visualization or metrics generation failures

---

## Main Classes

### SplitFieldsOperation

#### Constructor
```python
SplitFieldsOperation(
    name: str = "split_fields_operation",
    description: str = "Split dataset by fields",
    id_field: str = None,
    field_groups: Optional[Dict[str, List[str]]] = None,
    include_id_field: bool = True,
    output_format: str = OutputFormat.CSV.value,
    **kwargs
)
```
**Parameters:**
- `name`: Name of the operation.
- `description`: Short description.
- `id_field`: Field used as a unique identifier.
- `field_groups`: Mapping of group names to lists of field names.
- `include_id_field`: Whether to include the ID field in each output.
- `output_format`: Output file format ("csv" or "json").
- `**kwargs`: Additional configuration.

#### Key Attributes
- `id_field`: The unique identifier field.
- `field_groups`: Dictionary of field groupings.
- `include_id_field`: Boolean flag for ID field inclusion.
- `output_format`: Output format for files.
- `use_cache`, `force_recalculation`, `generate_visualization`, `include_timestamp`, `save_output`, `parallel_processes`, `batch_size`, `use_dask`, `use_encryption`, `encryption_key`: Various execution and output controls.

#### Public Methods

##### execute
```python
def execute(
    self,
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    progress_tracker: Optional[ProgressTracker] = None,
    **kwargs
) -> OperationResult
```
- **Parameters:**
    - `data_source`: Source for input data.
    - `task_dir`: Directory for outputs and logs.
    - `reporter`: Status reporter or logger.
    - `progress_tracker`: Optional progress tracker.
    - `**kwargs`: Dynamic overrides for operation parameters.
- **Returns:** `OperationResult` summarizing the outcome.
- **Raises:** Returns error status in `OperationResult` on failure.

##### _process_data
```python
def _process_data(
    self,
    df: pd.DataFrame,
    **kwargs
) -> Dict[str, pd.DataFrame]
```
- **Parameters:**
    - `df`: Input DataFrame.
    - `**kwargs`: Not used directly.
- **Returns:** Dictionary mapping group names to DataFrames.

##### _collect_metrics
```python
def _collect_metrics(
    self,
    input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
) -> Dict[str, Any]
```
- **Parameters:**
    - `input_data`: Original input data.
    - `output_data`: Resulting split data.
- **Returns:** Dictionary of metrics.

##### _save_metrics
```python
def _save_metrics(
    self,
    metrics: Dict[str, Any],
    task_dir: Path,
    timestamp: str,
    result: OperationResult
) -> Path
```
- **Parameters:**
    - `metrics`: Metrics dictionary.
    - `task_dir`: Output directory.
    - `timestamp`: Timestamp string.
    - `result`: OperationResult to update.
- **Returns:** Path to saved metrics file.

##### _save_output
```python
def _save_output(
    self,
    result_subsets: dict[str, pd.DataFrame],
    task_dir: Path,
    timestamp: Optional[str],
    result: OperationResult
)
```
- **Parameters:**
    - `result_subsets`: Output DataFrames.
    - `task_dir`: Output directory.
    - `timestamp`: Timestamp string.
    - `result`: OperationResult to update.

##### _generate_visualizations
```python
def _generate_visualizations(
    self,
    input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    output_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    task_dir: Path,
    timestamp: str,
    result: OperationResult
) -> None
```
- **Parameters:**
    - `input_data`: Original data.
    - `output_data`: Split data.
    - `task_dir`: Output directory.
    - `timestamp`: Timestamp string.
    - `result`: OperationResult to update.

##### _validate_parameters
```python
def _validate_parameters(
    self,
    df: pd.DataFrame
) -> bool
```
- **Parameters:**
    - `df`: DataFrame to validate.
- **Returns:** True if valid, False otherwise.

##### _set_common_operation_parameters
```python
def _set_common_operation_parameters(self, **kwargs)
```
- **Parameters:**
    - `**kwargs`: Dynamic parameter overrides.

##### _get_cache / _save_cache / _generate_data_hash
- Internal methods for caching and data fingerprinting.

---

## Dependency Resolution and Completion Validation

- **Dependency Resolution:** The operation expects a `DataSource` object, which abstracts the loading of input data. The `load_data_operation` utility is used to fetch the dataset, optionally by name.
- **Completion Validation:** The operation validates that all fields in `field_groups` exist in the input DataFrame and that the ID field (if required) is present. If validation fails, the operation aborts and returns an error result.

---

## Usage Examples

### Basic Usage
```python
from pamola_core.transformations.splitting.split_fields_op import SplitFieldsOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Define field groups for splitting
groups = {
    "demographics": ["age", "gender", "country"],
    "financial": ["income", "expenses"]
}

# Create the operation
split_op = SplitFieldsOperation(
    id_field="user_id",
    field_groups=groups,
    output_format="csv"
)

data_source = DataSource("/path/to/input.csv")
task_dir = Path("/tmp/task1")
reporter = None  # Replace with your reporter if needed

# Execute the operation
result = split_op.execute(data_source, task_dir, reporter)

# Access output artifacts
for artifact in result.artifacts:
    print(artifact["path"])
```

### Handling Failed Dependencies
```python
try:
    result = split_op.execute(data_source, task_dir, reporter)
    if result.status != "SUCCESS":
        print(f"Operation failed: {result.error_message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Continue-on-Error Mode with Logging
```python
# Example: Continue processing even if some fields are missing
split_op = SplitFieldsOperation(
    id_field="user_id",
    field_groups=groups
)
try:
    result = split_op.execute(data_source, task_dir, reporter)
except Exception as e:
    # Log and continue
    logger.warning(f"Split failed, continuing: {e}")
```

### Integration with BaseTask
```python
# In a pipeline, use SplitFieldsOperation as a step in a BaseTask
task = BaseTask(
    operation=split_op,
    ... # other task config
)
task.run()
```

---

## Error Handling and Exception Hierarchy

- All errors are reported via the `OperationResult` object, with status set to `ERROR` and a descriptive message.
- The logger records detailed error traces for debugging.
- No custom exception classes are defined in this module; standard exceptions are used.

---

## Configuration Requirements

- `field_groups` must be a non-empty dictionary mapping group names to lists of valid DataFrame columns.
- `id_field` (if used) must exist in the input data.
- `output_format` must be one of the supported formats ("csv", "json").
- The `task_dir` must be a valid, writable directory.

---

## Security Considerations and Best Practices

- **Output Path Security:** All output is written to subdirectories of the provided `task_dir`. Avoid using untrusted or user-supplied paths for `task_dir`.
- **Data Leakage:** When splitting sensitive data, ensure that field groups do not inadvertently combine identifying and sensitive fields.
- **Encryption:** Use the `use_encryption` and `encryption_key` parameters to encrypt outputs if required.

### Example: Security Failure and Handling
```python
# Security risk: Using an untrusted output directory
untrusted_dir = Path("/tmp/../../etc/")
try:
    split_op.execute(data_source, untrusted_dir, reporter)
except Exception as e:
    print(f"Security error: {e}")
# Always validate and sanitize output paths!
```

**Risks of Disabling Path Security:**
- Writing to arbitrary locations can lead to data leaks or system compromise. Always restrict output to controlled directories.

---

## Internal vs. External Dependencies

- **Internal Dependencies:** Use logical task IDs and pipeline-managed data sources for reproducibility and traceability.
- **External (Absolute Path) Dependencies:** Only use absolute paths for data not produced by the pipeline. Document and audit such dependencies for security and reproducibility.

---

## Best Practices

1. **Use Task IDs for Internal Dependencies:** For data flows within your project, use task IDs as dependencies to maintain logical connections.
2. **Use Absolute Paths Judiciously:** Only use absolute paths for truly external data that isn't produced by your task pipeline.
3. **Validate Field Groups:** Always ensure that all fields in `field_groups` exist in your input data before running the operation.
4. **Enable Caching for Large Datasets:** Use the caching feature to avoid redundant computation on large or frequently processed datasets.
5. **Monitor Metrics and Visualizations:** Review generated metrics and visualizations to verify correct operation and data integrity.
6. **Handle Errors Gracefully:** Check the `OperationResult` status and error messages to handle failures programmatically.
7. **Secure Output Locations:** Never use untrusted or user-supplied paths for output directories.
