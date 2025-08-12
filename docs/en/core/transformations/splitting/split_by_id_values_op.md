# SplitByIDValuesOperation Module

**Module Path:** `pamola_core/transformations/splitting/split_by_id_values_op.py`

---

## Overview

The `split_by_id_values_op` module provides the `SplitByIDValuesOperation` class, a flexible and robust transformation operation for splitting datasets by ID values or partitioning them into multiple subsets. This module is a core component of the PAMOLA Core framework, supporting advanced data processing pipelines with configurable partitioning, output formats, metrics collection, and visualization.

This operation is designed for use in privacy, anonymization, and data engineering workflows, where splitting data by identifiers or partitioning for downstream tasks is required.

---

## Key Features

- **Flexible Splitting Modes:**
  - Split by explicit ID value groups.
  - Automatic partitioning by equal size, random assignment, or modulo of ID values.
- **Multiple Output Formats:**
  - Supports CSV and JSON output.
- **Metrics and Visualization:**
  - Collects detailed metrics and generates visualizations for split results.
- **Caching and Reproducibility:**
  - Built-in caching to avoid redundant computation.
- **Parallel Processing Support:**
  - Optional Dask integration and parallel output saving.
- **Comprehensive Logging and Progress Tracking:**
  - Stepwise progress updates and detailed logging.
- **Security and Encryption:**
  - Optional output encryption and secure handling of sensitive data.

---

## Dependencies

### Standard Library
- `datetime`, `hashlib`, `json`, `time`, `enum`, `pathlib`, `typing`

### Third-Party Libraries
- `numpy`, `pandas`, `matplotlib`, `seaborn`

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

This module does not define custom exception classes. Instead, it raises standard Python exceptions (e.g., `Exception`) and logs errors using the PAMOLA logging system. Errors are reported via the `OperationResult` object with status and error messages.

**Example: Handling Operation Errors**
```python
try:
    result = split_op.execute(data_source, task_dir, reporter)
    if result.status != OperationStatus.SUCCESS:
        # Handle operation failure
        print(f"Operation failed: {result.error_message}")
except Exception as e:
    # Handle unexpected errors
    print(f"Critical error: {e}")
```

**When errors are likely to be raised:**
- Invalid configuration (e.g., missing `id_field`, invalid partition method).
- Data loading failures (empty or missing DataFrame).
- File I/O or serialization errors during output or metrics saving.

---

## Main Classes

### SplitByIDValuesOperation

#### Constructor
```python
SplitByIDValuesOperation(
    name: str = "split_by_id_values_operation",
    description: str = "Split dataset by ID values",
    id_field: str = None,
    value_groups: Optional[Dict[str, List[Any]]] = None,
    number_of_partitions: int = 0,
    partition_method: str = PartitionMethod.EQUAL_SIZE.value,
    output_format: str = OutputFormat.CSV.value,
    **kwargs
)
```
**Parameters:**
- `name`: Name of the operation.
- `description`: Description of the operation.
- `id_field`: Field name used to identify records.
- `value_groups`: Mapping of group names to lists of ID values.
- `number_of_partitions`: Number of partitions for automatic splitting.
- `partition_method`: Partitioning strategy (`equal_size`, `random`, `modulo`).
- `output_format`: Output file format (`csv`, `json`).
- `**kwargs`: Additional configuration.

#### Key Attributes
- `id_field`: The column used for splitting/partitioning.
- `value_groups`: Dictionary of group names to ID lists.
- `number_of_partitions`: Number of partitions for automatic splitting.
- `partition_method`: Partitioning method.
- `output_format`: Output file format.
- `logger`: Logger instance for operation.
- `input_dataset`, `original_df`: Internal state for input tracking.

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
  - `data_source`: Data source for loading input data.
  - `task_dir`: Directory for outputs and logs.
  - `reporter`: Reporting/logging object.
  - `progress_tracker`: Optional progress tracker.
  - `**kwargs`: Override configuration.
- **Returns:** `OperationResult` summarizing execution.
- **Raises:** Standard exceptions on critical errors (logged and reported).

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
  - `**kwargs`: Additional options.
- **Returns:** Dictionary mapping subset names to DataFrames.

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
- **Returns:** Structured metrics dictionary.

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
  - `result`: OperationResult to record artifact.
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
  - `result_subsets`: Output data.
  - `task_dir`: Output directory.
  - `timestamp`: Timestamp string.
  - `result`: OperationResult to record artifact.

##### _generate_visualizations
```python
def _generate_visualizations(
    self,
    input_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    output_data: Dict[str, pd.DataFrame],
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
  - `result`: OperationResult to record artifact.

##### _validate_parameters
```python
def _validate_parameters(self, df: pd.DataFrame) -> bool
```
- **Parameters:**
  - `df`: Input DataFrame.
- **Returns:** `True` if parameters are valid, else `False`.

##### _set_common_operation_parameters
```python
def _set_common_operation_parameters(self, **kwargs)
```
- **Parameters:**
  - `**kwargs`: Override configuration.

##### _initialize_logger
```python
def _initialize_logger(self, task_dir: Path)
```
- **Parameters:**
  - `task_dir`: Output/log directory.

##### _save_cache, _get_cache, _get_cache_parameters, _generate_data_hash
- Internal methods for caching and data fingerprinting.

---

## Dependency Resolution and Completion Validation

- **Dependency Resolution:**
  - The operation loads data from a `DataSource` object, which abstracts the source of input data (e.g., file, database, pipeline output).
  - Output artifacts (files, metrics, visualizations) are saved to structured directories under `task_dir`.
  - Caching is used to avoid redundant computation if the same input and configuration are detected.

- **Completion Validation:**
  - The operation validates configuration and input data before processing.
  - If validation fails (e.g., missing `id_field`, invalid partition method), the operation aborts and returns an error result.
  - Output and metrics are only saved if processing completes successfully.

---

## Usage Examples

### Basic Usage: Splitting by ID Groups
```python
from pamola_core.transformations.splitting.split_by_id_values_op import SplitByIDValuesOperation

# Example value groups for splitting
value_groups = {
    "group_A": [1, 2, 3],
    "group_B": [4, 5, 6]
}

# Create the operation
split_op = SplitByIDValuesOperation(
    id_field="user_id",
    value_groups=value_groups,
    output_format="csv"
)

# Execute the operation
result = split_op.execute(data_source, task_dir, reporter)

# Access output artifacts
for artifact in result.artifacts:
    print(artifact.path)
```

### Partitioning by Modulo
```python
split_op = SplitByIDValuesOperation(
    id_field="record_id",
    number_of_partitions=3,
    partition_method="modulo",
    output_format="json"
)
result = split_op.execute(data_source, task_dir, reporter)
```

### Handling Failed Dependencies
```python
result = split_op.execute(data_source, task_dir, reporter)
if result.status != OperationStatus.SUCCESS:
    # Log and handle the error
    logger.error(f"Split failed: {result.error_message}")
```

### Continue-on-Error Mode
```python
try:
    result = split_op.execute(data_source, task_dir, reporter, continue_on_error=True)
except Exception as e:
    # Handle critical errors
    print(f"Critical error: {e}")
```

### Integration with BaseTask
```python
# In a pipeline task definition
class MyTask(BaseTask):
    def run(self):
        split_op = SplitByIDValuesOperation(id_field="id", number_of_partitions=2)
        result = split_op.execute(self.data_source, self.task_dir, self.reporter)
        if result.status != OperationStatus.SUCCESS:
            self.logger.error("Split failed")
```

---

## Error Handling and Exception Hierarchy

- All errors are logged and reported via the `OperationResult` object.
- Critical errors (e.g., file I/O, invalid configuration) are raised as standard exceptions and caught in the `execute` method.
- The operation does not define custom exception classes but uses structured error reporting.

---

## Configuration Requirements

- **id_field**: Must be specified and present in the input DataFrame.
- **value_groups**: Optional; if provided, must map group names to valid ID values.
- **number_of_partitions**: Required if `value_groups` is not provided; must be a positive integer.
- **partition_method**: Must be one of `equal_size`, `random`, or `modulo`.
- **output_format**: Must be `csv` or `json`.

---

## Security Considerations and Best Practices

- **Output Encryption:**
  - Enable `use_encryption` and provide a valid `encryption_key` to encrypt output files.
- **Path Security:**
  - Avoid using absolute paths for outputs unless necessary. All outputs should be saved under the provided `task_dir`.
- **Risks of Disabling Path Security:**
  - Writing to arbitrary absolute paths can lead to data leaks or overwriting critical files.

**Example: Security Failure and Handling**
```python
# BAD: Writing output to an absolute path (risk of data leak)
output_path = Path("/tmp/critical_data.csv")
df.to_csv(output_path)  # May overwrite or expose sensitive data

# GOOD: Always use task_dir for outputs
output_path = task_dir / "output" / "partition_0.csv"
df.to_csv(output_path)
```

---

## Internal vs. External Dependencies

- **Internal Dependencies:**
  - Use logical task IDs and pipeline outputs for data flows within the project.
- **External (Absolute Path) Dependencies:**
  - Only use absolute paths for data not produced by the pipeline and ensure proper access controls.

---

## Best Practices

1. **Use Task IDs for Internal Dependencies:**
   - Maintain logical connections within your pipeline by referencing task outputs by ID.
2. **Use Absolute Paths Judiciously:**
   - Only for external data sources, and always validate access and security.
3. **Validate Configuration Early:**
   - Ensure all required parameters are set before execution.
4. **Enable Caching for Large Datasets:**
   - Use caching to avoid redundant computation on repeated runs.
5. **Monitor Logs and Metrics:**
   - Review logs and metrics for each run to ensure data integrity and correct partitioning.
6. **Secure Sensitive Outputs:**
   - Use encryption and avoid writing to insecure locations.
7. **Document Value Groups and Partitioning Logic:**
   - Clearly document how data is split for reproducibility and auditability.
