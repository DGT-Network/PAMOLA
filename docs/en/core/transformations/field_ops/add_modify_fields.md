# Add or Modify Fields Operation Module

## Overview

The `add_modify_fields.py` module provides a robust **AddOrModifyFieldsOperation** for the PAMOLA Core framework, extensible operation for adding or modifying fields in tabular datasets within the PAMOLA Core privacy-preserving AI data processing framework. It enables users to enrich or transform data columns based on lookup tables, constants, or (future) conditional logic, while supporting large-scale, chunked, and parallel processing. The operation is designed for seamless integration into PAMOLA pipelines, with comprehensive caching, encryption, metrics, visualization, and artifact management.

## Key Features

- **Flexible Field Operations**: Add or modify fields using constants, lookup tables, or (planned) conditional logic.
- **Batch and Parallel Processing**: Efficiently processes large datasets in chunks or with parallel workers or Dask.
- **Comprehensive Metrics**: Collects and saves detailed metrics for privacy and data quality assessment.
- **Visualization Support**: Generates visualizations comparing original and processed data distributions.
- **Reporting**: including performance and data impact.
- **Caching**: Supports operation-level caching to avoid redundant computation.
- **Encryption**: Optional encryption for output artifacts.
- **Memory Efficiency**: Explicit memory cleanup for large data operations.
- **Standardized Interfaces**: Follows PAMOLA operation framework for configuration, progress tracking, and result reporting.
- **Progress Tracking**: Integrates with hierarchical progress trackers for pipeline orchestration.
- **Error Handling**: Gracefully handles and reports errors at every stage.

---

## Dependencies

### Standard Library
- `logging`, `time`, `json`, `datetime`, `pathlib`, `typing`, `hashlib`, `gc`, `os`, `threading`, `contextvars`

### Internal Modules
- `pamola_core.utils.ops.op_config.OperationConfig`
- `pamola_core.utils.ops.op_data_source.DataSource`
- `pamola_core.utils.ops.op_data_writer.DataWriter`
- `pamola_core.utils.ops.op_registry.register`
- `pamola_core.utils.ops.op_result.OperationResult, OperationStatus`
- `pamola_core.utils.progress.HierarchicalProgressTracker`
- `pamola_core.transformations.base_transformation_op.TransformationOperation`
- `pamola_core.transformations.commons.processing_utils`
- `pamola_core.transformations.commons.metric_utils`
- `pamola_core.transformations.commons.visualization_utils`
- `pamola_core.utils.ops.op_cache.operation_cache`
- `pamola_core.common.constants`
- `pamola_core.utils.io.load_settings_operation`
- `pamola_core.utils.io.load_data_operation`
- `pamola_core.utils.io_helpers.crypto_utils.get_encryption_mode`
---

## Exception Classes

This module does not define custom exception classes, but raises standard exceptions and `NotImplementedError` for unimplemented features. All errors are caught and reported in the `OperationResult`.

### NotImplementedError
- **Description**: Raised when a requested operation type (e.g., `add_conditional`, `modify_conditional`, `modify_expression`) is not yet implemented.
- **Example Usage**:
  ```python
  try:
      op = AddOrModifyFieldsOperation(...)
      op.process_batch(batch)
  except NotImplementedError as e:
      # Handle or log the missing feature
      print(f"Feature not implemented: {e}")
  ```
- **When Raised**: If a field operation type is not supported by the current implementation.

### Example: Handling Data/Processing Errors
```python
result = op.execute(ds, task_dir, reporter, progress_tracker)
if result.status.name == "ERROR":
    print(f"Error: {result.error_message}")
```
**When raised**: Any unexpected error during data loading, validation, processing, metrics, or output will be caught and reported in the result.

---

## Main Classes

### AddOrModifyFieldsConfig
Configuration schema for the operation. Used internally for parameter validation.
#### Key Attributes
- `schema`: JSON schema for validating configuration parameters.

### AddOrModifyFieldsOperation
Implements the add/modify fields operation.

#### Constructor
```python
def __init__(
    field_operations: Optional[Dict[str, Dict[str, Any]]] = None,
    lookup_tables: Optional[Dict[str, Union[Path, Dict[Any, Any]]]] = None,
    output_format: str = "csv",
    name: str = "add_modify_fields_operation",
    description: str = "Add or modify fields",
    field_name: str = "",
    mode: str = "REPLACE",
    output_field_name: Optional[str] = None,
    column_prefix: str = "_",
    visualization_theme: Optional[str] = None,
    visualization_backend: Optional[str] = None,
    visualization_strict: bool = False,
    visualization_timeout: int = 120,
    chunk_size: int = 10000,
    use_dask: bool = False,
    npartitions: int = 2,
    use_vectorization: bool = False,
    parallel_processes: int = 2,
    use_cache: bool = True,
    use_encryption: bool = False,
    encryption_key: Optional[Union[str, Path]] = None,
    encryption_mode: Optional[str] = None,
)
```
**Parameters:**
- `field_operations`: Dict of field operation configs.
- `lookup_tables`: Dict of lookup tables (as dict or file path).
- `output_format`: Output file format (`csv`, `json`, `parquet`).
- `name`: Operation name.
- `description`: Operation description.
- `field_name`: Name of the field to transform.
- `mode`: `REPLACE` (in-place) or `ENRICH` (add new field).
- `output_field_name`: Name for new field (if `ENRICH`).
- `column_prefix`: Prefix for new columns.
- `batch_size`: Batch size for chunked processing.
- `use_cache`: Enable/disable caching.
- `use_dask`: Use Dask for distributed processing.
- `use_encryption`: Encrypt output files.
- `encryption_key`: Key or path for encryption.
- `encryption_mode`: mode for encryption.
- `visualization_*`: Visualization settings.
- `chunk_size`, `npartitions`, `use_vectorization`, `parallel_processes`: Performance tuning.

#### Key Attributes
- `field_operations`, `lookup_tables`, `output_format`, `mode`, `visualization_theme`, `visualization_backend`, `visualization_strict`, `visualization_timeout`, `chunk_size`, `use_dask`, `npartitions`, `use_vectorization`, `parallel_processes`, `use_cache`, `use_encryption`, `encryption_key`, `encryption_mode`

#### Public Methods

##### execute
```python
def execute(
    self,
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs
) -> OperationResult
```
- **Parameters:**
    - `data_source`: Source of input data.
    - `task_dir`: Directory for task artifacts.
    - `reporter`: Progress and artifact reporter.
    - `progress_tracker`: Progress tracker.
    - `**kwargs`: Additional options (see code for details).
- **Returns:** `OperationResult` with status, metrics, and artifacts.
- **Raises:** Standard exceptions for data loading, processing, or output errors.

##### process_batch
```python
def process_batch(
    self,
    batch: pd.DataFrame
) -> pd.DataFrame
```
- **Parameters:**
    - `batch`: DataFrame batch to process.
- **Returns:** Processed DataFrame.
- **Raises:** `NotImplementedError` for unsupported operation types.

##### process_value
```python
def process_value(
    self,
    value,
    **params
)
```
- **Parameters:**
    - `value`: Value to process.
    - `**params`: Additional parameters.
- **Returns:** Processed value.
- **Raises:** `NotImplementedError` (not implemented).

##### create_add_modify_fields_operation
```python
def create_add_modify_fields_operation(**kwargs) -> AddOrModifyFieldsOperation
```
- **Parameters:**
    - `**kwargs`: Operation parameters.
- **Returns:** Configured `AddOrModifyFieldsOperation` instance.

#### Other Methods
- `_prepare_directories`, `_check_cache`, `_get_and_validate_data`, `_generate_cache_key`, `_get_operation_parameters`, `_get_cache_parameters`, `_generate_data_hash`, `_process_dataframe`, `_calculate_all_metrics`, `_collect_metrics`, `_save_metrics`, `_handle_visualizations`, `_generate_visualizations`, `_save_output_data`, `_save_to_cache`, `_cleanup_memory`

## Dependency Resolution and Completion Validation
- **Dependency Resolution**: The operation can use lookup tables provided as file paths or in-memory dictionaries. It validates the presence and format of these dependencies before processing.
- **Completion Validation**: The operation checks for cached results and validates input data before proceeding. If data is missing or invalid, it returns an error result.
- **Caching**: Uses a hash of operation parameters and data characteristics to avoid redundant computation. If a cache hit is found, results are loaded from cache.
---

## Usage Examples

### Basic Usage
```python
from pamola_core.transformations.field_ops.add_modify_fields import create_add_modify_fields_operation
from pamola_core.utils.ops.op_data_source import DataSource

# Example field operations and lookup tables
field_ops = {
    "new_field": {"operation_type": "add_constant", "constant_value": 42},
    "category": {"operation_type": "add_from_lookup", "lookup_table_name": "cat_map"}
}
lookup_tables = {"cat_map": {"A": "Alpha", "B": "Beta"}}

# Create operation
op = create_add_modify_fields_operation(
    field_operations=field_ops,
    lookup_tables=lookup_tables,
    output_format="csv"
)

# Prepare data source and directories
source = DataSource(...)
task_dir = Path("/tmp/task")
reporter = ...

# Execute operation
result = op.execute(
    data_source=source,
    task_dir=task_dir,
    reporter=reporter
)

# Access metrics
print(result.metrics)
```

### Handling NotImplementedError
```python
try:
    op = create_add_modify_fields_operation(
        field_operations={"f": {"operation_type": "add_conditional"}}
    )
    op.process_batch(batch)
except NotImplementedError as e:
    # Log or handle the missing feature
    print(f"Feature not implemented: {e}")
```

### Continue-on-Error Mode with Logging
```python
try:
    result = op.execute(
        data_source=source,
        task_dir=task_dir,
        reporter=reporter,
        continue_on_error=True
    )
except Exception as e:
    # Log error and continue pipeline
    logger.warning(f"Operation failed: {e}")
```

### Accessing Metrics and Artifacts
```python
for artifact in result.artifacts:
    print(f"Artifact: {artifact.description} at {artifact.path}")

for metric, value in result.metrics.items():
    print(f"Metric: {metric} = {value}")
```

---

## Integration Notes

- Designed for use with PAMOLA pipeline tasks and the `BaseTask` class.
- Artifacts, metrics, and outputs are reported via the provided `reporter` object.
- Progress can be tracked using `HierarchicalProgressTracker`.
- Integrates with progress tracking, reporting, and artifact management.
- Can be used as a standalone operation or as part of a larger workflow.
---

## Error Handling and Exception Hierarchy

- **Standard Exceptions**: Raised for data loading, processing, or output errors (e.g., `ValueError`, `IOError`).
- **NotImplementedError**: Raised for unimplemented operation types.
- **Error Reporting**: All errors are logged and, where possible, included in the `OperationResult`.

## Configuration Requirements

- The config object must specify at least `field_operations` and (if needed) `lookup_tables`.
- Visualization and encryption settings are optional but recommended for production.
- Output format, batch size, and other options can be customized.
- Example config:
  ```python
  config = {
      "field_operations": {...},
      "lookup_tables": {...},
      "output_format": "csv",
      "batch_size": 10000
  }
  ```

## Security Considerations and Best Practices

- **Encryption**: Enable `use_encryption=True` and provide a secure `encryption_key` to protect output files.
- **Path Security**: Avoid using absolute paths for lookup tables unless necessary. 
- **Risks of Disabling Path Security**: Using external/absolute paths can expose the pipeline to data leaks or tampering. Always validate external data sources.

- **Security Failure Example**:
 ```python
  # BAD: Using an untrusted absolute path
  lookup_tables = {"external": Path("/tmp/untrusted/lookup.json")}
  op = create_add_modify_fields_operation(lookup_tables=lookup_tables)
  # This may fail or leak data if the file is not secure
  ```
  **How it is handled:** The operation will raise an error if the file is missing or unreadable, and logs a warning.
- **Risks of Disabling Path Security**: Using external absolute paths can lead to data leakage, non-reproducible pipelines, and security vulnerabilities. Always validate and restrict external file usage.
```

#### Internal vs. External Dependencies
- **Internal**: Use task IDs and pipeline-managed data for dependencies.
- **External**: Use absolute paths only for data not produced by the pipeline. Mark and document such dependencies clearly.

---

## Best Practices

1. **Use Logical Task IDs for Internal Dependencies**: Reference lookup tables and data produced within your pipeline by logical names, not absolute paths.
2. **Use Absolute Paths Only for External Data**: Only use absolute paths for data not produced by your pipeline, and ensure proper access controls.
3. **Enable Encryption for Sensitive Outputs**: Always enable encryption when processing or outputting sensitive data.
4. **Handle NotImplementedError Gracefully**: Check for and handle unimplemented operation types in your pipeline logic.
5. **Validate Configurations**: Use the provided schema and config class to validate all operation parameters before execution.
6. **Monitor Metrics and Artifacts**: Review generated metrics and visualizations to assess privacy and data quality impacts.
7. **Clean Up Resources**: Ensure memory is cleaned up after large operations to avoid resource leaks.
8. **Leverage Caching**: Use caching to speed up repeated operations.
