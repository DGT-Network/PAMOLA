# Remove Fields Operation Module

## Overview

The `remove_fields.py` module is part of the PAMOLA Core framework and provides a robust, configurable operation for removing one or more specified fields (columns) from a dataset. This operation is essential for privacy-preserving data processing, enabling users to drop sensitive or unnecessary fields either by explicit name or by pattern matching. The module is designed for integration into data pipelines, supporting large-scale, chunked, and parallel processing, with comprehensive metrics and artifact generation.

## Key Features

- **Explicit and Pattern-Based Field Removal**: Remove fields by name or using regular expressions.
- **Batch and Chunked Processing**: Efficiently handles large datasets with configurable batch sizes.
- **Parallel Processing Support**: Optionally process data in parallel for improved performance.
- **Comprehensive Metrics**: Collects and saves metrics for privacy impact and performance assessment.
- **Visualization Generation**: Produces visual comparisons of field counts before and after processing.
- **Caching**: Supports operation result caching for efficiency.
- **Encryption Support**: Optionally encrypts output files for security.
- **Memory-Efficient**: Explicit cleanup and garbage collection for large data.
- **Integration Ready**: Designed for use with PAMOLA's operation framework, including progress tracking and reporting.

## Dependencies

### Standard Library
- `logging`
- `time`
- `re`
- `datetime`
- `pathlib`
- `typing`
- `hashlib`
- `gc`

### Internal Modules
- `pamola_core.utils.ops.op_config`
- `pamola_core.utils.ops.op_data_source`
- `pamola_core.utils.ops.op_data_writer`
- `pamola_core.utils.ops.op_registry`
- `pamola_core.utils.ops.op_result`
- `pamola_core.utils.progress`
- `pamola_core.transformations.base_transformation_op`
- `pamola_core.transformations.commons.processing_utils`
- `pamola_core.transformations.commons.metric_utils`
- `pamola_core.transformations.commons.visualization_utils`
- `pamola_core.utils.ops.op_cache`

## Exception Classes

This module does not define custom exception classes. Instead, it raises standard Python exceptions (e.g., `ValueError`, `NotImplementedError`) and propagates exceptions from internal modules. All errors are logged, and critical errors are wrapped in `OperationResult` objects with status `ERROR`.

### Example: Handling a Processing Error
```python
try:
    processed_df = remove_fields_op._process_dataframe(df, progress_tracker)
except Exception as e:
    # Handle processing error
    logger.error(f"Processing error: {str(e)}")
    # Optionally, return an OperationResult with status ERROR
```
**When raised:** During batch or chunked processing, if an error occurs in the data transformation logic.

### Example: Handling NotImplementedError
```python
try:
    remove_fields_op.process_value(value)
except NotImplementedError as e:
    # This method is not implemented in RemoveFieldsOperation
    print("Not implemented:", e)
```
**When raised:** If `process_value` is called, as this method is not implemented for this operation.

## Main Classes

### RemoveFieldsConfig
Configuration schema for the Remove Fields operation.

#### Key Attributes
- `schema`: JSON schema for validating configuration parameters.

### RemoveFieldsOperation
Implements the field removal operation.

#### Constructor
```python
def __init__(
    self,
    fields_to_remove: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    output_format: str = "csv",
    name: str = "remove_fields_operation",
    description: str = "Remove fields from dataset",
    field_name: str = "",
    mode: str = "REPLACE",
    output_field_name: Optional[str] = None,
    column_prefix: str = "_",
    batch_size: int = 10000,
    use_cache: bool = True,
    use_dask: bool = False,
    use_encryption: bool = False,
    encryption_key: Optional[Union[str, Path]] = None
)
```
**Parameters:**
- `fields_to_remove`: List of field names to remove.
- `pattern`: Regex pattern for field selection.
- `output_format`: Output file format (`csv`, `json`, `parquet`).
- `name`: Operation name.
- `description`: Operation description.
- `field_name`: Field name to transform (not used in this op).
- `mode`: "REPLACE" or "ENRICH" (not used in this op).
- `output_field_name`: Name for output field if enriching (not used).
- `column_prefix`: Prefix for new columns (not used).
- `batch_size`: Batch size for processing.
- `use_cache`: Enable/disable caching.
- `use_dask`: Enable/disable Dask for distributed processing.
- `use_encryption`: Enable/disable output encryption.
- `encryption_key`: Key or path for encryption.

#### Key Attributes
- `fields_to_remove`: List of fields to remove.
- `pattern`: Regex pattern for field selection.
- `version`: Operation version.
- `execution_time`: Time taken for execution.
- `include_timestamp`: Whether to include timestamp in outputs.
- `is_encryption_required`: Whether encryption is required.

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
    - `data_source`: Source of data for the operation.
    - `task_dir`: Directory for task artifacts.
    - `reporter`: Reporter for progress and artifacts.
    - `progress_tracker`: Progress tracker.
    - `**kwargs`: Additional options (see code for details).
- **Returns:** `OperationResult` with status and artifacts.
- **Raises:** Logs and returns errors in `OperationResult`.

##### process_batch
```python
def process_batch(
    self,
    batch: pd.DataFrame
) -> pd.DataFrame
```
- **Parameters:**
    - `batch`: DataFrame batch to process.
- **Returns:** Processed DataFrame with specified fields removed.
- **Raises:** None (errors are propagated).

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
- **Returns:** Not implemented.
- **Raises:** `NotImplementedError`

##### _prepare_directories
```python
def _prepare_directories(
    self,
    task_dir: Path
) -> Dict[str, Path]
```
- **Parameters:**
    - `task_dir`: Root task directory.
- **Returns:** Dictionary of prepared directories.

##### _check_cache
```python
def _check_cache(
    self,
    data_source: DataSource,
    task_dir: Path,
    dataset_name: str = "main"
) -> Optional[OperationResult]
```
- **Parameters:**
    - `data_source`: Data source.
    - `task_dir`: Task directory.
    - `dataset_name`: Dataset name.
- **Returns:** Cached result or None.

##### _get_and_validate_data
```python
def _get_and_validate_data(
    self,
    data_source: DataSource,
    dataset_name: str = "main"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]
```
- **Parameters:**
    - `data_source`: Data source.
    - `dataset_name`: Dataset name.
- **Returns:** Tuple of DataFrame and error message.

##### _process_dataframe
```python
def _process_dataframe(
    self,
    df: pd.DataFrame,
    progress_tracker: Optional[HierarchicalProgressTracker]
) -> pd.DataFrame
```
- **Parameters:**
    - `df`: DataFrame to process.
    - `progress_tracker`: Progress tracker.
- **Returns:** Processed DataFrame.

##### _calculate_all_metrics
```python
def _calculate_all_metrics(
    self,
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame
) -> Dict[str, Any]
```
- **Parameters:**
    - `original_df`: Original DataFrame.
    - `processed_df`: Processed DataFrame.
- **Returns:** Dictionary of metrics.

##### _cleanup_memory
```python
def _cleanup_memory(
    self,
    original_df: Optional[pd.DataFrame],
    processed_df: Optional[pd.DataFrame]
) -> None
```
- **Parameters:**
    - `original_df`: Original DataFrame.
    - `processed_df`: Processed DataFrame.
- **Returns:** None

## Dependency Resolution and Completion Validation

The operation checks for required fields in the input DataFrame. If any specified fields to remove are missing, the operation returns an error message and does not proceed. Caching logic ensures that repeated operations on the same data/configuration use cached results for efficiency.

## Usage Examples

### Basic Usage: Remove Fields by Name
```python
from pamola_core.transformations.field_ops.remove_fields import RemoveFieldsOperation

# Create the operation to remove 'ssn' and 'email' fields
remove_fields_op = RemoveFieldsOperation(fields_to_remove=['ssn', 'email'])

# Process a DataFrame
processed_df = remove_fields_op.process_batch(input_df)
```

### Remove Fields by Pattern
```python
# Remove all fields containing 'temp' in their name
remove_fields_op = RemoveFieldsOperation(pattern='temp')
processed_df = remove_fields_op.process_batch(input_df)
```

### Using in a Pipeline with Progress Tracking
```python
# Assume data_source, task_dir, reporter, and progress_tracker are defined
result = remove_fields_op.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    progress_tracker=progress_tracker
)
# Access metrics
print(result.metrics)
```

### Handling Errors
```python
try:
    result = remove_fields_op.execute(data_source, task_dir, reporter)
    if result.status == 'ERROR':
        print("Operation failed:", result.error_message)
except Exception as e:
    print("Unexpected error:", e)
```

### Continue-on-Error Mode
```python
# In execute(), pass continue_on_error=True to skip errors and continue
result = remove_fields_op.execute(
    data_source=data_source,
    task_dir=task_dir,
    reporter=reporter,
    continue_on_error=True
)
```

## Integration Notes

- Designed for use with the PAMOLA operation framework and compatible with `BaseTask` and pipeline orchestration.
- Artifacts (metrics, outputs, visualizations) are saved to the task directory and reported via the provided reporter.
- Caching and encryption are handled transparently if enabled.

## Error Handling and Exception Hierarchy

- **Standard Exceptions**: Errors in processing, validation, or file operations are logged and returned in the `OperationResult`.
- **NotImplementedError**: Raised if `process_value` is called.
- **Validation Errors**: If required fields are missing, an error message is returned and processing is halted.

## Configuration Requirements

- The config object must specify either `fields_to_remove` (list of strings) or `pattern` (regex string).
- Optional parameters include output format, batch size, caching, encryption, and Dask usage.

## Security Considerations and Best Practices

- **Encryption**: Enable `use_encryption` and provide a valid `encryption_key` to secure output files.
- **Path Security**: Avoid using absolute paths for internal dependencies unless necessary. Absolute paths may expose sensitive locations or allow data leakage.

### Example: Security Failure and Handling
```python
# Security risk: using an absolute path for output
remove_fields_op = RemoveFieldsOperation()
try:
    remove_fields_op.execute(
        data_source=data_source,
        task_dir=Path('/tmp/unsafe_output'),  # Risk: world-writable directory
        reporter=reporter
    )
except Exception as e:
    logger.error("Security error: %s", e)
```
**Risk:** Output may be written to an insecure location. **Mitigation:** Always use controlled, project-specific directories for outputs.

## Internal vs. External Dependencies

- **Internal**: Use logical task IDs and project-relative paths for dependencies within the pipeline.
- **External**: Use absolute paths only for data not produced by the pipeline, and validate their security.

## Best Practices

1. **Use Field Names for Internal Data**: Prefer specifying fields to remove by name for clarity and maintainability.
2. **Use Patterns for Bulk Removal**: Use regex patterns for removing groups of related fields.
3. **Enable Caching for Large Pipelines**: Use caching to avoid redundant computation.
4. **Encrypt Sensitive Outputs**: Always enable encryption for outputs containing sensitive data.
5. **Validate Input Data**: Ensure all fields to be removed exist in the input data to avoid errors.
6. **Monitor Metrics**: Review generated metrics and visualizations to assess privacy impact.
7. **Clean Up Resources**: Let the operation handle memory cleanup, especially for large datasets.
8. **Integrate with Progress Tracking**: Use the provided progress tracker for better pipeline observability.
