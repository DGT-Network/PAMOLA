# PAMOLA Core: Base Transformation Operation Module

## Overview

The `base_transformation_op.py` module provides the foundational class for all data transformation operations within the PAMOLA Core framework. It defines a standardized interface, lifecycle, and utility set for implementing privacy-preserving, scalable, and auditable data transformations. This module is designed to be subclassed for custom transformation logic, ensuring consistency and best practices across the PAMOLA ecosystem.

---

## Key Features

- **Standardized Operation Lifecycle**: Built-in validation, execution, result handling, and artifact management.
- **Flexible Transformation Modes**: Supports both in-place (`REPLACE`) and enrichment (`ENRICH`) transformations.
- **Batch and Parallel Processing**: Configurable batch size, Dask-based distributed processing, and parallelization support.
- **Integrated Caching**: Optional operation-level caching for efficient repeated runs.
- **Output Encryption**: Optional encryption for sensitive output data.
- **Progress Tracking**: Hierarchical progress reporting for complex pipelines.
- **Artifact and Metrics Reporting**: Automatic generation and registration of metrics, visualizations, and output artifacts.
- **Extensible Hooks**: Easy subclassing for custom metrics, visualizations, and processing logic.

---

## Dependencies

### Standard Library
- `logging`, `time`, `datetime`, `pathlib`, `typing`, `hashlib`, `json`, `gc`

### Third-Party
- `pandas`

### Internal Modules
- `pamola_core.utils.io.load_data_operation`
- `pamola_core.utils.ops.op_cache.operation_cache`
- `pamola_core.utils.ops.op_data_source.DataSource`
- `pamola_core.utils.ops.op_data_writer.DataWriter`
- `pamola_core.utils.ops.op_base.BaseOperation`
- `pamola_core.utils.ops.op_result.OperationResult`, `OperationStatus`
- `pamola_core.utils.progress.HierarchicalProgressTracker`
- `pamola_core.transformations.commons.processing_utils`
- `pamola_core.transformations.commons.visualization_utils`

---

## Exception Classes

This module does not define custom exception classes directly, but it raises standard exceptions (e.g., `ValueError`, `TypeError`) in well-defined scenarios. Subclasses may introduce their own exceptions as needed.

### Example: Handling a Data Validation Error
```python
try:
    df = transformation_op._validate_and_get_dataframe(data_source, dataset_name)
except ValueError as e:
    # Handle missing field or data loading error
    logger.error(f"Validation failed: {e}")
```
**When Raised:**
- If the required field is missing from the DataFrame.
- If the data source fails to load.

---

## Main Class: `TransformationOperation`

### Constructor
```python
def __init__(
    self,
    field_name: str = "",
    name: str = "transformation_operation",
    mode: str = "REPLACE",
    output_field_name: Optional[str] = None,
    column_prefix: str = "_",
    description: str = "",
    batch_size: int = 10000,
    use_cache: bool = False,
    use_dask: bool = False,
    use_encryption: bool = False,
    encryption_key: Optional[Union[str, Path]] = None,
    output_format: str = "csv",
)
```
**Parameters:**
- `field_name`: Name of the field to transform.
- `name`: Operation name.
- `mode`: "REPLACE" (in-place) or "ENRICH" (add new field).
- `output_field_name`: Name for the new field (ENRICH mode).
- `column_prefix`: Prefix for new columns (ENRICH mode).
- `description`: Operation description.
- `batch_size`: Batch size for processing.
- `use_cache`: Enable/disable caching.
- `use_dask`: Enable/disable Dask parallelism.
- `use_encryption`: Enable/disable output encryption.
- `encryption_key`: Key or path for encryption.
- `output_format`: Output file format (csv, parquet, json).

### Key Attributes
- `field_name`, `mode`, `output_field_name`, `column_prefix`, `batch_size`, `use_cache`, `use_dask`, `use_encryption`, `encryption_key`, `output_format`, `version`, `parallel_processes`

### Public Methods

#### `execute`
```python
def execute(
    self,
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> OperationResult
```
**Parameters:**
- `data_source`: Source of input data.
- `task_dir`: Directory for task artifacts.
- `reporter`: Progress/artifact reporter.
- `progress_tracker`: Progress tracker instance.
- `**kwargs`: Additional options (e.g., `force_recalculation`, `encrypt_output`, `use_dask`, `generate_visualization`, `include_timestamp`, `save_output`, `parallel_processes`, `dataset_name`).

**Returns:**
- `OperationResult`: Encapsulates status, metrics, and artifacts.

**Raises:**
- `ValueError`, `Exception` (on data loading, validation, or processing errors).

#### `process_batch`
```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame
```
**Parameters:**
- `batch`: DataFrame batch to process.

**Returns:**
- Processed DataFrame batch.

**Raises:**
- `NotImplementedError` (must be implemented by subclasses).

#### `_validate_and_get_dataframe`
```python
def _validate_and_get_dataframe(self, data_source: DataSource, dataset_name: str) -> pd.DataFrame
```
**Parameters:**
- `data_source`: Data source.
- `dataset_name`: Name of the dataset.

**Returns:**
- Loaded and validated DataFrame.

**Raises:**
- `ValueError` (if data or field is missing).

#### `_prepare_output_fields`, `_report_operation_details`, `_process_dataframe`, `_calculate_all_metrics`, `_handle_visualizations`, `_generate_visualizations`, `_save_output_data`, `_generate_cache_key`, `_check_cache`, `_save_to_cache`, `_cleanup_memory`, `_generate_data_hash`, `_series_characteristics`, `_df_characteristics`, `_get_basic_parameters`, `_get_cache_parameters`, `_collect_metrics`

> See source code for full signatures and details. Most are intended for internal use or subclassing.

---

## Dependency Resolution and Completion Validation

- **Dependency Resolution**: The module expects data sources and dependencies to be managed via the `DataSource` abstraction. Caching and output artifact management are handled internally, with hooks for integration into larger pipelines.
- **Completion Validation**: The `OperationResult` object tracks status, metrics, and artifacts, allowing downstream tasks to validate completion and access outputs.

---

## Usage Examples

### Basic Transformation Operation
```python
from pamola_core.transformations.base_transformation_op import TransformationOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pathlib import Path

# Create a transformation operation (subclass must implement process_batch)
class MyCustomTransform(TransformationOperation):
    def process_batch(self, batch):
        # Example: Add 1 to a numeric column
        batch[self.field_name] = batch[self.field_name] + 1
        return batch

data_source = DataSource("/path/to/data.csv")
task_dir = Path("/tmp/task")
reporter = None  # Replace with actual reporter

op = MyCustomTransform(field_name="age", mode="REPLACE")
result = op.execute(data_source, task_dir, reporter)

# Access metrics and artifacts
print(result.metrics)
for artifact in result.artifacts:
    print(artifact.path)
```

### Handling Data Validation Errors
```python
try:
    op = MyCustomTransform(field_name="nonexistent_field")
    op.execute(data_source, task_dir, reporter)
except ValueError as e:
    print(f"Error: {e}")  # Field not found in DataFrame
```

### Using Caching and Parallel Processing
```python
op = MyCustomTransform(field_name="salary", use_cache=True, use_dask=True, batch_size=5000)
result = op.execute(data_source, task_dir, reporter, parallel_processes=4)
```

### Continue-on-Error Mode with Logging
```python
try:
    result = op.execute(data_source, task_dir, reporter, continue_on_error=True)
except Exception as e:
    logger.warning(f"Operation failed but pipeline continues: {e}")
```

---

## Integration Notes

- Designed for use within PAMOLA Core pipelines and with `BaseTask`-derived classes.
- Artifacts, metrics, and progress are reported via the provided `reporter` and `progress_tracker` interfaces.
- Subclassing is required for custom transformation logic (implement `process_batch`).

---

## Error Handling and Exception Hierarchy

- **ValueError**: Raised for missing fields, data loading failures, or invalid parameters.
- **TypeError**: Raised for invalid data types in hashing or processing.
- **NotImplementedError**: Raised if `process_batch` is not implemented in a subclass.
- **General Exception**: All other errors are logged and returned as `OperationResult(status=ERROR)`.

### Example: Handling a Security Failure
```python
try:
    op = MyCustomTransform(use_encryption=True)
    op.execute(data_source, task_dir, reporter)
except ValueError as e:
    print(f"Security error: {e}")  # Encryption key must be provided
```

---

## Configuration Requirements

- **Encryption**: If `use_encryption=True`, an `encryption_key` must be provided.
- **Output Format**: Supported formats are `csv`, `parquet`, and `json`.
- **Batch Size**: Adjust `batch_size` for large datasets to optimize memory and performance.

---

## Security Considerations and Best Practices

- **Always Provide an Encryption Key**: When enabling encryption, ensure a valid key or key path is supplied.
- **Risks of Disabling Path Security**: Disabling encryption or using insecure output directories may expose sensitive data. Always validate output paths and restrict access as needed.

### Example: Security Failure and Handling
```python
# Incorrect: Encryption enabled but no key provided
op = MyCustomTransform(use_encryption=True)
try:
    op.execute(data_source, task_dir, reporter)
except ValueError as e:
    print(f"Security error: {e}")
# Output: Security error: Encryption key must be provided when use_encryption is True
```

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical task IDs and DataSource objects for data produced within the pipeline.
- **External Dependencies**: Use absolute paths only for data not managed by the pipeline. Document and validate all external dependencies for reproducibility and security.

---

## Best Practices

1. **Subclass for Custom Logic**: Always implement `process_batch` in your transformation subclass.
2. **Use Caching for Efficiency**: Enable `use_cache` for expensive or repeated operations.
3. **Leverage Parallelism**: Use `use_dask` and adjust `parallel_processes` for large datasets.
4. **Secure Sensitive Outputs**: Enable `use_encryption` and provide a strong key for sensitive data.
5. **Report Artifacts and Metrics**: Use the `reporter` and `progress_tracker` for full pipeline observability.
6. **Validate All Inputs**: Always check that required fields exist in your data source.
7. **Use Task IDs for Internal Dependencies**: For data flows within your project, use task IDs as dependencies to maintain logical connections.
8. **Use Absolute Paths Judiciously**: Only use absolute paths for truly external data that isn't produced by your task pipeline.
