# PAMOLA Core: Merge Datasets Operation (`merge_datasets_op.py`) Module

## Overview

The `merge_datasets_op.py` module is part of the PAMOLA Core framework, providing a robust, privacy-preserving operation for merging datasets using various strategies. It is designed for use in data processing pipelines where secure, efficient, and auditable merging of tabular data is required. The operation supports in-place DataFrame modification, comprehensive metrics collection, and advanced visualization, all while maintaining data utility and privacy.

## Key Features

- **Multiple Merge Strategies**: Supports binning, rounding, and range-based merging.
- **Flexible Join Types**: Inner, left, right, and outer joins.
- **Relationship Detection**: Auto-detects one-to-one and one-to-many relationships.
- **Null Handling**: Configurable strategies for null values (preserve, exclude, error).
- **Chunked Processing**: Handles large datasets efficiently.
- **Comprehensive Metrics**: Collects detailed metrics for privacy and utility assessment.
- **Visualization**: Generates visualizations for overlap, size comparison, and join distribution.
- **Caching and Encryption**: Supports result caching and output encryption.
- **Memory Management**: Explicit cleanup for large datasets.

## Dependencies

### Standard Library
- `logging`
- `time`
- `datetime`
- `pathlib`
- `gc`

### Internal Modules
- `pamola_core.common.enum.relationship_type.RelationshipType`
- `pamola_core.transformations.commons.processing_utils.merge_dataframes`
- `pamola_core.transformations.commons.visualization_utils.sample_large_dataset`
- `pamola_core.transformations.commons.merging_utils` (visualization functions)
- `pamola_core.transformations.base_transformation_op.TransformationOperation`
- `pamola_core.utils.ops.op_cache.OperationCache`
- `pamola_core.utils.ops.op_config.OperationConfig`
- `pamola_core.utils.ops.op_data_source.DataSource`
- `pamola_core.utils.ops.op_data_writer.DataWriter`
- `pamola_core.utils.ops.op_registry.register`
- `pamola_core.utils.ops.op_result.OperationResult, OperationStatus`
- `pamola_core.utils.progress.HierarchicalProgressTracker`
- `pamola_core.utils.io.load_data_operation`
- `pamola_core.common.constants.Constants`

## Exception Classes

This module does not define custom exception classes, but raises standard exceptions (e.g., `ValueError`) for invalid configuration or unsupported relationships. Example handling:

```python
try:
    operation = MergeDatasetsOperation(...)
    result = operation.execute(...)
except ValueError as e:
    # Handle configuration or relationship errors
    print(f"Configuration error: {e}")
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

**When exceptions are raised:**
- `ValueError`: Raised for invalid relationship types, missing keys, or unsupported merge relationships (e.g., many-to-many).
- `Exception`: Used for unexpected errors during execution, metrics calculation, or output saving.

## Main Classes

### `MergeDatasetsOperationConfig`
Configuration schema for the merge operation.

#### Key Attributes
- `schema`: JSON schema for validating operation configuration.

### `MergeDatasetsOperation`
Implements the main logic for merging datasets.

#### Constructor
```python
def __init__(
    self,
    name: str = "merge_datasets_operation",
    description: str = "Merge datasets by key field",
    left_dataset_name: str = "main",
    right_dataset_name: str = None,
    right_dataset_path: Optional[Path] = None,
    left_key: str = None,
    right_key: Optional[str] = None,
    join_type: str = "left",
    relationship_type: str = "auto",
    suffixes: Tuple[str, str] = ("_x", "_y"),
    output_format: str = "csv",
    use_cache: bool = True,
    use_encryption: bool = False,
    encryption_key: Optional[Union[str, Path]] = None,
    use_dask: bool = False,
)
```
**Parameters:**
- `name`: Name of the operation.
- `description`: Description of the operation.
- `left_dataset_name`: Name of the left (main) dataset.
- `right_dataset_name`: Name of the right (lookup) dataset.
- `right_dataset_path`: Path to the right dataset (if not named).
- `left_key`: Key field in the left dataset.
- `right_key`: Key field in the right dataset.
- `join_type`: Join strategy ("inner", "left", "right", "outer").
- `relationship_type`: Relationship type ("auto", "one-to-one", "one-to-many").
- `suffixes`: Suffixes for overlapping columns.
- `output_format`: Output format ("csv", "json", "parquet").
- `use_cache`: Enable/disable caching.
- `use_encryption`: Enable/disable output encryption.
- `encryption_key`: Key for encryption.
- `use_dask`: Use Dask for distributed computation.

#### Key Attributes
- `config`: Operation configuration object.
- `right_df`: Cached right DataFrame.
- `operation_cache`: Cache handler.

#### Public Methods

##### `execute`
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
- `data_source`: Source of data for the operation.
- `task_dir`: Directory for task artifacts.
- `reporter`: Progress/artifact reporter.
- `progress_tracker`: Progress tracker.
- `**kwargs`: Additional options (e.g., `force_recalculation`, `encrypt_output`).

**Returns:**
- `OperationResult`: Result object with status, metrics, and artifacts.

**Raises:**
- `ValueError`: For invalid configuration or relationships.
- `Exception`: For unexpected errors.

##### `process_batch`
```python
def process_batch(self, batch_df: pd.DataFrame, **kwargs) -> pd.DataFrame
```
*Deprecated. No longer used for merging.*

##### `_detect_relationship_type_auto`
```python
def _detect_relationship_type_auto(
    self,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
) -> str
```
**Parameters:**
- `left_df`: Left DataFrame.
- `right_df`: Right DataFrame.
- `left_key`: Join key in left DataFrame.
- `right_key`: Join key in right DataFrame.

**Returns:**
- `str`: Detected relationship type.

**Raises:**
- `ValueError`: If relationship is not one-to-one or one-to-many.

##### `_validate_relationship`
```python
def _validate_relationship(self, left_df: pd.DataFrame, right_df: pd.DataFrame) -> None
```
**Raises:**
- `ValueError`: If relationship does not match configuration.

##### `_collect_metrics`
```python
def _collect_metrics(
    self,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> dict
```
**Returns:**
- `dict`: Metrics dictionary.

##### `_generate_visualizations`
```python
def _generate_visualizations(
    self,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    task_dir: Path,
    result: OperationResult,
    reporter: Any = None,
) -> dict
```
**Returns:**
- `dict`: Mapping of visualization types to file paths.

##### `_save_output_data`
```python
def _save_output_data(
    self,
    result_df: pd.DataFrame,
    task_dir: Path,
    include_timestamp_in_filenames: bool,
    is_encryption_required: bool,
    writer: DataWriter,
    result: OperationResult,
    reporter: Any,
    progress_tracker: Optional[HierarchicalProgressTracker],
    **kwargs,
) -> None
```

##### `_prepare_directories`
```python
def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]
```

##### `_cleanup_memory`
```python
def _cleanup_memory(
    self,
    processed_df: Optional[pd.DataFrame] = None,
    left_df: Optional[pd.DataFrame] = None,
    right_df: Optional[pd.DataFrame] = None,
) -> None
```

##### `_get_dataset`
```python
def _get_dataset(self, source: Any, dataset_name_or_path: Optional[str]) -> Optional[pd.DataFrame]
```

##### `_validate_input_params`
```python
def _validate_input_params(
    self,
    relationship_type: str,
    left_key: Optional[str],
    left_dataset_name: Optional[str],
    right_dataset_name: Optional[str],
    right_dataset_path: Optional[Path],
) -> None
```

##### `_check_cache`
```python
def _check_cache(self, df: pd.DataFrame) -> Optional[OperationResult]
```

##### `_save_to_cache`
```python
def _save_to_cache(
    self,
    original_data: Union[pd.Series, pd.DataFrame],
    transformed_data: Union[pd.Series, pd.DataFrame],
    task_dir: Path,
    metrics: Dict[str, Any] = None,
) -> bool
```

## Dependency Resolution and Completion Validation

- **Dependency Resolution**: The operation can load datasets by name (internal) or by absolute path (external). Internal dependencies are resolved via the pipeline's data source, while external dependencies are loaded from disk.
- **Completion Validation**: The operation validates that all required keys and datasets are present and that the relationship type is supported. If not, it raises a `ValueError`.

## Usage Examples

### Basic Merge Operation
```python
from pamola_core.transformations.merging.merge_datasets_op import MergeDatasetsOperation

# Create the operation
merge_op = MergeDatasetsOperation(
    left_dataset_name="main",
    right_dataset_name="lookup",
    left_key="id",
    right_key="ref_id",
    join_type="left",
    relationship_type="auto"
)

# Execute the operation
result = merge_op.execute(
    data_source=my_data_source,  # DataSource instance
    task_dir=Path("/tmp/task_dir"),
    reporter=my_reporter
)

# Access metrics
print(result.metrics)
```

### Handling Failed Dependencies
```python
try:
    merge_op = MergeDatasetsOperation(...)
    result = merge_op.execute(...)
except ValueError as e:
    # Handle missing keys or unsupported relationships
    print(f"Dependency error: {e}")
```

### Continue-on-Error Mode with Logging
```python
try:
    result = merge_op.execute(..., continue_on_error=True)
except Exception as e:
    logger.warning(f"Operation failed but continuing: {e}")
```

### Integration with BaseTask
```python
class MyTask(BaseTask):
    def run(self):
        merge_op = MergeDatasetsOperation(...)
        result = merge_op.execute(...)
        # Use result.artifacts, result.metrics, etc.
```

## Error Handling and Exception Hierarchy

- **ValueError**: Raised for invalid configuration, missing keys, or unsupported relationships.
- **Exception**: Used for unexpected errors during execution, metrics, or output.

## Configuration Requirements

- `left_dataset_name` (str): Name of the left dataset (required).
- `right_dataset_name` (str or None): Name of the right dataset (required if `right_dataset_path` is not set).
- `right_dataset_path` (str or None): Path to the right dataset (required if `right_dataset_name` is not set).
- `left_key` (str): Key column in the left dataset (required).
- `right_key` (str or None): Key column in the right dataset (defaults to `left_key`).
- `join_type` (str): Join type ("inner", "left", "right", "outer").
- `relationship_type` (str): "auto", "one-to-one", or "one-to-many".

## Security Considerations and Best Practices

- **Encryption**: Enable `use_encryption` and provide a secure `encryption_key` to protect output data.
- **Path Security**: Avoid using absolute paths for dependencies unless necessary. Absolute paths may expose sensitive data or break pipeline reproducibility.

### Security Failure Example
```python
# BAD: Using absolute path without encryption
merge_op = MergeDatasetsOperation(
    left_dataset_name="main",
    right_dataset_path="/tmp/external.csv",  # External, unencrypted
    left_key="id",
    right_key="ref_id"
)
# This may expose sensitive data if /tmp/external.csv is not secured.
```

### Secure Usage Example
```python
merge_op = MergeDatasetsOperation(
    left_dataset_name="main",
    right_dataset_path="/secure/external.csv",
    left_key="id",
    right_key="ref_id",
    use_encryption=True,
    encryption_key="my-strong-key"
)
```

**Risks of Disabling Path Security:**
- Data leakage from unprotected files.
- Inconsistent pipeline results if external files change.
- Reduced auditability and traceability.

## Internal vs. External Dependencies

- **Internal**: Use dataset names managed by the pipeline for reproducibility and auditability.
- **External**: Use absolute paths only for data not produced by the pipeline, and always secure with encryption if sensitive.

## Best Practices

1. **Use Dataset Names for Internal Dependencies**: Maintain logical connections and reproducibility.
2. **Use Absolute Paths Judiciously**: Only for truly external data.
3. **Enable Encryption for Sensitive Data**: Always encrypt outputs when handling private or regulated data.
4. **Validate Relationships**: Use `relationship_type="auto"` for automatic detection, but review logs for warnings.
5. **Monitor Metrics and Artifacts**: Use the provided metrics and visualizations to assess merge quality and privacy impact.
6. **Clean Up Memory**: For large datasets, ensure memory is released after processing.
