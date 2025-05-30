# PAMOLA Core: Aggregate Records Operation Module

## Overview

The `aggregate_records_op.py` module is a core component of the PAMOLA Core framework, designed to provide flexible, efficient, and privacy-preserving aggregation of tabular records. It enables users to group and aggregate data using both standard and custom aggregation functions, supporting large-scale data processing with robust metrics, visualization, and memory management. This operation is intended for use in privacy-preserving AI pipelines, data preprocessing, and analytics workflows.

## Key Features

- **Flexible Aggregation**: Supports standard aggregation functions (count, sum, mean, median, min, max, std, var, first, last, nunique) and custom aggregation functions per field.
- **Scalable Processing**: Efficiently processes data using either pandas or Dask, enabling handling of large datasets.
- **Robust Null Handling**: Ensures null values are managed appropriately during aggregation.
- **Comprehensive Metrics**: Collects detailed metrics on aggregation impact, group sizes, and field statistics.
- **Visualization Support**: Generates visualizations for group sizes, aggregation comparisons, and record counts.
- **Chunked Processing**: Supports chunked processing for memory efficiency.
- **Caching and Encryption**: Integrates with PAMOLA's caching and encryption utilities for secure, repeatable operations.
- **Standardized Interfaces**: Follows PAMOLA.CORE operation framework for input/output, progress tracking, and result reporting.

## Dependencies

### Standard Library
- `logging`
- `time`
- `datetime`
- `pathlib`
- `typing`

### Third-Party
- `pandas`

### Internal Modules
- `pamola_core.common.helpers.custom_aggregations_helper`
- `pamola_core.transformations.commons.processing_utils`
- `pamola_core.transformations.commons.visualization_utils`
- `pamola_core.transformations.commons.aggregation_utils`
- `pamola_core.transformations.base_transformation_op`
- `pamola_core.utils.ops.op_cache`
- `pamola_core.utils.ops.op_config`
- `pamola_core.utils.ops.op_data_source`
- `pamola_core.utils.ops.op_data_writer`
- `pamola_core.utils.ops.op_registry`
- `pamola_core.utils.ops.op_result`
- `pamola_core.utils.progress`
- `pamola_core.utils.io`

## Exception Classes

This module does not define custom exception classes. Instead, it raises standard Python exceptions (e.g., `ValueError`) for input validation and uses logging for warnings and errors. Example error handling is shown below.

### Example: Handling Input Validation Errors

```python
try:
    op = AggregateRecordsOperation(group_by_fields=[], aggregations={})
except ValueError as e:
    print(f"Configuration error: {e}")
```

**When Raised:**
- A `ValueError` is raised if required parameters (such as `group_by_fields`) are missing or invalid.

## Main Classes

### `AggregateRecordsOperationConfig`

Configuration schema for the aggregation operation.

#### Key Attributes
- `schema`: JSON schema for validating configuration parameters.

### `AggregateRecordsOperation`

#### Constructor

```python
def __init__(
    self,
    name: str = "aggregate_records_operation",
    description: str = "Group and aggregate records",
    group_by_fields: List[str] = None,
    aggregations: Dict[str, List[str]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
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
- `group_by_fields`: List of fields to group by.
- `aggregations`: Dictionary mapping field names to aggregation functions.
- `custom_aggregations`: Dictionary mapping field names to custom aggregation callables.
- `output_format`: Output file format (`csv`, `json`, or `parquet`).
- `use_cache`: Whether to enable result caching.
- `use_encryption`: Whether to encrypt output files.
- `encryption_key`: Key for encryption (if enabled).
- `use_dask`: Whether to use Dask for distributed computation.

#### Key Attributes
- `name`, `description`, `group_by_fields`, `aggregations`, `custom_aggregations`, `output_format`, `use_cache`, `use_encryption`, `encryption_key`, `use_dask`
- `config`: Instance of `AggregateRecordsOperationConfig`
- `operation_cache`: Instance of `OperationCache`

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
- `data_source`: Source of input data.
- `task_dir`: Directory for task artifacts.
- `reporter`: Reporter for progress and artifacts.
- `progress_tracker`: Progress tracker instance.
- `**kwargs`: Additional options (e.g., `force_recalculation`, `encrypt_output`).

**Returns:**
- `OperationResult`: Result object with status, metrics, and artifacts.

**Raises:**
- Returns an error result on failure (see error handling section).

##### `process_batch`

```python
def process_batch(self, batch_df: pd.DataFrame, **kwargs) -> pd.DataFrame
```

**Parameters:**
- `batch_df`: DataFrame to process.
- `**kwargs`: Additional options.

**Returns:**
- The input DataFrame (deprecated method).

##### `_collect_metrics`

```python
def _collect_metrics(self, df: pd.DataFrame, processed_df: pd.DataFrame) -> dict
```

**Parameters:**
- `df`: Original DataFrame.
- `processed_df`: Aggregated DataFrame.

**Returns:**
- Dictionary of metrics (record counts, field counts, execution time, etc.).

##### `_generate_visualizations`

```python
def _generate_visualizations(
    self,
    df: pd.DataFrame,
    processed_df: pd.DataFrame,
    task_dir: Path,
    result: OperationResult,
    reporter: Any = None,
) -> dict
```

**Parameters:**
- `df`: Original DataFrame.
- `processed_df`: Aggregated DataFrame.
- `task_dir`: Directory for outputs.
- `result`: Operation result object.
- `reporter`: Optional reporter.

**Returns:**
- Dictionary of visualization file paths.

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

**Parameters:**
- `result_df`: DataFrame to save.
- `task_dir`: Output directory.
- `include_timestamp_in_filenames`: Whether to add timestamp to filenames.
- `is_encryption_required`: Whether to encrypt output.
- `writer`: DataWriter instance.
- `result`: OperationResult instance.
- `reporter`: Reporter instance.
- `progress_tracker`: Progress tracker.
- `**kwargs`: Additional options.

##### `_validate_input_params`

```python
def _validate_input_params(
    self,
    group_by_fields: List[str],
    aggregations: Dict[str, List[str]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
) -> None
```

**Parameters:**
- `group_by_fields`: Fields to group by.
- `aggregations`: Aggregation functions.
- `custom_aggregations`: Custom aggregation functions.

**Raises:**
- `ValueError` if parameters are invalid.

## Dependency Resolution and Completion Validation

The operation validates dependencies by ensuring all required fields and aggregation functions are present and supported. It checks for the existence of group-by fields in the input DataFrame and validates aggregation function names against allowed sets. Caching logic ensures that repeated operations with the same parameters and data can reuse previous results, improving efficiency.

## Usage Examples

### Basic Aggregation

```python
from pamola_core.transformations.grouping.aggregate_records_op import AggregateRecordsOperation

# Define group-by fields and aggregations
op = AggregateRecordsOperation(
    group_by_fields=["user_id"],
    aggregations={"purchase_amount": ["sum", "mean"]},
    output_format="csv"
)

# Execute operation
result = op.execute(data_source, task_dir, reporter)

# Access metrics
print(result.metrics)
```

### Handling Input Validation Errors

```python
try:
    op = AggregateRecordsOperation(group_by_fields=[], aggregations={})
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Using Custom Aggregations

```python
def custom_mode(series):
    return series.mode().iloc[0] if not series.mode().empty else None

op = AggregateRecordsOperation(
    group_by_fields=["category"],
    aggregations={"value": ["sum"]},
    custom_aggregations={"value": [custom_mode]}
)
```

### Continue-on-Error with Logging

```python
try:
    result = op.execute(data_source, task_dir, reporter, continue_on_error=True)
except Exception as e:
    logger.error(f"Operation failed: {e}")
```

### Integration with BaseTask

```python
class MyTask(BaseTask):
    def run(self):
        op = AggregateRecordsOperation(...)
        result = op.execute(self.data_source, self.task_dir, self.reporter)
        # Use result.artifacts, result.metrics, etc.
```

## Error Handling and Exception Hierarchy

- **ValueError**: Raised for invalid configuration or unsupported aggregation functions.
- **General Exception**: All other errors are caught in `execute` and returned as error results with logging.

## Configuration Requirements

- `group_by_fields`: List of strings (required)
- `aggregations`: Dict mapping field names to list of aggregation function names (required)
- `custom_aggregations`: Dict mapping field names to list of custom functions (optional)
- `output_format`: One of `csv`, `json`, `parquet` (default: `csv`)
- `use_cache`, `use_encryption`, `encryption_key`, `use_dask`: Optional flags

## Security Considerations and Best Practices

- **Encryption**: Enable `use_encryption` and provide a secure `encryption_key` to protect sensitive output data.
- **Cache Security**: Ensure cache directories are secured and not accessible to unauthorized users.
- **Input Validation**: Always validate input parameters to prevent injection of unsupported aggregation functions.

### Example: Security Failure and Handling

```python
# Risk: Disabling encryption for sensitive data
op = AggregateRecordsOperation(
    group_by_fields=["user_id"],
    aggregations={"ssn": ["first"]},
    use_encryption=False  # Not recommended for sensitive fields
)

# Best practice: Enable encryption
op = AggregateRecordsOperation(
    group_by_fields=["user_id"],
    aggregations={"ssn": ["first"]},
    use_encryption=True,
    encryption_key="my-strong-key"
)
```

**Risks of Disabling Path Security:**
- Output files may be written to insecure locations.
- Sensitive data may be exposed if encryption is not enabled.

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical task IDs and data flows within the PAMOLA pipeline for reproducibility and traceability.
- **External Dependencies**: Use absolute paths only for data not produced by the pipeline. Ensure such paths are secure and access is controlled.

## Best Practices

1. **Use Logical Task IDs**: For internal data flows, always use task IDs to maintain clear dependencies.
2. **Validate All Parameters**: Ensure all group-by fields and aggregation functions are valid and supported.
3. **Enable Encryption for Sensitive Data**: Always enable encryption when processing or outputting sensitive information.
4. **Leverage Caching**: Use caching to speed up repeated operations, but ensure cache security.
5. **Handle Errors Gracefully**: Use try/except blocks and check `OperationResult.status` for error handling.
6. **Document Custom Aggregations**: Clearly document any custom aggregation functions used for maintainability.
7. **Limit Absolute Paths**: Only use absolute paths for external data, and document their use for auditability.
