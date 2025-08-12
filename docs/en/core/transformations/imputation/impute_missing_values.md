# Impute Missing Values Operation Module

---

## Overview

The `impute_missing_values.py` module provides a robust, privacy-preserving operation for imputing missing or invalid values in tabular datasets within the PAMOLA Core framework. It supports a wide range of imputation strategies, per-field configuration, and is designed for scalable, auditable, and secure data processing pipelines. The operation integrates with PAMOLA's task and dependency management, metrics, and artifact reporting systems.

## Key Features

- **Flexible Imputation Strategies**: Supports mean, median, mode, constant, interpolation, and more, per field and data type (numeric, categorical, datetime, string).
- **Chunked and Parallel Processing**: Efficiently processes large datasets in batches or with parallelism.
- **Comprehensive Metrics**: Collects and saves detailed metrics for privacy and quality assessment.
- **Visualization Support**: Generates before/after distribution visualizations for audit and reporting.
- **Caching**: Deterministic cache keys for fast re-execution and reproducibility.
- **Encryption Support**: Optional encryption for output data and metrics.
- **Integration with PAMOLA Task Framework**: Designed for use in pipeline tasks, with full support for progress tracking, reporting, and artifact management.
- **Memory-Efficient**: Explicit cleanup and chunked processing to minimize memory usage.

## Dependencies

- **Standard Library**: `datetime`, `pathlib`, `typing`, `logging`, `gc`, `hashlib`, `time`
- **Third-Party**: `pandas`, `numpy`
- **Internal Modules**:
  - `pamola_core.utils.ops.op_config`, `op_data_source`, `op_data_writer`, `op_registry`, `op_result`
  - `pamola_core.utils.progress`
  - `pamola_core.transformations.base_transformation_op`
  - `pamola_core.transformations.commons.processing_utils`, `metric_utils`, `visualization_utils`

## Exception Classes

This module does **not** define custom exception classes directly. Instead, it raises standard Python exceptions (e.g., `ValueError`, `TypeError`) and exceptions from internal modules (such as `OperationResult`, `OperationStatus`, and cache/metrics/visualization utilities). For error handling best practices, see below.

**Example: Handling Operation Errors**

```python
try:
    result = operation.execute(data_source, task_dir, reporter)
    if result.status != OperationStatus.SUCCESS:
        print(f"Operation failed: {result.error_message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

- Errors are typically reported via the `OperationResult` object, with status and error_message fields.
- For cache, metrics, or output errors, see the respective internal modules for their exception types.

## Main Classes

### ImputeMissingValuesConfig

Configuration schema for the operation. Used internally for parameter validation.

### ImputeMissingValuesOperation

#### Constructor

```python
def __init__(
    field_strategies: Optional[Dict[str, Dict[str, Any]]] = None,
    invalid_values: Optional[Dict[str, List[Any]]] = None,
    output_format: str = "csv",
    name: str = "impute_missing_values_operation",
    description: str = "Impute missing or invalid values",
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
- `field_strategies`: Per-field imputation strategies and parameters
- `invalid_values`: Per-field list of values to treat as missing
- `output_format`: Output file format (`csv`, `json`, `parquet`)
- `name`, `description`: Operation metadata
- `field_name`, `mode`, `output_field_name`, `column_prefix`: Field selection and output control
- `batch_size`: Batch size for chunked processing
- `use_cache`, `use_dask`, `use_encryption`, `encryption_key`: Performance and security options

#### Key Attributes
- `field_strategies`, `invalid_values`, `version`
- `execution_time`, `include_timestamp`, `is_encryption_required`
- `use_cache`, `use_dask`, `parallel_processes`, `batch_size`

#### Public Methods

##### execute

```python
def execute(
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs
) -> OperationResult
```
- **Parameters:**
    - `data_source`: Data source object
    - `task_dir`: Output directory
    - `reporter`: Progress/artifact reporter
    - `progress_tracker`: Progress tracker
    - `**kwargs`: Additional options (see code for details)
- **Returns:** `OperationResult` (status, metrics, artifacts)
- **Raises:** Standard and internal exceptions (see error handling)

##### process_batch

```python
def process_batch(
    batch: pd.DataFrame
) -> pd.DataFrame
```
- **Parameters:**
    - `batch`: DataFrame batch to process
- **Returns:** Processed DataFrame

##### process_value

```python
def process_value(
    value,
    **params
)
```
- **Parameters:**
    - `value`: Single value to process
    - `**params`: Additional parameters
- **Raises:** `NotImplementedError` (not implemented in this operation)

##### (Internal) _get_and_validate_data, _generate_cache_key, _process_dataframe, _calculate_all_metrics, _save_metrics, _handle_visualizations, _save_output_data, _save_to_cache, _cleanup_memory

See code for full signatures and descriptions.

## Dependency Resolution and Completion Validation

- **DataSource**: The operation expects a `DataSource` object for input data. It validates data loading and reports errors via `OperationResult`.
- **Cache**: Uses a deterministic cache key based on operation parameters and data characteristics. Checks for cached results before processing.
- **Completion**: The operation reports status, metrics, and artifacts via the `OperationResult` and `reporter` objects, integrating with the PAMOLA pipeline's completion and reporting logic.

## Usage Examples

### Basic Usage in a Pipeline Task

```python
from pamola_core.transformations.imputation.impute_missing_values import ImputeMissingValuesOperation
from pamola_core.utils.ops.op_data_source import DataSource
from pathlib import Path

# Create operation
operation = ImputeMissingValuesOperation(
    field_strategies={
        "age": {"imputation_strategy": "mean"},
        "gender": {"imputation_strategy": "mode"}
    },
    invalid_values={"age": [-1, None], "gender": [""]},
    output_format="csv"
)

# Prepare data source and directories
source = DataSource(...)
task_dir = Path("DATA/processed/t_1I_impute")

# Execute operation
result = operation.execute(source, task_dir, reporter=None)
if result.status == "SUCCESS":
    print("Imputation completed!")
else:
    print(f"Error: {result.error_message}")
```

### Accessing Metrics and Artifacts

```python
# After execution
for metric, value in result.metrics.items():
    print(f"{metric}: {value}")
for artifact in result.artifacts:
    print(f"Artifact: {artifact['path']} ({artifact['description']})")
```

### Handling Absolute Path Dependencies

- Use absolute paths only for external data not produced by the pipeline.
- For internal dependencies, use task IDs and managed paths.

### Continue-on-Error Mode

- The operation itself does not implement continue-on-error for imputation, but upstream task and dependency managers can be configured to allow pipeline continuation on non-critical errors.

## Integration Notes

- **With BaseTask**: Instantiate and configure the operation as part of a `BaseTask` subclass. Use the `OperationResult` to propagate status, metrics, and artifacts to the pipeline controller.
- **With DependencyManager**: Use dependency outputs as input data sources for the operation. Validate dependencies before execution.

## Error Handling and Exception Hierarchy

- **Standard Exceptions**: Most errors are raised as standard Python exceptions (e.g., `ValueError`, `IOError`).
- **Internal Exceptions**: For operation execution, caching, and configuration, exceptions from `pamola_core.utils.ops` and related modules may be raised (e.g., `OpsError`, `ConfigError`, `CacheError`).
- **Best Practice**: Always catch and log exceptions at the pipeline or task level to ensure robust error reporting and recovery.

## Configuration Requirements

- The config object (or parameters) must specify at least one field and strategy in `field_strategies`.
- `invalid_values` should be provided for fields with custom missing value definitions.
- Output format, batch size, and security options should be set according to pipeline requirements.

## Security Considerations and Best Practices

- **Path Security**: All file and directory paths should be validated using PAMOLA's path security utilities. Never allow user-supplied or unvalidated paths.
- **Encryption**: Enable `use_encryption=True` and provide a secure key for sensitive outputs.
- **Absolute Path Risks**: Disabling path security or allowing arbitrary absolute paths can expose the system to directory traversal, data leakage, or overwrite attacks.

**Example: Security Failure and Handling**

```python
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError

try:
    validate_path_security("../../../etc/passwd", strict_mode=True)
except PathSecurityError as e:
    print(f"Path security error: {e}")
```

- **Risk**: If path security is disabled, malicious paths could be accessed or written, leading to data breaches or system compromise.

## Best Practices

1. **Use Task IDs for Internal Dependencies**: For data flows within your project, use task IDs as dependencies to maintain logical connections.
2. **Use Absolute Paths Judiciously**: Only use absolute paths for truly external data that isn't produced by your task pipeline.
3. **Validate All Configuration**: Ensure all constraint and path configurations are correct before running the operation.
4. **Handle Exceptions Gracefully**: Catch and log exceptions at the task or pipeline level; do not allow unhandled exceptions to propagate.
5. **Enable Encryption for Sensitive Outputs**: Use `use_encryption=True` and provide a secure key for sensitive data.
6. **Monitor Metrics and Artifacts**: Always review operation metrics and output artifacts for auditing and compliance.
7. **Restrict Path Access**: Never allow user-supplied or unvalidated paths for whitelist/blacklist files.