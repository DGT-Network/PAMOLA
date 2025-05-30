# Clean Invalid Values Operation Module

**Module Path:** `pamola_core.transformations.cleaning.clean_invalid_values`

---

## Overview

The `clean_invalid_values` module provides a robust, configurable operation for cleaning and nullifying invalid values in tabular datasets within the PAMOLA Core framework. It is designed to enforce data quality and compliance by applying field-level constraints, whitelists, blacklists, and null-replacement strategies. The operation is highly extensible, supports batch and parallel processing, and integrates with the broader PAMOLA pipeline for secure, auditable, and reproducible data transformations.

---

## Key Features

- **Constraint-based Cleaning:** Supports numeric, categorical, datetime, and string constraints (e.g., min/max, allowed/disallowed values, patterns).
- **Whitelist/Blacklist Enforcement:** Enforces value inclusion/exclusion from external files or dictionaries.
- **Null Replacement Strategies:** Flexible imputation (mean, median, mode, random sample, or custom value).
- **Batch and Parallel Processing:** Efficiently processes large datasets in chunks or parallel jobs.
- **Caching and Reproducibility:** Deterministic cache keys for operation results, with cache validation and storage.
- **Metrics and Reporting:** Collects and saves detailed metrics and visualizations for auditing and monitoring.
- **Secure Output Handling:** Optional encryption and strict path security for output artifacts.
- **Integration Ready:** Designed for seamless use in PAMOLA pipelines and with `BaseTask`.

---

## Dependencies

### Standard Library
- `os`, `time`, `hashlib`, `datetime`, `logging`, `json`, `functools`, `traceback`, `re`

### Third-Party
- `pandas` (DataFrame operations)
- `numpy` (random sampling, numeric operations)

### Internal Modules
- `pamola_core.transformations.commons.processing_utils` (batch/parallel processing)
- `pamola_core.transformations.commons.metric_utils` (metrics)
- `pamola_core.transformations.commons.visualization_utils` (visualizations)
- `pamola_core.utils.ops.op_cache` (operation caching)
- `pamola_core.utils.io` (data I/O)
- `pamola_core.config` (configuration)

---

## Exception Classes

This module does not define custom exceptions directly, but relies on standard exceptions and those from internal PAMOLA modules. Key exceptions you may encounter:

- **ValueError**: Raised for invalid parameter values or data types.
- **NotImplementedError**: Raised for unimplemented features (e.g., custom constraint functions).
- **IOError / OSError**: Raised when reading/writing files (e.g., whitelist/blacklist files).
- **Exception**: Used for unexpected errors, with detailed logging and error reporting.

### Example: Handling NotImplementedError
```python
try:
    op = CleanInvalidValuesOperation(field_constraints={...})
    op.process_batch(df)
except NotImplementedError as e:
    print(f"Feature not implemented: {e}")
```
**When raised:** If a constraint type such as `custom_function` is specified but not implemented.

### Example: Handling File Errors
```python
try:
    op = CleanInvalidValuesOperation(whitelist_path={"col": "missing.txt"})
    op.process_batch(df)
except (IOError, OSError) as e:
    print(f"File error: {e}")
```
**When raised:** If a whitelist or blacklist file cannot be opened.

---

## Main Class: `CleanInvalidValuesOperation`

### Constructor
```python
def __init__(
    self,
    field_constraints: Optional[Dict[str, Dict[str, Any]]] = None,
    whitelist_path: Optional[Dict[str, Path]] = None,
    blacklist_path: Optional[Dict[str, Path]] = None,
    null_replacement: Optional[Union[str, Dict[str, Any]]] = None,
    output_format: str = "csv",
    name: str = "clean_invalid_values_operation",
    description: str = "Clean values violating constraints",
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
- `field_constraints`: Field-level constraint definitions.
- `whitelist_path`: Dict mapping columns to whitelist files.
- `blacklist_path`: Dict mapping columns to blacklist files.
- `null_replacement`: Strategy or mapping for null value replacement.
- `output_format`: Output file format (csv, json, parquet).
- `name`, `description`: Operation metadata.
- `field_name`, `mode`, `output_field_name`, `column_prefix`: Field transformation options.
- `batch_size`: Batch size for processing.
- `use_cache`, `use_dask`, `use_encryption`, `encryption_key`: Performance and security options.

### Key Attributes
- `field_constraints`, `whitelist_path`, `blacklist_path`, `null_replacement`, `version`
- `execution_time`, `include_timestamp`, `is_encryption_required`
- `_temp_data`: Temporary storage for cleanup

### Public Methods

#### execute
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
    - `data_source`: Source of input data
    - `task_dir`: Directory for outputs and artifacts
    - `reporter`: Progress and artifact reporter
    - `progress_tracker`: Progress tracking object
    - `**kwargs`: Additional options (see docstring)
- **Returns:** `OperationResult` with status, metrics, and artifacts
- **Raises:** Exception (with detailed logging)

#### process_batch
```python
def process_batch(
    self,
    batch: pd.DataFrame
) -> pd.DataFrame
```
- **Parameters:**
    - `batch`: DataFrame batch to process
- **Returns:** Processed DataFrame
- **Raises:** NotImplementedError (for unimplemented constraint types)

#### process_value
```python
def process_value(
    self,
    value,
    **params
)
```
- **Parameters:**
    - `value`: Single value to process
    - `**params`: Additional parameters
- **Returns:** Processed value
- **Raises:** NotImplementedError

#### _process_dataframe
```python
def _process_dataframe(
    self,
    df: pd.DataFrame,
    progress_tracker: Optional[HierarchicalProgressTracker]
) -> pd.DataFrame
```
- **Parameters:**
    - `df`: DataFrame to process
    - `progress_tracker`: Progress tracker
- **Returns:** Processed DataFrame

#### _generate_cache_key, _generate_data_hash
- Deterministically generate cache keys and data hashes for reproducibility and caching.

#### _calculate_all_metrics, _collect_metrics
- Compute and collect operation metrics (e.g., generalization ratio, execution time).

#### _save_metrics, _save_output_data, _save_to_cache
- Save metrics, output data, and cache results securely.

#### _handle_visualizations, _generate_visualizations
- Generate and save visualizations for before/after data comparison.

#### _cleanup_memory
- Explicitly free memory after operation completion.

---

## Dependency Resolution and Completion Validation

- **Dependency Resolution:**
    - Whitelist and blacklist files are resolved via provided paths. If a path is invalid or inaccessible, an exception is raised and logged.
    - Field constraints are validated at initialization; missing or malformed constraints may result in a `ValueError`.
- **Completion Validation:**
    - The operation checks for successful processing, metrics calculation, and output saving. Errors in any step are logged and reported in the `OperationResult`.
    - Caching is validated by generating a deterministic key from operation parameters and data characteristics.

---

## Usage Examples

### Basic Usage
```python
from pamola_core.transformations.cleaning.clean_invalid_values import CleanInvalidValuesOperation

# Define constraints for columns
field_constraints = {
    "age": {"constraint_type": "min_value", "min_value": 0},
    "status": {"constraint_type": "allowed_values", "allowed_values": ["A", "B"]}
}

# Create operation instance
op = CleanInvalidValuesOperation(field_constraints=field_constraints)

# Process a DataFrame batch
cleaned = op.process_batch(df)
```

### Handling Failed Dependencies
```python
try:
    op = CleanInvalidValuesOperation(whitelist_path={"col": "/bad/path.txt"})
    op.process_batch(df)
except (IOError, OSError) as e:
    # Handle missing or inaccessible whitelist file
    print(f"Dependency error: {e}")
```

### Continue-on-Error Mode with Logging
```python
try:
    op = CleanInvalidValuesOperation(...)
    result = op.execute(data_source, task_dir, reporter, progress_tracker, continue_on_error=True)
except Exception as e:
    # Log and continue
    logger.warning(f"Operation failed: {e}")
```

### Accessing Metrics and Artifacts
```python
result = op.execute(data_source, task_dir, reporter)
if result.status == OperationStatus.SUCCESS:
    print(result.metrics)
    for artifact in result.artifacts:
        print(artifact.path)
```

---

## Integration Notes

- **With BaseTask:**
    - The operation is designed for use in PAMOLA pipeline tasks. Instantiate and configure as part of a `BaseTask` subclass, passing the operation to the task's execution logic.
    - Use the `OperationResult` to propagate status, metrics, and artifacts to the pipeline controller.
- **Absolute Path Dependencies:**
    - Use absolute paths only for external data not produced by the pipeline. Internal dependencies should use logical task IDs and managed paths.

---

## Error Handling and Exception Hierarchy

- **Standard Exceptions:** Most errors are raised as standard Python exceptions (e.g., `ValueError`, `IOError`).
- **Internal Exceptions:** For operation execution, caching, and configuration, exceptions from `pamola_core.utils.ops` and related modules may be raised (e.g., `OpsError`, `ConfigError`, `CacheError`).
- **Best Practice:** Always catch and log exceptions at the pipeline or task level to ensure robust error reporting and recovery.

---

## Configuration Requirements

- The `config` object (usually `CleanInvalidValuesConfig`) must define all required fields for constraints, whitelists, blacklists, and null replacement.
- Paths must be valid and accessible; missing files will cause errors.
- For encryption, a valid key or key path must be provided if `use_encryption` is enabled.

---

## Security Considerations and Best Practices

- **Path Security:**
    - Always validate whitelist/blacklist paths. Avoid using untrusted or user-supplied paths without validation.
    - Disabling path security (e.g., by passing unchecked absolute paths) can expose the system to directory traversal or data leakage risks.

#### Example: Security Failure and Handling
```python
try:
    op = CleanInvalidValuesOperation(whitelist_path={"col": "/etc/passwd"})
    op.process_batch(df)
except Exception as e:
    print(f"Security error: {e}")
```
**Risk:** If path security is disabled, sensitive system files could be read. Always restrict allowed paths.

---

## Best Practices

1. **Use Logical Task IDs for Internal Dependencies:** Reference other pipeline outputs by task ID, not by absolute path, to ensure reproducibility and portability.
2. **Use Absolute Paths Only for External Data:** Only use absolute paths for data not produced by the pipeline.
3. **Validate All Configuration:** Ensure all constraint and path configurations are correct before running the operation.
4. **Handle Exceptions Gracefully:** Catch and log exceptions at the task or pipeline level; do not allow unhandled exceptions to propagate.
5. **Enable Encryption for Sensitive Outputs:** Use `use_encryption=True` and provide a secure key for sensitive data.
6. **Monitor Metrics and Artifacts:** Always review operation metrics and output artifacts for auditing and compliance.
7. **Restrict Path Access:** Never allow user-supplied or unvalidated paths for whitelist/blacklist files.
