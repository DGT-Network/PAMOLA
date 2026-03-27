# Error Codes Reference

**Module:** `pamola_core.errors.codes.registry`
**Class:** `ErrorCode`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Error Code Naming Convention](#error-code-naming-convention)
3. [Error Code Categories](#error-code-categories)
4. [Complete Code Listing](#complete-code-listing)
5. [Code Usage Guidelines](#code-usage-guidelines)
6. [Adding New Error Codes](#adding-new-error-codes)
7. [Best Practices](#best-practices)
8. [Related Components](#related-components)

## Overview

The `ErrorCode` registry provides a centralized, standardized collection of error codes used throughout PAMOLA Core. Each error code is a string constant following a naming convention: `<CATEGORY>_<SPECIFIC_ERROR>`.

**Key Characteristics:**
- All codes are class variables of type `ClassVar[str]`
- Codes are uppercase with underscores separating words
- One-to-one correspondence with message templates in `ErrorMessages`
- Metadata (severity, category, retry_allowed) available via `get_error_metadata()`
- Used by `BasePamolaError` to standardize error reporting

**Example Codes:**
```python
ErrorCode.DATA_LOAD_FAILED      # String: "DATA_LOAD_FAILED"
ErrorCode.FIELD_NOT_FOUND       # String: "FIELD_NOT_FOUND"
ErrorCode.TASK_INIT_FAILED      # String: "TASK_INIT_FAILED"
ErrorCode.PROCESSING_FAILED     # String: "PROCESSING_FAILED"
```

## Error Code Naming Convention

All error codes follow this pattern:

```
<CATEGORY>_<SPECIFIC_ERROR>
```

**Components:**

| Component | Format | Examples |
|-----------|--------|----------|
| **CATEGORY** | Uppercase, single word | DATA, FIELD, PARAM, TASK, CRYPTO, RESOURCE |
| **SPECIFIC_ERROR** | Uppercase, descriptive phrase | LOAD_FAILED, NOT_FOUND, INVALID, MISSING |

**Valid Examples:**
- `DATA_LOAD_FAILED` - Data category, load operation failed
- `FIELD_NOT_FOUND` - Field category, not found issue
- `PARAM_INVALID` - Parameter category, invalid value
- `TASK_INIT_FAILED` - Task category, initialization failed
- `ENCRYPTION_ERROR` - Crypto category, encryption issue

**Invalid Examples:**
- `data_load` - Use UPPERCASE_UNDERSCORES
- `DATALOAD_ERROR` - Separate words with underscore
- `LOAD_FAILED` - Include category for clarity
- `DATA_LOAD_FAILED_CRITICAL` - Severity in category name (use metadata instead)

## Error Code Categories

PAMOLA error codes are organized into logical categories:

| Category | Count | Purpose |
|----------|-------|---------|
| **Data Operations** | 7 | Loading, validation, writing data |
| **Field Validation** | 4 | Field existence, type, value checks |
| **Parameter Validation** | 6 | Parameter validity and type checks |
| **Conditional & Range** | 4 | Format and range validation |
| **File Operations** | 5 | File access, format, corruption checks |
| **Multiple Errors** | 1 | Aggregating multiple errors |
| **Processing** | 6 | Operation execution and timeouts |
| **Resources** | 4 | Memory, disk, CPU constraints |
| **Caching** | 5 | Cache read/write operations |
| **Artifacts** | 4 | Artifact validation and storage |
| **Visualization** | 3 | Chart/graph generation |
| **Cryptography** | 14+ | Encryption, keys, hashing |
| **Tasks** | 17+ | Task execution lifecycle |
| **NLP & LLM** | 9+ | Language model operations |
| **Filesystem** | 3+ | Path validation and security |

## Complete Code Listing

### Data Operations (7 codes)

```python
DATA_LOAD_FAILED            # "DATA_LOAD_FAILED"
DATA_SOURCE_INVALID         # "DATA_SOURCE_INVALID"
DATA_VALIDATION_ERROR       # "DATA_VALIDATION_ERROR"
DATA_EMPTY                  # "DATA_EMPTY"
DATA_WRITE_FAILED           # "DATA_WRITE_FAILED"
DATA_FRAME_PROCESSING_ERROR # "DATA_FRAME_PROCESSING_ERROR"
```

| Code | Message Template | Typical Cause |
|------|------------------|---------------|
| `DATA_LOAD_FAILED` | "Failed to load data from source '{source}': {reason}" | File not found, connection timeout, format error |
| `DATA_SOURCE_INVALID` | "Data source '{source}' is invalid or not supported: {reason}" | Invalid path, unsupported protocol |
| `DATA_VALIDATION_ERROR` | "Data validation failed for '{context}': {reason}" | Schema mismatch, constraint violation |
| `DATA_EMPTY` | "Dataset is empty; cannot perform operation '{operation}'" | No rows in DataFrame |
| `DATA_WRITE_FAILED` | "Failed to write data to '{destination}': {reason}" | Permission denied, disk full |
| `DATA_FRAME_PROCESSING_ERROR` | "DataFrame processing failed for operation '{operation}': {reason}" | Invalid column, type error |

### Field Validation (4 codes)

```python
FIELD_NOT_FOUND      # "FIELD_NOT_FOUND"
FIELD_TYPE_ERROR     # "FIELD_TYPE_ERROR"
FIELD_VALUE_ERROR    # "FIELD_VALUE_ERROR"
COLUMN_NOT_FOUND     # "COLUMN_NOT_FOUND"
```

| Code | Message Template | Used For |
|------|------------------|----------|
| `FIELD_NOT_FOUND` | "Field '{field_name}' not found in data. Available fields: {available_fields}" | Accessing non-existent field |
| `FIELD_TYPE_ERROR` | "Field '{field_name}' has invalid type. Expected: {expected}, actual: {actual}" | Type mismatch |
| `FIELD_VALUE_ERROR` | "Invalid value {value} for field '{field_name}': {reason}" | Value constraint violation |
| `COLUMN_NOT_FOUND` | "Column '{column_name}' not found in DataFrame. Available columns: {available_columns}" | Missing DataFrame column |

### Parameter Validation (6 codes)

```python
PARAM_MISSING           # "PARAM_MISSING"
PARAM_INVALID           # "PARAM_INVALID"
PARAM_TYPE_ERROR        # "PARAM_TYPE_ERROR"
STRATEGY_INVALID        # "STRATEGY_INVALID"
MODE_INVALID            # "MODE_INVALID"
NULL_STRATEGY_INVALID   # "NULL_STRATEGY_INVALID"
```

### Conditional & Range Validation (4 codes)

```python
VALIDATION_CONDITIONAL_FAILED   # "VALIDATION_CONDITIONAL_FAILED"
VALIDATION_RANGE_FAILED         # "VALIDATION_RANGE_FAILED"
VALIDATION_FORMAT_INVALID       # "VALIDATION_FORMAT_INVALID"
VALIDATION_MARKER_FAILED        # "VALIDATION_MARKER_FAILED"
```

### File Operations (5 codes)

```python
FILE_ERROR          # "FILE_ERROR"
FILE_NOT_FOUND      # "FILE_NOT_FOUND"
FILE_FORMAT_INVALID # "FILE_FORMAT_INVALID"
FILE_ACCESS_DENIED  # "FILE_ACCESS_DENIED"
FILE_CORRUPTED      # "FILE_CORRUPTED"
```

### Multiple Errors (1 code)

```python
MULTIPLE_ERRORS  # "MULTIPLE_ERRORS" - For aggregated validation errors
```

### Processing Operations (6 codes)

```python
PROCESSING_FAILED        # "PROCESSING_FAILED"
PROCESSING_BATCH_FAILED  # "PROCESSING_BATCH_FAILED"
PROCESSING_CHUNK_FAILED  # "PROCESSING_CHUNK_FAILED"
PROCESSING_TIMEOUT       # "PROCESSING_TIMEOUT"
PROCESSING_INTERRUPTED   # "PROCESSING_INTERRUPTED"
FEATURE_NOT_IMPLEMENTED  # "FEATURE_NOT_IMPLEMENTED"
```

### Resource Management (4 codes)

```python
RESOURCE_MEMORY_EXCEEDED    # "RESOURCE_MEMORY_EXCEEDED"
RESOURCE_DISK_FULL          # "RESOURCE_DISK_FULL"
RESOURCE_CPU_THROTTLED      # "RESOURCE_CPU_THROTTLED"
RESOURCE_NOT_FOUND          # "RESOURCE_NOT_FOUND"
```

### Caching (5 codes)

```python
CACHE_READ_FAILED    # "CACHE_READ_FAILED"
CACHE_WRITE_FAILED   # "CACHE_WRITE_FAILED"
CACHE_KEY_INVALID    # "CACHE_KEY_INVALID"
CACHE_EXPIRED        # "CACHE_EXPIRED"
CACHE_CORRUPTED      # "CACHE_CORRUPTED"
```

### Artifacts (4 codes)

```python
ARTIFACT_VALIDATION_FAILED  # "ARTIFACT_VALIDATION_FAILED"
ARTIFACT_NOT_FOUND          # "ARTIFACT_NOT_FOUND"
ARTIFACT_WRITE_FAILED       # "ARTIFACT_WRITE_FAILED"
ARTIFACT_CORRUPTED          # "ARTIFACT_CORRUPTED"
```

### Visualization (3 codes)

```python
VISUALIZATION_ERROR          # "VISUALIZATION_ERROR"
VISUALIZATION_RENDERING_FAILED
VISUALIZATION_INVALID_CONFIG
```

### Cryptography (14+ codes)

```python
ENCRYPTION_ERROR                # "ENCRYPTION_ERROR"
DECRYPTION_ERROR                # "DECRYPTION_ERROR"
ENCRYPTION_INITIALIZATION_ERROR  # "ENCRYPTION_INITIALIZATION_ERROR"
KEY_GENERATION_ERROR            # "KEY_GENERATION_ERROR"
KEY_LOADING_ERROR               # "KEY_LOADING_ERROR"
MASTER_KEY_ERROR                # "MASTER_KEY_ERROR"
TASK_KEY_ERROR                  # "TASK_KEY_ERROR"
CRYPTO_KEY_ERROR                # "CRYPTO_KEY_ERROR"
KEY_STORE_ERROR                 # "KEY_STORE_ERROR"
CRYPTO_ERROR                    # "CRYPTO_ERROR"
PSEUDONYMIZATION_ERROR          # "PSEUDONYMIZATION_ERROR"
HASH_COLLISION_ERROR            # "HASH_COLLISION_ERROR"
DATA_REDACTION_ERROR            # "DATA_REDACTION_ERROR"
FORMAT_ERROR                    # "FORMAT_ERROR"
PROVIDER_ERROR                  # "PROVIDER_ERROR"
MODE_ERROR                      # "MODE_ERROR"
LEGACY_MIGRATION_ERROR          # "LEGACY_MIGRATION_ERROR"
AGE_TOOL_ERROR                  # "AGE_TOOL_ERROR"
```

### Task Execution (17+ codes)

```python
TASK_ERROR                      # "TASK_ERROR"
TASK_INIT_FAILED                # "TASK_INIT_FAILED"
TASK_EXECUTION_FAILED           # "TASK_EXECUTION_FAILED"
TASK_FINALIZATION_FAILED        # "TASK_FINALIZATION_FAILED"
TASK_DEPENDENCY_ERROR           # "TASK_DEPENDENCY_ERROR"
DEPENDENCY_MISSING              # "DEPENDENCY_MISSING"
TASK_DEPENDENCY_FAILED          # "TASK_DEPENDENCY_FAILED"
TASK_REGISTRY_ERROR             # "TASK_REGISTRY_ERROR"
EXECUTION_ERROR                 # "EXECUTION_ERROR"
EXECUTION_LOG_ERROR             # "EXECUTION_LOG_ERROR"
CHECKPOINT_ERROR                # "CHECKPOINT_ERROR"
STATE_SERIALIZATION_FAILED      # "STATE_SERIALIZATION_FAILED"
STATE_RESTORATION_FAILED        # "STATE_RESTORATION_FAILED"
TASK_CONTEXT_ERROR              # "TASK_CONTEXT_ERROR"
MAX_RETRIES_EXCEEDED            # "MAX_RETRIES_EXCEEDED"
NON_RETRIABLE_ERROR             # "NON_RETRIABLE_ERROR"
OPERATION_ERROR                 # "OPERATION_ERROR"
```

### NLP & LLM (9+ codes)

```python
NLP_ERROR                  # "NLP_ERROR"
PROMPT_VALIDATION_ERROR    # "PROMPT_VALIDATION_ERROR"
LLM_ERROR                  # "LLM_ERROR"
LLM_CONNECTION_ERROR       # "LLM_CONNECTION_ERROR"
LLM_GENERATION_ERROR       # "LLM_GENERATION_ERROR"
LLM_RESPONSE_ERROR         # "LLM_RESPONSE_ERROR"
MODEL_NOT_AVAILABLE        # "MODEL_NOT_AVAILABLE"
MODEL_LOAD_ERROR           # "MODEL_LOAD_ERROR"
UNSUPPORTED_LANGUAGE       # "UNSUPPORTED_LANGUAGE"
```

### Filesystem (3+ codes)

```python
PATH_VALIDATION_ERROR       # "PATH_VALIDATION_ERROR"
PATH_SECURITY_ERROR         # "PATH_SECURITY_ERROR"
DIRECTORY_MANAGER_ERROR     # "DIRECTORY_MANAGER_ERROR"
DIRECTORY_CREATION_ERROR    # "DIRECTORY_CREATION_ERROR"
```

## Code Usage Guidelines

### 1. Always Use ErrorCode Registry

Import and use constants, never string literals:

```python
# Good - use registry
from pamola_core.errors.codes.registry import ErrorCode

raise BasePamolaError(
    message="Failed",
    error_code=ErrorCode.DATA_LOAD_FAILED
)

# Avoid - string literals
raise BasePamolaError(
    message="Failed",
    error_code="DATA_LOAD_FAILED"  # Avoid this
)
```

### 2. Match Error Code to Exception Type

Use codes appropriate for the exception being raised:

```python
# Good - task error code for task exception
from pamola_core.errors.exceptions.tasks import TaskInitializationError
raise TaskInitializationError(
    task_name="load",
    reason="Missing config",
    error_code=ErrorCode.TASK_INIT_FAILED
)

# Avoid - unrelated code
raise TaskInitializationError(
    task_name="load",
    reason="Missing config",
    error_code=ErrorCode.FIELD_NOT_FOUND  # Wrong category
)
```

### 3. Use Specific Codes

Choose the most specific code that describes the error:

```python
# Good - specific code
raise BasePamolaError(
    message="Field 'age' not found",
    error_code=ErrorCode.FIELD_NOT_FOUND
)

# Avoid - too generic
raise BasePamolaError(
    message="Field 'age' not found",
    error_code=ErrorCode.DATA_VALIDATION_ERROR  # Too broad
)
```

### 4. Error Code with ErrorHandler

Pass error codes to ErrorHandler for validation:

```python
from pamola_core.errors.error_handler import ErrorHandler

handler = ErrorHandler(logger)
result = handler.handle_error(
    error=e,
    error_code=ErrorCode.DATA_LOAD_FAILED,  # Validated
    context={"source": "file.csv"}
)
```

## Adding New Error Codes

To add a new error code to the framework:

### 1. Add to ErrorCode Registry

Edit `pamola_core/errors/codes/registry.py`:

```python
class ErrorCode:
    # =========================================================================
    # NEW CATEGORY (N codes)
    # =========================================================================
    NEW_ERROR_NAME: ClassVar[str] = "NEW_ERROR_NAME"
```

**Naming Guidelines:**
- Follow `<CATEGORY>_<SPECIFIC_ERROR>` pattern
- Use descriptive names that clearly indicate the error
- Avoid abbreviations unless universally understood
- Keep consistent with related codes

### 2. Add Message Template

Edit `pamola_core/errors/messages/registry.py`:

```python
class ErrorMessages:
    NEW_ERROR_NAME: ClassVar[str] = (
        "Error message with {placeholder} for formatting"
    )
```

**Template Guidelines:**
- Be specific and actionable
- Include placeholders for context
- Use consistent terminology
- Write for developers (technical language OK)

### 3. Add Error Metadata

Edit `pamola_core/errors/codes/metadata.py`:

```python
"NEW_ERROR_NAME": {
    "category": "new_category",
    "severity": "error",  # critical, error, warning, info
    "retry_allowed": False,
    "user_facing": False
}
```

### 4. Add Recovery Suggestions (Optional)

Edit `pamola_core/errors/context/recovery_data.yaml`:

```yaml
suggestions:
  NEW_ERROR_NAME:
    - First recovery step
    - Second recovery step
    - Third recovery step
```

### 5. Create Exception Class (Optional)

If needed, create exception class in appropriate module:

```python
from pamola_core.errors.base import auto_exception, BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode

@auto_exception(
    default_error_code=ErrorCode.NEW_ERROR_NAME,
    message_params=["field1", "field2"],
    detail_params=["field1", "field2"]
)
class NewException(BasePamolaError):
    """Description of the exception."""
    pass
```

## Best Practices

### 1. Group Related Codes by Category

Organize code in logical categories that match your domain:

```python
class ErrorCode:
    # =========================================================================
    # AUTHENTICATION & AUTHORIZATION (3 codes)
    # =========================================================================
    AUTH_INVALID_CREDENTIALS
    AUTH_EXPIRED_TOKEN
    AUTH_PERMISSION_DENIED
```

### 2. Be Consistent with Naming

Use same terminology for similar errors:

```python
# Good - consistent naming
FIELD_NOT_FOUND
COLUMN_NOT_FOUND
PARAMETER_NOT_FOUND

# Avoid - inconsistent
FIELD_NOT_FOUND
COLUMN_MISSING
PARAMETER_ABSENT
```

### 3. Don't Include Severity in Code

Use error metadata for severity, not the code name:

```python
# Good - severity in metadata
ERROR_CODE = "DATA_LOAD_FAILED"
metadata = {"severity": "critical"}

# Avoid - severity in code name
ERROR_CODE = "CRITICAL_DATA_LOAD_FAILED"
```

### 4. Pair with Message Template

Every error code must have a corresponding message template:

```python
# In ErrorCode registry
DATA_LOAD_FAILED: ClassVar[str] = "DATA_LOAD_FAILED"

# In ErrorMessages registry (REQUIRED)
DATA_LOAD_FAILED: ClassVar[str] = "Failed to load data: {reason}"
```

### 5. Use for Metrics and Monitoring

Reference error codes in monitoring and alerting:

```python
# Track error frequency by code
error_counts = {}
for result in results:
    if result.error:
        code = result.metrics.get("error_code")
        error_counts[code] = error_counts.get(code, 0) + 1

# Alert on specific codes
if error_counts.get(ErrorCode.RESOURCE_MEMORY_EXCEEDED, 0) > 10:
    send_alert("Memory pressure detected")
```

## Related Components

- **ErrorMessages** (`pamola_core.errors.messages.registry`) - Templates paired with codes
- **ErrorCode Metadata** (`pamola_core.errors.codes.metadata`) - Severity, category, retry info
- **BasePamolaError** (`pamola_core.errors.base`) - Uses codes in exception
- **ErrorHandler** (`pamola_core.errors.error_handler`) - Validates codes
- **ErrorContext** (`pamola_core.errors.context.suggestions`) - Recovery suggestions by code
