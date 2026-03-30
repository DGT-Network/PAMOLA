# ErrorHandler Documentation

**Module:** `pamola_core.errors.error_handler`
**Class:** `ErrorHandler`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Class Signature](#class-signature)
3. [Constructor Parameters](#constructor-parameters)
4. [Core Methods](#core-methods)
5. [Usage Examples](#usage-examples)
6. [Decorators](#decorators)
7. [Context Manager](#context-manager)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Related Components](#related-components)

## Overview

`ErrorHandler` is the central error handling component in PAMOLA Core. It provides a unified interface for:

- **Error validation** against `ErrorCode` and `ErrorMessages` registries
- **Structured logging** with full context and recovery suggestions
- **OperationResult creation** with error status and metrics
- **Error code resolution** when exceptions own their codes
- **Recovery suggestions** from `ErrorContext`
- **Decorator-based handling** for both sync and async functions
- **Context manager support** for automatic error handling

The ErrorHandler pattern ensures consistent error handling across all PAMOLA operations while maintaining detailed diagnostic information for troubleshooting.

## Class Signature

```python
class ErrorHandler:
    """
    Centralized error handling with structured logging.

    Features:
    - Validates error codes and message templates
    - Provides standardized technical messages
    - Adds recovery suggestions
    - Creates structured OperationResult
    - Logs with full context
    - Supports both sync and async functions
    """
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logger` | `logging.Logger` | Required | Logger instance for error logging (should be created with `logging.getLogger(__name__)`) |
| `operation_name` | `Optional[str]` | `None` | Name of the operation being performed. Included in logs and error details for context |
| `enable_verbose_logging` | `bool` | `True` | Enable detailed debug logging including full tracebacks and recovery suggestions |

## Core Methods

### `handle_error()`

Primary method for handling exceptions with structured logging and optional recovery.

**Signature:**
```python
def handle_error(
    self,
    error: Exception,
    error_code: str,
    context: Optional[Dict[str, Any]] = None,
    raise_error: bool = False,
    message_template: Optional[str] = None,
    message_kwargs: Optional[Dict[str, Any]] = None,
) -> OperationResult:
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `error` | `Exception` | Required | The exception that occurred |
| `error_code` | `str` | Required | Standard error code from `ErrorCode` registry |
| `context` | `Optional[Dict]` | `None` | Additional context information (operation metadata, values, etc.) |
| `raise_error` | `bool` | `False` | If `True`, re-raises the error after logging. If `False`, returns OperationResult |
| `message_template` | `Optional[str]` | `None` | Error message template name from `ErrorMessages` (e.g., "DATA_LOAD_FAILED") |
| `message_kwargs` | `Optional[Dict]` | `None` | Parameters for formatting the message template |

**Returns:**
`OperationResult` - Structured result with:
- `status=OperationStatus.ERROR`
- `error_message` with formatted message
- `error_trace` with full traceback
- `exception` reference to original exception
- `metrics` with error metadata (code, type, severity, category, retry info)
- Recovery suggestions in nested metrics

**Raises:**
- `ValueError` if error_code is invalid or message_template doesn't exist
- Re-raises original `error` if `raise_error=True`

**Example:**
```python
import logging
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.codes.registry import ErrorCode

logger = logging.getLogger(__name__)
handler = ErrorHandler(logger, operation_name="data_loading")

try:
    load_data()
except Exception as e:
    result = handler.handle_error(
        error=e,
        error_code=ErrorCode.DATA_LOAD_FAILED,
        context={"source": "file.csv"},
        message_template="DATA_LOAD_FAILED",
        message_kwargs={"source": "file.csv", "reason": str(e)},
        raise_error=False
    )
    return result  # OperationResult with error status
```

### `standardize_result()`

Attach standardized error metadata to an existing `OperationResult`.

**Signature:**
```python
def standardize_result(
    self,
    result: OperationResult,
    error_code: str,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> OperationResult:
```

**Use Case:** When a function creates its own `OperationResult` but didn't include standardized error codes/metadata.

**Example:**
```python
result = OperationResult(status=OperationStatus.ERROR)
result.error_message = "Processing failed"

# Enhance with standardized metadata
result = handler.standardize_result(
    result=result,
    error_code=ErrorCode.PROCESSING_FAILED,
    context={"operation": "transform"}
)
```

### `create_error()`

Create a properly formatted `BasePamolaError` with standardized message.

**Signature:**
```python
def create_error(
    self,
    error_code: str,
    message_kwargs: Dict[str, Any],
    exception_class: type = BasePamolaError,
    details: Optional[Dict[str, Any]] = None,
) -> BasePamolaError:
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `error_code` | `str` | Required | Error code from `ErrorCode` registry |
| `message_kwargs` | `Dict[str, Any]` | Required | Parameters for message template formatting |
| `exception_class` | `type` | `BasePamolaError` | Exception class to instantiate (must be `BasePamolaError` subclass) |
| `details` | `Optional[Dict]` | `None` | Additional details to attach |

**Returns:** Formatted `BasePamolaError` instance ready to raise

**Raises:**
- `ValidationError` if exception_class is not `BasePamolaError` subclass
- `InvalidParameterError` if error code invalid, template not found, or missing parameters

**Example:**
```python
error = handler.create_error(
    error_code=ErrorCode.FIELD_NOT_FOUND,
    message_kwargs={"field_name": "age", "available_fields": "name, email"},
    exception_class=FieldNotFoundError,
    details={"dataset_columns": ["name", "email"]}
)
raise error
```

### `get_stats()`

Get error handling statistics.

**Returns:** `Dict[str, int]` with:
- `total_errors` - Total errors handled
- `fallback_codes_used` - Times invalid error codes fell back to default

**Example:**
```python
stats = handler.get_stats()
print(f"Handled {stats['total_errors']} errors")
```

### `reset_stats()`

Reset error handling statistics to zero.

```python
handler.reset_stats()
```

## Usage Examples

### Basic Error Handling

```python
import logging
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.codes.registry import ErrorCode

logger = logging.getLogger(__name__)
handler = ErrorHandler(logger, operation_name="csv_loading")

def load_csv(filepath: str):
    try:
        with open(filepath) as f:
            return f.read()
    except FileNotFoundError as e:
        result = handler.handle_error(
            error=e,
            error_code=ErrorCode.FILE_NOT_FOUND,
            context={"filepath": filepath},
            raise_error=False
        )
        return result
```

### Re-Raising with Context

```python
try:
    process_data()
except Exception as e:
    handler.handle_error(
        error=e,
        error_code=ErrorCode.PROCESSING_FAILED,
        context={"operation": "transform", "row_count": 1000},
        raise_error=True  # Re-raise after logging
    )
```

### Error Code Detection from Exception

```python
from pamola_core.errors.exceptions.validation import FieldNotFoundError

try:
    operation()
except FieldNotFoundError as e:
    # Exception owns its error code
    result = handler.handle_error(
        error=e,
        error_code=ErrorCode.FIELD_NOT_FOUND,
        context={"field": e.field_name}
    )
    # Handler detects e.error_code and uses it instead
```

### Creating Properly Formatted Errors

```python
from pamola_core.errors.exceptions.validation import InvalidParameterError

try:
    validate_config(config)
except ConfigError:
    error = handler.create_error(
        error_code=ErrorCode.PARAM_INVALID,
        message_kwargs={
            "param_name": "timeout",
            "value": 0,
            "constraint": "must be > 0"
        },
        exception_class=InvalidParameterError
    )
    raise error
```

### Enhancing Existing Results

```python
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus

# Function returned result without proper error metadata
result = OperationResult(status=OperationStatus.ERROR)
result.error_message = "Data validation failed"

# Enhance with standardized metadata
result = handler.standardize_result(
    result=result,
    error_code=ErrorCode.DATA_VALIDATION_ERROR,
    message="Data validation failed for dataset 'users'",
    context={"rows_invalid": 42, "total_rows": 1000}
)
```

## Decorators

### `capture_errors()`

Decorator for wrapping functions with standardized error handling.

**Signature:**
```python
def capture_errors(
    self,
    error_code: str,
    rethrow: bool = False,
    context: Optional[Dict[str, Any]] = None,
    message_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
```

**Supports both sync and async functions automatically.**

**Example - Sync:**
```python
@handler.capture_errors(
    error_code=ErrorCode.PROCESSING_FAILED,
    rethrow=False
)
def process_data(data):
    return data.transform()

# Usage
result = process_data(df)  # Returns OperationResult on error, data on success
```

**Example - Async:**
```python
@handler.capture_errors(
    error_code=ErrorCode.DATA_LOAD_FAILED,
    rethrow=True
)
async def load_data_async(source: str):
    return await fetch_from_source(source)

# Usage
try:
    data = await load_data_async("database://prod")
except Exception as e:
    # Error was logged and re-raised
    pass
```

**Example with Context:**
```python
@handler.capture_errors(
    error_code=ErrorCode.PROCESSING_FAILED,
    context={"operation": "batch_transform"},
    rethrow=False
)
def batch_process(items: List[dict]):
    return [process(item) for item in items]
```

## Context Manager

### `error_context()`

Context manager for automatic error handling within a code block.

**Signature:**
```python
@contextmanager
def error_context(
    self,
    error_code: str,
    context: Optional[Dict[str, Any]] = None,
    message_kwargs: Optional[Dict[str, Any]] = None,
    suppress: bool = False,
):
```

**Example - With Re-raising:**
```python
with handler.error_context(
    error_code=ErrorCode.DATA_LOAD_FAILED,
    context={"source": "file.csv"},
    suppress=False
):
    load_data()
# Exception is logged and re-raised
```

**Example - With Suppression:**
```python
with handler.error_context(
    error_code=ErrorCode.CACHE_READ_FAILED,
    suppress=True
):
    cached_data = cache.get("key")
    # If exception occurs, it's logged but not raised
    # Function continues execution
```

**Example - Complex Error Handling:**
```python
def load_with_fallback():
    # Try primary source
    with handler.error_context(
        error_code=ErrorCode.DATA_LOAD_FAILED,
        context={"source": "primary"},
        suppress=True
    ):
        return load_from_primary()

    # Try fallback source
    with handler.error_context(
        error_code=ErrorCode.DATA_LOAD_FAILED,
        context={"source": "fallback"},
        suppress=False
    ):
        return load_from_fallback()
```

## Best Practices

### 1. Initialize Once Per Module

Create a single handler instance per module:

```python
# At module level
logger = logging.getLogger(__name__)
handler = ErrorHandler(logger, operation_name="data_processing")

# Reuse throughout module
def operation1():
    try:
        ...
    except Exception as e:
        handler.handle_error(e, ErrorCode.PROCESSING_FAILED)

def operation2():
    try:
        ...
    except Exception as e:
        handler.handle_error(e, ErrorCode.DATA_LOAD_FAILED)
```

### 2. Provide Rich Context

Include all relevant debugging information:

```python
result = handler.handle_error(
    error=e,
    error_code=ErrorCode.PROCESSING_FAILED,
    context={
        "operation": "transform",
        "row_count": 10000,
        "columns_processed": 50,
        "duration_ms": 5000,
        "memory_used_mb": 256
    }
)
```

### 3. Use Error Codes Consistently

Match error codes to exception types:

```python
# Good - appropriate code
from pamola_core.errors.exceptions.tasks import TaskInitializationError

try:
    setup_task()
except Exception as e:
    handler.handle_error(e, ErrorCode.TASK_INIT_FAILED)

# Avoid - wrong category
handler.handle_error(e, ErrorCode.FIELD_NOT_FOUND)  # Wrong!
```

### 4. Leverage Message Templates

Let ErrorHandler format messages from templates:

```python
# Good - use message template
result = handler.handle_error(
    error=e,
    error_code=ErrorCode.DATA_LOAD_FAILED,
    message_template="DATA_LOAD_FAILED",
    message_kwargs={"source": "file.csv", "reason": str(e)}
)

# Also acceptable - let handler infer template
result = handler.handle_error(
    error=e,
    error_code=ErrorCode.DATA_LOAD_FAILED
)
```

### 5. Choose Appropriate Handler Action

Decide whether to re-raise or return result:

```python
# Critical operation - re-raise
handler.handle_error(
    error=e,
    error_code=ErrorCode.TASK_INIT_FAILED,
    raise_error=True  # Must fail loudly
)

# Recoverable operation - return result
result = handler.handle_error(
    error=e,
    error_code=ErrorCode.CACHE_READ_FAILED,
    raise_error=False  # Can continue with empty cache
)
```

### 6. Use Decorators for Simple Cases

Simplify code with decorators:

```python
# Verbose approach
def load_data(source):
    try:
        return fetch_data(source)
    except Exception as e:
        return handler.handle_error(e, ErrorCode.DATA_LOAD_FAILED)

# Cleaner with decorator
@handler.capture_errors(ErrorCode.DATA_LOAD_FAILED, rethrow=False)
def load_data(source):
    return fetch_data(source)
```

## Troubleshooting

### Issue: Error Code Validation Fails

**Symptom:** `ValueError` about invalid error code.

**Cause:** Error code not in `ErrorCode` registry.

**Solution:**
```python
# Use registered codes
from pamola_core.errors.codes.registry import ErrorCode

handler.handle_error(
    error=e,
    error_code=ErrorCode.PROCESSING_FAILED  # From registry
)
```

### Issue: Message Template Not Found

**Symptom:** `InvalidParameterError` about missing template.

**Cause:** Message template name doesn't exist in `ErrorMessages`.

**Solution:**
```python
# Check template exists
from pamola_core.errors.messages.registry import ErrorMessages

# Either use existing template
handler.handle_error(e, ErrorCode.DATA_LOAD_FAILED,
                     message_template="DATA_LOAD_FAILED")

# Or don't specify template
handler.handle_error(e, ErrorCode.DATA_LOAD_FAILED)
```

### Issue: Missing Message Parameters

**Symptom:** Template has `{placeholder}` but parameter not provided.

**Cause:** Incomplete message_kwargs dictionary.

**Solution:**
```python
# Provide all required parameters
result = handler.handle_error(
    error=e,
    error_code=ErrorCode.FIELD_NOT_FOUND,
    message_template="FIELD_NOT_FOUND",
    message_kwargs={
        "field_name": "age",  # Required
        "available_fields": "name, email"  # Required
    }
)
```

### Issue: Exception Owns Error Code

**Symptom:** Handler uses different code than exception.

**Cause:** Exception has `error_code` attribute that differs from handler's parameter.

**Solution:** This is intentional - exception's code takes precedence:

```python
# Exception owns the code
error = FieldNotFoundError(field_name="age")  # Owns FIELD_NOT_FOUND

result = handler.handle_error(
    error=error,
    error_code=ErrorCode.DATA_VALIDATION_ERROR  # Will use FIELD_NOT_FOUND instead
)
```

## Related Components

- **BasePamolaError** (`pamola_core.errors.base`) - Exception class that handler works with
- **ErrorCode** (`pamola_core.errors.codes.registry`) - Error codes validated by handler
- **ErrorMessages** (`pamola_core.errors.messages.registry`) - Message templates used for formatting
- **ErrorContext** (`pamola_core.errors.context.suggestions`) - Recovery suggestions included in results
- **OperationResult** (`pamola_core.utils.ops.op_result`) - Result container populated by handler
- **BaseOperation** (`pamola_core.utils.ops.op_base`) - Operations use ErrorHandler for error handling
- **BaseTask** (`pamola_core.utils.tasks.base_task`) - Tasks wrap operations with error handling
