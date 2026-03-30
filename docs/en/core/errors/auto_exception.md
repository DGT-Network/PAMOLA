# auto_exception Decorator Documentation

**Module:** `pamola_core.errors.base`
**Function:** `auto_exception`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Function Signature](#function-signature)
3. [Decorator Parameters](#decorator-parameters)
4. [Generated `__init__` Signature](#generated-__init__-signature)
5. [Behavior Details](#behavior-details)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)

## Overview

The `auto_exception` decorator automatically generates exception `__init__` methods, eliminating boilerplate code when creating domain-specific exception classes. It handles:

- **Message formatting** from `ErrorMessages` templates
- **Parameter collection** from function arguments into the `details` dictionary
- **Error code management** with defaults and overrides
- **Custom message builders** for complex message logic
- **Parent class initialization** for exception hierarchy support

This decorator is essential for creating concise, maintainable exception classes throughout PAMOLA without repeating the same initialization logic.

## Function Signature

```python
def auto_exception(
    default_error_code: str,
    message_params: Optional[List[str]] = None,
    detail_params: Optional[List[str]] = None,
    custom_message_builder: Optional[Callable] = None,
    parent_class: Optional[Type[BasePamolaError]] = None,
) -> Callable[[Type[BasePamolaError]], Type[BasePamolaError]]:
    """
    Decorator to auto-generate exception __init__ method.

    Returns decorator function that modifies exception class.
    """
```

## Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_error_code` | `str` | Required | Error code to use if none provided in exception instantiation. Must exist in `ErrorCode` registry |
| `message_params` | `Optional[List[str]]` | `None` | Parameter names to extract from kwargs for message template formatting. If `None`, defaults to empty list |
| `detail_params` | `Optional[List[str]]` | `None` | Parameter names to include in the `details` dictionary. If `None`, defaults to `message_params` |
| `custom_message_builder` | `Optional[Callable]` | `None` | Custom function `(error_code_val: str, **params) -> str` for complex message logic. If provided, overrides template-based message building |
| `parent_class` | `Optional[Type[BasePamolaError]]` | `None` | Parent exception class to call `__init__` on. If `None`, defaults to `BasePamolaError` |

## Generated `__init__` Signature

The decorator creates an `__init__` method with this signature:

```python
def __init__(
    self,
    message: Optional[str] = None,
    *,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **params,
):
```

**Parameters:**
- `message` (optional): Explicit message to use. If `None`, built from template or custom builder
- `error_code` (keyword-only, optional): Override the default error code. If `None`, uses `default_error_code`
- `details` (keyword-only, optional): Pre-built details dictionary. Merged with collected params
- `**params` (keyword, variable): Additional parameters matching `message_params` and `detail_params`

## Behavior Details

### Message Building Logic

1. **If `message` is provided:** Use it as-is (skip template formatting)
2. **Else if `custom_message_builder` is provided:** Call builder function with error code and params
3. **Else if `message_params` is non-empty:** Format template from `ErrorMessages` with params
4. **Else:** Use `params.get("reason", "Unknown error")`

### Details Population Logic

1. Start with `details` parameter if provided, else empty dict
2. Add all params where:
   - Key is in `detail_params`
   - Value is not `None`
3. Pass merged `details` to parent class

### Parameter Resolution

- Extract values for `message_params` from `**params`
- If parameter missing, substitute `"<unknown>"` in message
- Extract values for `detail_params` from `**params`
- Skip params with `None` values in details dictionary

## Usage Examples

### Simple Exception with Message Template

```python
from pamola_core.errors.base import auto_exception, BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode

@auto_exception(
    default_error_code=ErrorCode.DATA_LOAD_FAILED,
    message_params=["source", "reason"],
    detail_params=["source", "reason"]
)
class DataLoadError(BasePamolaError):
    """Error during data loading."""
    pass

# Usage - message formatted from template automatically
try:
    load_data()
except Exception as e:
    raise DataLoadError(
        source="database://prod/users",
        reason=f"Connection timeout: {e}"
    )

# The exception will:
# 1. Format message from ErrorMessages.DATA_LOAD_FAILED template
# 2. Include "source" and "reason" in details
```

### Exception with Custom Message Builder

```python
def build_dependency_message(error_code_val: str, **params) -> str:
    from pamola_core.errors.messages.registry import ErrorMessages
    dependency = params.get("dependency_name", "<unknown>")
    reason = params.get("reason")
    required_by = params.get("required_by")

    if reason:
        reason_display = reason
    elif required_by:
        reason_display = f"required by {required_by}"
    else:
        reason_display = "missing or unavailable"

    return ErrorMessages.format(
        error_code_val,
        dependency=dependency,
        reason=reason_display
    )

@auto_exception(
    default_error_code=ErrorCode.DEPENDENCY_MISSING,
    custom_message_builder=build_dependency_message,
    detail_params=["dependency_name", "required_by", "reason"]
)
class DependencyMissingError(BasePamolaError):
    """Exception for missing dependencies."""
    pass

# Usage
raise DependencyMissingError(
    dependency_name="numpy",
    required_by="data_loader",
    reason="Module import failed"
)
```

### Exception with Explicit Message

```python
@auto_exception(
    default_error_code=ErrorCode.PROCESSING_FAILED,
    message_params=["operation"],
    detail_params=["operation", "duration_ms"]
)
class ProcessingError(BasePamolaError):
    """Error during data processing."""
    pass

# Usage 1: Auto-format from template
error1 = ProcessingError(operation="transform", duration_ms=5000)

# Usage 2: Provide explicit message
error2 = ProcessingError(
    message="Custom processing error message",
    operation="transform",
    duration_ms=5000
)
```

### Exception with Pre-Built Details

```python
@auto_exception(
    default_error_code=ErrorCode.VALIDATION_FORMAT_INVALID,
    message_params=["field_name"],
    detail_params=["field_name"]
)
class FormatValidationError(BasePamolaError):
    """Format validation error."""
    pass

# Usage - merge pre-built details with parameters
error = FormatValidationError(
    field_name="birth_date",
    details={
        "expected_format": "YYYY-MM-DD",
        "sample_invalid": ["2024-13-45", "invalid-date"]
    }
)
# Final details will include: field_name, expected_format, sample_invalid
```

### Exception with Error Code Override

```python
@auto_exception(
    default_error_code=ErrorCode.TASK_EXECUTION_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"]
)
class TaskError(BasePamolaError):
    """Base task error."""
    pass

# Usage 1: Default error code
error1 = TaskError(task_name="load_data", reason="File not found")

# Usage 2: Override error code
error2 = TaskError(
    task_name="load_data",
    reason="File not found",
    error_code=ErrorCode.FILE_NOT_FOUND  # Override
)
```

### Exception Hierarchy

```python
@auto_exception(
    default_error_code=ErrorCode.EXECUTION_ERROR,
    message_params=["operation"],
    detail_params=["operation"]
)
class BaseExecutionError(BasePamolaError):
    """Base execution error."""
    pass

# Subclass inheriting auto_exception behavior
@auto_exception(
    default_error_code=ErrorCode.MAX_RETRIES_EXCEEDED,
    message_params=["operation", "max_retries"],
    detail_params=["operation", "max_retries", "reason"],
    parent_class=BaseExecutionError
)
class MaxRetriesError(BaseExecutionError):
    """Error when retries exhausted."""
    pass

# Usage
raise MaxRetriesError(
    operation="fetch_user",
    max_retries=3,
    reason="Timeout on every attempt"
)
```

### Validation Error Aggregation

```python
@auto_exception(
    default_error_code=ErrorCode.FIELD_TYPE_ERROR,
    message_params=["field_name", "expected", "actual"],
    detail_params=["field_name", "expected", "actual", "convertible"]
)
class FieldTypeError(BasePamolaError):
    """Field type validation error."""
    pass

# Usage in validation pipeline
errors = []
for field, value in data.items():
    if not is_valid_type(field, value):
        errors.append(
            FieldTypeError(
                field_name=field,
                expected=schema[field]["type"],
                actual=type(value).__name__,
                convertible=can_convert(value, schema[field]["type"])
            )
        )

if errors:
    from pamola_core.errors.exceptions.validation import raise_if_errors
    raise_if_errors(errors)
```

## Best Practices

### 1. Define message_params for Template Formatting

Always specify `message_params` matching placeholders in your error message template:

```python
# ErrorMessages.FIELD_NOT_FOUND = "Field '{field_name}' not found. Available: {available_fields}"

@auto_exception(
    default_error_code=ErrorCode.FIELD_NOT_FOUND,
    message_params=["field_name", "available_fields"],  # Matches template placeholders
    detail_params=["field_name", "available_fields"]
)
class FieldNotFoundError(BasePamolaError):
    pass
```

### 2. Include All Debugging Info in detail_params

Capture everything needed for debugging, not just what appears in the message:

```python
@auto_exception(
    default_error_code=ErrorCode.PROCESSING_FAILED,
    message_params=["operation"],  # What's in the message
    detail_params=["operation", "duration_ms", "row_count", "memory_used"]  # For debugging
)
class ProcessingError(BasePamolaError):
    pass

# Usage
raise ProcessingError(
    operation="transform",
    duration_ms=5000,
    row_count=10000,
    memory_used=256.5
)
```

### 3. Use Custom Message Builder for Complex Logic

Only use `custom_message_builder` when template formatting isn't enough:

```python
# Good - simple template formatting
@auto_exception(
    default_error_code=ErrorCode.PARAM_INVALID,
    message_params=["param_name", "value"]
)
class ParamError(BasePamolaError):
    pass

# Good - complex conditional logic needs custom builder
def build_dependency_message(error_code_val: str, **params) -> str:
    # Complex logic here
    return formatted_message

@auto_exception(
    default_error_code=ErrorCode.DEPENDENCY_MISSING,
    custom_message_builder=build_dependency_message
)
class DependencyError(BasePamolaError):
    pass
```

### 4. Avoid Redundant Parameters

Don't include the same parameter in both `message_params` and `detail_params` if it's just for the message:

```python
# Good - only in message_params if only used for message
@auto_exception(
    default_error_code=ErrorCode.DATA_LOAD_FAILED,
    message_params=["source", "reason"],  # Used in message
    detail_params=["source", "attempted_reconnects"]  # Different for details
)
class DataLoadError(BasePamolaError):
    pass
```

### 5. Provide Meaningful Default Error Codes

Choose error codes that accurately reflect the exception:

```python
# Good - specific, descriptive code
@auto_exception(
    default_error_code=ErrorCode.TASK_INIT_FAILED,
    ...
)
class TaskInitializationError(BasePamolaError):
    pass

# Avoid - generic, unclear
@auto_exception(
    default_error_code=ErrorCode.PROCESSING_FAILED,  # Too generic
    ...
)
class TaskInitializationError(BasePamolaError):
    pass
```

## Troubleshooting

### Issue: Message Not Formatting

**Symptom:** Message contains literal `{field_name}` instead of formatted value.

**Cause:** Parameter not in `message_params` or not passed to exception.

**Solution:**
```python
# Check message_params includes all template placeholders
@auto_exception(
    default_error_code=ErrorCode.FIELD_NOT_FOUND,
    message_params=["field_name", "available_fields"],  # Both required
)
class FieldNotFoundError(BasePamolaError):
    pass

# Pass all required parameters
raise FieldNotFoundError(
    field_name="age",
    available_fields="name, email"  # Don't forget!
)
```

### Issue: Missing Parameters Don't Fail

**Symptom:** Exception created with missing parameters, message shows `<unknown>`.

**Cause:** By design - auto_exception substitutes `<unknown>` for missing params.

**Solution:** Validate parameters before raising:
```python
required_params = ["field_name", "available_fields"]
if not all(k in kwargs for k in required_params):
    missing = [k for k in required_params if k not in kwargs]
    raise ValueError(f"Missing required params: {missing}")

raise FieldNotFoundError(**kwargs)
```

### Issue: Details Dictionary Ignored

**Symptom:** Pre-built details not appearing in exception.

**Cause:** Forgetting to pass `details` parameter.

**Solution:**
```python
# Correct - pass details as keyword argument
error = FieldNotFoundError(
    field_name="age",
    details={"available_fields": ["name", "email"]}
)

# Details will be merged with parameters
```

### Issue: Parent Class __init__ Not Called

**Symptom:** Exception doesn't initialize properly, missing attributes.

**Cause:** Parent class not specified if using custom exception hierarchy.

**Solution:**
```python
# If subclassing from custom parent:
@auto_exception(
    default_error_code=ErrorCode.TASK_EXECUTION_FAILED,
    message_params=["task_name"],
    parent_class=BaseTaskError  # Must specify parent
)
class TaskError(BaseTaskError):
    pass
```

## Related Components

- **BasePamolaError** (`pamola_core.errors.base`) - Base class all decorated exceptions inherit from
- **ErrorCode** (`pamola_core.errors.codes.registry`) - Registry of error codes
- **ErrorMessages** (`pamola_core.errors.messages.registry`) - Message template registry used for formatting
- **ErrorHandler** (`pamola_core.errors.error_handler`) - Centralized handler that validates exceptions created with auto_exception
- **Validation Exceptions** (`pamola_core.errors.exceptions.validation`) - Multiple examples of auto_exception usage
- **Task Exceptions** (`pamola_core.errors.exceptions.tasks`) - Multiple examples of custom message builders
