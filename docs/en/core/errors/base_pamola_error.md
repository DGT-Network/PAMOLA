# BasePamolaError Documentation

**Module:** `pamola_core.errors.base`
**Class:** `BasePamolaError`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Class Signature](#class-signature)
3. [Constructor Parameters](#constructor-parameters)
4. [Methods](#methods)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Components](#related-components)

## Overview

`BasePamolaError` is the root exception class for all PAMOLA errors. It extends Python's built-in `Exception` class and adds structured error handling capabilities including error codes, standardized messages, and rich contextual details for debugging.

All PAMOLA exceptions (e.g., `DataError`, `ValidationError`, `TaskInitializationError`) inherit from `BasePamolaError` to ensure:
- Consistent error handling across the framework
- Standardized logging and telemetry
- Structured error serialization via `to_dict()`
- Integration with error metadata and recovery suggestions

## Class Signature

```python
class BasePamolaError(Exception):
    """
    Base exception for all PAMOLA errors.

    All PAMOLA exceptions should inherit from this class to ensure consistent
    error handling, logging, and telemetry across the framework.

    Attributes
    ----------
        message (str): Human-readable error message
        error_code (str): Standardized error code from ErrorCode registry
        details (Dict[str, Any]): Additional structured context for debugging
    """
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | Required | Human-readable error message explaining what went wrong |
| `error_code` | `Optional[str]` | `None` | Standardized error code from `ErrorCode` registry. If `None`, uses class name as fallback |
| `details` | `Optional[Dict[str, Any]]` | `None` | Structured context dictionary for debugging. Contains operation metadata, field names, values, etc. If `None`, defaults to empty dict |

## Methods

### `__init__(message, error_code=None, details=None)`

Initializes the exception with message, error code, and structured details.

**Parameters:**
```python
def __init__(
    self,
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
```

**Behavior:**
- Stores `message`, `error_code`, and `details` as instance attributes
- If `error_code` is `None`, uses `self.__class__.__name__` as fallback
- If `details` is `None`, initializes to empty dictionary
- Calls parent `Exception.__init__()` with the message

**Example:**
```python
error = BasePamolaError(
    message="Operation failed",
    error_code="PROCESSING_FAILED",
    details={"operation": "data_transform", "rows": 1000}
)
```

### `to_dict() -> Dict[str, Any]`

Converts the exception to a structured dictionary for logging and serialization.

**Returns:**
| Field | Type | Description |
|-------|------|-------------|
| `error_type` | `str` | Exception class name (e.g., "BasePamolaError") |
| `message` | `str` | The error message |
| `error_code` | `str` | The error code (or class name fallback) |
| `details` | `Dict[str, Any]` | The structured details dictionary |
| `severity` | `str` | Severity level from error metadata (critical, error, warning, info, debug) |
| `category` | `str` | Error category from metadata (data, processing, validation, etc.) |
| `retry_allowed` | `bool` | Whether the operation should be retried |
| `user_facing` | `bool` | Whether the error should be shown to end users |

**How It Works:**
1. Retrieves error metadata via `get_error_metadata(self.error_code)`
2. Builds dictionary with error information and metadata
3. Includes severity, category, and retry information for structured logging

**Example:**
```python
error = BasePamolaError(
    message="Data validation failed",
    error_code="DATA_VALIDATION_ERROR",
    details={"field": "age", "reason": "negative value"}
)

result = error.to_dict()
# {
#     "error_type": "BasePamolaError",
#     "message": "Data validation failed",
#     "error_code": "DATA_VALIDATION_ERROR",
#     "details": {"field": "age", "reason": "negative value"},
#     "severity": "error",
#     "category": "data",
#     "retry_allowed": False,
#     "user_facing": True
# }
```

### `__repr__() -> str`

Returns a string representation of the exception.

**Format:**
```
{ClassName}(code={error_code}, message={first_50_chars}...)
```

**Example:**
```python
error = BasePamolaError("This is a long error message", error_code="TEST_ERROR")
repr(error)  # "BasePamolaError(code=TEST_ERROR, message=This is a long error message...)"
```

## Usage Examples

### Basic Error Creation

```python
from pamola_core.errors import BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode

# Create and raise a simple error
error = BasePamolaError(
    message="Operation failed",
    error_code=ErrorCode.PROCESSING_FAILED,
    details={"operation": "transform", "record_count": 1000}
)
raise error
```

### Error with Rich Debugging Context

```python
import json
from datetime import datetime

error = BasePamolaError(
    message="Failed to load data from source",
    error_code=ErrorCode.DATA_LOAD_FAILED,
    details={
        "source": "database://prod/users",
        "timestamp": datetime.now().isoformat(),
        "connection_timeout": 30,
        "retry_count": 3,
        "last_error": "timeout: no connection established"
    }
)

# Serialize for logging
error_dict = error.to_dict()
json_str = json.dumps(error_dict, default=str, indent=2)
```

### Catching and Inspecting Errors

```python
try:
    perform_operation()
except BasePamolaError as e:
    # Access structured information
    print(f"Error Code: {e.error_code}")
    print(f"Message: {e.message}")
    print(f"Details: {e.details}")

    # Serialize for storage/transmission
    error_data = e.to_dict()

    # Check severity for alerting
    if error_data["severity"] == "critical":
        send_alert(error_data)
```

### Subclassing for Domain-Specific Errors

```python
from pamola_core.errors import BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode

class DataSourceError(BasePamolaError):
    """Error connecting to a data source."""

    def __init__(self, source_name: str, reason: str):
        super().__init__(
            message=f"Failed to connect to source '{source_name}': {reason}",
            error_code=ErrorCode.DATA_SOURCE_INVALID,
            details={
                "source_name": source_name,
                "reason": reason,
                "error_type": self.__class__.__name__
            }
        )

# Usage
raise DataSourceError("postgresql://prod", "Connection refused")
```

### Chaining Errors

```python
import traceback

try:
    load_csv("data.csv")
except FileNotFoundError as original:
    error = BasePamolaError(
        message=f"Could not load CSV file: {original}",
        error_code=ErrorCode.FILE_NOT_FOUND,
        details={
            "file_path": "data.csv",
            "original_error": str(original),
            "traceback": traceback.format_exc()
        }
    )
    raise error from original
```

## Best Practices

### 1. Always Provide Error Code

Use codes from the `ErrorCode` registry to ensure standardization and enable metadata lookups.

```python
# Good
error = BasePamolaError(
    message="Data validation failed",
    error_code=ErrorCode.DATA_VALIDATION_ERROR  # From registry
)

# Avoid - no error code
error = BasePamolaError(message="Data validation failed")
```

### 2. Use Meaningful Message Text

Messages should explain what happened in language developers will understand, not cryptic codes.

```python
# Good - descriptive
message = "Failed to load CSV file 'sales.csv': File does not exist"

# Avoid - too generic
message = "Operation failed"
```

### 3. Populate Details with Context

Include all information needed for debugging, not just the immediate error.

```python
# Good - rich context
details = {
    "file_path": "/data/sales.csv",
    "expected_columns": ["id", "name", "amount"],
    "actual_columns": ["id", "name"],
    "row_count": 0,
    "encoding": "utf-8"
}

# Avoid - sparse details
details = {}
```

### 4. Don't Expose Sensitive Information

Filter out passwords, API keys, and personal data from details.

```python
# Good - sensitive data removed
details = {
    "database": "prod_database",
    "connection_timeout": 30,
    "error": "Connection failed"
    # password, API key NOT included
}

# Avoid - sensitive data exposed
details = {
    "connection_string": "postgresql://user:password123@prod:5432/db"
}
```

### 5. Use Exception Subclasses

Create domain-specific subclasses to simplify error handling and add custom logic.

```python
from pamola_core.errors import BasePamolaError

class ValidationError(BasePamolaError):
    """Base validation error for the module."""
    pass

class FieldValidationError(ValidationError):
    """Error validating a field."""
    pass

# Usage
try:
    validate_email(email)
except FieldValidationError as e:
    # Handle field-level validation errors
    pass
except ValidationError as e:
    # Handle other validation errors
    pass
```

## Troubleshooting

### Issue: Error Code Not Found in Metadata

**Symptom:** `to_dict()` returns `None` for severity/category fields.

**Cause:** Error code doesn't exist in `ErrorCode` registry or metadata files.

**Solution:**
```python
# Verify code exists
from pamola_core.errors.codes.registry import ErrorCode
print(hasattr(ErrorCode, "MY_ERROR"))  # Should be True

# Use existing codes from registry
raise BasePamolaError(
    message="Error",
    error_code=ErrorCode.PROCESSING_FAILED  # Use registered codes
)
```

### Issue: Large Details Dictionary

**Symptom:** Memory usage high due to large details dictionary.

**Cause:** Storing entire DataFrames or large objects in details.

**Solution:**
```python
# Good - store only summaries
details = {
    "row_count": 1000,
    "column_count": 50,
    "memory_usage_mb": 256,
    "dtypes": ["int64", "float64", "object"]
}

# Avoid - storing entire DataFrame
details = {
    "dataframe": df  # Don't do this
}
```

### Issue: Message Parameter Formatting

**Symptom:** Message doesn't format correctly in `to_dict()`.

**Cause:** Message contains literal `{` or `}` that shouldn't be formatted.

**Solution:**
```python
# Good - escape braces
message = "Invalid JSON: {{missing 'name' key}}"

# Or use raw string
message = r"Regex pattern {.*} is invalid"

# Or use %s formatting
message = f"Missing field: {field_name}"
```

## Related Components

- **ErrorCode** (`pamola_core.errors.codes.registry`) - Standardized error codes
- **ErrorMessages** (`pamola_core.errors.messages.registry`) - Message templates
- **ErrorHandler** (`pamola_core.errors.error_handler`) - Centralized error handling with validation
- **auto_exception** (`pamola_core.errors.base`) - Decorator to auto-generate exception classes
- **OperationResult** (`pamola_core.utils.ops.op_result`) - Container that ErrorHandler populates from exceptions
- **ErrorContext** (`pamola_core.errors.context.suggestions`) - Recovery suggestions for error codes
