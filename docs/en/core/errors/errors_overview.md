# PAMOLA Errors Module Documentation

**Module:** `pamola_core.errors`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Error Hierarchy](#error-hierarchy)
6. [Quick Start](#quick-start)
7. [Best Practices](#best-practices)
8. [Related Components](#related-components)
9. [Summary Analysis](#summary-analysis)

## Overview

The `pamola_core.errors` module provides a comprehensive error handling framework for PAMOLA Core. It establishes a standardized error system with:

- **Centralized error codes** using `ErrorCode` registry for all errors across the framework
- **Standardized error messages** via `ErrorMessages` registry with template-based formatting
- **Structured exception hierarchy** with `BasePamolaError` as the root for all PAMOLA exceptions
- **Contextual error information** through metadata, recovery suggestions, and structured details
- **Unified error handling** via `ErrorHandler` for consistent logging, validation, and recovery support
- **Decorator-based exception generation** using `auto_exception` to eliminate boilerplate code

This module enables developers to create, raise, and handle errors consistently across the entire PAMOLA ecosystem while maintaining detailed diagnostic information for troubleshooting.

## Key Features

| Feature | Description |
|---------|-------------|
| **ErrorCode Registry** | Centralized string-based error codes with standardized naming (CATEGORY_SPECIFIC_ERROR) |
| **ErrorMessages Registry** | Technical message templates with parameter substitution for all error scenarios |
| **BasePamolaError** | Foundation exception class with error code, message, and structured details support |
| **auto_exception Decorator** | Eliminates boilerplate by auto-generating exception `__init__` methods with message formatting |
| **ErrorHandler** | Centralized error handling with validation, logging, suggestions, and OperationResult creation |
| **Recovery Suggestions** | Context-aware recovery suggestions from recovery_data.yaml loaded via ErrorContext |
| **OperationResult Integration** | Automatic conversion of errors to structured OperationResult for consistent API responses |
| **Structured Details** | All exceptions support details dictionary for rich contextual information beyond the message |
| **Type Safety** | Full type hints for runtime validation of error codes and message templates |
| **Exception Categories** | Organized exception hierarchy covering data, validation, processing, resources, crypto, NLP, and more |

## Architecture

The module follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│              User Code / Operations                          │
├─────────────────────────────────────────────────────────────┤
│ ErrorHandler (centralized error handling + validation)       │
├─────────────────────────────────────────────────────────────┤
│ Specialized Exceptions (task, validation, crypto, nlp, etc.) │
├─────────────────────────────────────────────────────────────┤
│ BasePamolaError (root exception class)                       │
├─────────────────────────────────────────────────────────────┤
│ Error Registries & Context                                   │
│ - ErrorCode (codes)                                          │
│ - ErrorMessages (templates)                                  │
│ - ErrorContext (suggestions)                                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Interactions:**
- `ErrorCode` and `ErrorMessages` are parallel registries: each error code has a corresponding message template
- `BasePamolaError` uses both registries to validate and format errors
- `auto_exception` decorator bridges specialized exceptions to the registries
- `ErrorHandler` validates against registries and provides recovery suggestions via `ErrorContext`
- `ErrorContext` loads recovery strategies from `recovery_data.yaml` on first access

## Core Components

### 1. ErrorCode Registry
**Location:** `pamola_core.errors.codes.registry`

String-based error code registry with standardized naming convention: `<CATEGORY>_<SPECIFIC_ERROR>`.

**Example Codes:**
```
DATA_LOAD_FAILED
FIELD_NOT_FOUND
PARAM_INVALID
TASK_INIT_FAILED
PROCESSING_FAILED
```

### 2. ErrorMessages Registry
**Location:** `pamola_core.errors.messages.registry`

Technical message templates with placeholders for parameter substitution.

**Example:**
```python
FIELD_NOT_FOUND = "Field '{field_name}' not found in data. Available fields: {available_fields}"
```

### 3. BasePamolaError
**Location:** `pamola_core.errors.base`

Root exception class for all PAMOLA exceptions. Supports structured error details and provides `to_dict()` for serialization.

### 4. auto_exception Decorator
**Location:** `pamola_core.errors.base`

Decorator that automatically generates exception `__init__` methods with message formatting, parameter collection, and details population.

### 5. ErrorHandler
**Location:** `pamola_core.errors.error_handler`

Centralized error handling providing:
- Error code validation
- Message template formatting
- Recovery suggestions
- OperationResult creation
- Structured logging
- Sync and async decorator support

### 6. ErrorContext
**Location:** `pamola_core.errors.context.suggestions`

Provides recovery suggestions loaded from `recovery_data.yaml`. Supports per-error-code and category-level suggestions with LRU caching.

## Error Hierarchy

PAMOLA exceptions are organized by domain:

| Category | Exceptions | Count |
|----------|-----------|-------|
| **Core Domain** | DataError, ProcessingError, CacheError, VisualizationError, etc. | 10 |
| **Cryptography** | EncryptionError, KeyGenerationError, PseudonymizationError, HashCollisionError, etc. | 19 |
| **Tasks** | TaskInitializationError, TaskExecutionError, DependencyError, MaxRetriesExceededError, etc. | 17 |
| **Validation** | FieldNotFoundError, FieldTypeError, InvalidParameterError, FileValidationError, etc. | 22 |
| **Resources** | ResourceError, DateTimeParsingError, FakeDataError, MappingError, etc. | 8 |
| **NLP** | NLPError, LLMError, ModelLoadError, PromptValidationError, etc. | 9 |
| **Filesystem** | PathValidationError, PathSecurityError, DirectoryManagerError | 3 |

**Root:** `BasePamolaError` (from `pamola_core.errors.base`)

All specialized exceptions inherit from `BasePamolaError` to ensure consistent error handling, logging, and telemetry across the framework.

## Quick Start

### Raising a Simple Error

```python
from pamola_core.errors import BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode

# Manually construct
raise BasePamolaError(
    message="Operation failed",
    error_code=ErrorCode.PROCESSING_FAILED,
    details={"operation": "data_transform", "record_count": 1000}
)
```

### Using auto_exception Decorator

```python
from pamola_core.errors.base import BasePamolaError, auto_exception
from pamola_core.errors.codes.registry import ErrorCode

@auto_exception(
    default_error_code=ErrorCode.DATA_LOAD_FAILED,
    message_params=["source", "reason"],
    detail_params=["source", "operation", "reason"]
)
class DataError(BasePamolaError):
    """Error during data loading operations."""
    pass

# Usage - message built from template automatically
raise DataError(source="database", reason="Connection timeout", operation="fetch_rows")
```

### Using ErrorHandler

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
        message_kwargs={"source": "file.csv", "reason": str(e)}
    )
    # result is OperationResult with error status and metadata
```

### Using Error Context Manager

```python
with handler.error_context(
    error_code=ErrorCode.DATA_LOAD_FAILED,
    context={"file": "data.csv"},
    suppress=False  # Re-raise after handling
):
    load_data()
```

### Decorator-Based Error Handling

```python
@handler.capture_errors(
    error_code=ErrorCode.PROCESSING_FAILED,
    rethrow=False
)
def process_data(data):
    return data.transform()
```

## Best Practices

### 1. Choose the Right Exception
Use domain-specific exceptions from `pamola_core.errors.exceptions`:
- `DataError` for data loading/processing failures
- `ValidationError` subclasses for validation failures
- `TaskInitializationError` for task setup failures
- `EncryptionError` for crypto operations

### 2. Include Error Context
Always populate the `details` parameter with relevant debugging information:
```python
raise DataError(
    message="Failed to load CSV",
    error_code=ErrorCode.DATA_LOAD_FAILED,
    details={
        "file_path": "/data/sales.csv",
        "row_count": 1000,
        "columns": ["id", "name", "amount"]
    }
)
```

### 3. Use ErrorHandler for Centralized Control
- Initialize once per module or class
- Enables consistent logging levels and recovery suggestions
- Automatically validates error codes
- Converts to OperationResult for API consistency

### 4. Leverage Message Templates
Rather than building messages manually, use message parameters:
```python
# Good: Uses template from ErrorMessages
raise ValidationError(
    message=ErrorMessages.format(
        ErrorCode.FIELD_NOT_FOUND,
        field_name="user_id",
        available_fields="id, name, email"
    )
)

# Also good: Let auto_exception handle it
raise FieldNotFoundError(field_name="user_id", available_fields=[...])
```

### 5. Structured Error Details
Use `to_dict()` to serialize errors for logging/APIs:
```python
error = BasePamolaError("msg", error_code=ErrorCode.PROCESSING_FAILED)
serialized = error.to_dict()
# {
#   "error_type": "BasePamolaError",
#   "message": "msg",
#   "error_code": "PROCESSING_FAILED",
#   "details": {},
#   "severity": "error",
#   "category": "processing",
#   ...
# }
```

### 6. Multiple Validation Errors
Aggregate validation errors before raising:
```python
from pamola_core.errors.exceptions.validation import (
    MultipleValidationErrors,
    raise_if_errors
)

errors = []
if not field_exists:
    errors.append(FieldNotFoundError("user_id"))
if not type_valid:
    errors.append(FieldTypeError("age", "int", "str"))

raise_if_errors(errors)  # Raises MultipleValidationErrors with all errors grouped
```

### 7. Recovery Suggestions
ErrorHandler automatically includes recovery suggestions from recovery_data.yaml:
```python
result = handler.handle_error(error, ErrorCode.DATA_LOAD_FAILED)
# Recovery suggestions automatically included in result metrics and logs
```

## Related Components

- **OperationResult** (`pamola_core.utils.ops.op_result`) - Status container that ErrorHandler populates on errors
- **BaseOperation** (`pamola_core.utils.ops.op_base`) - Operations use ErrorHandler for consistent error handling
- **BaseTask** (`pamola_core.utils.tasks.base_task`) - Task execution wraps operations with error handling
- **Validation Module** (`pamola_core.errors.exceptions.validation`) - Comprehensive validation error types
- **Crypto Module** (`pamola_core.errors.exceptions.crypto`) - Encryption/decryption specific errors
- **Task Module** (`pamola_core.errors.exceptions.tasks`) - Task execution and dependency errors

## Summary Analysis

The `pamola_core.errors` module is a sophisticated error handling framework that:

1. **Standardizes** error codes and messages across the entire framework for consistency
2. **Reduces boilerplate** through auto_exception decorator and template-based messages
3. **Enables visibility** with structured error details, recovery suggestions, and logging
4. **Ensures reliability** through error validation and OperationResult integration
5. **Supports both** synchronous and asynchronous error handling via decorators
6. **Integrates deeply** with the operation/task framework via ErrorHandler

The framework encourages developers to use domain-specific exceptions with the ErrorHandler pattern, resulting in maintainable, debuggable, and user-friendly error handling across PAMOLA Core.
