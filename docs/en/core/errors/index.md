# PAMOLA Core Errors Module - Complete Documentation

**Location:** `pamola_core.errors`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Contents

This directory contains comprehensive documentation for PAMOLA's error handling framework.

### Core Documentation Files

1. **[errors_overview.md](./errors_overview.md)** - START HERE
   - Overview of the entire errors module
   - Key features and architecture
   - Error hierarchy and organization
   - Quick start patterns
   - Best practices and related components

2. **[base_pamola_error.md](./base_pamola_error.md)** - Foundation
   - `BasePamolaError` class documentation
   - Constructor and methods
   - Structured error serialization via `to_dict()`
   - Exception hierarchy
   - Error creation patterns

3. **[auto_exception.md](./auto_exception.md)** - Exception Generation
   - `auto_exception` decorator documentation
   - Eliminates boilerplate for exception classes
   - Message formatting and parameter collection
   - Custom message builders
   - Exception hierarchy support

4. **[task_initialization_error.md](./task_initialization_error.md)** - Task Errors
   - `TaskInitializationError` class documentation
   - When to use and common causes
   - Integration with task framework
   - Real-world usage patterns

5. **[error_codes.md](./error_codes.md)** - Error Code Reference
   - Complete registry of all error codes
   - Naming conventions and categories
   - Error code organization (data, validation, processing, etc.)
   - Guidelines for using and adding new codes

6. **[error_handler.md](./error_handler.md)** - Centralized Handling
   - `ErrorHandler` class documentation
   - Methods: `handle_error()`, `create_error()`, `standardize_result()`
   - Decorators: `capture_errors()`
   - Context manager: `error_context()`
   - Integration patterns

## Learning Path

### For Newcomers
1. Start with [errors_overview.md](./errors_overview.md) for conceptual understanding
2. Read [base_pamola_error.md](./base_pamola_error.md) to understand the foundation
3. Review [error_codes.md](./error_codes.md) to learn available error codes
4. Check [error_handler.md](./error_handler.md) for practical usage patterns

### For Exception Creation
1. [auto_exception.md](./auto_exception.md) - Creating custom exception classes
2. [base_pamola_error.md](./base_pamola_error.md) - Subclassing and inheritance
3. [error_codes.md](./error_codes.md) - Choosing appropriate error codes

### For Error Handling
1. [error_handler.md](./error_handler.md) - Centralized error handling
2. [errors_overview.md](./errors_overview.md) - Best practices
3. [task_initialization_error.md](./task_initialization_error.md) - Real-world example

## Quick Reference

### Creating Exceptions

**Option 1: Manual (Direct Instantiation)**
```python
from pamola_core.errors import BasePamolaError
from pamola_core.errors.codes.registry import ErrorCode

raise BasePamolaError(
    message="Operation failed",
    error_code=ErrorCode.PROCESSING_FAILED,
    details={"operation": "transform"}
)
```

**Option 2: Using auto_exception Decorator**
```python
@auto_exception(
    default_error_code=ErrorCode.DATA_LOAD_FAILED,
    message_params=["source", "reason"]
)
class DataLoadError(BasePamolaError):
    pass

raise DataLoadError(source="file.csv", reason="File not found")
```

### Handling Errors

**Option 1: Using ErrorHandler**
```python
handler = ErrorHandler(logger, operation_name="loading")

try:
    load_data()
except Exception as e:
    result = handler.handle_error(
        error=e,
        error_code=ErrorCode.DATA_LOAD_FAILED,
        raise_error=True
    )
```

**Option 2: Using Decorator**
```python
@handler.capture_errors(ErrorCode.DATA_LOAD_FAILED, rethrow=True)
def load_data():
    return fetch_data()
```

**Option 3: Using Context Manager**
```python
with handler.error_context(ErrorCode.DATA_LOAD_FAILED):
    load_data()
```

## Module Architecture

```
pamola_core.errors/
├── base.py
│   ├── BasePamolaError      # Root exception class
│   └── auto_exception       # Decorator for auto-generating __init__
├── codes/
│   ├── registry.py          # ErrorCode constants
│   ├── metadata.py          # Error metadata (severity, category, etc.)
│   └── utils.py             # Code validation utilities
├── messages/
│   ├── registry.py          # ErrorMessages templates
│   └── utils.py             # Message formatting utilities
├── context/
│   ├── suggestions.py       # ErrorContext for recovery suggestions
│   └── recovery_data.yaml   # Recovery suggestions data
├── exceptions/
│   ├── validation.py        # Validation error classes
│   ├── tasks.py             # Task execution error classes
│   ├── crypto.py            # Cryptography error classes
│   ├── core_domain.py       # Core domain error classes
│   ├── resources.py         # Resource management error classes
│   ├── nlp.py               # NLP/LLM error classes
│   └── filesystem.py        # Filesystem error classes
├── error_handler.py         # ErrorHandler class
└── __init__.py              # Public API exports
```

## Key Components Summary

| Component | Module | Purpose |
|-----------|--------|---------|
| **BasePamolaError** | `base.py` | Root exception class with error code, message, details |
| **auto_exception** | `base.py` | Decorator to generate exception `__init__` methods |
| **ErrorCode** | `codes/registry.py` | Centralized error code registry |
| **ErrorMessages** | `messages/registry.py` | Message template registry |
| **ErrorHandler** | `error_handler.py` | Centralized error handling and logging |
| **ErrorContext** | `context/suggestions.py` | Recovery suggestions for error codes |
| **Validation Exceptions** | `exceptions/validation.py` | Field, parameter, file validation errors |
| **Task Exceptions** | `exceptions/tasks.py` | Task execution lifecycle errors |
| **Crypto Exceptions** | `exceptions/crypto.py` | Encryption/decryption errors |

## Common Tasks

### Raise a Domain-Specific Error
See: [auto_exception.md](./auto_exception.md) → Usage Examples

### Get Error Metadata
See: [error_codes.md](./error_codes.md) → Code Usage Guidelines

### Handle Multiple Validation Errors
See: [base_pamola_error.md](./base_pamola_error.md) → Best Practices

### Add Recovery Suggestions
See: [error_handler.md](./error_handler.md) → Core Methods

### Create Custom Exception Class
See: [auto_exception.md](./auto_exception.md) → Decorator Parameters

### Debug Error Codes
See: [error_codes.md](./error_codes.md) → Troubleshooting

## Error Code Categories

The complete error code registry includes codes for:

- **Data Operations** (7 codes) - Loading, validation, writing
- **Field Validation** (4 codes) - Field existence, type, value checks
- **Parameter Validation** (6 codes) - Parameter validity
- **File Operations** (5 codes) - File access and format checks
- **Processing** (6 codes) - Operation execution
- **Resources** (4 codes) - Memory, disk, CPU
- **Caching** (5 codes) - Cache operations
- **Cryptography** (18+ codes) - Encryption, keys, hashing
- **Task Execution** (18+ codes) - Task lifecycle
- **NLP & LLM** (9+ codes) - Language model operations
- **Filesystem** (3+ codes) - Path validation

See [error_codes.md](./error_codes.md) for complete listing.

## Integration Points

The errors module integrates with:

- **OperationResult** - ErrorHandler converts exceptions to results
- **BaseOperation** - Operations use ErrorHandler for error handling
- **BaseTask** - Tasks wrap operations with error handling
- **Logging** - Structured logging via ErrorHandler
- **Metrics** - Error metadata captured in result metrics

## Best Practices Summary

1. **Use ErrorCode Registry** - Never hardcode error code strings
2. **Leverage ErrorHandler** - Centralized, consistent error handling
3. **Create Domain-Specific Exceptions** - Use auto_exception for custom classes
4. **Include Rich Details** - Provide debugging context in details dict
5. **Validate Before Raising** - Check parameters and dependencies
6. **Use Message Templates** - Let ErrorMessages format messages
7. **Distinguish Error Phases** - Use appropriate exception types
8. **Handle Errors Consistently** - Follow established patterns

## Related Documentation

- `../../project-overview-pdr.md` - Project overview and requirements
- `../../code-standards.md` - Code style and standards
- `../../system-architecture.md` - System architecture
- `../../../README.md` - Project README

## Support & Troubleshooting

For specific issues:
- Error code not found → See [error_codes.md](./error_codes.md) → Troubleshooting
- Message formatting issues → See [auto_exception.md](./auto_exception.md) → Troubleshooting
- ErrorHandler problems → See [error_handler.md](./error_handler.md) → Troubleshooting
- Exception creation → See [base_pamola_error.md](./base_pamola_error.md) → Troubleshooting

## Version History

**v1.0 (2026-03-23)** - Initial comprehensive documentation
- Complete API documentation for all error classes
- Usage examples and best practices
- Complete error code reference
- ErrorHandler integration guide
