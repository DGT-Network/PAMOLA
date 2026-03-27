# TaskInitializationError Documentation

**Module:** `pamola_core.errors.exceptions.tasks`
**Class:** `TaskInitializationError`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Class Signature](#class-signature)
3. [Constructor Parameters](#constructor-parameters)
4. [Usage Examples](#usage-examples)
5. [Common Causes](#common-causes)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Exceptions](#related-exceptions)

## Overview

`TaskInitializationError` is raised when a task fails to initialize properly. It indicates that the task setup phase encountered an error before execution could begin. This is one of the most common exceptions in PAMOLA's task execution framework.

**Key Characteristics:**
- Inherits from `BasePamolaError`
- Generated using the `auto_exception` decorator
- Default error code: `ErrorCode.TASK_INIT_FAILED`
- Message parameters: `task_name`, `reason`
- Part of the task execution exception hierarchy

## Class Signature

```python
@auto_exception(
    default_error_code=ErrorCode.TASK_INIT_FAILED,
    message_params=["task_name", "reason"],
    detail_params=["task_name", "reason"]
)
class TaskInitializationError(BasePamolaError):
    """Exception raised when task initialization fails."""
    pass
```

**Decorator Behavior:**
- Auto-generates `__init__` with message formatting
- Formats message from `ErrorMessages.TASK_INIT_FAILED` template
- Includes task_name and reason in details dictionary
- Allows error_code override if needed

## Constructor Parameters

The decorator creates an `__init__` method with these parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | `Optional[str]` | No | Explicit message. If provided, overrides template formatting. If `None`, uses template |
| `task_name` | `str` (kwarg) | Yes | Name of the task that failed to initialize |
| `reason` | `str` (kwarg) | Yes | Specific reason for initialization failure |
| `error_code` | `Optional[str]` (kwarg) | No | Override the default `TASK_INIT_FAILED` code |
| `details` | `Optional[Dict[str, Any]]` (kwarg) | No | Additional context to merge with task_name and reason |

## Usage Examples

### Basic Usage

```python
from pamola_core.errors.exceptions.tasks import TaskInitializationError

try:
    task = Task(name="load_data", config={})
except Exception as e:
    raise TaskInitializationError(
        task_name="load_data",
        reason=f"Missing required configuration: {e}"
    )
```

### With Configuration Context

```python
config = {
    "source": "database://prod",
    "timeout": 30
}

try:
    task = DataLoadTask(config=config)
except KeyError as e:
    raise TaskInitializationError(
        task_name="DataLoadTask",
        reason=f"Missing config key: {e}",
        error_code="TASK_INIT_FAILED",  # Can override if needed
        details={
            "provided_keys": list(config.keys()),
            "error_type": type(e).__name__
        }
    )
```

### Catching and Inspecting

```python
try:
    task = setup_task("transform_data", config)
except TaskInitializationError as e:
    print(f"Task: {e.details.get('task_name')}")
    print(f"Reason: {e.details.get('reason')}")
    print(f"Message: {e.message}")

    # Serialize for logging
    error_dict = e.to_dict()
    logger.error(f"Task init failed: {error_dict}")
```

### In Task Factory Pattern

```python
from pamola_core.errors.exceptions.tasks import TaskInitializationError

class TaskFactory:
    def create_task(self, task_type: str, config: dict):
        try:
            if task_type == "load":
                return DataLoadTask(config)
            elif task_type == "transform":
                return TransformTask(config)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except ValueError as e:
            raise TaskInitializationError(
                task_name=task_type,
                reason=str(e)
            )
        except Exception as e:
            raise TaskInitializationError(
                task_name=task_type,
                reason=f"Initialization failed: {e}",
                details={
                    "exception_type": type(e).__name__,
                    "config_keys": list(config.keys())
                }
            )

# Usage
factory = TaskFactory()
try:
    task = factory.create_task("load", {"source": "data.csv"})
except TaskInitializationError as e:
    logger.error(f"Failed to create task: {e.message}")
```

### With ErrorHandler Integration

```python
import logging
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.codes.registry import ErrorCode
from pamola_core.errors.exceptions.tasks import TaskInitializationError

logger = logging.getLogger(__name__)
handler = ErrorHandler(logger, operation_name="task_setup")

def setup_task(task_name: str, config: dict):
    try:
        # Initialize task with validation
        task = initialize_task(task_name, config)
        return task
    except TaskInitializationError as e:
        # Let handler manage the error
        result = handler.handle_error(
            error=e,
            error_code=ErrorCode.TASK_INIT_FAILED,
            context={"task_name": task_name, "config_size": len(config)},
            raise_error=True
        )
        # Error is re-raised, but also logged with context
    except Exception as e:
        # Convert other exceptions to TaskInitializationError
        raise TaskInitializationError(
            task_name=task_name,
            reason=f"Unexpected error during initialization: {e}"
        )
```

### Decorator-Based Error Handling

```python
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.codes.registry import ErrorCode

handler = ErrorHandler(logger, operation_name="task_init")

@handler.capture_errors(
    error_code=ErrorCode.TASK_INIT_FAILED,
    rethrow=True
)
def create_task(task_name: str, config: dict):
    """Setup task - any exception becomes TaskInitializationError."""
    return Task(name=task_name, config=config)

# Usage
try:
    task = create_task("load_data", config)
except TaskInitializationError as e:
    print(f"Failed: {e.message}")
```

### Validating Configuration Before Raising

```python
def validate_and_initialize(task_name: str, config: dict):
    """Validate config and initialize task."""
    required_keys = ["source", "destination"]
    missing_keys = [k for k in required_keys if k not in config]

    if missing_keys:
        raise TaskInitializationError(
            task_name=task_name,
            reason=f"Missing required configuration keys: {', '.join(missing_keys)}",
            details={
                "required_keys": required_keys,
                "provided_keys": list(config.keys()),
                "missing_keys": missing_keys
            }
        )

    # If validation passes, initialize
    return Task(name=task_name, config=config)
```

## Common Causes

| Cause | Example | Solution |
|-------|---------|----------|
| **Missing Configuration** | Task requires "source" but config is empty | Validate config keys before initialization |
| **Invalid Config Type** | Expected dict, got string | Type-check configuration |
| **Invalid Dependencies** | Required module not installed | Check dependency availability |
| **Permission Denied** | Cannot access config file | Check file permissions |
| **Resource Unavailable** | Database connection failed | Verify connectivity before init |
| **Invalid Task Name** | Task type not registered | Check against available task types |
| **Conflicting Config** | Two incompatible config options | Validate config compatibility |
| **Unsupported Version** | Task requires newer version | Check version compatibility |

## Best Practices

### 1. Provide Specific Reasons

Include details about what went wrong, not just "initialization failed":

```python
# Good - specific reason
raise TaskInitializationError(
    task_name="load_data",
    reason="Missing required config key 'source_path'"
)

# Avoid - vague reason
raise TaskInitializationError(
    task_name="load_data",
    reason="Configuration error"
)
```

### 2. Include Validation Context in Details

Add information useful for debugging beyond what's in the message:

```python
raise TaskInitializationError(
    task_name="transform_task",
    reason="Invalid transformation strategy",
    details={
        "provided_strategy": config.get("strategy"),
        "valid_strategies": ["drop", "mean_impute", "forward_fill"],
        "strategy_type": type(config.get("strategy")).__name__
    }
)
```

### 3. Catch All Initialization Exceptions

Wrap initialization code to catch unexpected errors:

```python
try:
    # Initialization code that might fail
    validate_config(config)
    load_dependencies(task_name)
    initialize_resources(task_name)
    return create_task_instance(task_name, config)
except TaskInitializationError:
    # Re-raise expected errors as-is
    raise
except Exception as e:
    # Wrap unexpected errors
    raise TaskInitializationError(
        task_name=task_name,
        reason=f"Unexpected error: {type(e).__name__}: {e}"
    )
```

### 4. Use in Task Factory Pattern

Leverage in factory methods for consistent error handling:

```python
class TaskFactory:
    task_types = {
        "load": DataLoadTask,
        "transform": TransformTask,
        "validate": ValidationTask
    }

    @classmethod
    def create(cls, task_type: str, config: dict):
        if task_type not in cls.task_types:
            raise TaskInitializationError(
                task_name=task_type,
                reason=f"Unsupported task type",
                details={"supported_types": list(cls.task_types.keys())}
            )

        try:
            return cls.task_types[task_type](config)
        except Exception as e:
            raise TaskInitializationError(
                task_name=task_type,
                reason=f"Failed to instantiate: {e}"
            )
```

### 5. Distinguish from Execution Errors

Use `TaskInitializationError` for setup issues, not runtime failures:

```python
# Good - initialization phase
if "source" not in config:
    raise TaskInitializationError(
        task_name="load",
        reason="Missing source config"
    )

# Run task
result = task.execute()

# Different error - execution phase
if result.status == "failed":
    raise TaskExecutionError(
        task_name="load",
        reason="Execution timeout"
    )
```

## Troubleshooting

### Issue: Error Code Not Found

**Symptom:** `to_dict()` returns `None` for severity/category.

**Cause:** Using custom error code not in `ErrorCode` registry.

**Solution:**
```python
# Use registered error codes
raise TaskInitializationError(
    task_name="load_data",
    reason="Config missing",
    error_code=ErrorCode.TASK_INIT_FAILED  # From registry
)
```

### Issue: Message Not Formatting

**Symptom:** Message contains `{task_name}` instead of actual task name.

**Cause:** Not passing task_name as keyword argument.

**Solution:**
```python
# Correct - keyword arguments
raise TaskInitializationError(
    task_name="my_task",
    reason="Failed"
)

# Wrong - positional arguments
raise TaskInitializationError("my_task", "Failed")  # Won't work
```

### Issue: Details Lost

**Symptom:** Exception raised but details are empty in to_dict().

**Cause:** Not passing details or forgotten parameters.

**Solution:**
```python
# Include all relevant debugging info
raise TaskInitializationError(
    task_name=task_name,
    reason=error_msg,
    details={
        "config": config,
        "available_fields": list(config.keys()),
        "error_type": type(e).__name__
    }
)
```

## Related Exceptions

| Exception | When to Use | Relationship |
|-----------|------------|--------------|
| **TaskExecutionError** | During task execution | Raised after init succeeds |
| **TaskFinalizationError** | During task finalization | Raised during cleanup |
| **TaskDependencyError** | Dependency not satisfied | Can occur during init |
| **DependencyMissingError** | Required module missing | Specific cause of init failure |
| **InvalidParameterError** | Config parameter invalid | Common cause of init failure |
| **ValidationError** | Config validation fails | Can lead to init failure |

**Exception Hierarchy:**
```
BasePamolaError
├── TaskError
    ├── TaskInitializationError ← You are here
    ├── TaskExecutionError
    ├── TaskFinalizationError
    ├── TaskDependencyError
    └── ... (other task-related)
```

## Integration with Task Framework

`TaskInitializationError` is integrated with PAMOLA's task execution framework:

1. **BaseTask** (`pamola_core.utils.tasks.base_task`) - Initialization failures during `__init__` or `setup()`
2. **Task Registry** (`pamola_core.utils.tasks.registry`) - Registration failures when registering new task types
3. **Task Dependencies** - Checked during initialization, failures raise related errors
4. **ErrorHandler** - Converts init errors to structured OperationResult
5. **Checkpointing** - Init state checkpointed for recovery

**Typical Task Lifecycle:**
```
create_task() -> TaskInitializationError caught here
    ↓
setup() -> TaskInitializationError can occur here
    ↓
execute() -> TaskExecutionError occurs here
    ↓
finalize() -> TaskFinalizationError occurs here
```
