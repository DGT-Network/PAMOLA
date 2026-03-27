# TaskRunner Documentation

**Module:** `pamola_core.utils.tasks.task_runner`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Classes](#core-classes)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)
10. [Summary](#summary)

## Overview

The `TaskRunner` class extends `BaseTask` to provide comprehensive operation sequence management and execution. It orchestrates privacy-enhancing operations with support for task configuration, operation registration, field-specific scoping, encryption, and reproducibility control.

TaskRunner is the primary interface for executing sequences of anonymization, profiling, transformation, and metric operations on datasets. It handles the complete lifecycle including configuration validation, operation setup, execution with checkpointing, and result collection.

## Key Features

- **Operation Sequence Management**: Define and execute ordered operation sequences
- **Flexible Configuration**: Support for field-specific and dataset-level operations
- **Encryption Support**: Integrated encryption/decryption for sensitive data
- **Reproducibility Control**: Deterministic operation execution via TaskContext seed propagation
- **Operation Classification**: Built-in knowledge of operation categories and requirements
- **Progress Tracking**: Visual progress tracking throughout execution
- **Error Handling**: Graceful error handling with continue-on-error support
- **Checkpoint Support**: Resume capability from operation checkpoints
- **Artifact Management**: Automatic collection of operation outputs
- **Metric Aggregation**: Comprehensive metrics collection across operations

## Architecture

```
TaskRunner (extends BaseTask)
  ├── Operation Management
  │   ├── operations_sequence: predefined execution order
  │   ├── operations_always_output_finals: operations producing final outputs
  │   ├── function_maps: operations supporting field mapping
  │   └── operations_not_include_field_name: dataset-level operations
  │
  ├── Configuration
  │   ├── operation_configs: user-provided operation configs
  │   ├── additional_options: execution parameters
  │   ├── data_types: per-dataset data types
  │   └── encryption_keys: per-dataset encryption keys
  │
  ├── Reproducibility (FR-EP3-CORE-043)
  │   └── TaskContext: manages seed propagation
  │
  ├── Execution Pipeline
  │   ├── configure_operations(): parse configs
  │   ├── _initialize_data_source(): load input datasets
  │   ├── _run_operations(): execute sequence
  │   └── _prepare_operation_parameters(): inject parameters
  │
  └── Tracking
      ├── lst_result: operation result metadata
      ├── lst_final_output: output dataset paths
      ├── results: operation results
      └── metrics: aggregated metrics
```

## Dependencies

| Module | Purpose |
|--------|---------|
| `BaseTask` | Base class for task management |
| `TaskContext` | Seed propagation for reproducibility (FR-EP3-CORE-043) |
| `DataSource` | Data source management |
| `All operation classes` | Anonymization, profiling, transformation, metrics operations |
| `OperationStatus` | Status tracking for operations |
| `OperationResult` | Operation result wrapper |
| `logging` | Logging infrastructure |

## Core Classes

### TaskRunner

Orchestrates execution of a sequence of operations on datasets.

#### Constructor

```python
def __init__(
    self,
    task_id: str,
    task_type: str,
    description: str,
    input_datasets: Optional[Dict[str, str]] = None,
    data_types: Optional[Dict[str, Any]] = None,
    auxiliary_datasets: Optional[Dict[str, str]] = None,
    operation_configs: Optional[List[Dict[str, Any]]] = None,
    additional_options: Optional[Dict[str, Any]] = None,
    use_encryption: Optional[bool] = False,
    encryption_keys: Optional[Dict[str, str]] = None,
    task_dir: Optional[str] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_id` | str | Required | Unique task identifier |
| `task_type` | str | Required | Task type (anonymization, profiling, etc.) |
| `description` | str | Required | Human-readable task description |
| `input_datasets` | Dict[str, str] | None | Map of dataset names to file paths |
| `data_types` | Dict[str, Any] | None | Data type metadata per dataset |
| `auxiliary_datasets` | Dict[str, str] | None | Auxiliary data files (lookups, etc.) |
| `operation_configs` | List[Dict] | None | List of operation configurations |
| `additional_options` | Dict[str, Any] | None | Execution options (seed, encoding, etc.) |
| `use_encryption` | bool | False | Enable encryption support |
| `encryption_keys` | Dict[str, str] | None | Encryption keys per dataset |
| `task_dir` | str | None | Override task directory path |

**Initialization Behavior:**

1. Calls parent `BaseTask.__init__()` to set up infrastructure
2. Initializes operation tracking lists
3. Creates `TaskContext` with seed from additional_options
4. Registers all known operation classes
5. Initializes result and metric tracking
6. Stores operation configurations and options

#### Methods

##### configure_operations()

```python
def configure_operations(self) -> bool
```

**Purpose:** Parse operation configurations and register operations for execution.

**Returns:** True if all operations configured successfully, False if errors

**Process:**

1. Iterates through `operation_configs`
2. For each operation:
   - Extracts class name, parameters, scope, and target fields
   - **Injects per-operation seed** (FR-EP3-CORE-043)
   - Validates operation is in function_maps
   - For field-scoped operations: creates one op per target field
   - For dataset-level ops: creates single operation
   - Records metadata in `lst_result`

**Error Handling:**
- Raises exception if operation class not found
- Raises exception if operation registration fails
- Logs detailed error messages for debugging

**Example Configuration:**

```python
operation_configs = [
    {
        "operation": "anonymization",
        "class_name": "RandomMaskingOperation",
        "task_operation_id": "op_001",
        "task_operation_order_index": 0,
        "dataset_name": "main",
        "parameters": {"masking_char": "*"},
        "scope": {
            "type": "field",
            "target": ["email", "phone"]  # Field-specific
        }
    },
    {
        "operation": "analysis",
        "class_name": "DataAttributeProfilerOperation",
        "task_operation_id": "op_002",
        "task_operation_order_index": 1,
        "dataset_name": "main",
        "parameters": {},
        "scope": {"type": "dataset"}  # Dataset-level
    }
]

runner = TaskRunner(..., operation_configs=operation_configs)
runner.configure_operations()
```

##### _run_operations()

```python
def _run_operations(self, start_idx: int = 0) -> bool
```

**Purpose:** Execute registered operations in sequence with error handling and checkpointing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_idx` | int | 0 | Starting operation index (for resume) |

**Returns:** True if all operations succeeded, False on critical error

**Execution Flow:**

1. Retrieves execution options (encoding, delimiter, etc.)
2. Iterates through operations starting from `start_idx`
3. For each operation:
   - Prepares operation parameters
   - Sets dataset name based on operation type
   - Calls `operation_executor.execute_with_retry()`
   - Collects artifacts and metrics
   - Creates checkpoint
   - Handles errors per `continue_on_error` policy
4. Updates final dataset tracking
5. Records result status

**Dataset Tracking:**
- Maintains `final_dataset_name` and `final_dataset_path`
- Updates for operations in `operations_always_output_finals`
- Tracks `lst_final_output` for all final outputs

**Error Handling:**
- Catches `KeyboardInterrupt` separately for proper shutdown
- Logs operation errors with context
- Returns False on critical errors
- Continues on non-critical errors if configured
- Sets `error_info` and `status` appropriately

**Checkpointing:**
- Creates automatic checkpoint after each operation
- Stores operation index and metrics
- Enables resume capability

##### _initialize_data_source()

```python
def _initialize_data_source(self) -> None
```

**Purpose:** Load input and auxiliary datasets into DataSource.

**Process:**

1. Creates new `DataSource` instance
2. Processes `input_datasets`:
   - Validates and normalizes paths
   - Adds to data source
   - Registers encryption keys
   - Records data type metadata
3. Processes `auxiliary_datasets`:
   - Same validation and registration
   - Marked as auxiliary for read-only access
4. Validates encryption if enabled

**Error Handling:**
- Raises `TaskInitializationError` on validation failure
- Logs detailed path and encryption issues
- Shows progress during initialization

### Operation Classification Lists

TaskRunner maintains several lists for operation classification:

#### operations_sequence

Predefined execution order for operations that modify outputs:

```python
operations_sequence = [
    "FakeEmailOperation",
    "FakeNameOperation",
    "FakeOrganizationOperation",
    "FakePhoneOperation",
    "AggregateRecordsOperation",
    "ImputeMissingValuesOperation",
    "AddOrModifyFieldsOperation",
    "RemoveFieldsOperation",
    "CleanInvalidValuesOperation",
    "NumericGeneralizationOperation",
    "DateTimeGeneralizationOperation",
    "AttributeSuppressionOperation",
    "CategoricalGeneralizationOperation",
    "UniformTemporalNoiseOperation",
    "UniformNumericNoiseOperation",
    "RecordSuppressionOperation",
    "CellSuppressionOperation",
    "FullMaskingOperation",
    "PartialMaskingOperation",
]
```

#### operations_always_output_finals

Operations that always produce final outputs:

```python
operations_always_output_finals = [
    "MergeDatasetsOperation",
    "SplitByIDValuesOperation",
    "SplitFieldsOperation",
]
```

#### function_maps

All operations with configurable parameters:

```python
function_maps = [
    # All anonymization operations
    # All profiling operations
    # All transformation operations
    # All metric operations
    # All fake data operations
]
```

#### operations_not_include_field_name

Operations that work on entire datasets:

```python
operations_not_include_field_name = [
    "KAnonymityProfilerOperation",
    "DataAttributeProfilerOperation",
    "CorrelationOperation",
    # ... other dataset-level operations
]
```

## Usage Examples

### Example 1: Basic Anonymization Task

```python
from pamola_core.utils.tasks.task_runner import TaskRunner

# Define input datasets
input_datasets = {
    "main": "data/customers.csv"
}

# Define operations to execute
operation_configs = [
    {
        "operation": "anonymization",
        "class_name": "FullMaskingOperation",
        "task_operation_id": "op_001",
        "task_operation_order_index": 0,
        "dataset_name": "main",
        "parameters": {"masking_char": "X"},
        "scope": {
            "type": "field",
            "target": ["email", "phone"]
        }
    },
    {
        "operation": "anonymization",
        "class_name": "NumericGeneralizationOperation",
        "task_operation_id": "op_002",
        "task_operation_order_index": 1,
        "dataset_name": "main",
        "parameters": {"levels": 3},
        "scope": {
            "type": "field",
            "target": ["age", "income"]
        }
    }
]

# Create and execute task
runner = TaskRunner(
    task_id="anon_task_001",
    task_type="anonymization",
    description="Anonymize customer PII",
    input_datasets=input_datasets,
    operation_configs=operation_configs
)

# Configure operations
runner.configure_operations()

# Execute operations (inherited from BaseTask)
success = runner.run()

if success:
    print("Task completed successfully")
    print(f"Results: {runner.results}")
else:
    print(f"Task failed: {runner.error_info}")
```

### Example 2: Reproducible Task with Seed

```python
# Use fixed seed for reproducibility
runner = TaskRunner(
    task_id="reproducible_anon_001",
    task_type="anonymization",
    description="Reproducible anonymization",
    input_datasets={"main": "data/input.csv"},
    operation_configs=operation_configs,
    additional_options={
        "seed": 42,  # Enable deterministic execution
        "encoding": "utf-8",
        "sep": ","
    }
)

# Run task - same input + same seed = same output
result = runner.run()

# Run again with same configuration
runner2 = TaskRunner(
    task_id="reproducible_anon_002",
    task_type="anonymization",
    description="Reproducible anonymization",
    input_datasets={"main": "data/input.csv"},
    operation_configs=operation_configs,
    additional_options={"seed": 42}
)

result2 = runner2.run()

# Results should be identical
assert results_are_identical(result, result2)  # True with seed=42
```

### Example 3: Complex Multi-Operation Task

```python
# Profiling + Anonymization + Metrics
operation_configs = [
    # Profiling
    {
        "operation": "analysis",
        "class_name": "DataAttributeProfilerOperation",
        "task_operation_id": "prof_001",
        "task_operation_order_index": 0,
        "dataset_name": "main",
        "parameters": {},
        "scope": {"type": "dataset"}
    },
    # Anonymization
    {
        "operation": "anonymization",
        "class_name": "FullMaskingOperation",
        "task_operation_id": "anon_001",
        "task_operation_order_index": 1,
        "dataset_name": "main",
        "parameters": {"masking_char": "*"},
        "scope": {
            "type": "field",
            "target": ["ssn", "email"]
        }
    },
    # Metrics
    {
        "operation": "metrics",
        "class_name": "PrivacyMetricOperation",
        "task_operation_id": "metric_001",
        "task_operation_order_index": 2,
        "dataset_name": "main",
        "parameters": {"privacy_level": "high"},
        "scope": {"type": "dataset"}
    }
]

runner = TaskRunner(
    task_id="complete_workflow_001",
    task_type="complete_workflow",
    description="Comprehensive data processing",
    input_datasets={"main": "data/input.csv"},
    operation_configs=operation_configs,
    additional_options={
        "seed": 123,
        "continue_on_error": False
    }
)

runner.configure_operations()
success = runner.run()
```

### Example 4: Encrypted Dataset Processing

```python
runner = TaskRunner(
    task_id="encrypted_task_001",
    task_type="anonymization",
    description="Process encrypted data",
    input_datasets={
        "main": "data/encrypted_customers.csv.enc",
        "lookup": "data/reference_data.csv"
    },
    operation_configs=operation_configs,
    use_encryption=True,
    encryption_keys={
        "main": "secret_key_main",
        "lookup": "secret_key_lookup"
    },
    additional_options={"seed": 42}
)

runner.configure_operations()
success = runner.run()
```

### Example 5: Error Handling with Continue-On-Error

```python
runner = TaskRunner(
    task_id="robust_task_001",
    task_type="anonymization",
    description="Continue despite errors",
    input_datasets={"main": "data/input.csv"},
    operation_configs=operation_configs,
    additional_options={
        "continue_on_error": True  # Continue even if ops fail
    }
)

runner.configure_operations()
success = runner.run()

# Check which operations succeeded
for i, op_result in enumerate(runner.lst_result):
    result = op_result['operation_result']
    if result.status == OperationStatus.ERROR:
        print(f"Operation {i} failed: {result.error_message}")
    else:
        print(f"Operation {i} succeeded")
```

## Best Practices

1. **Always Configure Before Running**
   - Call `configure_operations()` before `run()`
   - Validates all operations and parameters
   - Catches configuration errors early

2. **Use Seeds for Production**
   - Set `seed` in additional_options for reproducibility
   - Document seed used for audit trail
   - Critical for regulatory compliance

3. **Validate Datasets**
   - Ensure input_datasets paths are valid
   - Verify data types match dataset content
   - Test with sample data first

4. **Handle Errors Gracefully**
   - Check return value from `run()`
   - Review `error_info` and `lst_result` on failure
   - Log errors for debugging

5. **Use Field Scoping Effectively**
   - Scope operations to specific fields when possible
   - Reduces memory usage and processing time
   - Makes results more precise

6. **Leverage Checkpointing**
   - Enable automatic checkpoints
   - Resume long-running tasks from checkpoints
   - Avoid reprocessing completed operations

7. **Monitor Metrics**
   - Collect metrics from operations
   - Review metrics after execution
   - Use for quality validation

8. **Clean Up Resources**
   - Use context managers for safe cleanup
   - Release unused datasets explicitly
   - Monitor memory during execution

## Troubleshooting

### Issue: Operation Not Found

**Cause:** Operation class name not in function_maps

**Solution:**
```python
# Verify operation exists
from pamola_core.anonymization import FullMaskingOperation
print(f"Operation name: {FullMaskingOperation.__name__}")

# Use exact class name in config
operation_configs = [{
    "class_name": "FullMaskingOperation",  # Must match __name__
    # ... rest of config
}]
```

### Issue: Seed Not Applied

**Cause:** Operation not using seed from TaskContext

**Solution:**
```python
# TaskContext injects seed into parameters
# Operation must use it
class MyOperation(BaseOperation):
    def execute(self, data_source, task_dir, reporter, **kwargs):
        seed = kwargs.get('seed')
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
```

### Issue: Dataset Not Found

**Cause:** Input dataset path incorrect or doesn't exist

**Solution:**
```python
from pathlib import Path

# Verify path exists
dataset_path = Path("data/customers.csv")
assert dataset_path.exists(), f"File not found: {dataset_path}"

# Use absolute path if possible
input_datasets = {
    "main": str(Path("/full/path/to/customers.csv").resolve())
}
```

### Issue: Out of Memory

**Cause:** Large dataset + multiple operations

**Solution:**
```python
# Use field scoping to limit data loaded
"scope": {
    "type": "field",
    "target": ["email", "phone"]  # Only load needed fields
}

# Or configure chunking in operation parameters
"parameters": {"chunk_size": 10000}
```

## Related Components

| Component | Purpose |
|-----------|---------|
| `BaseTask` | Parent class providing infrastructure |
| `TaskContext` | Manages seed propagation |
| `DataSource` | Manages input/auxiliary datasets |
| `OperationExecutor` | Executes individual operations |
| `All Operation Classes` | Anonymization, profiling, etc. |
| `ProgressManager` | Progress tracking |
| `ContextManager` | Checkpoint management |

## Summary

The `TaskRunner` class is the primary orchestrator for executing privacy-enhancing operation sequences. It combines configuration management, operation registration, and execution lifecycle into a unified interface.

Key strengths:
- Comprehensive operation classification
- Built-in reproducibility support (FR-EP3-CORE-043)
- Flexible field and dataset scoping
- Robust error handling and recovery
- Automatic checkpointing
- Integrated metric collection

Use TaskRunner for:
- Anonymization workflows
- Data profiling and analysis
- Complex multi-operation pipelines
- Reproducible batch processing
- Regulated data processing with audit trails
