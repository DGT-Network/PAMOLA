# `run` Command Documentation

**Module:** `pamola_core.cli.commands.run`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Classes/Methods](#core-classesmethods)
6. [Usage Examples](#usage-examples)
7. [Configuration Formats](#configuration-formats)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Related Components](#related-components)
11. [Summary Analysis](#summary-analysis)

## Overview

The `run` command executes PAMOLA operations in two distinct modes: **Task-mode** (full pipeline from JSON definition) and **Single-Operation mode** (individual operation with minimal config). It implements FR-EP3-CORE-023 and leverages the TaskRunner engine for robust execution with logging and error handling.

**Command Name:** `pamola-core run`

## Key Features

- **Dual Execution Modes** — Full pipeline task execution or single operation execution
- **JSON-Driven Configuration** — Reads task definitions and operation configs from JSON files
- **Flexible Input Handling** — Supports CSV, Parquet, and other data formats
- **Random Seed Support** — Optional reproducible results with `--seed`
- **Auto Output Directory Management** — Creates output directories, supports relative/absolute paths
- **Rich Progress Feedback** — Colored status messages with operation counts and summaries
- **Error Recovery** — Validates inputs, provides actionable error messages
- **Lazy Operation Discovery** — Registers operations at runtime for faster startup

## Architecture

### Dual-Mode Execution Flow

```
run() command invoked
    ├─ Mode Selection
    │   ├─ if --task provided → Task-mode
    │   │   ├─ Parse task JSON
    │   │   ├─ Validate required fields
    │   │   ├─ Instantiate TaskRunner
    │   │   └─ Run task pipeline
    │   ├─ elif --op provided → Single-Op mode
    │   │   ├─ Validate --input provided
    │   │   ├─ Load operation config JSON
    │   │   ├─ Wrap in minimal TaskRunner
    │   │   └─ Run single operation
    │   └─ else → Error: Provide --task or --op
    │
    └─ Execution
        ├─ Discover operations (populate registry)
        ├─ Instantiate TaskRunner
        ├─ Run pipeline/operation
        ├─ Handle success/failure
        └─ Exit with appropriate code
```

## Dependencies

**External:**
- `typer` — CLI framework
- `rich.console.Console` — Rich text output
- `pathlib.Path` — File path handling

**Internal:**
- `pamola_core.utils.tasks.task_runner.TaskRunner` — Pipeline execution engine
- `pamola_core.utils.ops.op_registry.discover_operations` — Operation discovery
- `pamola_core.cli.utils.exit_codes` — Exit code constants

## Core Classes/Methods

### Main Command Function

#### `run(task, output, seed, op, config, input_data)`

Execute a task pipeline or single operation.

**Signature:**
```python
def run(
    task: Optional[Path] = typer.Option(None, "--task", "-t", ...),
    output: Optional[Path] = typer.Option(None, "--output", "-o", ...),
    seed: Optional[int] = typer.Option(None, "--seed", ...),
    op: Optional[str] = typer.Option(None, "--op", ...),
    config: Optional[Path] = typer.Option(None, "--config", ...),
    input_data: Optional[Path] = typer.Option(None, "--input", ...),
) -> None
```

**Parameters:**

| Parameter | Type | Mode | Purpose |
|-----------|------|------|---------|
| `task` | Path (optional) | Task-mode | Path to task JSON definition |
| `output` | Path (optional) | Both | Output directory for results |
| `seed` | int (optional) | Both | Random seed for reproducibility |
| `op` | str (optional) | Single-Op | Operation class name (e.g., `AttributeSuppressionOperation`) |
| `config` | Path (optional) | Single-Op | Path to operation config JSON |
| `input_data` | Path (optional) | Single-Op | Path to input CSV/Parquet file |

**Behavior:**
1. If `--task` provided, call `_run_task()` (task-mode)
2. Else if `--op` provided, call `_run_single_op()` (single-op mode)
3. Else, display error and exit with EXIT_ERROR

---

### Task-Mode Functions

#### `_run_task(task_path, output, seed)`

Execute a full pipeline task from JSON definition.

**Signature:**
```python
def _run_task(task_path: Path, output: Optional[Path], seed: Optional[int]) -> None
```

**Parameters:**
- `task_path` — Path to task JSON file
- `output` — Optional output directory
- `seed` — Optional random seed

**JSON Structure:**
```json
{
  "task_id": "my_task",
  "task_type": "anonymization",
  "description": "Apply anonymization to customer data",
  "input_datasets": {
    "customers": "data/customers.csv"
  },
  "auxiliary_datasets": {
    "lookup": "data/lookup_table.csv"
  },
  "data_types": {
    "customers": "csv"
  },
  "operations": [
    {
      "operation": "anonymization",
      "class_name": "AttributeSuppressionOperation",
      "parameters": {
        "keep_probability": 0.7
      },
      "scope": {
        "target": ["sensitive_column"]
      },
      "dataset_name": "customers",
      "task_operation_id": "op_001",
      "task_operation_order_index": 1
    }
  ],
  "additional_options": {
    "log_level": "info"
  }
}
```

**Required Fields:**
- `input_datasets` — At least one input file mapping
- `operations` — At least one operation definition

**Flow:**
1. Parse JSON with error handling for invalid syntax
2. Extract task metadata (task_id, task_type, description, etc.)
3. Validate `input_datasets` and `operations` are non-empty
4. Handle `--seed` override in `additional_options`
5. Resolve output directory:
   - If `output` is just a name (no path separators), place under `./output/`
   - Otherwise, use provided path
6. Display task summary (filename, operation count, output location)
7. Discover operations to populate registry
8. Instantiate TaskRunner with all extracted parameters
9. Run task and check success status
10. Display success/failure message with appropriate exit code

**Exit Codes:**
- `0` — Task completed successfully
- `EXIT_VALIDATION` (2) — JSON parsing error or missing required fields
- `EXIT_ERROR` (1) — Task execution failed

---

#### `_run_single_op(op_name, config_path, input_path, output, seed)`

Execute a single operation with minimal configuration.

**Signature:**
```python
def _run_single_op(
    op_name: str,
    config_path: Optional[Path],
    input_path: Optional[Path],
    output: Optional[Path],
    seed: Optional[int],
) -> None
```

**Parameters:**
- `op_name` — Operation class name
- `config_path` — Optional operation config JSON path
- `input_path` — Input data file path (required)
- `output` — Optional output directory
- `seed` — Optional random seed

**Config Format:**
Supports two config structures:

**Format 1: Structured (Recommended)**
```json
{
  "operation": "AttributeSuppressionOperation",
  "parameters": {
    "keep_probability": 0.7
  },
  "scope": {
    "target": ["email", "phone"]
  }
}
```

**Format 2: Flat (Legacy)**
```json
{
  "keep_probability": 0.7,
  "scope": {
    "target": ["email", "phone"]
  }
}
```

**Flow:**
1. Validate `--input` is provided (required)
2. Load config JSON if `--config` provided (optional)
3. Parse config structure (structured vs flat)
4. Extract `parameters` and `scope` from config
5. Handle `--seed` override
6. Determine output directory:
   - If `output` provided, use it
   - Otherwise, default to `./output/single_<op_name_lowercase>`
7. Display operation summary
8. Discover operations to populate registry
9. Wrap operation in minimal TaskRunner with:
   - Single operation config
   - Input dataset mapping
   - Scope and parameter passthrough
10. Run task and check result
11. Display completion message with exit code

**Exit Codes:**
- `0` — Operation completed successfully
- `EXIT_ERROR` (1) — Missing input or operation failed
- `EXIT_VALIDATION` (2) — Invalid config JSON

---

## Usage Examples

### Example 1: Task-Mode with Task JSON

Execute a full pipeline from task definition.

**Command:**
```bash
pamola-core run --task task.json --output ./results
```

**Files:**
- `task.json` — Full pipeline definition with multiple operations
- Output: `./results/` — Processed data, logs, and metadata

**Example task.json:**
```json
{
  "task_id": "anonymize_customers",
  "task_type": "anonymization",
  "description": "Anonymize customer PII",
  "input_datasets": {
    "customers": "data/customers.csv"
  },
  "operations": [
    {
      "class_name": "AttributeSuppressionOperation",
      "parameters": {"keep_probability": 0.5},
      "scope": {"target": ["ssn", "credit_card"]},
      "dataset_name": "customers"
    }
  ]
}
```

**Output:**
```
Task: task.json  |  Operations: 1  |  Output: ./results

✓ Task completed successfully.
```

---

### Example 2: Task-Mode with Random Seed

Execute task with fixed seed for reproducible results.

**Command:**
```bash
pamola-core run --task task.json --output ./results --seed 42
```

**Result:** Same input always produces identical output (useful for testing and validation)

---

### Example 3: Single-Operation Mode with Config

Execute a single operation with full configuration.

**Command:**
```bash
pamola-core run --op AttributeSuppressionOperation \
  --config config.json \
  --input data/customers.csv \
  --output ./anonymized
```

**Files:**
- `config.json`:
  ```json
  {
    "operation": "AttributeSuppressionOperation",
    "parameters": {"keep_probability": 0.7},
    "scope": {"target": ["email", "phone"]}
  }
  ```
- `data/customers.csv` — Input file
- Output: `./anonymized/` — Results

**Output:**
```
Single-Operation: AttributeSuppressionOperation  |  Input: data/customers.csv

✓ Operation completed successfully.
```

---

### Example 4: Single-Operation Mode with Minimal Config

Execute operation with minimal parameters.

**Command:**
```bash
pamola-core run --op AggregateRecordsOperation \
  --input data.csv
```

**Result:** Uses operation defaults, outputs to `./output/single_aggregaterecordsoperation/`

---

### Example 5: Piping with Validation

Validate before running.

**Command:**
```bash
pamola-core validate-config --task task.json && \
  pamola-core run --task task.json --output ./results
```

**Result:** Only runs if validation passes

---

## Configuration Formats

### Task JSON Structure

**Top-level Keys:**

| Key | Type | Required | Purpose |
|-----|------|----------|---------|
| `task_id` | string | No | Unique task identifier (defaults to filename) |
| `task_type` | string | No | Task category (default: "anonymization") |
| `description` | string | No | Human-readable task description |
| `input_datasets` | object | **Yes** | Mapping of dataset names to file paths |
| `auxiliary_datasets` | object | No | Optional auxiliary/lookup datasets |
| `data_types` | object | No | Explicit data type mappings (e.g., "csv") |
| `operations` | array | **Yes** | Array of operation configs (minimum 1) |
| `additional_options` | object | No | Extra options (e.g., log_level, seed override) |

### Operation Config Structure

**Structured Format (Recommended):**

| Key | Type | Required | Purpose |
|-----|------|----------|---------|
| `operation` | string | No | Operation category or type |
| `class_name` | string | **Yes** | Full operation class name |
| `parameters` | object | No | Operation-specific parameters |
| `scope` | object | No | Field targeting scope (target array) |
| `dataset_name` | string | No | Input dataset name |

**Example:**
```json
{
  "class_name": "AttributeSuppressionOperation",
  "parameters": {
    "keep_probability": 0.8
  },
  "scope": {
    "target": ["credit_card", "ssn"]
  }
}
```

## Best Practices

1. **Always validate before running** — Use `pamola-core validate-config --task <file>` before execution
2. **Use task-mode for complex pipelines** — Multiple operations benefit from task definition
3. **Use single-op mode for quick tests** — Test individual operations without full JSON setup
4. **Organize output directories** — Use meaningful names: `./results/anonymization_v1/`
5. **Set seeds for reproducibility** — Always use `--seed` for testing and CI/CD
6. **Check operation schemas** — Run `pamola-core schema <op>` before creating configs
7. **Handle relative paths carefully** — `--input data.csv` resolves relative to cwd
8. **Review output logs** — Check `<output_dir>/logs/` for execution details
9. **Use JSON validation** — Validate JSON syntax before passing to CLI:
   ```bash
   jq . < task.json > /dev/null && pamola-core run --task task.json
   ```
10. **Monitor exit codes** — Check exit code in scripts: `echo $?` (0=success, 2=validation error, 1=runtime error)

## Troubleshooting

### Issue: "Provide either --task or --op"

**Cause:** Neither `--task` nor `--op` option provided.

**Solution:**
```bash
# Task-mode
pamola-core run --task task.json --output ./results

# Single-op mode
pamola-core run --op MyOperation --config cfg.json --input data.csv
```

---

### Issue: "Task JSON must define 'input_datasets'"

**Cause:** Missing required `input_datasets` key in task JSON.

**Solution:**
```bash
# Validate first
pamola-core validate-config --task task.json

# Add input_datasets to task.json:
{
  "input_datasets": {
    "primary": "path/to/file.csv"
  },
  "operations": [...]
}
```

---

### Issue: "Invalid JSON in task file"

**Cause:** Malformed JSON syntax.

**Solution:**
```bash
# Validate JSON
jq . < task.json

# Common issues:
# - Missing commas between fields
# - Trailing commas
# - Unquoted strings
# - Single quotes instead of double quotes
```

---

### Issue: "--input is required for single-op mode"

**Cause:** Missing `--input` option in single-operation mode.

**Solution:**
```bash
pamola-core run --op AttributeSuppressionOperation \
  --config config.json \
  --input data.csv  # Add this
```

---

### Issue: "Operation failed" with no details

**Cause:** Missing verbose logging for diagnosis.

**Solution:**
```bash
# Enable debug logging
pamola-core --verbose run --task task.json

# Check output directory for logs
cat ./results/logs/*.log

# Validate config first
pamola-core validate-config --task task.json --format json
```

---

### Issue: Output directory not created

**Cause:** Insufficient permissions or invalid path.

**Solution:**
```bash
# Use simple relative path
pamola-core run --task task.json --output results

# Verify directory is writable
mkdir -p results && touch results/test.txt && rm results/test.txt

# Use absolute path
pamola-core run --task task.json --output /tmp/results
```

---

## Related Components

- **TaskRunner** — `pamola_core.utils.tasks.task_runner` — Core execution engine
- **Operation Registry** — `pamola_core.utils.ops.op_registry` — Operation discovery
- **List-Ops Command** — `pamola-core list-ops` — Discover available operations
- **Schema Command** — `pamola-core schema` — Inspect operation parameters
- **Validate Command** — `pamola-core validate-config` — Pre-execution validation

## Summary Analysis

The `run` command provides flexible operation execution:

1. **Dual modes** address different use cases (full pipelines vs quick tests)
2. **JSON configuration** enables reproducible, version-controlled task definitions
3. **Comprehensive error handling** validates inputs before execution
4. **Flexible output management** supports various directory organization strategies
5. **Reproducibility support** via `--seed` for testing and auditing
6. **Rich feedback** provides actionable status messages throughout execution
7. **Integration ready** with exit codes and JSON validation for scripting

The command is designed for both interactive use and programmatic automation in CI/CD pipelines.
