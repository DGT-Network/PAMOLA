# `validate-config` Command Documentation

**Module:** `pamola_core.cli.commands.validate`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Classes/Methods](#core-classesmethods)
6. [Validation Rules](#validation-rules)
7. [Usage Examples](#usage-examples)
8. [Output Formats](#output-formats)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Related Components](#related-components)
12. [Summary Analysis](#summary-analysis)

## Overview

The `validate-config` command validates operation configs and task JSON files before execution. It performs comprehensive schema validation, parameter checking, and configuration error reporting. The command helps catch configuration issues early, reducing runtime failures and improving development velocity.

**Command Name:** `pamola-core validate-config`

## Key Features

- **Dual Validation** — Validate single-operation configs or full task pipelines
- **Comprehensive Error Reporting** — Clear, actionable error messages
- **Required Parameter Checking** — Ensures all mandatory parameters are provided
- **Unknown Parameter Detection** — Flags unsupported parameters in configs
- **JSON Syntax Validation** — Detects malformed JSON before processing
- **Runtime Parameter Handling** — Skips required check for runtime-injected parameters (e.g., `field_name`)
- **Multiple Output Formats** — Human-readable table (default) and JSON for scripting
- **Operation Registry Integration** — Validates against registered operation schemas
- **Graceful Error Handling** — Returns structured error information

## Architecture

### Validation Flow

```
validate_config(config, task, fmt) invoked
    ├─ Input Validation
    │   ├─ Check config or task provided
    │   └─ Load and parse JSON
    │       ├─ Valid → Continue
    │       └─ Invalid JSON → Error with line/column info
    │
    ├─ Schema Routing
    │   ├─ if task → _validate_task()
    │   └─ if config → _validate_op_config()
    │
    └─ Output
        ├─ if fmt == json → Emit structured JSON
        └─ if fmt == table → Display errors or success message
```

### Validation Tiers

**Tier 1: JSON Syntax**
- Validates JSON is well-formed
- Reports parse errors with context

**Tier 2: Schema Structure**
- Validates required top-level keys exist
- Checks data types match expectations

**Tier 3: Operation Validation**
- Discovers operations in registry
- Validates operation class names
- Checks parameters against operation metadata

**Tier 4: Required Parameter Checking**
- Identifies missing required parameters
- Skips parameters provided at runtime
- Skips scope.target-provided parameters

## Dependencies

**External:**
- `typer` — CLI framework
- `rich.console.Console` — Rich text output
- `pathlib.Path` — File path handling
- `json` — JSON parsing
- `inspect` — Operation signature inspection

**Internal:**
- `pamola_core.utils.ops.op_registry` — Operation discovery and metadata
  - `discover_operations()`
  - `get_operation_class()`
  - `get_operation_metadata()`
- `pamola_core.cli.utils.exit_codes` — Exit code constants

## Core Classes/Methods

### Output Format Enum

```python
class OutputFormat(str, Enum):
    table = "table"  # Human-readable table (default)
    json = "json"    # Machine-readable JSON (for scripting)
```

### Main Command Function

#### `validate_config(config, task, fmt)`

Validate operation config or task JSON file.

**Signature:**
```python
def validate_config(
    config: Optional[Path] = typer.Option(None, "--config", ...),
    task: Optional[Path] = typer.Option(None, "--task", ...),
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", "-f", ...),
) -> None
```

**Parameters:**
- `config` (Path, optional) — Path to single-operation config JSON
- `task` (Path, optional) — Path to task pipeline JSON
- `fmt` (OutputFormat, optional) — Output format: `table` or `json` (default: `table`)

**Return Value:** None (prints to stdout, exits with code 0 or error code)

**Exit Codes:**
- `0` — Valid configuration
- `EXIT_ERROR` (1) — Missing both config and task options
- `EXIT_VALIDATION` (2) — JSON parsing error or validation errors

**Behavior:**
1. Validate that either `--config` or `--task` is provided
2. Load and parse JSON file
3. Route to appropriate validation function based on input type
4. Collect all validation errors (non-fatal approach)
5. If errors found, output them and exit with EXIT_VALIDATION
6. If valid, display success message and exit with 0

---

### Validation Functions

#### `_validate_task(data)`

Validate a task pipeline JSON.

**Signature:**
```python
def _validate_task(data: dict) -> list
```

**Parameters:**
- `data` — Parsed task JSON object

**Return Value:** List of error strings (empty if valid)

**Validation Rules:**

| Rule | Error Message | Condition |
|------|---------------|-----------|
| Root type | "Root must be a JSON object." | `data` is not dict |
| Input datasets | "Missing 'input_datasets' key or empty." | `input_datasets` missing or empty |
| Operations list | "Missing 'operations' key or empty operations list." | `operations` missing or empty |
| Operation count | (implicit) | At least 1 operation required |

**Per-Operation Validation:**

For each operation in `operations` array:

| Check | Error Message | Details |
|-------|---------------|---------|
| Class name | `"Operation {i}: missing 'class_name' key."` | `class_name` field required |
| Class exists | `"Operation {i}: '{op_name}' not found in registry."` | Operation must be registered |
| Parameters | `"Operation {i} ({op_name}): unknown parameter(s): {list}."` | Unknown params only if `**kwargs` not accepted |
| Required params | `"Operation {i} ({op_name}): required parameter '{param}' is missing."` | Check required params unless in `scope.target` |

**Special Handling:**
- Skips unknown parameter check if operation accepts `**kwargs` (has `*args, **kwargs` in signature)
- Skips required parameter check if parameter is in runtime injection set (`field_name`)
- Continues validation for remaining operations even after errors found

---

#### `_validate_op_config(data)`

Validate a single-operation config JSON.

**Signature:**
```python
def _validate_op_config(data: dict) -> list
```

**Parameters:**
- `data` — Parsed operation config JSON object

**Return Value:** List of error strings (empty if valid)

**Validation Rules:**

| Rule | Error Message | Condition |
|------|---------------|-----------|
| Operation key | "Missing 'operation' key in config." | `operation` field required |
| Class exists | `"Operation '{op_name}' not found in registry."` | Operation must be registered |
| Parameters | `"Unknown parameter(s): {list}."` | Unknown params unless `**kwargs` accepted |
| Required params | `"Required parameter '{param}' is missing."` | Required params unless in runtime set |

**Special Handling:**
- Runtime parameters (`field_name`) only skipped if `scope.target` is non-empty
- Unknown parameter check skips if operation accepts `**kwargs`
- Validates operation existence before checking parameters

---

#### `_emit(fmt, payload)`

Output validation result in specified format.

**Signature:**
```python
def _emit(fmt: OutputFormat, payload: dict) -> None
```

**Parameters:**
- `fmt` — Output format (json only; table handled by `console.print()`)
- `payload` — Result dict with keys: `valid`, `file`, `errors`

**Output:**
- If `fmt == OutputFormat.json` — Pretty-printed JSON to stdout
- Otherwise — No output (handled by caller)

---

## Validation Rules

### JSON Syntax Rules

1. **Well-formed JSON** — File must be valid JSON (no trailing commas, etc.)
2. **UTF-8 Encoding** — File should use UTF-8 encoding
3. **Root Object** — Root must be a JSON object (not array or primitive)

### Task JSON Rules

1. **Required Keys**
   - `input_datasets` — Must exist and be non-empty dict
   - `operations` — Must exist and be non-empty array

2. **Optional Keys** (but recommended)
   - `task_id` — Unique identifier
   - `task_type` — Task category (default: "anonymization")
   - `description` — Human-readable description
   - `auxiliary_datasets` — Optional lookup files
   - `data_types` — Optional explicit type mappings
   - `additional_options` — Extra configuration

3. **Operations Array Rules**
   - Each operation must have `class_name` field
   - `class_name` value must match registered operation
   - Parameters must be defined in operation signature or via `**kwargs`
   - Required parameters must be provided (unless runtime-injected)

### Operation Config Rules

1. **Required Keys**
   - `operation` — Operation class name (or use `class_name` in structured format)

2. **Optional Keys**
   - `parameters` — Operation-specific parameters
   - `scope` — Field targeting scope
   - `dataset_name` — Input dataset name

3. **Parameter Rules**
   - All parameters must be defined in operation signature
   - Unless operation accepts `**kwargs` (flexible parameter acceptance)
   - Required parameters must be provided (unless runtime-injected)

### Runtime Parameter Injection

Certain parameters are injected at runtime and skip required checks:

- `field_name` — Injected when `scope.target` is provided
- Other parameters can be marked in the validator's `runtime_params` set

## Usage Examples

### Example 1: Validate Task JSON (Table Output)

Validate a task pipeline and display results.

**Command:**
```bash
pamola-core validate-config --task task.json
```

**Output (Valid):**
```
✓ Valid: task.json
```

**Output (Invalid):**
```
✗ Validation failed — 2 issue(s) in task.json

  • Missing 'input_datasets' key or empty.
  • Operation 1: 'UnknownOperation' not found in registry.
```

---

### Example 2: Validate Operation Config (JSON Output)

Validate operation config with JSON output for scripting.

**Command:**
```bash
pamola-core validate-config --config config.json --format json
```

**Output (Valid):**
```json
{
  "valid": true,
  "file": "config.json",
  "errors": []
}
```

**Output (Invalid):**
```json
{
  "valid": false,
  "file": "config.json",
  "errors": [
    "Missing 'operation' key in config.",
    "Required parameter 'keep_probability' is missing."
  ]
}
```

---

### Example 3: Validate Before Running

Validate config, then run if valid.

**Command:**
```bash
pamola-core validate-config --task task.json && \
  pamola-core run --task task.json --output ./results
```

**Output (Valid):**
```
✓ Valid: task.json
Task: task.json  |  Operations: 3  |  Output: ./results

✓ Task completed successfully.
```

**Output (Invalid):**
```
✗ Validation failed — 1 issue(s) in task.json

  • Operation 2 (MyOperation): required parameter 'threshold' is missing.
```

---

### Example 4: Comprehensive Task Validation

Show all validation errors at once.

**Command:**
```bash
pamola-core validate-config --task task.json --format json | jq '.errors[]'
```

**Output:**
```
"Missing 'input_datasets' key or empty."
"Operation 1: missing 'class_name' key."
"Operation 2: 'AttributeSuppressionOperation': unknown parameter 'invalid_param'."
```

---

### Example 5: Validate with Malformed JSON

Show JSON syntax error handling.

**Command:**
```bash
# task.json has trailing comma
cat task.json  # {"operations": [1,]}

pamola-core validate-config --task task.json
```

**Output:**
```
✗ Invalid JSON in task.json

  Expecting value: line 1 column 22 (char 21)
```

---

### Example 6: Pipeline Multiple Configs

Validate multiple config files.

**Command:**
```bash
for config in configs/*.json; do
  echo "Validating $config..."
  pamola-core validate-config --config "$config" || exit 1
done
```

**Output:**
```
Validating configs/op1.json...
✓ Valid: configs/op1.json
Validating configs/op2.json...
✓ Valid: configs/op2.json
```

---

## Output Formats

### Table Format (Default)

Human-readable output for interactive use.

**Valid Configuration:**
```
✓ Valid: /path/to/config.json
```

**Invalid Configuration:**
```
✗ Validation failed — N issue(s) in /path/to/config.json

  • Error message 1
  • Error message 2
  • Error message 3
```

**Features:**
- Green checkmark for valid
- Red X for invalid
- File path shown
- Error count displayed
- Bulleted error list

### JSON Format

Structured output for programmatic use.

**Schema:**
```json
{
  "valid": true|false,
  "file": "path/to/file.json",
  "errors": ["error1", "error2"]
}
```

**Exit Code Handling:**
- Include exit code in script logic:
  ```bash
  result=$(pamola-core validate-config --config config.json --format json)
  valid=$(echo "$result" | jq '.valid')
  if [ "$valid" = "true" ]; then
    echo "Config is valid"
  fi
  ```

## Best Practices

1. **Always validate before running** — Use `validate-config` as gate before `run` command
2. **Use table format interactively** — Default format provides clear feedback
3. **Use JSON format in scripts** — Integrate with CI/CD and automation
4. **Validate early in development** — Check configs as you write them
5. **Review all errors at once** — Non-stopping validation shows all issues
6. **Check operation schemas first** — Run `pamola-core schema <op>` before creating configs
7. **Use jq for advanced validation** — Extract specific error types:
   ```bash
   pamola-core validate-config --task task.json --format json | jq '.errors[] | select(contains("required"))'
   ```
8. **Set CI/CD gates** — Fail builds if validation returns non-zero:
   ```bash
   pamola-core validate-config --task task.json || exit 1
   ```
9. **Document required parameters** — Include comments in task JSON about mandatory fields
10. **Test edge cases** — Validate configs with missing optional params to understand defaults

## Troubleshooting

### Issue: "Provide --config or --task"

**Cause:** Neither option provided.

**Solution:**
```bash
# Provide either --config or --task
pamola-core validate-config --config operation_config.json
# or
pamola-core validate-config --task task_pipeline.json
```

---

### Issue: "Invalid JSON in <file>"

**Cause:** Malformed JSON syntax.

**Solution:**
```bash
# Validate JSON syntax with jq
jq . < config.json

# Use online JSON validator
# Common issues:
# - Missing commas between fields
# - Trailing commas
# - Unquoted string values
# - Single quotes instead of double quotes
# - Unescaped special characters
```

---

### Issue: "Unknown operation: <name>"

**Cause:** Operation class name not found in registry.

**Solution:**
```bash
# List available operations
pamola-core list-ops

# Search for similar names
pamola-core list-ops --format json | jq '.[] | select(.name | contains("Aggregate"))'

# Verify exact spelling and casing
pamola-core schema AttributeSuppressionOperation  # not Attribute_Suppression_Operation
```

---

### Issue: "Required parameter '<param>' is missing"

**Cause:** Mandatory operation parameter not provided in config.

**Solution:**
```bash
# Check operation schema to see parameter requirements
pamola-core schema AttributeSuppressionOperation

# Add missing parameter to config:
{
  "operation": "AttributeSuppressionOperation",
  "parameters": {
    "keep_probability": 0.7  # Required parameter
  }
}
```

---

### Issue: "Unknown parameter(s): <list>"

**Cause:** Config contains parameters not defined in operation signature.

**Solution:**
```bash
# Check operation schema for valid parameters
pamola-core schema MyOperation --format json | jq '.parameters | keys'

# Remove unknown parameters from config
# or verify spelling:
# - "keep_probablity" → "keep_probability"
# - "target_field" → "target_fields"
```

---

### Issue: Validation passes but run fails

**Cause:** Runtime errors not caught by schema validation (e.g., missing input files).

**Solution:**
```bash
# Run with verbose logging
pamola-core --verbose run --task task.json

# Check that all input files exist
# Verify file paths are correct (absolute or relative to cwd)
# Check file permissions and encoding
```

---

## Related Components

- **Operation Registry** — `pamola_core.utils.ops.op_registry` — Source of truth for operation metadata
- **Schema Command** — `pamola-core schema` — Inspect operation parameter requirements
- **List-Ops Command** — `pamola-core list-ops` — Discover available operations
- **Run Command** — `pamola-core run` — Execute validated configurations
- **TaskRunner** — `pamola_core.utils.tasks.task_runner` — Execution engine

## Summary Analysis

The `validate-config` command provides critical pre-execution validation:

1. **Comprehensive error reporting** — All errors found in one pass, not failing on first issue
2. **Clear error messages** — Actionable guidance for fixing configuration problems
3. **Dual output formats** — Interactive table output and JSON for automation
4. **Smart parameter handling** — Understands runtime-injected parameters and flexible signatures
5. **Early error detection** — Catches issues before expensive operation execution
6. **Integration ready** — Exit codes and JSON output enable CI/CD pipelines
7. **Developer friendly** — Helps developers understand operation requirements and catch errors

The command is essential for reliable task execution and should be used before every `run` invocation in production workflows.
