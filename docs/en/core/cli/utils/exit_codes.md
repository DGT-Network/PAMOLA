# CLI Exit Codes Documentation

**Module:** `pamola_core.cli.utils.exit_codes`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Exit Code Reference](#exit-code-reference)
3. [Usage in Commands](#usage-in-commands)
4. [Script Integration Examples](#script-integration-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Related Components](#related-components)
8. [Summary Analysis](#summary-analysis)

## Overview

The `exit_codes` module defines standardized exit codes used across all PAMOLA.CORE CLI commands. Exit codes enable reliable script integration and error handling by providing machine-readable completion status that shells and automation tools can check.

**Module Location:** `pamola_core/cli/utils/exit_codes.py`

**Purpose:** Provide consistent, meaningful exit codes across all CLI commands for:
- Shell script integration
- CI/CD pipeline automation
- Error handling and recovery
- Operational monitoring and alerts

## Exit Code Reference

### EXIT_OK (0)

**Value:** `0`

**Meaning:** Command completed successfully.

**When Used:**
- Command executed without errors
- All validations passed
- All operations completed as expected

**Shell Interpretation:**
```bash
pamola-core list-ops
echo $?  # Output: 0
```

**Script Handling:**
```bash
if pamola-core list-ops; then
  echo "List completed successfully"
fi
```

**Common Scenarios:**
- `list-ops` — Operations listed successfully
- `schema` — Schema displayed successfully
- `validate-config` — Configuration is valid
- `run` — Task/operation completed successfully

---

### EXIT_ERROR (1)

**Value:** `1`

**Meaning:** Runtime or unexpected error occurred.

**When Used:**
- File I/O errors (cannot read input file, cannot write output)
- Operation execution failures
- Unexpected exceptions
- Missing required input data
- Invalid file paths or permissions
- Database connection errors
- Memory or resource exhaustion

**Shell Interpretation:**
```bash
pamola-core run --task nonexistent.json
echo $?  # Output: 1
```

**Script Handling:**
```bash
if ! pamola-core run --task task.json; then
  echo "Task failed with error code: $?"
  exit 1
fi
```

**Common Scenarios:**
- `list-ops` — Failed to load operations catalog/registry
- `run` — Task execution encountered unexpected error
- `run` — Missing input data file
- `schema` — Unknown operation class
- Any command — Unhandled exception

---

### EXIT_VALIDATION (2)

**Value:** `2`

**Meaning:** Configuration or schema validation failure.

**When Used:**
- JSON syntax errors
- Missing required configuration keys
- Invalid parameter values
- Unknown parameters in config
- Missing required parameters
- Type mismatches
- Configuration structure violations

**Shell Interpretation:**
```bash
pamola-core validate-config --config invalid.json
echo $?  # Output: 2
```

**Script Handling:**
```bash
if pamola-core validate-config --task task.json; then
  pamola-core run --task task.json
else
  echo "Configuration validation failed"
  exit 2
fi
```

**Common Scenarios:**
- `validate-config` — JSON parsing error
- `validate-config` — Missing required fields
- `validate-config` — Unknown operation or parameters
- `run` — Invalid task JSON syntax
- `run` — Missing operation config keys

---

## Exit Code Mapping Table

| Code | Constant | Meaning | Category | Recoverable |
|------|----------|---------|----------|-------------|
| 0 | EXIT_OK | Success | Normal | N/A |
| 1 | EXIT_ERROR | Runtime error | Error | Maybe |
| 2 | EXIT_VALIDATION | Validation failed | Configuration | Yes |

## Usage in Commands

### In `list_ops.py`

```python
from pamola_core.cli.utils.exit_codes import EXIT_ERROR

try:
    ops_data = _load_from_catalog(category)
except Exception as e:
    console.print(f"[red]✗ Failed to load operations:[/red] {e}")
    raise typer.Exit(EXIT_ERROR)  # Exit with code 1
```

**Scenarios:**
- EXIT_ERROR — Catalog/registry load failure
- EXIT_OK — Operations listed successfully

---

### In `run.py`

```python
from pamola_core.cli.utils.exit_codes import EXIT_ERROR, EXIT_VALIDATION

# Validation error (bad JSON)
try:
    task_def = json.loads(task_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    console.print(f"[red]✗ Invalid JSON in task file:[/red] {e}")
    raise typer.Exit(EXIT_VALIDATION)  # Exit with code 2

# Runtime error (missing field)
if not input_datasets:
    console.print("[red]✗ Task JSON must define 'input_datasets'.[/red]")
    raise typer.Exit(EXIT_VALIDATION)  # Exit with code 2

# Execution failure
try:
    success = runner.run()
except Exception as e:
    console.print(f"\n[red]✗ Task failed:[/red] {e}")
    raise typer.Exit(EXIT_ERROR)  # Exit with code 1
```

**Scenarios:**
- EXIT_VALIDATION — JSON parse error, missing required fields
- EXIT_ERROR — Task execution failure, file I/O errors
- EXIT_OK — Task completed successfully

---

### In `validate.py`

```python
from pamola_core.cli.utils.exit_codes import EXIT_ERROR, EXIT_VALIDATION

# Missing option
if not config and not task:
    console.print("[red]✗ Provide --config or --task.[/red]")
    raise typer.Exit(EXIT_ERROR)  # Exit with code 1

# JSON parse error
try:
    data = json.loads(target.read_text(encoding="utf-8"))
except json.JSONDecodeError as e:
    console.print(f"[red]✗ Invalid JSON in[/red] [cyan]{target}[/cyan]")
    raise typer.Exit(EXIT_VALIDATION)  # Exit with code 2

# Validation errors
errors = _validate_task(data)
if errors:
    for err in errors:
        console.print(f"  [red]•[/red] {err}")
    raise typer.Exit(EXIT_VALIDATION)  # Exit with code 2
```

**Scenarios:**
- EXIT_ERROR — Missing required options
- EXIT_VALIDATION — JSON errors, validation failures
- EXIT_OK — Configuration is valid

---

### In `schema.py`

```python
from pamola_core.cli.utils.exit_codes import EXIT_ERROR

op_cls = get_operation_class(operation)
if op_cls is None:
    console.print(f"[red]✗ Unknown operation:[/red] {operation}")
    raise typer.Exit(EXIT_ERROR)  # Exit with code 1
```

**Scenarios:**
- EXIT_ERROR — Unknown operation
- EXIT_OK — Schema displayed successfully

---

## Script Integration Examples

### Example 1: Simple Success Check

```bash
#!/bin/bash

pamola-core validate-config --task task.json
if [ $? -eq 0 ]; then
    echo "Validation passed"
else
    echo "Validation failed"
    exit 1
fi
```

---

### Example 2: Distinguishing Error Types

```bash
#!/bin/bash

pamola-core validate-config --task task.json
exit_code=$?

case $exit_code in
    0)
        echo "Configuration is valid"
        # Proceed with execution
        pamola-core run --task task.json
        ;;
    2)
        echo "Configuration validation failed (invalid JSON or missing fields)"
        # Offer to fix configuration
        exit 2
        ;;
    1)
        echo "Unexpected error occurred"
        # Critical error, exit
        exit 1
        ;;
esac
```

---

### Example 3: Conditional Pipeline

```bash
#!/bin/bash

# Validate and run only if validation passes
pamola-core validate-config --task task.json && \
  pamola-core run --task task.json --output ./results || \
  echo "Task failed"
```

**Execution:**
- If validate returns 0, proceed to run
- If validate returns non-zero, stop and print error message

---

### Example 4: Error Recovery

```bash
#!/bin/bash

run_task() {
    local task=$1
    pamola-core run --task "$task"
    return $?
}

attempt=1
max_attempts=3
exit_code=1

while [ $attempt -le $max_attempts ] && [ $exit_code -ne 0 ]; do
    echo "Attempt $attempt of $max_attempts..."
    run_task "task.json"
    exit_code=$?

    if [ $exit_code -eq 2 ]; then
        echo "Configuration error - not retrying"
        exit 2
    elif [ $exit_code -eq 1 ]; then
        echo "Runtime error - retrying"
        ((attempt++))
        sleep 5
    fi
done

if [ $exit_code -eq 0 ]; then
    echo "Task succeeded"
else
    echo "Task failed after $max_attempts attempts"
    exit 1
fi
```

**Logic:**
- Configuration errors (EXIT_VALIDATION) are not retried
- Runtime errors (EXIT_ERROR) trigger retry logic
- Success (EXIT_OK) terminates loop

---

### Example 5: CI/CD Pipeline Integration

```bash
#!/bin/bash
set -e  # Exit on first error

echo "Step 1: List available operations"
pamola-core list-ops --format json > ops.json || exit 1

echo "Step 2: Validate configuration"
pamola-core validate-config --task task.json || {
    echo "Validation failed"
    exit 2
}

echo "Step 3: Run task"
pamola-core run --task task.json --output ./results || {
    exit_code=$?
    echo "Execution failed with exit code: $exit_code"
    exit $exit_code
}

echo "All steps completed successfully"
```

---

### Example 6: Python Integration

```python
#!/usr/bin/env python3

import subprocess
import sys

def run_task(task_file):
    """Run task and handle exit codes."""
    result = subprocess.run(
        ["pamola-core", "run", "--task", task_file],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Task completed successfully")
        return True
    elif result.returncode == 2:
        print("Configuration error (validation failed)")
        print(result.stderr)
        # Could attempt to fix config and retry
        return False
    elif result.returncode == 1:
        print("Runtime error occurred")
        print(result.stderr)
        # Could log error and alert monitoring
        return False
    else:
        print(f"Unknown exit code: {result.returncode}")
        return False

if __name__ == "__main__":
    success = run_task("task.json")
    sys.exit(0 if success else 1)
```

---

## Best Practices

1. **Always check exit codes** — Use `$?` (shell) or `result.returncode` (Python) after CLI commands
2. **Distinguish error types** — Handle validation errors differently from runtime errors
3. **Validate before executing** — Check exit code 2 before running expensive operations
4. **Log exit codes** — Include exit code in error messages for debugging
5. **Retry on EXIT_ERROR** — Only retry for runtime errors, not validation errors
6. **Document expected codes** — Specify which commands can return which exit codes
7. **Chain commands carefully** — Use `&&` to stop on first error, `;` to continue
8. **Set -e in scripts** — Use `set -e` to exit bash script on first non-zero exit code
9. **Test error paths** — Verify scripts handle all three exit codes
10. **Monitor exit codes** — Track exit codes in logs for alerting and analytics

## Troubleshooting

### Issue: "Command appears to succeed but returns non-zero"

**Cause:** Exit code not correctly propagated.

**Solution:**
```bash
# Check the actual exit code
pamola-core list-ops
echo "Exit code: $?"

# Ensure command is the last in pipeline
# (last command's exit code is used in shell)
# BAD: pamola-core list-ops | grep something
# GOOD: pamola-core list-ops > output.txt && grep something output.txt
```

---

### Issue: Script stops even when not expecting errors

**Cause:** `set -e` is enabled or && chain too long.

**Solution:**
```bash
# Remove or narrow set -e scope
set +e  # Disable for specific commands

# Or handle individual command failures:
pamola-core validate-config --task task.json || {
    # Handle error here without stopping entire script
    echo "Validation failed"
}
```

---

### Issue: Can't distinguish between error types in scripts

**Cause:** Treating all non-zero exits as same error.

**Solution:**
```bash
# Use case statement
pamola-core validate-config --task task.json
case $? in
    0) echo "Valid" ;;
    2) echo "Validation failed" ;;
    1) echo "Unexpected error" ;;
esac
```

---

## Related Components

- **CLI Commands** — All commands in `pamola_core/cli/commands/` use these exit codes
- **Main Application** — `pamola_core/cli/main.py` — Root CLI application
- **Typer Framework** — Typer framework handles exit code propagation to shell

## Summary Analysis

The `exit_codes` module provides essential CLI integration features:

1. **Clear error semantics** — Three distinct codes for success, runtime errors, and validation errors
2. **Broad adoption** — Used consistently across all CLI commands
3. **Shell integration** — Exit codes enable reliable bash/shell scripting
4. **CI/CD ready** — Supports automated pipeline decision-making
5. **Language agnostic** — Works with bash, Python, Node.js, and any other language
6. **Debugging aid** — Error types can be identified immediately without parsing output
7. **Retry logic enablement** — Validation errors can be skipped in retry loops

The module is small but critical for reliable CLI automation and operational reliability.
