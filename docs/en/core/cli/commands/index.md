# CLI Commands Reference

**Location:** `pamola_core/cli/commands/`
**Last Updated:** 2026-03-23

## Overview

This directory contains documentation for all PAMOLA.CORE CLI commands. Each command provides specific functionality for operation discovery, configuration validation, schema inspection, and task execution.

## Quick Reference

| Command | Purpose | Documentation |
|---------|---------|---------------|
| `list-ops` | Discover and list operations by category | [list_ops.md](./list_ops.md) |
| `run` | Execute task pipelines or single operations | [run.md](./run.md) |
| `validate-config` | Validate JSON configs and task files | [validate.md](./validate.md) |
| `schema` | Display operation parameter schemas | [schema.md](./schema.md) |

## Command Structure

All commands are registered as sub-commands of the root `pamola-core` application.

### Root Application
```
pamola-core
├── --version, -V          Show version and exit
├── --verbose, -v          Enable debug logging
├── list-ops (sub-app)
├── run (sub-app)
├── validate-config (sub-app)
└── schema (command)
```

## Usage by Workflow

### Workflow 1: Explore Operations

Start by discovering what operations are available.

```bash
# 1. List all operations
pamola-core list-ops

# 2. Filter by category
pamola-core list-ops --category profiling

# 3. Inspect specific operation
pamola-core schema MyOperation

# 4. Export for documentation
pamola-core list-ops --format json > operations.json
```

**Commands Used:** `list-ops`, `schema`

---

### Workflow 2: Create and Validate Config

Build a task definition and validate before execution.

```bash
# 1. Create task.json (manually)
cat > task.json << 'EOF'
{
  "task_id": "my_task",
  "input_datasets": {"data": "input.csv"},
  "operations": [...]
}
EOF

# 2. Validate configuration
pamola-core validate-config --task task.json

# 3. Fix any errors (if needed)
# Edit task.json and re-validate

# 4. Execute validated task
pamola-core run --task task.json --output ./results
```

**Commands Used:** `schema`, `validate-config`, `run`

---

### Workflow 3: Quick Single Operation Test

Test an operation quickly without full pipeline.

```bash
# 1. Find operation
pamola-core list-ops | grep -i suppress

# 2. Check parameters
pamola-core schema AttributeSuppressionOperation

# 3. Create minimal config
cat > config.json << 'EOF'
{
  "operation": "AttributeSuppressionOperation",
  "parameters": {"keep_probability": 0.7}
}
EOF

# 4. Validate and run
pamola-core validate-config --config config.json
pamola-core run --op AttributeSuppressionOperation \
  --config config.json --input data.csv
```

**Commands Used:** `list-ops`, `schema`, `validate-config`, `run`

---

### Workflow 4: CI/CD Integration

Integrate CLI into automated pipelines.

```bash
#!/bin/bash
set -e

echo "Validating configuration..."
pamola-core validate-config --task task.json --format json

echo "Running task..."
pamola-core run --task task.json --output ./results --seed 42

echo "Success!"
```

**Commands Used:** `validate-config`, `run`

## Command Details

### `list-ops` — Operation Discovery

**Syntax:**
```bash
pamola-core list-ops [--category <name>] [--format table|json]
```

**Purpose:** Discover and list all registered PAMOLA operations

**Key Features:**
- Filter by category (profiling, anonymization, transformations, metrics, fake_data, attacks)
- Output as rich table (default) or JSON
- Shows operation name, category, version, description
- Displays available categories

**Examples:**
```bash
# List all operations
pamola-core list-ops

# Filter by category
pamola-core list-ops --category profiling

# JSON output for scripting
pamola-core list-ops --format json | jq '.[] | .name'
```

See [list_ops.md](./list_ops.md) for comprehensive documentation.

---

### `run` — Task and Operation Execution

**Syntax:**
```bash
# Task-mode (full pipeline)
pamola-core run --task <path> [--output <dir>] [--seed <int>]

# Single-operation mode
pamola-core run --op <name> --config <path> --input <path> [--output <dir>] [--seed <int>]
```

**Purpose:** Execute task pipelines or individual operations

**Key Features:**
- Task-mode: Run full operation pipelines from JSON
- Single-op mode: Test individual operations quickly
- Output directory management with smart path handling
- Random seed support for reproducibility
- Comprehensive error reporting with logs

**Examples:**
```bash
# Run full task pipeline
pamola-core run --task task.json --output ./results

# Run single operation
pamola-core run --op AttributeSuppressionOperation \
  --config config.json --input data.csv

# With reproducible seed
pamola-core run --task task.json --seed 42
```

See [run.md](./run.md) for comprehensive documentation.

---

### `validate-config` — Configuration Validation

**Syntax:**
```bash
pamola-core validate-config [--config <path>|--task <path>] [--format table|json]
```

**Purpose:** Validate operation configs and task JSON files

**Key Features:**
- JSON syntax validation
- Schema structure validation
- Required parameter checking
- Unknown parameter detection
- Operation registry validation
- Multiple output formats (table, JSON)
- Non-stopping validation (reports all errors)

**Examples:**
```bash
# Validate operation config
pamola-core validate-config --config config.json

# Validate task pipeline
pamola-core validate-config --task task.json

# JSON output for scripting
pamola-core validate-config --task task.json --format json
```

See [validate.md](./validate.md) for comprehensive documentation.

---

### `schema` — Parameter Schema Inspection

**Syntax:**
```bash
pamola-core schema <operation> [--format pretty|json]
```

**Purpose:** Display parameter schema for any operation

**Key Features:**
- Operation metadata (name, version, module, category)
- Parameter details (type, required, default)
- Rich table formatting (default) or JSON
- Supports jq filtering for advanced queries
- Error handling with helpful suggestions

**Examples:**
```bash
# Show schema in pretty format
pamola-core schema AggregateRecordsOperation

# Show schema as JSON
pamola-core schema AggregateRecordsOperation --format json

# Find all required parameters
pamola-core schema MyOp --format json | jq '.parameters | to_entries[] | select(.value.required) | .key'
```

See [schema.md](./schema.md) for comprehensive documentation.

## Command Relationships

```
list-ops
  ↓ (find operation name)
schema
  ↓ (understand parameters)
validate-config
  ↓ (verify config)
run
  ↓ (execute)
(results)
```

## Global Options

All commands respect these global options from root application:

### `--version` / `-V`
Display version and exit.

```bash
pamola-core --version
# Output: pamola-core 0.0.1
```

### `--verbose` / `-v`
Enable debug logging for any command.

```bash
pamola-core --verbose run --task task.json
# Shows detailed logs during execution
```

### `-h` / `--help`
Display help for any command.

```bash
pamola-core run --help
pamola-core list-ops --help
```

## Error Handling

### Exit Codes

All commands use standardized exit codes:

| Code | Meaning | Example |
|------|---------|---------|
| 0 | Success | Valid configuration, task completed |
| 1 | Runtime error | File I/O error, execution failure |
| 2 | Validation error | Invalid JSON, missing required field |

See `../utils/exit_codes.md` for details.

### Error Messages

Commands provide clear, actionable error messages:

```bash
# Clear indication of what's wrong
✗ Unknown operation: NonExistentOp
  Run pamola-core list-ops to see available operations.

# Validation errors with context
✗ Validation failed — 1 issue(s) in task.json
  • Operation 1: required parameter 'threshold' is missing.
```

## Integration Patterns

### Shell Scripts

```bash
#!/bin/bash
set -e

# Validate before running
pamola-core validate-config --task task.json

# Execute with error handling
pamola-core run --task task.json --output ./results || {
    echo "Task failed"
    exit 1
}
```

### Python Automation

```python
import subprocess
import json

# Get list of operations
result = subprocess.run(
    ["pamola-core", "list-ops", "--format", "json"],
    capture_output=True, text=True
)
operations = json.loads(result.stdout)

# Validate config
subprocess.run(
    ["pamola-core", "validate-config", "--task", "task.json"],
    check=True
)

# Run task
subprocess.run(
    ["pamola-core", "run", "--task", "task.json"],
    check=True
)
```

### CI/CD Pipelines

```yaml
# Example GitHub Actions workflow
- name: List operations
  run: pamola-core list-ops --format json

- name: Validate configuration
  run: pamola-core validate-config --task task.json

- name: Execute task
  run: pamola-core run --task task.json --output ./results
```

## Best Practices

1. **Always validate before running** — Use `validate-config` as a gate before execution
2. **Inspect schemas first** — Run `schema` before creating operation configs
3. **Use category filtering** — Filter with `--category` to find relevant operations
4. **Enable verbose on errors** — Add `--verbose` flag when debugging
5. **Check exit codes** — Use `$?` or equivalent to check command success
6. **Export for documentation** — Use JSON output to generate operation references
7. **Seed for reproducibility** — Use `--seed` flag in testing and CI/CD
8. **Organize output directories** — Use meaningful names: `./results/anonymization_v1/`
9. **Validate JSON first** — Use `jq` to verify JSON syntax before passing to CLI
10. **Review logs** — Check output directories for detailed execution logs

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Unknown operation" | Run `pamola-core list-ops` to find correct name |
| "Invalid JSON" | Validate with `jq . < file.json` |
| "Missing required field" | Run `pamola-core schema <op>` to check parameters |
| "Command not found" | Ensure PAMOLA.CORE is installed: `pip install pamola-core` |
| "Permission denied" | Check file permissions on input/output directories |

For detailed troubleshooting, see individual command documentation.

## File Organization

```
commands/
├── index.md (this file)
├── list_ops.md
├── run.md
├── schema.md
└── validate.md
```

## Related Documentation

- **CLI Overview** — `../cli_overview.md` — Full CLI documentation
- **Exit Codes** — `../utils/exit_codes.md` — Exit code reference
- **Project Overview** — `../../../project-overview-pdr.md` — System architecture

## Summary

The CLI commands provide a complete toolkit for:
- Discovering what operations are available
- Understanding operation requirements
- Validating configurations
- Executing data processing tasks

Use them in sequence: discover → understand → validate → execute

For detailed information about any command, see its individual documentation file.
