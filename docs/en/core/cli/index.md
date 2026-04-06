# PAMOLA.CORE CLI Documentation

**Module:** `pamola_core.cli`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Quick Start

Welcome to the PAMOLA.CORE CLI documentation. The CLI provides command-line access to privacy-preserving data operations.

### Install

```bash
pip install pamola-core
```

### First Steps

```bash
# See available commands
pamola-core --help

# List all operations
pamola-core list-ops

# Explore an operation
pamola-core schema AttributeSuppressionOperation

# Validate a configuration
pamola-core validate-config --task task.json

# Run a task
pamola-core run --task task.json --output ./results
```

## Documentation Structure

### Main Documentation
- **[CLI Overview](./cli_overview.md)** — Complete CLI system architecture, features, and global options

### Commands
- **[Commands Directory](./commands/index.md)** — Index of all CLI commands
  - **[`list-ops`](./commands/list_ops.md)** — Discover and list operations
  - **[`run`](./commands/run.md)** — Execute tasks and operations
  - **[`schema`](./commands/schema.md)** — View operation parameter schemas
  - **[`validate-config`](./commands/validate.md)** — Validate configurations

### Utilities
- **[Utils Directory](./utils/index.md)** — CLI utility modules
  - **[Exit Codes](./utils/exit_codes.md)** — Exit code reference and integration

## What is PAMOLA.CORE CLI?

PAMOLA.CORE is a **privacy-preserving data processing framework** with a command-line interface for:

- **Discovering operations** — Find available data operations by category
- **Inspecting schemas** — Understand operation parameters and requirements
- **Validating configurations** — Catch configuration errors before execution
- **Running tasks** — Execute complex data pipelines with single commands
- **Single operations** — Test individual operations quickly
- **Reproducible results** — Set random seeds for deterministic processing

## Use Cases

### Data Science & Analytics
Process sensitive datasets while maintaining privacy standards.

```bash
# List privacy operations
pamola-core list-ops --category anonymization

# Anonymize customer dataset
pamola-core run --task anonymization_task.json --output ./results
```

### Development & Testing
Test individual operations during development.

```bash
# Understand operation parameters
pamola-core schema MyOperation

# Quick operation test
pamola-core run --op MyOperation --config cfg.json --input data.csv
```

### CI/CD Automation
Integrate privacy operations into automated pipelines.

```bash
# Validate config in pipeline
pamola-core validate-config --task task.json || exit 1

# Run with reproducible seed
pamola-core run --task task.json --seed 42
```

### Configuration Management
Build and validate data processing pipelines.

```bash
# Validate before deploying
pamola-core validate-config --task production_task.json

# Check JSON is valid
jq . < task.json && pamola-core run --task task.json
```

## Command Roadmap

```
User Goal                 → Command Sequence
─────────────────────────────────────────────
Explore operations        → list-ops
Understand operation      → list-ops → schema
Create configuration      → schema → validate-config
Run single operation      → schema → run (single-op mode)
Run full pipeline         → validate-config → run (task-mode)
Automated validation      → validate-config (--format json)
Troubleshoot errors       → schema → validate-config (--verbose)
Integration into scripts  → list-ops (--format json), validate-config (--format json)
```

## Common Workflows

### Workflow 1: Explore and Use Operations

```bash
# 1. Find what's available
pamola-core list-ops

# 2. Filter by category
pamola-core list-ops --category profiling

# 3. Understand specific operation
pamola-core schema AnalyzeDateOperation

# 4. Use in task
# (Create task.json with operation, then validate and run)
pamola-core validate-config --task task.json
pamola-core run --task task.json --output ./results
```

**Commands:** list-ops → schema → validate-config → run

---

### Workflow 2: Quick Single-Operation Test

```bash
# 1. Check parameters
pamola-core schema AttributeSuppressionOperation

# 2. Create minimal config
# (Edit config.json with required parameters)

# 3. Validate
pamola-core validate-config --config config.json

# 4. Run
pamola-core run --op AttributeSuppressionOperation \
  --config config.json --input data.csv
```

**Commands:** schema → validate-config → run

---

### Workflow 3: Validate Configuration (No Execution)

```bash
# Check if task JSON is valid
pamola-core validate-config --task task.json

# Show validation errors as JSON
pamola-core validate-config --task task.json --format json | jq '.errors[]'

# Validate before committing to version control
pamola-core validate-config --task task.json || {
    echo "Configuration invalid"
    exit 1
}
```

**Commands:** validate-config

---

### Workflow 4: Export Operation Metadata

```bash
# Get all operations as JSON
pamola-core list-ops --format json > operations.json

# Get specific operation schema
pamola-core schema AggregateRecordsOperation --format json > schema.json

# Parse with jq
pamola-core list-ops --format json | jq '.[] | select(.category == "metrics")'
```

**Commands:** list-ops, schema (with JSON output)

---

## File Organization

```
docs/en/core/cli/
├── index.md (this file)
├── cli_overview.md (detailed overview)
├── commands/
│   ├── index.md (commands index)
│   ├── list_ops.md
│   ├── run.md
│   ├── schema.md
│   └── validate.md
└── utils/
    ├── index.md (utils index)
    └── exit_codes.md
```

## Getting Help

### In-CLI Help

```bash
# Show all commands
pamola-core --help

# Show specific command help
pamola-core list-ops --help
pamola-core run --help
pamola-core schema --help
pamola-core validate-config --help

# Enable debug output
pamola-core --verbose <command>
```

### Documentation References

| Need | See |
|------|-----|
| Complete CLI overview | [cli_overview.md](./cli_overview.md) |
| How to list operations | [commands/list_ops.md](./commands/list_ops.md) |
| How to run tasks | [commands/run.md](./commands/run.md) |
| How to inspect schemas | [commands/schema.md](./commands/schema.md) |
| How to validate configs | [commands/validate.md](./commands/validate.md) |
| Exit code meanings | [utils/exit_codes.md](./utils/exit_codes.md) |
| Command index | [commands/index.md](./commands/index.md) |
| Utils index | [utils/index.md](./utils/index.md) |

## Key Concepts

### Operations
Reusable data processing units. Each operation:
- Has a unique class name (e.g., `AttributeSuppressionOperation`)
- Belongs to a category (profiling, anonymization, transformations, metrics, attacks, fake_data)
- Has configurable parameters
- Processes data and produces results

### Tasks
Pipelines of operations. Each task:
- Defines input datasets
- Specifies a sequence of operations
- Configures operation parameters
- Produces output datasets
- Defined in JSON format

### Configurations
JSON files defining how operations work. Two formats:
1. **Task Config** — Full pipeline definition with multiple operations
2. **Operation Config** — Single operation parameters

### Validation
Pre-execution checking. Validates:
- JSON syntax is correct
- Required fields are present
- Operations exist and are registered
- Parameters match operation signatures

### Exit Codes
Machine-readable completion status:
- `0` — Success
- `1` — Runtime error
- `2` — Validation error

## Best Practices

1. **Always validate before running** — Use `validate-config` before `run`
2. **Inspect schemas** — Run `schema` before creating operations configs
3. **Use meaningful output names** — Organize results with descriptive directories
4. **Set seeds for testing** — Use `--seed` for reproducible results
5. **Check exit codes** — Verify command success in scripts
6. **Use JSON for automation** — Export data with `--format json`
7. **Enable verbose on errors** — Add `--verbose` for detailed error info
8. **Review operation categories** — Filter with `--category` to find relevant ops
9. **Validate JSON early** — Use `jq` before passing to CLI
10. **Check logs** — Review output directory logs for execution details

## Integration Examples

### Bash Script

```bash
#!/bin/bash
set -e

# Validate and run
pamola-core validate-config --task task.json
pamola-core run --task task.json --output ./results

echo "Task completed successfully"
```

### Python Script

```python
import subprocess
import json

# List operations
result = subprocess.run(
    ["pamola-core", "list-ops", "--format", "json"],
    capture_output=True, text=True, check=True
)
operations = json.loads(result.stdout)
print(f"Found {len(operations)} operations")
```

### CI/CD Pipeline (GitHub Actions)

```yaml
- name: Validate configuration
  run: pamola-core validate-config --task task.json

- name: Run task
  run: pamola-core run --task task.json --seed 42 --output ./results
```

## Troubleshooting

### "Command not found"
Install PAMOLA.CORE:
```bash
pip install pamola-core
```

### "Unknown operation"
List available operations:
```bash
pamola-core list-ops
```

### "Invalid JSON"
Validate JSON syntax:
```bash
jq . < file.json
```

### "Validation failed"
Check schema and review errors:
```bash
pamola-core schema <operation>
pamola-core validate-config --task file.json --format json
```

### "Operation failed"
Enable debug logging:
```bash
pamola-core --verbose run --task task.json
```

For more troubleshooting, see individual command documentation.

## Version Information

- **PAMOLA.CORE Version:** 0.0.1
- **CLI Framework:** Typer
- **Python Support:** 3.10–3.12
- **Last Updated:** 2026-03-23

## Related Documentation

- **Project Overview** — `../../../project-overview-pdr.md`
- **System Architecture** — `../../../system-architecture.md`
- **Code Standards** — `../../../code-standards.md`
- **Codebase Summary** — `../../../codebase-summary.md`

## Summary

The PAMOLA.CORE CLI provides comprehensive command-line access to privacy-preserving data operations through four main commands:

1. **list-ops** — Discover available operations
2. **schema** — Inspect operation parameters
3. **validate-config** — Validate configurations
4. **run** — Execute tasks and operations

Use the command index or individual command documentation for detailed information about any command. Start with `list-ops` to explore what's available!
