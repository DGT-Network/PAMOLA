# CLI Overview Documentation

**Module:** `pamola_core.cli`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Components](#core-components)
6. [Entry Point](#entry-point)
7. [Global Options](#global-options)
8. [Command Structure](#command-structure)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Related Components](#related-components)
12. [Summary Analysis](#summary-analysis)

## Overview

The `pamola_core.cli` module provides a command-line interface for PAMOLA.CORE, a privacy-preserving data processing framework. It enables users to list available operations, execute data anonymization tasks, validate configurations, and inspect operation schemas—all from the terminal.

The CLI is built using **Typer**, a modern Python CLI framework that wraps Click and provides intuitive command structure with built-in help generation and rich text formatting.

**Entry point:** `pamola-core` (defined in `pyproject.toml`)

## Key Features

- **Operation Discovery**: List all registered PAMOLA operations by category
- **Task Execution**: Run complex multi-operation pipelines from JSON definitions
- **Single-Operation Mode**: Execute individual operations with minimal configuration
- **Configuration Validation**: Validate task JSON and operation configs before execution
- **Schema Inspection**: Display parameter schemas for any operation
- **Structured Output**: Support for both human-readable and machine-readable (JSON) output formats
- **Rich Text Formatting**: Colored output, tables, and visual indicators for better UX
- **Lazy Command Loading**: Sub-commands load only when invoked for fast startup

## Architecture

The CLI follows a hierarchical command structure:

```
pamola-core (root application)
├── list-ops (typer sub-app)
│   ├── --category (filter by category)
│   └── --format (table/json output)
├── run (typer sub-app)
│   ├── Task-mode
│   │   ├── --task (JSON file path)
│   │   ├── --output (output directory)
│   │   └── --seed (random seed)
│   └── Single-Op mode
│       ├── --op (operation class name)
│       ├── --config (operation config JSON)
│       ├── --input (input data file)
│       └── --seed (random seed)
├── validate-config (typer sub-app)
│   ├── --config (operation config JSON)
│   ├── --task (task pipeline JSON)
│   └── --format (table/json output)
└── schema (single command)
    ├── operation (positional argument)
    └── --format (pretty/json output)
```

### Module Organization

| File | Purpose |
|------|---------|
| `main.py` | Root application, global options, command registration |
| `commands/list_ops.py` | Operation discovery and listing command |
| `commands/run.py` | Task execution (task-mode and single-op mode) |
| `commands/schema.py` | Operation schema inspection |
| `commands/validate.py` | Configuration and task validation |
| `utils/exit_codes.py` | Standardized exit codes for CLI commands |

## Dependencies

**Core Dependencies:**
- `typer` — CLI framework
- `rich` — Rich text formatting and rendering

**Internal Dependencies:**
- `pamola_core.catalogs` — Operations catalog (optional, falls back to runtime registry)
- `pamola_core.utils.ops.op_registry` — Operation registry and discovery
- `pamola_core.utils.tasks.task_runner` — Task execution engine

## Core Components

### 1. Root Application (`main.py`)

Defines the main Typer application with global options and lazy sub-command registration.

**Key Elements:**
- Application name: `pamola-core`
- Rich markup mode enabled for colored output
- Help options: `-h`, `--help`
- Global options: `--version`, `--verbose`

**Global Callbacks:**
- `_version_callback()` — Display version and exit
- `_verbose_callback()` — Enable debug logging
- `_root()` — Root command group

**Sub-Command Registration:**
Lazy imports ensure fast startup time:
```python
def _register_commands() -> None:
    from pamola_core.cli.commands.list_ops import app as list_ops_app
    from pamola_core.cli.commands.run import app as run_app
    from pamola_core.cli.commands.validate import app as validate_app
    from pamola_core.cli.commands.schema import show_schema

    app.add_typer(list_ops_app, name="list-ops")
    app.add_typer(run_app, name="run")
    app.add_typer(validate_app, name="validate-config")
    app.command("schema", ...)(show_schema)
```

### 2. Exit Codes (`utils/exit_codes.py`)

Standardized exit codes used across all CLI commands:

| Code | Meaning | Usage |
|------|---------|-------|
| `EXIT_OK` (0) | Success | Normal completion |
| `EXIT_ERROR` (1) | Runtime/unexpected error | File I/O, operation failure, unhandled exceptions |
| `EXIT_VALIDATION` (2) | Config/schema validation failure | JSON parsing errors, missing required fields, invalid parameters |

## Entry Point

The CLI is invoked via the command:
```bash
pamola-core [command] [options]
```

Configured in `pyproject.toml`:
```toml
[project.scripts]
pamola-core = "pamola_core.cli.main:app"
```

## Global Options

### `--version` / `-V`
Display PAMOLA.CORE version and exit.

**Example:**
```bash
pamola-core --version
# Output: pamola-core 0.0.1
```

### `--verbose` / `-v`
Enable debug logging for all commands.

**Example:**
```bash
pamola-core --verbose list-ops
# Displays DEBUG-level logs during operation discovery
```

## Command Structure

### Command 1: `list-ops`
List all registered PAMOLA operations.

**Syntax:**
```bash
pamola-core list-ops [--category <name>] [--format <format>]
```

**Options:**
- `--category`, `-c` — Filter by category (profiling, anonymization, transformations, metrics, attacks, fake_data)
- `--format`, `-f` — Output format: `table` (default) or `json`

**Examples:**
```bash
# List all operations
pamola-core list-ops

# Filter by category
pamola-core list-ops --category profiling

# JSON output for scripting
pamola-core list-ops --format json
```

See [`./commands/list_ops.md`](./commands/list_ops.md) for detailed documentation.

---

### Command 2: `run`
Execute a task pipeline or single operation.

**Syntax:**
```bash
# Task-mode
pamola-core run --task <path> [--output <dir>] [--seed <int>]

# Single-Operation mode
pamola-core run --op <name> --config <path> --input <path> [--output <dir>] [--seed <int>]
```

**Examples:**
```bash
# Run a full task pipeline
pamola-core run --task task.json --output ./results

# Run a single operation
pamola-core run --op AttributeSuppressionOperation --config cfg.json --input data.csv
```

See [`./commands/run.md`](./commands/run.md) for detailed documentation.

---

### Command 3: `validate-config`
Validate operation config or task JSON files.

**Syntax:**
```bash
pamola-core validate-config [--config <path>] [--task <path>] [--format <format>]
```

**Options:**
- `--config` — Path to single-operation config JSON
- `--task` — Path to task pipeline JSON
- `--format`, `-f` — Output format: `table` (default) or `json`

**Examples:**
```bash
# Validate operation config
pamola-core validate-config --config config.json

# Validate task JSON
pamola-core validate-config --task task.json

# JSON output for scripting
pamola-core validate-config --task task.json --format json
```

See [`./commands/validate.md`](./commands/validate.md) for detailed documentation.

---

### Command 4: `schema`
Display the parameter schema for an operation.

**Syntax:**
```bash
pamola-core schema <operation> [--format <format>]
```

**Arguments:**
- `operation` — Operation class name (e.g., `AggregateRecordsOperation`)

**Options:**
- `--format`, `-f` — Output format: `pretty` (default) or `json`

**Examples:**
```bash
# Show schema in pretty format
pamola-core schema AggregateRecordsOperation

# Show schema as JSON
pamola-core schema AggregateRecordsOperation --format json
```

See [`./commands/schema.md`](./commands/schema.md) for detailed documentation.

## Best Practices

1. **Use `validate-config` before `run`** — Always validate JSON files before execution to catch configuration errors early
2. **Inspect schemas first** — Run `pamola-core schema <op>` to understand required parameters before creating config files
3. **List operations by category** — Use `pamola-core list-ops --category <name>` to find operations relevant to your task
4. **Enable verbose mode for debugging** — Add `-v` flag when troubleshooting task execution failures
5. **Use JSON output for scripting** — Integrate CLI output into automation with `--format json`
6. **Provide meaningful output directories** — Use `--output` to organize results by project or date
7. **Set random seeds for reproducibility** — Use `--seed` when results must be deterministic

## Troubleshooting

### Issue: "Unknown operation: <name>"

**Cause:** Operation class name is misspelled or not registered.

**Solution:**
```bash
# List all available operations
pamola-core list-ops

# Search for operations containing a keyword
pamola-core list-ops | grep -i keyword

# Inspect schema to verify correct name
pamola-core schema <corrected-name>
```

### Issue: "Task JSON must define 'input_datasets'"

**Cause:** Missing required key in task definition.

**Solution:**
```bash
# Validate the task before running
pamola-core validate-config --task task.json

# Review error message and add missing keys
# Ensure task.json contains:
# {
#   "task_id": "...",
#   "input_datasets": {"name": "path/to/file.csv"},
#   "operations": [...]
# }
```

### Issue: "Invalid JSON in config file"

**Cause:** Malformed JSON syntax.

**Solution:**
```bash
# Validate JSON syntax
pamola-core validate-config --config config.json

# Use a JSON validator: jq, VS Code JSON validator, or online tools
jq . < config.json

# Check for common issues:
# - Missing commas between fields
# - Trailing commas in arrays/objects
# - Unquoted strings
# - Invalid escape sequences
```

### Issue: "Operation failed" with no details

**Cause:** Insufficient logging to diagnose the problem.

**Solution:**
```bash
# Run with verbose logging to see detailed error trace
pamola-core --verbose run --task task.json

# Check execution logs in output directory for more details
cat ./output/logs/task.log
```

## Related Components

- **Task Runner** — `pamola_core.utils.tasks.task_runner` — Executes operation pipelines
- **Operation Registry** — `pamola_core.utils.ops.op_registry` — Discovers and manages operations
- **Operations Catalog** — `pamola_core.catalogs` — Central registry of all operations (optional, auto-discovers at runtime)
- **Base Operation** — `pamola_core.utils.ops.op_base.BaseOperation` — Abstract base for all operations

**Related Commands:**
- See [`./commands/`](./commands/) directory for detailed command documentation
- See [`./utils/`](./utils/) directory for utility modules

## Summary Analysis

The `pamola_core.cli` module is a well-structured CLI application that:

1. **Provides user-friendly operation discovery** through the `list-ops` command with filtering and multiple output formats
2. **Enables flexible task execution** with both full-pipeline and single-operation modes
3. **Implements robust validation** to catch configuration errors before execution
4. **Offers schema inspection** for developers building task definitions
5. **Uses standardized exit codes** for reliable scripting and error handling
6. **Leverages lazy command loading** for minimal startup overhead
7. **Delivers rich, formatted output** with colored text and structured tables

The architecture supports both interactive use (humans running commands) and programmatic use (scripts and automation), making it versatile for diverse workflows.
