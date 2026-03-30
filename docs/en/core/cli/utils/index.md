# CLI Utilities Reference

**Location:** `pamola_core/cli/utils/`
**Last Updated:** 2026-03-23

## Overview

This directory contains utility modules that support CLI command implementation. These utilities provide shared functionality for error handling, exit code management, and common CLI operations.

## Module Reference

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `exit_codes` | Standardized exit codes for CLI commands | [exit_codes.md](./exit_codes.md) |

## Module Details

### `exit_codes.py` — CLI Exit Codes

**Purpose:** Define standardized exit codes used across all CLI commands

**Exports:**

```python
EXIT_OK = 0        # Success
EXIT_ERROR = 1     # Runtime / unexpected error
EXIT_VALIDATION = 2  # Config / schema validation failure
```

**Usage in Commands:**

All CLI commands use these exit codes for consistent error reporting:

```python
from pamola_core.cli.utils.exit_codes import EXIT_OK, EXIT_ERROR, EXIT_VALIDATION

# Success case
raise typer.Exit(EXIT_OK)  # or just exit naturally

# Runtime error
raise typer.Exit(EXIT_ERROR)

# Validation error
raise typer.Exit(EXIT_VALIDATION)
```

**Integration Points:**

- `pamola_core/cli/commands/list_ops.py` — Uses EXIT_ERROR
- `pamola_core/cli/commands/run.py` — Uses EXIT_ERROR, EXIT_VALIDATION
- `pamola_core/cli/commands/validate.py` — Uses EXIT_ERROR, EXIT_VALIDATION
- `pamola_core/cli/commands/schema.py` — Uses EXIT_ERROR

**Exit Code Semantics:**

| Code | When to Use | Examples |
|------|------------|----------|
| 0 | Command succeeded | Operation listed, config valid, task completed |
| 1 | Unexpected runtime error | File I/O error, registry load failure, execution failure |
| 2 | Configuration validation error | Invalid JSON, missing required field, schema mismatch |

**Shell Script Integration:**

```bash
# Check specific exit code
pamola-core validate-config --task task.json
if [ $? -eq 2 ]; then
    echo "Configuration error"
elif [ $? -eq 1 ]; then
    echo "Unexpected error"
else
    echo "Valid configuration"
fi

# Use with conditional execution
pamola-core validate-config --task task.json && \
  pamola-core run --task task.json || exit 1
```

**Best Practices:**

1. Use EXIT_VALIDATION for predictable configuration errors
2. Use EXIT_ERROR for unexpected runtime failures
3. Always check exit codes in shell scripts
4. Use different exit codes to enable smart retry logic
5. Document which exit codes each command can return

See [exit_codes.md](./exit_codes.md) for comprehensive documentation.

## Architecture

### CLI Utility Dependencies

```
CLI Commands
    ↓
exit_codes (utilities)
    ↓
System Shell / CI/CD
```

Commands import utilities and use them for consistent behavior:

```python
# In any command file
from pamola_core.cli.utils.exit_codes import EXIT_ERROR, EXIT_VALIDATION

# Use in command logic
if error_condition:
    raise typer.Exit(EXIT_ERROR)
elif validation_failure:
    raise typer.Exit(EXIT_VALIDATION)
```

## File Organization

```
utils/
├── __init__.py (empty)
├── index.md (this file)
└── exit_codes.py
```

## Cross-Module Patterns

### Exit Code Usage Pattern

All CLI commands follow this pattern:

```python
from pamola_core.cli.utils.exit_codes import EXIT_ERROR, EXIT_VALIDATION

def command():
    # Check input validity
    if not valid_input:
        console.print("[red]✗ Error message[/red]")
        raise typer.Exit(EXIT_ERROR)

    # Validate configuration
    try:
        data = parse_config()
    except ParseError as e:
        console.print(f"[red]✗ Parse error:[/red] {e}")
        raise typer.Exit(EXIT_VALIDATION)

    # Perform operation
    try:
        result = execute_operation()
    except RuntimeError as e:
        console.print(f"[red]✗ Execution failed:[/red] {e}")
        raise typer.Exit(EXIT_ERROR)

    # Success
    console.print("[green]✓ Operation completed[/green]")
```

## Future Extensions

The `utils` directory can be extended with:

- **Input Validation** — Shared validation functions for CLI arguments
- **Output Formatting** — Common formatting utilities for rich output
- **Logging Helpers** — Centralized logging configuration
- **Error Handling** — Common exception handling patterns
- **Progress Indicators** — Shared progress bar utilities

## Related Documentation

- **CLI Overview** — `../cli_overview.md` — Full CLI system documentation
- **Commands** — `../commands/` — Individual command documentation
- **Project Overview** — `../../../project-overview-pdr.md` — System architecture

## Summary

The `utils` module provides foundational utilities for CLI reliability:

1. **Standardized exit codes** — Consistent error reporting across all commands
2. **Shell integration** — Enables reliable scripting and automation
3. **Error semantics** — Distinguishes between configuration and runtime errors
4. **Extensible design** — Foundation for additional utility modules

The exit codes module is the primary utility currently in use, providing critical infrastructure for CLI error handling and operational reliability.
