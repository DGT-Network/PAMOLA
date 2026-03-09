# PAMOLA.CORE CLI Reference

> Entry point: `pamola-core`
> Defined in `pyproject.toml` → `pamola_core.cli.main:app`

---

## Global Options

These options apply to every command.

| Option | Short | Description |
|---|---|---|
| `--version` | `-V` | Show version and exit |
| `--verbose` | `-v` | Enable DEBUG logging output |
| `--help` | `-h` | Show help and exit |

```bash
pamola-core --version
pamola-core --verbose list-ops
```

---

## Commands Overview

```
pamola-core
├── list-ops          List all registered operations
├── run               Execute a task pipeline or single operation
├── validate-config   Validate a task or operation config JSON
└── schema            Show parameter schema for an operation
```

---

## `list-ops` — List Operations

**REQ:** FR-EP3-CORE-022

```bash
pamola-core list-ops [OPTIONS]
```

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--category` | `-c` | `str` | — | Filter by category (`profiling`, `anonymization`, `field`, `dataframe`) |
| `--format` | `-f` | `table\|json` | `table` | Output format |

### Examples

```bash
# List all operations (table)
pamola-core list-ops

# Filter by category
pamola-core list-ops --category profiling
pamola-core list-ops --category anonymization

# JSON output (for scripting)
pamola-core list-ops --format json

# Combined
pamola-core list-ops --category profiling --format json
```

---

## `run` — Execute Task or Operation

**REQ:** FR-EP3-CORE-023

```bash
pamola-core run [OPTIONS]
```

### Mode 1 — Task Pipeline (from JSON)

| Option | Short | Type | Required | Description |
|---|---|---|---|---|
| `--task` | `-t` | `path` | ✓ | Path to task JSON file |
| `--output` | `-o` | `path` | — | Output directory (auto-generated if omitted) |
| `--seed` | — | `int` | — | Random seed for reproducibility |

```bash
pamola-core run --task configs/demo_task.json
pamola-core run --task configs/demo_task.json --output ./results
pamola-core run --task configs/demo_task.json --output ./results --seed 42
```

### Mode 2 — Single Operation

| Option | Short | Type | Required | Description |
|---|---|---|---|---|
| `--op` | — | `str` | ✓ | Operation class name |
| `--input` | — | `path` | ✓ | Input CSV/Parquet file |
| `--config` | — | `path` | — | Operation parameters JSON |
| `--output` | `-o` | `path` | — | Output directory |

```bash
pamola-core run --op AttributeSuppressionOperation \
                --input data/sample.csv \
                --config configs/op_config.json \
                --output ./results
```

### Task JSON Format

```json
{
  "task_id": "my_task",
  "task_type": "anonymization",
  "description": "My anonymization pipeline",

  "input_datasets": {
    "main": "data/sample.csv"
  },
  "auxiliary_datasets": {},

  "additional_options": {
    "encoding": "utf-8",
    "sep": ",",
    "delimiter": ",",
    "quotechar": "\"",
    "orient": "columns"
  },

  "operations": [
    {
      "task_operation_id": "op_001",
      "task_operation_order_index": 1,
      "operation": "anonymization",
      "class_name": "AttributeSuppressionOperation",
      "dataset_name": "main",
      "parameters": {
        "suppression_mode": "REMOVE",
        "save_suppressed_schema": true
      },
      "scope": {
        "target": ["email", "phone"]
      }
    },
    {
      "task_operation_id": "op_002",
      "task_operation_order_index": 2,
      "operation": "anonymization",
      "class_name": "FullMaskingOperation",
      "dataset_name": "main",
      "parameters": {
        "mode": "REPLACE"
      },
      "scope": {
        "target": ["name"]
      }
    }
  ]
}
```

**Key fields:**

| Field | Required | Description |
|---|---|---|
| `task_id` | ✓ | Unique task identifier |
| `task_type` | ✓ | Task type (`anonymization`, `profiling`, etc.) |
| `input_datasets` | ✓ | Dict of `{ "alias": "path/to/file.csv" }` |
| `auxiliary_datasets` | — | Secondary datasets |
| `additional_options` | — | CSV encoding/delimiter settings |
| `operations` | ✓ | Ordered list of operation configs |

**Each operation:**

| Field | Required | Description |
|---|---|---|
| `class_name` | ✓ | Operation class (see `list-ops`) |
| `operation` | ✓ | Operation type label |
| `dataset_name` | ✓ | Which dataset alias to process |
| `parameters` | — | Operation-specific config |
| `scope.target` | — | List of field names to apply the operation to |
| `task_operation_id` | — | Unique ID for this step |
| `task_operation_order_index` | — | Execution order (1-based) |

---

## `validate-config` — Validate JSON Config

**REQ:** FR-EP3-CORE-024

```bash
pamola-core validate-config [OPTIONS]
```

| Option | Type | Description |
|---|---|---|
| `--config` | `path` | Path to single-operation config JSON |
| `--task` | `path` | Path to task pipeline JSON |

```bash
# Validate a task pipeline
pamola-core validate-config --task configs/demo_task.json

# Validate a single-operation config
pamola-core validate-config --config configs/op_config.json
```

---

## `schema` — Show Operation Schema

**REQ:** FR-EP3-CORE-025

```bash
pamola-core schema OPERATION [OPTIONS]
```

| Argument / Option | Type | Description |
|---|---|---|
| `OPERATION` | `str` (positional) | Operation class name |
| `--format` / `-f` | `pretty\|json` | Output format (default: `pretty`) |

```bash
# Pretty table (default)
pamola-core schema AttributeSuppressionOperation

# JSON output (for tooling/docs)
pamola-core schema AttributeSuppressionOperation --format json
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Runtime / unexpected error |
| `2` | Validation failure (bad JSON, missing fields, unknown operation) |

---

## Debugging

### VSCode (breakpoints)

Open **Run & Debug** (`Ctrl+Shift+D`), choose a launch config from `.vscode/launch.json`, set breakpoints, press **F5**.

Available configs:
- `CLI: --version`
- `CLI: list-ops`
- `CLI: list-ops --category profiling`
- `CLI: run --task (demo)`
- `CLI: run --op (single)`
- `CLI: validate-config`
- `CLI: schema <operation>`

### Terminal (verbose logging)

```bash
pamola-core --verbose <command>

# Examples
pamola-core --verbose list-ops
pamola-core --verbose run --task configs/demo_task.json
```

---

## Demo Config

A working demo is available at [`configs/demo_task.json`](../../configs/demo_task.json).

```bash
pamola-core run --task configs/demo_task.json --output ./output/demo
```

> Update `input_datasets.main` in the JSON to point to your actual CSV file.
