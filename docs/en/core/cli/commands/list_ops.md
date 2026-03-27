# `list-ops` Command Documentation

**Module:** `pamola_core.cli.commands.list_ops`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Classes/Methods](#core-classesmethods)
6. [Usage Examples](#usage-examples)
7. [Output Formats](#output-formats)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Related Components](#related-components)
11. [Summary Analysis](#summary-analysis)

## Overview

The `list-ops` command discovers and displays all registered PAMOLA operations. It supports filtering by category and multiple output formats (human-readable tables and JSON). The command implements FR-EP3-CORE-022 and falls back to runtime registry scanning if the operations catalog is unavailable.

**Command Name:** `pamola-core list-ops`

## Key Features

- **Operation Discovery** — Scans all registered operations from the operations catalog or runtime registry
- **Category Filtering** — Filter operations by category (profiling, anonymization, transformations, metrics, attacks, fake_data)
- **Multiple Output Formats** — Support for rich tables (default) and JSON output
- **Catalog Support** — Reads from `operations_catalog.yaml` (NFR-EP3-CORE-120) with fallback to runtime discovery
- **Rich Text Output** — Color-coded tables with styled columns
- **Graceful Fallback** — Automatically uses runtime registry if catalog is unavailable
- **Category Summary** — Displays available categories when listing all operations

## Architecture

### Data Flow

```
list_ops() command invoked
    ↓
Load operations from catalog (try)
    ├─ Success → Format and display
    └─ Failure → Fall back to runtime discovery
        ↓
    Discover operations at runtime
        ↓
    Build operation metadata (name, category, version, description)
        ↓
    Apply category filter (if specified)
        ↓
    Render as table or JSON
```

### Output Enum

```python
class OutputFormat(str, Enum):
    table = "table"     # Human-readable Rich table
    json = "json"       # Machine-readable JSON array
```

## Dependencies

**External:**
- `typer` — CLI framework for command definition
- `rich.console.Console` — Rich text output
- `rich.table.Table` — Table rendering

**Internal:**
- `pamola_core.catalogs.get_operations_catalog()` — Load operations from catalog
- `pamola_core.utils.ops.op_registry` — Operation discovery and metadata
  - `discover_operations()`
  - `list_operations()`
  - `get_operation_class()`
  - `get_operation_metadata()`
  - `get_operation_version()`
- `pamola_core.cli.utils.exit_codes` — Standardized exit codes

## Core Classes/Methods

### Main Command Function

#### `list_ops(category, fmt)`

List all registered PAMOLA operations with optional filtering.

**Signature:**
```python
def list_ops(
    category: Optional[str] = typer.Option(None, "--category", "-c", ...),
    fmt: OutputFormat = typer.Option(OutputFormat.table, "--format", "-f", ...)
) -> None
```

**Parameters:**
- `category` (str, optional) — Filter by category (profiling, anonymization, transformations, metrics, attacks, fake_data)
- `fmt` (OutputFormat) — Output format: `table` or `json` (default: `table`)

**Return Value:** None (prints to stdout, exits with code 0 or EXIT_ERROR/EXIT_VALIDATION)

**Exit Codes:**
- `0` — Success
- `EXIT_ERROR` (1) — Failed to load operations
- `0` — Empty result with message

**Behavior:**
1. Attempts to load operations from catalog using `_load_from_catalog()`
2. If no operations found, displays a message and exits with code 0
3. If `fmt == OutputFormat.json`, outputs JSON array
4. If `fmt == OutputFormat.table`, renders rich table
5. When listing all operations (no filter), displays available categories at the end

**Example:**
```python
# CLI invocation
pamola-core list-ops --category profiling --format json
```

---

### Loader Functions

#### `_load_from_catalog(category)`

Load operations from catalog with fallback to runtime discovery.

**Signature:**
```python
def _load_from_catalog(category: Optional[str]) -> list
```

**Parameters:**
- `category` (str, optional) — Optional category filter

**Return Value:** List of operation dicts with keys: `name`, `category`, `module`, `version`, `description`

**Flow:**
1. Try `pamola_core.catalogs.get_operations_catalog()` (if available)
2. If successful, extract metadata from catalog entries
3. On failure, fall back to runtime discovery:
   - Call `discover_operations("pamola_core")`
   - Call `list_operations(category=category)`
   - For each operation, fetch class, metadata, version, and docstring
   - Extract first line of docstring as description
4. Apply category filter if specified
5. Sort by (category, name) and return

**Exception Handling:**
- Catches all exceptions during catalog load, silently falls back to runtime discovery
- Runtime discovery does not raise exceptions; gracefully handles missing metadata

---

#### `_render_table(ops_data, category)`

Render operations as a rich-formatted table.

**Signature:**
```python
def _render_table(ops_data: list, category: Optional[str]) -> None
```

**Parameters:**
- `ops_data` (list) — List of operation dicts
- `category` (str, optional) — Current category filter (for display in title)

**Return Value:** None (prints to stdout)

**Output:**
- Table with columns: Operation, Category, Version, Description
- Styled with colors: cyan (operation name), magenta (category), green (version), dim (description)
- Shows total operation count at the bottom
- Title includes category name if filtered

**Example Output:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Operation                    ┃ Category  ┃ Version  ┃ Description       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ AttributeSuppressionOp       │ suppress  │ 1.0      │ Suppress attr...  │
│ AggregateRecordsOperation    │ transform │ 1.0      │ Aggregate records │
└─────────────────────────────┴──────────┴─────────┴───────────────────┘
Total: 2 operation(s)
```

## Usage Examples

### Example 1: List All Operations

Display all registered operations in table format.

**Command:**
```bash
pamola-core list-ops
```

**Output:**
```
PAMOLA Operations

Operation                          Category          Version  Description
─────────────────────────────────  ─────────────────  ───────  ──────────────────
AttributeSuppressionOperation      anonymization     1.0      Suppress attributes
AggregateRecordsOperation          transformations   1.0      Aggregate records
...

Categories: anonymization, fake_data, metrics, profiling, transformations
  |  Use --category <name> to filter
```

---

### Example 2: Filter by Category

Display only profiling operations.

**Command:**
```bash
pamola-core list-ops --category profiling
```

**Output:**
```
PAMOLA Operations — profiling

Operation              Category   Version  Description
───────────────────────  ─────────  ───────  ──────────────────
AnalyzeDateOperation   profiling  1.0      Analyze date fields
AnalyzeEmailOperation  profiling  1.0      Analyze email fields
AnalyzeTextOperation   profiling  1.0      Analyze text fields
...

Total: 14 operation(s)
```

---

### Example 3: JSON Output for Scripting

Export all operations as JSON for programmatic use.

**Command:**
```bash
pamola-core list-ops --format json
```

**Output:**
```json
[
  {
    "name": "AttributeSuppressionOperation",
    "category": "anonymization",
    "module": "pamola_core.anonymization.suppression",
    "version": "1.0",
    "description": "Suppress sensitive attributes from dataset"
  },
  {
    "name": "AggregateRecordsOperation",
    "category": "transformations",
    "module": "pamola_core.transformations.grouping",
    "version": "1.0",
    "description": "Aggregate records by grouping key"
  }
]
```

---

### Example 4: Filter by Category + JSON Output

Get anonymization operations as JSON.

**Command:**
```bash
pamola-core list-ops --category anonymization --format json | jq '.[] | .name'
```

**Output:**
```json
"AttributeSuppressionOperation"
"CellSuppressionOperation"
"RecordSuppressionOperation"
"FullMaskingOperation"
"PartialMaskingOperation"
...
```

## Output Formats

### Table Format (Default)

Human-readable Rich table with colors and styling.

**Columns:**
1. **Operation** — Class name (cyan, no wrap)
2. **Category** — Category name (magenta, centered)
3. **Version** — Semantic version (green, centered)
4. **Description** — First line of docstring (dim, wrapped)

**Features:**
- Color-coded for easy visual scanning
- Shows total count at bottom
- Category summary when listing all operations
- Dynamically sized based on content

### JSON Format

Machine-readable JSON array of operation objects.

**Structure:**
```json
[
  {
    "name": "string",
    "category": "string",
    "module": "string",
    "version": "string",
    "description": "string"
  }
]
```

**Use Cases:**
- Script integration (curl, jq, grep)
- CI/CD pipelines
- Automation tools
- Programmatic inspection

## Best Practices

1. **Explore before you code** — Run `pamola-core list-ops` to understand available operations before writing task definitions
2. **Use category filtering** — Organize operations by domain using `--category` to reduce cognitive load
3. **Inspect schemas** — After listing operations, run `pamola-core schema <op>` to understand parameters
4. **Validate before running** — Use `pamola-core validate-config` before executing tasks
5. **Export for documentation** — Use JSON output to generate operation catalogs or reference docs
6. **Pipeline JSON filtering** — Combine JSON output with `jq` for advanced filtering:
   ```bash
   pamola-core list-ops --format json | jq '.[] | select(.category == "anonymization") | .name'
   ```
7. **Track operation versions** — Monitor version field in JSON output to identify breaking changes

## Troubleshooting

### Issue: "Failed to load operations: <error>"

**Cause:** Catalog file missing or corrupted; runtime discovery also failed.

**Solution:**
```bash
# Ensure operation modules are installed
pip install -e .

# Check if operations are discoverable
python -c "from pamola_core.utils.ops.op_registry import discover_operations; discover_operations('pamola_core'); print('Success')"

# Run with verbose logging
pamola-core --verbose list-ops
```

### Issue: No operations found in category

**Cause:** Category name is incorrect or no operations registered in that category.

**Solution:**
```bash
# List all available categories
pamola-core list-ops

# Check spelling of category
pamola-core list-ops --category profiling  # not "profiling_operations"

# Count operations by category
pamola-core list-ops --format json | jq 'group_by(.category) | map({category: .[0].category, count: length})'
```

### Issue: JSON output not valid

**Cause:** Terminal encoding issue or incomplete output.

**Solution:**
```bash
# Save to file and verify
pamola-core list-ops --format json > ops.json

# Validate JSON
jq . < ops.json

# Check file encoding
file ops.json

# Retry with explicit UTF-8 encoding
PYTHONIOENCODING=utf-8 pamola-core list-ops --format json
```

### Issue: Missing descriptions in table

**Cause:** Operations without docstrings or first-line descriptions.

**Solution:**
- Use JSON format to see raw metadata:
  ```bash
  pamola-core list-ops --format json | jq '.[] | select(.description == "") | .name'
  ```
- Consider inspecting operation source code for undocumented operations

## Related Components

- **Operation Registry** — `pamola_core.utils.ops.op_registry` — Core discovery engine
- **Operations Catalog** — `pamola_core.catalogs` — Central operation metadata source
- **Schema Command** — `pamola-core schema` — Inspect operation parameters
- **Validate Command** — `pamola-core validate-config` — Validate operation configs
- **Run Command** — `pamola-core run` — Execute operations

## Summary Analysis

The `list-ops` command provides essential operation discovery functionality:

1. **Dual-source loading** — Prefers catalog (faster) with automatic fallback to runtime discovery (reliable)
2. **Flexible filtering** — Category filtering enables focused exploration of operation subsets
3. **User and machine-friendly output** — Table format for humans, JSON for automation
4. **Graceful error handling** — Never fails due to missing catalog; always falls back
5. **Rich formatting** — Color-coded output improves UX and reduces scanning time
6. **Integrated metadata** — Displays operation name, category, version, and description in one view

The command is designed for both exploratory use (discovering what's available) and programmatic use (scripting, automation, documentation generation).
