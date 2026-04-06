# `schema` Command Documentation

**Module:** `pamola_core.cli.commands.schema`
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

The `schema` command displays the parameter schema for any registered PAMOLA operation. It shows operation metadata (name, version, category, module) and a detailed parameter list with types, requirements, and defaults. The command supports both pretty-printed and JSON output formats.

**Command Name:** `pamola-core schema`

## Key Features

- **Parameter Inspection** — View all parameters for any operation
- **Type Information** — Display parameter type annotations
- **Requirement Status** — Indicate required vs optional parameters
- **Default Values** — Show parameter defaults for optional fields
- **Multiple Output Formats** — Pretty-printed tables (default) and JSON for scripting
- **Metadata Display** — Show operation version, category, and module location
- **Rich Text Output** — Color-coded table with styled columns
- **Error Handling** — Graceful handling of unknown operations with helpful suggestions

## Architecture

### Schema Retrieval Flow

```
show_schema(operation, fmt) invoked
    ↓
Discover operations (populate registry)
    ↓
Lookup operation by class name
    ├─ Found → Continue
    └─ Not found → Display error & suggest list-ops command
    ↓
Retrieve metadata: version, module, category, parameters
    ↓
Format and display
    ├─ JSON format → Serialize to JSON
    └─ Pretty format → Render rich table
```

### Schema Format Enum

```python
class SchemaFormat(str, Enum):
    pretty = "pretty"  # Human-readable table (default)
    json = "json"      # Machine-readable JSON
```

## Dependencies

**External:**
- `typer` — CLI framework
- `rich.console.Console` — Rich text output
- `rich.table.Table` — Table rendering
- `json` — JSON serialization

**Internal:**
- `pamola_core.utils.ops.op_registry` — Operation discovery and metadata retrieval
  - `discover_operations()`
  - `get_operation_class()`
  - `get_operation_metadata()`
  - `get_operation_version()`
- `pamola_core.cli.utils.exit_codes` — Exit code constants

## Core Classes/Methods

### Main Command Function

#### `show_schema(operation, fmt)`

Display the parameter schema for an operation.

**Signature:**
```python
def show_schema(
    operation: str = typer.Argument(..., help="Operation class name, e.g. AggregateRecordsOperation."),
    fmt: SchemaFormat = typer.Option(
        SchemaFormat.pretty,
        "--format",
        "-f",
        help="Output format: pretty (default) or json.",
    ),
) -> None
```

**Parameters:**
- `operation` (str, required) — Operation class name (e.g., `AttributeSuppressionOperation`)
- `fmt` (SchemaFormat, optional) — Output format: `pretty` or `json` (default: `pretty`)

**Return Value:** None (prints to stdout, exits with code 0 or EXIT_ERROR)

**Exit Codes:**
- `0` — Success
- `EXIT_ERROR` (1) — Unknown operation

**Behavior:**
1. Discover operations to populate registry
2. Look up operation class by name
3. If not found, display error and suggest using `list-ops` command, then exit with EXIT_ERROR
4. Retrieve metadata (version, module, category, parameters)
5. If `fmt == SchemaFormat.json`, call helper to format as JSON
6. If `fmt == SchemaFormat.pretty`, call `_render_table()` to format as table
7. Output result and exit with code 0

---

### Rendering Function

#### `_render_table(op_name, version, meta, params)`

Render operation schema as a rich-formatted table.

**Signature:**
```python
def _render_table(op_name: str, version: str, meta: dict, params: dict) -> None
```

**Parameters:**
- `op_name` — Operation class name
- `version` — Semantic version string
- `meta` — Operation metadata dict (category, module)
- `params` — Parameters dict with type/requirement info

**Return Value:** None (prints to stdout)

**Output Format:**
```
OperationName  v1.0  category

module.location

Parameter         Type           Required  Default
─────────────────  ─────────────  ─────────  ─────────────
param1             str            ✓
param2             int
param3             float                     0.5
```

**Table Structure:**
- Header line: Operation name (bold cyan), version (dim), category (magenta)
- Module line: Module path (dim)
- Blank line
- Table with columns:
  - **Parameter** — Parameter name (cyan, no wrap)
  - **Type** — Type annotation (yellow)
  - **Required** — Red checkmark if required, empty if optional
  - **Default** — Default value if available, empty if required

**Special Cases:**
- If no parameters defined: Display "[yellow]No parameters defined.[/yellow]"
- If type annotation is missing: Display "—" placeholder
- If required: Leave default blank
- If optional: Show default value

---

## Usage Examples

### Example 1: Display Schema in Pretty Format

Show operation parameters in human-readable format.

**Command:**
```bash
pamola-core schema AggregateRecordsOperation
```

**Output:**
```
AggregateRecordsOperation  v1.0  transformations

pamola_core.transformations.grouping

Parameter               Type              Required  Default
──────────────────────  ────────────────  ─────────  ──────────
grouping_key            str               ✓
aggregation_config      dict                        {}
output_format           str                         "compact"
include_metadata        bool                        True
```

---

### Example 2: Display Schema as JSON

Export operation schema for programmatic use.

**Command:**
```bash
pamola-core schema AggregateRecordsOperation --format json
```

**Output:**
```json
{
  "operation": "AggregateRecordsOperation",
  "version": "1.0",
  "module": "pamola_core.transformations.grouping",
  "category": "transformations",
  "parameters": {
    "grouping_key": {
      "type": "str",
      "required": true,
      "default": null
    },
    "aggregation_config": {
      "type": "dict",
      "required": false,
      "default": {}
    },
    "output_format": {
      "type": "str",
      "required": false,
      "default": "compact"
    },
    "include_metadata": {
      "type": "bool",
      "required": false,
      "default": true
    }
  }
}
```

---

### Example 3: Query Schema with jq

Extract specific parameter information using jq.

**Command:**
```bash
pamola-core schema AggregateRecordsOperation --format json | \
  jq '.parameters | keys'
```

**Output:**
```json
[
  "grouping_key",
  "aggregation_config",
  "output_format",
  "include_metadata"
]
```

---

### Example 4: Find Required Parameters

Identify all required parameters for an operation.

**Command:**
```bash
pamola-core schema AttributeSuppressionOperation --format json | \
  jq '.parameters | to_entries[] | select(.value.required) | .key'
```

**Output:**
```
keep_probability
```

---

### Example 5: Compare Schemas

Display schemas for multiple operations side-by-side (manual comparison).

**Commands:**
```bash
# Operation 1
pamola-core schema AttributeSuppressionOperation --format json > op1.json

# Operation 2
pamola-core schema FullMaskingOperation --format json > op2.json

# Compare parameters
diff <(jq '.parameters | keys' op1.json) <(jq '.parameters | keys' op2.json)
```

---

### Example 6: Unknown Operation

Show error message when operation not found.

**Command:**
```bash
pamola-core schema NonexistentOperation
```

**Output:**
```
✗ Unknown operation: NonexistentOperation
  Run pamola-core list-ops to see available operations.
```

---

## Output Formats

### Pretty Format (Default)

Human-readable table with rich styling.

**Components:**
1. **Header** — Operation name (cyan, bold), version (dim), category (magenta)
2. **Module** — Module path where operation is defined (dim text)
3. **Table** — Parameter details with 4 columns

**Column Details:**

| Column | Styling | Content |
|--------|---------|---------|
| Parameter | cyan, no wrap | Parameter name as defined in `__init__` |
| Type | yellow | Type annotation from signature |
| Required | red | Red checkmark (✓) or empty |
| Default | green | Default value or empty |

**Table Features:**
- Show lines between rows for clarity
- Dynamic column widths based on content
- Highlight enabled for visual distinction

### JSON Format

Machine-readable JSON structure for scripting and integration.

**Root Object Keys:**
- `operation` (string) — Operation class name
- `version` (string) — Semantic version
- `module` (string) — Python module path
- `category` (string) — Operation category
- `parameters` (object) — Parameter details

**Parameters Object Structure:**
Each parameter key maps to an object with:
- `type` (string) — Type annotation string
- `required` (boolean) — Whether parameter is required
- `default` (any) — Default value or null

**Example:**
```json
{
  "operation": "AttributeSuppressionOperation",
  "version": "1.0",
  "module": "pamola_core.anonymization.suppression",
  "category": "anonymization",
  "parameters": {
    "keep_probability": {
      "type": "float",
      "required": true,
      "default": null
    }
  }
}
```

## Best Practices

1. **Inspect before configuring** — Always run `pamola-core schema <op>` before creating operation configs
2. **Check required parameters** — Identify mandatory fields to avoid validation errors
3. **Review defaults** — Understand which parameters have defaults for optional configuration
4. **Use JSON for automation** — Export schemas with `--format json` for integration into scripts
5. **Compare related operations** — Use JSON output to find parameter name variations across operations
6. **Document your configs** — Reference schema output when documenting task definitions
7. **Validate parameter types** — Ensure your config values match the expected types
8. **Filter with jq** — Combine with jq for advanced schema inspection:
   ```bash
   pamola-core schema MyOp --format json | jq '.parameters | with_entries(select(.value.required))'
   ```
9. **Check version compatibility** — Note operation version when storing configs
10. **Review module location** — Understand where operations are defined for debugging

## Troubleshooting

### Issue: "Unknown operation: <name>"

**Cause:** Operation class name is misspelled, not registered, or from a different module.

**Solution:**
```bash
# List all available operations to find correct name
pamola-core list-ops

# Search for operations by keyword
pamola-core list-ops --format json | jq '.[] | select(.name | contains("Suppress"))'

# Verify correct spelling and casing
pamola-core schema AttributeSuppressionOperation  # not attribute_suppression_operation
```

---

### Issue: "No parameters defined"

**Cause:** Operation has no configurable parameters (uses defaults only).

**Solution:**
- Review operation source code for inherited parameters
- Check parent class schema if operation extends BaseOperation
- Verify operation can be instantiated without arguments:
  ```bash
  python -c "from pamola_core.anonymization.suppression import AttributeSuppressionOperation; op = AttributeSuppressionOperation()"
  ```

---

### Issue: Invalid JSON output format

**Cause:** Terminal encoding or output piping issue.

**Solution:**
```bash
# Save to file and verify encoding
pamola-core schema MyOp --format json > schema.json

# Validate JSON
jq . < schema.json

# Check for encoding issues
file schema.json

# Use explicit UTF-8 encoding
PYTHONIOENCODING=utf-8 pamola-core schema MyOp --format json
```

---

### Issue: Type annotation shows "NoneType" or similar

**Cause:** Parameter type hint is missing or uses non-standard notation.

**Solution:**
- Review operation source code docstring for parameter descriptions
- Check inspect library output:
  ```bash
  python -c "import inspect; from pamola_core import <op>; print(inspect.signature(<op>.__init__))"
  ```
- Note in task definition and use appropriate JSON types

---

## Related Components

- **Operation Registry** — `pamola_core.utils.ops.op_registry` — Core metadata source
- **List-Ops Command** — `pamola-core list-ops` — Discover available operations
- **Validate Command** — `pamola-core validate-config` — Validate configs against schemas
- **Run Command** — `pamola-core run` — Execute operations with the schema
- **Base Operation** — `pamola_core.utils.ops.op_base.BaseOperation` — Base class with parameter framework

## Summary Analysis

The `schema` command provides essential parameter inspection:

1. **Comprehensive metadata** — Operation name, version, category, module, and all parameters
2. **Dual output formats** — Pretty tables for humans, JSON for automation
3. **Clear parameter documentation** — Type, requirement status, and defaults in one view
4. **Integration ready** — JSON output enables scripting and tool integration
5. **Error guidance** — Helpful messages when operations aren't found
6. **Rich formatting** — Color-coded, visually organized table output

The command is critical for developers building task definitions and understanding operation capabilities before configuration and execution.
