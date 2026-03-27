# Schema Helpers Module Documentation

**Package:** `pamola_core.utils.schema_helpers`
**Version:** 1.0
**Last Updated:** 2026-03-23
**Type:** Internal (Non-Public API)

## Overview

The `schema_helpers` package provides utilities for manipulating, transforming, and generating JSON schemas used throughout PAMOLA.CORE. These helpers support schema flattening, form generation, schema building, and UI schema conversion for dynamic form creation and validation.

The package is essential for bridging operational configurations and user-facing forms, enabling automatic UI generation from operation schemas.

## Architecture

```
pamola_core.utils.schema_helpers/
├── Core Functions
│   ├── schema_utils.py - Schema flattening and manipulation
│   ├── form_builder.py - JSON schema to Formily conversion
│   └── schema_builder.py - Auto schema generation
│
└── __init__.py - Public API (empty - internal only)
```

## Component Files

| File | Purpose | Key Functions |
|------|---------|---|
| `schema_utils.py` | Schema manipulation | `flatten_schema()`, schema validation, property generation |
| `form_builder.py` | Form generation | `convert_json_schema_to_formily()`, Formily schema conversion |
| `schema_builder.py` | Schema auto-generation | `generate_schema_json()`, config-to-schema conversion |

## Key Concepts

### Schema Flattening

JSON Schema composition often uses `allOf` to combine multiple schemas. The `flatten_schema()` function merges these into a single-level schema:

```python
# Input: Schema with allOf composition
{
    "allOf": [
        {"properties": {"field1": {"type": "string"}}},
        {"properties": {"field2": {"type": "integer"}}}
    ]
}

# Output: Flattened schema
{
    "type": "object",
    "properties": {
        "field1": {"type": "string"},
        "field2": {"type": "integer"}
    }
}
```

### Formily Schema Conversion

Formily is a form framework used by PAMOLA's web interfaces. The `convert_json_schema_to_formily()` function transforms standard JSON schemas into Formily format for dynamic form generation:

```python
# Input: Standard JSON Schema
{
    "type": "object",
    "properties": {
        "email": {"type": "string", "format": "email"},
        "age": {"type": "integer"}
    }
}

# Output: Formily Schema
{
    "type": "object",
    "properties": {
        "email": {
            "type": "string",
            "x-component": "Input",
            "x-rules": [{"required": true}]
        },
        "age": {
            "type": "number",
            "x-component": "NumberInput"
        }
    }
}
```

### Operation Config Schemas

Schema helpers support automatic schema generation from operation configuration classes:

```python
from pamola_core.utils.schema_helpers.schema_utils import generate_schema_json

# Generate schema from operation config
schema = generate_schema_json(MyOperationConfig)

# Schema includes all fields, types, defaults, and constraints
```

## Usage Patterns

### Pattern 1: Flatten Complex Schema

```python
from pamola_core.utils.schema_helpers.schema_utils import flatten_schema

complex_schema = {
    "allOf": [
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        },
        {
            "type": "object",
            "properties": {
                "age": {"type": "integer"}
            }
        }
    ]
}

flat = flatten_schema(complex_schema)
# Result: merged properties with all fields
```

### Pattern 2: Convert to Formily for UI Generation

```python
from pamola_core.utils.schema_helpers.form_builder import convert_json_schema_to_formily

schema = {
    "type": "object",
    "properties": {
        "masking_char": {
            "type": "string",
            "title": "Masking Character",
            "default": "X"
        },
        "case_sensitive": {
            "type": "boolean",
            "title": "Case Sensitive",
            "default": False
        }
    }
}

formily_schema = convert_json_schema_to_formily(schema)
# Result: Formily-compatible schema for form rendering
```

### Pattern 3: Generate Schema from Operation Config

```python
from pamola_core.anonymization import FullMaskingOperation
from pamola_core.utils.schema_helpers.schema_utils import generate_schema_json

# Extract operation config
operation = FullMaskingOperation()
config = operation.config

# Generate JSON schema
schema = generate_schema_json(config.__class__)

# Schema is ready for UI form generation
```

### Pattern 4: Filter Properties

```python
from pamola_core.utils.schema_helpers.schema_utils import flatten_schema

schema = {
    "allOf": [...],
    "properties": {
        "field1": {...},
        "field2": {...},
        "field3": {...}
    }
}

# Filter out specific fields
flat = flatten_schema(schema, unused_fields=["field2"])
# Result: field1 and field3 only
```

## Schema Structure

### Standard JSON Schema Format

```python
{
    "type": "object",
    "title": "Operation Configuration",
    "description": "Configuration for the operation",
    "properties": {
        "field_name": {
            "type": "string|integer|boolean|array|object",
            "title": "Human-Readable Title",
            "description": "Field description",
            "default": default_value,
            "enum": [possible, values],
            "pattern": "regex pattern",  # for strings
            "minimum": min_value,  # for numbers
            "maximum": max_value,
            "items": {...}  # for arrays
        }
    },
    "required": ["field1", "field2"],
    "allOf": [...]  # composition
}
```

### Formily Schema Extensions

Formily adds UI-specific metadata:

```python
{
    "type": "object",
    "properties": {
        "field_name": {
            "type": "string",
            "title": "Field Title",
            "x-component": "Input",  # UI component
            "x-component-props": {  # component properties
                "maxLength": 100,
                "placeholder": "Enter value"
            },
            "x-rules": [  # validation rules
                {"required": true, "message": "Required field"}
            ],
            "x-display": "visible|hidden|none",
            "x-visible": "expression"
        }
    }
}
```

## API Reference

### schema_utils.py

#### flatten_schema()

```python
def flatten_schema(
    schema: dict,
    unused_fields: Optional[List[str]] = None
) -> dict
```

**Purpose:** Flatten JSON schema with allOf composition into single-level schema.

**Parameters:**
- `schema`: JSON schema to flatten
- `unused_fields`: List of property names to exclude

**Returns:** Flattened schema with merged properties

**Example:**
```python
flat = flatten_schema(schema, unused_fields=["internal_field"])
```

#### generate_schema_json()

```python
def generate_schema_json(
    config_class: Type[OperationConfig]
) -> Dict[str, Any]
```

**Purpose:** Generate JSON schema from operation config class.

**Parameters:**
- `config_class`: Operation config class

**Returns:** JSON schema representing config structure

### form_builder.py

#### convert_json_schema_to_formily()

```python
def convert_json_schema_to_formily(
    schema: Dict[str, Any],
    operation_config_type: Optional[str] = None
) -> Dict[str, Any]
```

**Purpose:** Convert JSON schema to Formily-compatible format.

**Parameters:**
- `schema`: JSON schema
- `operation_config_type`: Optional operation type for grouping

**Returns:** Formily schema with UI components and validation

**Example:**
```python
formily_schema = convert_json_schema_to_formily(schema)
```

### schema_builder.py

#### Various schema generation functions

Generates schemas for specific operation types:
- `generate_all_op_schemas()` - All operations
- Per-operation schema generators
- Config-to-schema conversion

## Integration Points

`schema_helpers` is used by:

- **Web UI**: Form generation and rendering
- **Operation Registry**: Schema exposure via Sphinx documentation
- **API Endpoints**: Configuration validation
- **CLI**: Parameter validation
- **Test Frameworks**: Dynamic test case generation

## Best Practices

1. **Use Schema Flattening for Complex Schemas**
   - Flatten allOf compositions before validation
   - Simplifies downstream schema processing
   - Improves performance

2. **Generate Formily Schemas for Web UI**
   - Always convert to Formily for web forms
   - Provides consistent user experience
   - Enables advanced validation

3. **Document Field Constraints**
   - Include title and description for all fields
   - Specify min/max for numeric fields
   - Provide enum values for restricted fields

4. **Filter Unused Fields**
   - Use `unused_fields` parameter to hide internal fields
   - Reduces complexity for UI generation
   - Improves user experience

5. **Maintain Schema Consistency**
   - Use schema generation from config classes
   - Avoid manual schema creation
   - Keeps schema and code in sync

## Common Patterns

### Field Type Mapping

Schema helpers automatically map Python types to JSON Schema types:

| Python Type | JSON Schema Type |
|-------------|-----------------|
| `str` | "string" |
| `int` | "integer" |
| `float` | "number" |
| `bool` | "boolean" |
| `List[T]` | "array" with items |
| `Dict[K, V]` | "object" |
| `Optional[T]` | Type with nullable |

### Component Mapping

Formily component selection based on field type:

| Field Type | Default Component |
|-----------|-----------------|
| "string" | "Input" |
| "integer" | "NumberInput" |
| "number" | "NumberInput" |
| "boolean" | "Checkbox" |
| "array" | "ArrayField" |
| "object" | "ObjectField" |

## Troubleshooting

### Issue: allOf Schemas Not Flattening

**Solution:**
```python
# Ensure allOf is present and contains dicts
if "allOf" not in schema:
    schema = {"allOf": [schema]}

flat = flatten_schema(schema)
```

### Issue: Missing Components in Formily Schema

**Solution:**
```python
# Ensure schema has type and properties
if "x-component" not in field:
    # Add appropriate component based on type
    field["x-component"] = get_component_for_type(field["type"])
```

### Issue: Form Not Validating Correctly

**Solution:**
```python
# Check that x-rules are set for required fields
for prop_name, prop_schema in schema["properties"].items():
    if prop_name in schema.get("required", []):
        if "x-rules" not in prop_schema:
            prop_schema["x-rules"] = [{"required": True}]
```

## Related Documentation

- [form_builder.py](./form_builder.md) - Formily schema conversion details
- [schema_utils.py](./schema_utils.md) - Schema manipulation utilities
- [schema_builder.py](./schema_builder.md) - Auto-generation of schemas
- [Web UI Documentation](../../../web-ui/) - Using generated forms

## Summary

The `schema_helpers` package provides essential infrastructure for schema management and form generation in PAMOLA.CORE. It bridges the gap between operational configuration and user-facing interfaces.

Key strengths:
- Transparent schema flattening
- Automatic Formily conversion
- Config-based schema generation
- Extensible component mapping
- Full validation support

Use `schema_helpers` for:
- Generating forms from operation configs
- Validating configuration data
- Creating UI forms dynamically
- Documenting operation parameters
- Building configuration templates
