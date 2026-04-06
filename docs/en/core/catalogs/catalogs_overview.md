# Catalogs Module Documentation

**Module:** `pamola_core.catalogs`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)
8. [Summary Analysis](#summary-analysis)

## Overview

The `catalogs` module provides centralized access to the PAMOLA.CORE operation catalog, which serves as a single source of truth for all available operations. This module implements requirements NFR-EP3-CORE-120 (single source of truth for operation catalog) and NFR-EP3-CORE-124 (Studio/Processing consume catalogs via CORE API).

### Purpose

The catalogs module:
- Loads and exposes `operations_catalog.yaml`, the authoritative registry of all operations
- Provides programmatic access to operation metadata (name, category, module, version, description)
- Enables Studio and Processing components to discover available operations
- Supports both YAML parsing (with PyYAML) and fallback parsing (without external dependencies)
- Uses caching to minimize I/O overhead when accessing catalog data

### Use Cases

- **Operation Discovery**: List all available operations and their metadata
- **Dynamic Operation Loading**: Locate operation implementations by name
- **Integration with Studio**: Provide operation metadata to frontend components
- **Operation Validation**: Verify operation names before instantiation

## Key Features

### 1. **Cached Catalog Loading**
- LRU cache (maxsize=1) for single catalog instance
- Efficient repeated access without disk I/O
- Fallback parser for environments without PyYAML

### 2. **YAML Support with Fallback**
- Primary: Full YAML parsing via PyYAML library
- Fallback: Minimal YAML list parser for environments without PyYAML

### 3. **Operation Entry Lookup**
- Query individual operations by class name
- Returns complete metadata for matched operation
- Supports case-sensitive matching

### 4. **Error Handling**
- Clear error messages when catalog file is missing
- Graceful degradation to fallback parser
- Validation of catalog structure

## Architecture

### Module Structure

```
pamola_core/catalogs/
├── __init__.py              # Public API exports
└── catalog_loader.py        # Core catalog loading logic
```

### Catalog File Location

```
pamola_core/catalogs/operations_catalog.yaml
```

The catalog YAML file is stored alongside the loader module and packaged with the library.

### Data Flow

```
applications/
  └── get_operations_catalog()
      └── [LRU Cache: 1 entry]
          └── operations_catalog.yaml
              ├── PyYAML parser (if available)
              └── fallback parser (minimal YAML support)
```

### Entry Structure

Each operation entry in the catalog contains:
- `name`: Operation class name (e.g., 'FullMaskingOperation')
- `category`: Anonymization category (e.g., 'masking', 'generalization')
- `module`: Full module path (e.g., 'pamola_core.anonymization.masking.operations')
- `version`: Operation version
- `description`: Human-readable description

## Core Components

### Public API

| Function | Returns | Description |
|----------|---------|-------------|
| `get_operations_catalog()` | `List[Dict[str, Any]]` | Load all catalog entries with LRU caching |
| `get_operation_entry(name: str)` | `Dict[str, Any] \| None` | Look up single operation by class name |

### get_operations_catalog()

Loads and returns all entries from `operations_catalog.yaml`.

```python
def get_operations_catalog() -> List[Dict[str, Any]]:
    """
    Return all entries from operations_catalog.yaml.

    Returns
    -------
    list of dict
        Each entry has: name, category, module, version, description.

    Raises
    ------
    FileNotFoundError
        If operations_catalog.yaml is missing.
    """
```

**Characteristics:**
- Decorated with `@lru_cache(maxsize=1)` for single-instance caching
- Attempts PyYAML parsing first; falls back to minimal parser if unavailable
- Raises `FileNotFoundError` if catalog file missing and no fallback available
- Returns empty list if fallback parser encounters missing file

### get_operation_entry(name: str)

Looks up a single operation by class name.

```python
def get_operation_entry(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up a single operation entry by class name.

    Parameters
    ----------
    name : str
        Operation class name (e.g. 'FullMaskingOperation').

    Returns
    -------
    dict or None
        Operation entry if found; None otherwise.
    """
```

**Characteristics:**
- Linear search through catalog entries
- Case-sensitive matching on `name` field
- Returns `None` if no match found

## Usage Examples

### Example 1: List All Operations

```python
from pamola_core.catalogs import get_operations_catalog

# Load all available operations
catalog = get_operations_catalog()

# Display operation names
for op in catalog:
    print(f"{op['name']}: {op['description']}")

# Output example:
# FullMaskingOperation: Replace values with a fixed mask
# RandomMaskingOperation: Replace with random masks based on field type
# ...
```

### Example 2: Look Up a Specific Operation

```python
from pamola_core.catalogs import get_operation_entry

# Find operation by class name
entry = get_operation_entry('FullMaskingOperation')

if entry:
    print(f"Category: {entry['category']}")
    print(f"Module: {entry['module']}")
    print(f"Version: {entry['version']}")
else:
    print("Operation not found in catalog")
```

### Example 3: Discover Operations by Category

```python
from pamola_core.catalogs import get_operations_catalog

catalog = get_operations_catalog()

# Find all masking operations
masking_ops = [op for op in catalog if op['category'] == 'masking']
print(f"Available masking operations: {len(masking_ops)}")

for op in masking_ops:
    print(f"  - {op['name']}")
```

### Example 4: Dynamic Operation Loading

```python
from pamola_core.catalogs import get_operation_entry
import importlib

# Get operation metadata
entry = get_operation_entry('FullMaskingOperation')

if entry:
    # Dynamically import the module
    module = importlib.import_module(entry['module'])

    # Get the operation class
    OpClass = getattr(module, entry['name'])

    # Instantiate operation
    operation = OpClass()
```

## Best Practices

### 1. **Cache Usage**
- Call `get_operations_catalog()` once and reuse the result if accessing multiple operations
- The LRU cache ensures only one catalog instance in memory
- Subsequent calls return the cached result without disk I/O

### 2. **Error Handling**
```python
try:
    catalog = get_operations_catalog()
except FileNotFoundError as e:
    print(f"Catalog not found: {e}")
    # Implement fallback logic
```

### 3. **Safe Operation Lookup**
```python
from pamola_core.catalogs import get_operation_entry

op_name = "UnknownOperation"
entry = get_operation_entry(op_name)

if entry is None:
    print(f"Operation '{op_name}' not found in catalog")
else:
    print(f"Found: {entry['name']} in {entry['module']}")
```

### 4. **Catalog File Management**
- The `operations_catalog.yaml` file must be present in the `pamola_core/catalogs/` directory
- File is included in package distribution (see setup.py/pyproject.toml)
- Regenerate catalog after adding new operations to ensure discovery

## Related Components

- **`pamola_core.utils.ops.op_base.BaseOperation`**: Base class for all operations
- **`pamola_core.anonymization.*`**: Anonymization operation implementations
- **`pamola_core.transformations.*`**: Data transformation operations
- **`pamola_core.profiling.*`**: Data profiling operations
- **Studio API**: Consumes catalog for operation discovery

## Summary Analysis

The catalogs module provides a lightweight, dependency-flexible mechanism for discovering PAMOLA operations. Key characteristics:

- **Single Source of Truth**: All operation metadata centralized in `operations_catalog.yaml`
- **Flexible Parsing**: Works with or without PyYAML
- **Efficient Access**: LRU caching prevents repeated disk I/O
- **Extensible**: New operations automatically available once added to catalog
- **Non-Intrusive API**: Simple functions with clear contracts
- **Error Resilient**: Graceful fallback when PyYAML unavailable
