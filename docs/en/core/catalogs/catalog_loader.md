# Catalog Loader Module Documentation

**Module:** `pamola_core.catalogs.catalog_loader`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Core Functions](#core-functions)
3. [Implementation Details](#implementation-details)
4. [Catalog YAML Format](#catalog-yaml-format)
5. [Caching Mechanism](#caching-mechanism)
6. [Fallback Parser](#fallback-parser)
7. [Error Handling](#error-handling)
8. [Usage Examples](#usage-examples)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Technical Summary](#technical-summary)

## Overview

The `catalog_loader.py` module implements the core mechanism for loading and caching PAMOLA operation catalog metadata from `operations_catalog.yaml`. It provides two entry points with intelligent YAML parsing and automatic fallback support.

### Module Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `_CATALOG_DIR` | `Path(__file__).parent` | Catalogs module directory |
| `_OPS_CATALOG_PATH` | `_CATALOG_DIR / "operations_catalog.yaml"` | Catalog file location |

## Core Functions

### get_operations_catalog()

**Signature:**
```python
@lru_cache(maxsize=1)
def get_operations_catalog() -> List[Dict[str, Any]]:
```

**Purpose:** Load and cache all operation entries from the catalog.

**Returns:**
- `List[Dict[str, Any]]`: List of operation entry dictionaries
- Each entry contains: `name`, `category`, `module`, `version`, `description`

**Raises:**
- `FileNotFoundError`: If `operations_catalog.yaml` is missing and PyYAML is available

**Behavior:**
1. First invocation: loads from YAML file using PyYAML or fallback parser
2. Subsequent invocations: returns cached result (single instance due to `maxsize=1`)
3. Logs debug message with operation count

**Example:**
```python
catalog = get_operations_catalog()  # Loads from disk
catalog_again = get_operations_catalog()  # Returns cached result
```

### get_operation_entry(name: str)

**Signature:**
```python
def get_operation_entry(name: str) -> Optional[Dict[str, Any]]:
```

**Purpose:** Look up a single operation entry by class name.

**Parameters:**
- `name` (str): Operation class name (e.g., 'FullMaskingOperation')

**Returns:**
- `Dict[str, Any]`: Operation entry if found
- `None`: If no matching entry exists

**Behavior:**
1. Calls `get_operations_catalog()` to get cached catalog
2. Linear search through entries matching `entry.get("name")`
3. Returns first match or `None`

**Example:**
```python
entry = get_operation_entry('FullMaskingOperation')
if entry:
    print(f"Found: {entry['module']}")
else:
    print("Operation not found")
```

### _load_catalog_fallback(path: Path)

**Signature:**
```python
def _load_catalog_fallback(path: Path) -> List[Dict[str, Any]]:
```

**Purpose:** Minimal YAML parser for environments without PyYAML.

**Parameters:**
- `path` (Path): Path to catalog YAML file

**Returns:**
- `List[Dict[str, Any]]`: Parsed operation entries (may be empty if file not found)

**Behavior:**
1. Returns empty list if file does not exist
2. Iterates through lines parsing simple YAML list format
3. Detects entries starting with `- name:`
4. Extracts key-value pairs after colons

**Limitations:**
- Only handles flat key-value pairs (no nested structures)
- Does not validate YAML syntax
- Assumes simple YAML list format

## Implementation Details

### Caching Strategy

**LRU Cache:**
```python
@lru_cache(maxsize=1)
def get_operations_catalog() -> List[Dict[str, Any]]:
```

- **maxsize=1**: Only one catalog instance cached (all calls share same object)
- **Key benefit**: Eliminates repeated disk I/O for catalog access
- **Memory efficiency**: Single cached list regardless of call frequency
- **Thread-safe**: LRU cache handles concurrent access automatically

**Cache Invalidation:**
- Cache persists for application lifetime (no automatic expiration)
- Manual reset not currently supported
- Requires restart to reload changed catalog file

### YAML Parsing Strategy

**Two-tier approach:**

1. **Primary (PyYAML):**
   ```python
   try:
       import yaml
       # ...
       data = yaml.safe_load(f)
       operations = data.get("operations", [])
   ```
   - Full YAML parsing with schema validation
   - Supports nested structures and complex formats
   - Raises clear errors on malformed YAML

2. **Fallback (Minimal Parser):**
   ```python
   except ImportError:
       return _load_catalog_fallback(_OPS_CATALOG_PATH)
   ```
   - Used when PyYAML not available
   - No external dependencies
   - Handles simple YAML lists

### File Location Strategy

The catalog file is located at module-relative path:
```python
_CATALOG_DIR = Path(__file__).parent
_OPS_CATALOG_PATH = _CATALOG_DIR / "operations_catalog.yaml"
```

**Advantages:**
- Works correctly after package installation
- Survives relocation of package directory
- Compatible with zip-packaged installations
- Deterministic relative to module location

## Catalog YAML Format

Expected structure in `operations_catalog.yaml`:

```yaml
operations:
  - name: FullMaskingOperation
    category: masking
    module: pamola_core.anonymization.masking.operations
    version: 1.0.0
    description: Replace values with a fixed mask

  - name: RandomMaskingOperation
    category: masking
    module: pamola_core.anonymization.masking.operations
    version: 1.0.0
    description: Replace with random masks based on field type

  # ... more operations
```

**Required Fields per Entry:**
- `name`: Operation class name (string)
- `category`: Anonymization category (string)
- `module`: Full Python module path (string)
- `version`: Semantic version string (string)
- `description`: Human-readable description (string)

**Optional Fields:**
- Any additional metadata (ignored by loader but preserved in entry dict)

## Caching Mechanism

### How Caching Works

1. **First call:**
   ```python
   catalog = get_operations_catalog()
   # Loads from disk, caches result
   ```

2. **Subsequent calls:**
   ```python
   catalog = get_operations_catalog()
   # Returns cached instance (no disk I/O)
   ```

3. **Multiple accessors share same object:**
   ```python
   list1 = get_operations_catalog()
   list2 = get_operations_catalog()
   assert list1 is list2  # True - same object
   ```

### Cache Inspection

To check cache status:
```python
from pamola_core.catalogs.catalog_loader import get_operations_catalog

# Cache info: CacheInfo(hits=0, misses=0, maxsize=1, currsize=0)
info = get_operations_catalog.cache_info()
print(f"Cache hits: {info.hits}, misses: {info.misses}")

# After calling:
get_operations_catalog()
info = get_operations_catalog.cache_info()
# Cache info: CacheInfo(hits=0, misses=1, maxsize=1, currsize=1)

# After second call:
get_operations_catalog()
info = get_operations_catalog.cache_info()
# Cache info: CacheInfo(hits=1, misses=1, maxsize=1, currsize=1)
```

## Fallback Parser

The minimal parser handles this YAML format:

```yaml
operations:
- name: FullMaskingOperation
  category: masking
  module: pamola_core.anonymization.masking.operations
  version: 1.0.0
  description: Replace values with a fixed mask
```

### Parsing Algorithm

1. Iterate through file lines
2. Detect entry start: line stripped starts with `- name:`
3. Extract name value after colon
4. For subsequent lines with `:` (not comments):
   - Split on first colon
   - Extract key and value
   - Strip quotes from value
5. When next `- name:` encountered, save current entry and start new

### Supported Format

- **Entry markers:** `- name: ValueHere`
- **Key-value pairs:** `key: value` or `key: "value"`
- **Comments:** Lines starting with `#` (ignored)
- **Whitespace:** Stripped before processing

### Unsupported Features

- Nested structures (maps within maps)
- Lists as values
- YAML anchors and aliases
- Multi-line strings
- Complex YAML types

## Error Handling

### FileNotFoundError Scenario

When PyYAML is available and catalog missing:
```python
get_operations_catalog()
# Raises: FileNotFoundError
# Message: "operations_catalog.yaml not found at [...].
#           Run catalog generation or check package installation."
```

### Fallback Parser Behavior

When PyYAML unavailable and catalog missing:
```python
_load_catalog_fallback(path)  # Path doesn't exist
# Returns: []  # Empty list (no error raised)
```

### Import Error Handling

```python
try:
    import yaml
except ImportError:
    # Fallback parser used automatically
    return _load_catalog_fallback(_OPS_CATALOG_PATH)
```

## Usage Examples

### Example 1: Basic Catalog Access

```python
from pamola_core.catalogs.catalog_loader import get_operations_catalog

# Load catalog
catalog = get_operations_catalog()

# Display count
print(f"Total operations: {len(catalog)}")

# Iterate entries
for op in catalog[:3]:
    print(f"  {op['name']}: {op['category']}")
```

### Example 2: Lookup by Name

```python
from pamola_core.catalogs.catalog_loader import get_operation_entry

# Find operation
op = get_operation_entry('RandomMaskingOperation')

if op:
    print(f"Module: {op['module']}")
    print(f"Version: {op['version']}")
else:
    print("Not found")
```

### Example 3: Filter Operations by Category

```python
from pamola_core.catalogs.catalog_loader import get_operations_catalog

catalog = get_operations_catalog()

# Get masking operations
masking = [op for op in catalog if op['category'] == 'masking']
print(f"Masking operations: {len(masking)}")
```

### Example 4: Check Cache Performance

```python
from pamola_core.catalogs.catalog_loader import get_operations_catalog
import time

# First call (cache miss)
start = time.time()
cat1 = get_operations_catalog()
t1 = time.time() - start

# Second call (cache hit)
start = time.time()
cat2 = get_operations_catalog()
t2 = time.time() - start

print(f"First call: {t1*1000:.2f}ms (disk I/O)")
print(f"Second call: {t2*1000:.2f}ms (from cache)")

# Same object
assert cat1 is cat2
```

## Best Practices

### 1. **Reuse Cached Catalog**

```python
# Good: Load once, reuse
catalog = get_operations_catalog()
for name in ['Op1', 'Op2', 'Op3']:
    entry = next((op for op in catalog if op['name'] == name), None)
```

```python
# Avoid: Calling function repeatedly
for name in ['Op1', 'Op2', 'Op3']:
    entry = get_operation_entry(name)  # Reuses cache but less efficient
```

### 2. **Handle Missing Operations Gracefully**

```python
op_name = user_input
entry = get_operation_entry(op_name)

if entry is None:
    raise ValueError(f"Operation '{op_name}' not found in catalog")
```

### 3. **Validate Before Dynamic Import**

```python
from pamola_core.catalogs.catalog_loader import get_operation_entry
import importlib

op_name = 'SomeOperation'
entry = get_operation_entry(op_name)

if entry is None:
    print(f"Unknown operation: {op_name}")
else:
    module = importlib.import_module(entry['module'])
    OpClass = getattr(module, op_name)
```

## Troubleshooting

### Issue: FileNotFoundError on get_operations_catalog()

**Symptoms:**
```
FileNotFoundError: operations_catalog.yaml not found at [path]
```

**Causes:**
1. Catalog file not included in package distribution
2. Package installed incorrectly
3. Package location moved or deleted

**Solutions:**
1. Verify `operations_catalog.yaml` exists in `pamola_core/catalogs/`
2. Reinstall package: `pip install --force-reinstall pamola-core`
3. Check installation: `python -c "from pamola_core.catalogs import get_operations_catalog; print(get_operations_catalog())"`

### Issue: Empty Catalog Returned

**Symptoms:**
```python
catalog = get_operations_catalog()
assert len(catalog) == 0  # Unexpected
```

**Causes:**
1. Catalog YAML file empty or malformed
2. Fallback parser used but YAML format incompatible
3. PyYAML present but catalog structure unexpected

**Solutions:**
1. Check `operations_catalog.yaml` content
2. Verify YAML structure starts with `operations:`
3. Ensure entries have `name:` field

### Issue: get_operation_entry() Returns None

**Symptoms:**
```python
entry = get_operation_entry('MyOperation')
assert entry is None  # But operation should exist
```

**Causes:**
1. Operation name misspelled (case-sensitive)
2. Operation not added to catalog
3. Cached old catalog (package not reloaded)

**Solutions:**
1. Verify operation class name matches catalog entry exactly
2. Check catalog file includes operation
3. Restart Python interpreter or clear cache if modified catalog

## Technical Summary

**catalog_loader.py** provides robust operation catalog discovery with intelligent fallback support:

- **Single-instance caching** via LRU decorator ensures minimal overhead
- **Two-tier parsing** (PyYAML + fallback) works in any environment
- **Relative path resolution** ensures portability
- **Clear error messages** guide troubleshooting
- **Simple linear search** suitable for typical catalog sizes
- **Zero external dependencies** (optional PyYAML for enhanced features)

The module successfully abstracts catalog access complexity, enabling Studio and Processing components to reliably discover PAMOLA operations without implementation concerns.
