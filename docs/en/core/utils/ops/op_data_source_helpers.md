# DataSource Helper Functions Documentation

**Module:** `pamola_core.utils.ops.op_data_source_helpers`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Functions](#core-functions)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)
10. [Summary](#summary)

## Overview

The `op_data_source_helpers` module provides supplementary utility functions used by the `DataSource` class. These functions are factored out to maintain clean separation of concerns and focus on value-added functionality not directly available in other modules.

Key responsibilities include schema validation, memory optimization recommendations, DataFrame chunking, and DataFrame analysis. These utilities are designed to support efficient data processing while maintaining memory constraints and data integrity.

## Key Features

- **Schema Validation & Type Compatibility**: Validate DataFrames against expected schemas with type compatibility checks
- **Memory Management**: System and process memory monitoring with optimization recommendations
- **DataFrame Chunking**: Memory-efficient chunking for large DataFrames with progress tracking
- **DataFrame Analysis**: Comprehensive analysis of structure, memory usage, and optimization opportunities
- **Sample Creation**: Representative sampling of large DataFrames with stratification support
- **Type Compatibility**: Enhanced type handling for pandas/numpy type conversions

## Architecture

The module provides independent utility functions organized by functionality:

```
op_data_source_helpers
├── Memory Management
│   ├── get_system_memory()
│   ├── get_process_memory_usage()
│   ├── optimize_memory_usage()
│   └── generate_dataframe_chunks()
│
├── Schema Validation
│   ├── validate_schema()
│   └── is_compatible_dtype()
│
├── DataFrame Analysis
│   ├── analyze_dataframe()
│   └── create_sample_dataframe()
│
└── Utility Types
    └── Type compatibility categories (int, float, string, etc.)
```

## Dependencies

| Module | Purpose |
|--------|---------|
| `pandas` | DataFrame operations |
| `psutil` | System memory monitoring |
| `pamola_core.utils.logging` | Logging infrastructure |
| `pamola_core.utils.progress` | `track_operation_safely()` for progress |
| `pamola_core.utils.io` | `optimize_dataframe_memory()` |
| `gc` | Garbage collection for memory cleanup |

## Core Functions

### Memory Management Functions

#### get_system_memory()

```python
def get_system_memory() -> Dict[str, float]
```

**Purpose:** Get system memory information.

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `total_gb` | float | Total system RAM in GB |
| `available_gb` | float | Available RAM in GB |
| `used_gb` | float | Used RAM in GB |
| `percent` | float | Percentage of RAM in use |

**Example:**
```python
memory = get_system_memory()
print(f"System has {memory['total_gb']:.1f}GB RAM, {memory['available_gb']:.1f}GB free")
```

#### get_process_memory_usage()

```python
def get_process_memory_usage() -> Dict[str, float]
```

**Purpose:** Get current process memory usage.

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `rss_mb` | float | Resident Set Size in MB |
| `vms_mb` | float | Virtual Memory Size in MB |

**Example:**
```python
memory = get_process_memory_usage()
print(f"Current process using {memory['rss_mb']:.1f}MB RSS")
```

#### optimize_memory_usage()

```python
def optimize_memory_usage(
    dataframes: Dict[str, pd.DataFrame],
    threshold_percent: float = 80.0,
    release_func: Optional[Callable[[str], bool]] = None,
    logger: Optional[Logger] = None
) -> Dict[str, Any]
```

**Purpose:** Analyze and optimize memory usage for multiple DataFrames.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataframes` | Dict[str, pd.DataFrame] | Required | Named DataFrames to optimize |
| `threshold_percent` | float | 80.0 | Memory usage threshold to trigger optimization |
| `release_func` | Callable | None | Function to release DataFrame by name |
| `logger` | Logger | None | Logger instance |

**Returns:** Dictionary with optimization results:

```python
{
    "status": "ok" | "optimized",
    "system_memory": {
        "total_gb": float,
        "available_gb": float,
        "used_gb": float,
        "percent": float
    },
    "initial_memory": {
        "total_mb": float,
        "usage_percent": float,
        "dataframes": Dict[str, float]
    },
    "final_memory": {
        "total_mb": float,
        "usage_percent": float,
        "dataframes": Dict[str, float]
    },
    "optimizations": Dict[str, Any],  # Optimization results per DataFrame
    "released_dataframes": List[str]   # DataFrames released
}
```

**Behavior:**

1. Gets system memory information
2. Calculates current DataFrame memory usage
3. Compares against threshold
4. If exceeds threshold:
   - Optimizes all DataFrames using `io.optimize_dataframe_memory()`
   - Releases DataFrames in order of size if still above threshold
   - Calls `gc.collect()` for cleanup
5. Returns detailed optimization report

**Example:**
```python
from pamola_core.utils.ops.op_data_source_helpers import optimize_memory_usage

dfs = {
    "main": df1,
    "auxiliary": df2,
    "temp": df3
}

result = optimize_memory_usage(
    dataframes=dfs,
    threshold_percent=75.0,
    release_func=lambda name: dfs.pop(name, None) is not None
)

print(f"Initial: {result['initial_memory']['total_mb']:.1f}MB")
print(f"Final: {result['final_memory']['total_mb']:.1f}MB")
print(f"Released: {result['released_dataframes']}")
```

#### generate_dataframe_chunks()

```python
def generate_dataframe_chunks(
    df: pd.DataFrame,
    chunk_size: int,
    columns: Optional[List[str]] = None,
    logger: Optional[Logger] = None,
    show_progress: bool = True
) -> Generator[pd.DataFrame, None, None]
```

**Purpose:** Generate chunks from a DataFrame for efficient processing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | pd.DataFrame | Required | DataFrame to chunk |
| `chunk_size` | int | Required | Size of each chunk (rows) |
| `columns` | List[str] | None | Specific columns to include |
| `logger` | Logger | None | Logger instance |
| `show_progress` | bool | True | Show progress during chunking |

**Yields:** DataFrame chunks of specified size

**Example:**
```python
from pamola_core.utils.ops.op_data_source_helpers import generate_dataframe_chunks

large_df = pd.read_csv("huge_file.csv")

# Process in chunks
for chunk in generate_dataframe_chunks(
    large_df,
    chunk_size=10000,
    columns=["email", "phone", "name"]
):
    # Process chunk
    process_sensitive_fields(chunk)
    print(f"Processed chunk with {len(chunk)} rows")
```

### Schema Validation Functions

#### validate_schema()

```python
def validate_schema(
    actual_schema: Dict[str, Any],
    expected_schema: Dict[str, Any],
    logger: Optional[Logger] = None
) -> Tuple[bool, List[str]]
```

**Purpose:** Validate a DataFrame schema against an expected schema.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `actual_schema` | Dict | Actual schema to validate |
| `expected_schema` | Dict | Expected schema to validate against |
| `logger` | Logger | Logger instance |

**Schema Format:**

```python
schema = {
    "columns": ["col1", "col2", "col3"],
    "dtypes": {
        "col1": "int64",
        "col2": "object",
        "col3": "float64"
    },
    "constraints": [
        {"type": "non_null", "column": "col1"},
        {"type": "unique", "column": "col1"}
    ],
    "null_counts": {"col1": 0, "col2": 5},
    "unique_counts": {"col1": 1000},
    "num_rows": 1000
}
```

**Returns:** Tuple of (is_valid, error_messages)

**Validations Performed:**

1. All expected columns present
2. Column data types compatible
3. Non-null constraint validation
4. Uniqueness constraint validation

**Example:**
```python
from pamola_core.utils.ops.op_data_source_helpers import validate_schema, analyze_dataframe

df = pd.read_csv("data.csv")
analysis = analyze_dataframe(df)

expected = {
    "columns": ["id", "email"],
    "dtypes": {"id": "int64", "email": "object"},
    "constraints": [{"type": "unique", "column": "id"}]
}

is_valid, errors = validate_schema(analysis, expected)
if not is_valid:
    print(f"Schema validation errors: {errors}")
```

#### is_compatible_dtype()

```python
def is_compatible_dtype(actual: str, expected: str) -> bool
```

**Purpose:** Check if two data types are compatible with enhanced type handling.

**Returns:** True if compatible, False otherwise

**Type Compatibility Rules:**

| Category | Compatible Types |
|----------|-------------------|
| Numeric | int, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float, float16, float32, float64, double |
| String | object, string, str |
| DateTime | datetime, datetime64, timestamp, datetime64[ns] |
| Date | date, datetime, datetime64, timestamp |
| Time | time, timedelta, timedelta64 |
| Boolean | bool, boolean |
| Category | category, categorical |

**Compatibility Rules:**

- Float can hold integer values
- String can hold any type when needed
- Category compatible with underlying type
- Parametrized types (e.g., datetime64[ns]) compared by base type

**Example:**
```python
from pamola_core.utils.ops.op_data_source_helpers import is_compatible_dtype

assert is_compatible_dtype("float64", "int32")  # True - float can hold int
assert is_compatible_dtype("object", "string")  # True - both are string types
assert is_compatible_dtype("int32", "float64")  # False - int can't hold float
```

### DataFrame Analysis Functions

#### analyze_dataframe()

```python
def analyze_dataframe(
    df: pd.DataFrame,
    logger: Optional[Logger] = None
) -> Dict[str, Any]
```

**Purpose:** Analyze DataFrame structure and provide insights.

**Returns:** Analysis dictionary:

```python
{
    "shape": {"rows": int, "columns": int},
    "memory_usage": {
        "total_mb": float,
        "by_column": Dict[str, float]
    },
    "column_types": Dict[str, str],
    "null_counts": Dict[str, int],
    "unique_counts": Dict[str, int],
    "potential_optimizations": [
        {
            "column": str,
            "current_type": str,
            "suggested_type": str,
            "estimated_savings_mb": float,
            ...
        }
    ],
    "sample_values": Dict[str, str]
}
```

**Optimizations Suggested:**

1. **Object → Category**: For columns with < 50% unique values
2. **int64 → Smaller Type**: For columns fitting in int8/int16/int32
3. **Category → Object**: For high-cardinality (> 80% unique) categoricals

**Example:**
```python
from pamola_core.utils.ops.op_data_source_helpers import analyze_dataframe

df = pd.read_csv("data.csv")
analysis = analyze_dataframe(df)

print(f"Shape: {analysis['shape']['rows']} rows, {analysis['shape']['columns']} columns")
print(f"Memory: {analysis['memory_usage']['total_mb']:.2f}MB")

for opt in analysis['potential_optimizations']:
    print(f"Column '{opt['column']}': {opt['current_type']} → {opt['suggested_type']}")
    print(f"  Estimated savings: {opt['estimated_savings_mb']:.2f}MB")
```

#### create_sample_dataframe()

```python
def create_sample_dataframe(
    df: pd.DataFrame,
    sample_size: int = 1000,
    random_seed: int = 42,
    preserve_dtypes: bool = True,
    logger: Optional[Logger] = None
) -> pd.DataFrame
```

**Purpose:** Create a representative sample of a DataFrame.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | pd.DataFrame | Required | Source DataFrame |
| `sample_size` | int | 1000 | Number of rows in sample |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `preserve_dtypes` | bool | True | Preserve data types in sample |
| `logger` | Logger | None | Logger instance |

**Sampling Strategy:**

1. If df smaller than sample_size, returns full DataFrame
2. Identifies stratification columns (5-100 unique values)
3. If found, uses stratified sampling to preserve distributions
4. Falls back to simple random sampling if stratification fails
5. Preserves data types if requested

**Example:**
```python
from pamola_core.utils.ops.op_data_source_helpers import create_sample_dataframe

df = pd.read_csv("huge_dataset.csv")  # 10M rows

# Create representative sample
sample = create_sample_dataframe(
    df,
    sample_size=10000,
    random_seed=42,
    preserve_dtypes=True
)

print(f"Original: {len(df)} rows, Sample: {len(sample)} rows")
# Distribution of stratification column should be similar
```

## Usage Examples

### Example 1: Monitor and Optimize Memory Usage

```python
from pamola_core.utils.ops.op_data_source_helpers import (
    get_system_memory,
    get_process_memory_usage,
    optimize_memory_usage
)

# Check system state
sys_mem = get_system_memory()
proc_mem = get_process_memory_usage()

print(f"System: {sys_mem['available_gb']:.1f}GB available")
print(f"Process: {proc_mem['rss_mb']:.1f}MB RSS")

# Optimize if needed
dataframes = {"df1": df1, "df2": df2}
result = optimize_memory_usage(
    dataframes,
    threshold_percent=75.0
)

if result['status'] == 'optimized':
    print(f"Optimized from {result['initial_memory']['total_mb']:.1f}MB to {result['final_memory']['total_mb']:.1f}MB")
```

### Example 2: Process Large DataFrame in Chunks

```python
from pamola_core.utils.ops.op_data_source_helpers import generate_dataframe_chunks

df = pd.read_csv("sensitive_data.csv")

# Process anonymization in chunks
anonymized_chunks = []
for chunk in generate_dataframe_chunks(df, chunk_size=5000):
    # Apply anonymization operation
    anonymized = apply_anonymization(chunk)
    anonymized_chunks.append(anonymized)

result_df = pd.concat(anonymized_chunks, ignore_index=True)
```

### Example 3: Create and Validate Sample

```python
from pamola_core.utils.ops.op_data_source_helpers import (
    create_sample_dataframe,
    analyze_dataframe,
    validate_schema
)

# Create sample for testing
df = pd.read_csv("production_data.csv")
sample = create_sample_dataframe(df, sample_size=5000)

# Analyze both
sample_analysis = analyze_dataframe(sample)
full_analysis = analyze_dataframe(df)

# Validate sample is representative
expected = {
    "columns": sample_analysis["column_types"].keys(),
    "dtypes": sample_analysis["column_types"]
}

is_valid, errors = validate_schema(sample_analysis, expected)
print(f"Sample is representative: {is_valid}")
```

## Best Practices

1. **Always Check Memory Before Processing**
   - Call `get_system_memory()` to verify available resources
   - Use `optimize_memory_usage()` proactively in large operations

2. **Use Chunking for Large DataFrames**
   - Don't load entire massive files at once
   - Process in memory-efficient chunks
   - Monitor progress with `show_progress=True`

3. **Validate Schema Before Processing**
   - Use `validate_schema()` before operations
   - Catches data quality issues early
   - Prevents errors in downstream operations

4. **Create Samples for Testing**
   - Use `create_sample_dataframe()` for testing operations
   - Stratified sampling preserves distributions
   - Fast iteration on large datasets

5. **Monitor Type Compatibility**
   - Use `is_compatible_dtype()` to understand conversions
   - Prevents silent data loss from incompatible types

6. **Act on Analysis Recommendations**
   - Review `analyze_dataframe()` optimization suggestions
   - Implement suggested type changes for memory savings
   - Especially important for repeated operations

## Troubleshooting

### Issue: Memory Optimization Not Effective

```python
# Check why optimization isn't helping
analysis = analyze_dataframe(df)
print(f"Current memory: {analysis['memory_usage']['total_mb']:.1f}MB")
print("Suggested optimizations:")
for opt in analysis['potential_optimizations']:
    print(f"  - {opt['column']}: {opt['estimated_savings_mb']:.1f}MB")

# If no optimizations available, consider chunking
for chunk in generate_dataframe_chunks(df, chunk_size=10000):
    process(chunk)
```

### Issue: Chunking Memory Still Growing

```python
# Force garbage collection between chunks
import gc
from pamola_core.utils.ops.op_data_source_helpers import generate_dataframe_chunks

for chunk in generate_dataframe_chunks(df, chunk_size=5000):
    result = process(chunk)
    save_results(result)
    del result
    gc.collect()  # Force cleanup
```

### Issue: Sample Not Representative

```python
# Use larger sample or verify distribution
from pamola_core.utils.ops.op_data_source_helpers import create_sample_dataframe

sample = create_sample_dataframe(
    df,
    sample_size=50000,  # Larger sample
    random_seed=42,
    preserve_dtypes=True
)

# Check distribution of key columns
print("Distribution in full dataset:")
print(df['status'].value_counts(normalize=True))
print("\nDistribution in sample:")
print(sample['status'].value_counts(normalize=True))
```

## Related Components

| Component | Purpose |
|-----------|---------|
| `DataSource` | Uses these helpers for data management |
| `DataReader` | Coordinates with these utilities |
| `BaseOperation` | Uses helpers for memory management |
| `pamola_core.utils.io` | Provides `optimize_dataframe_memory()` |
| `HierarchicalProgressTracker` | Used by chunking functions |

## Summary

The `op_data_source_helpers` module provides essential utilities for efficient and safe DataFrame management within the PAMOLA framework. By offering memory monitoring, schema validation, intelligent chunking, and analysis capabilities, it enables operations to handle large datasets responsibly.

Key strengths:
- Comprehensive memory management
- Robust schema validation with type compatibility
- Efficient chunking with progress tracking
- Intelligent sampling preserving distributions
- Detailed analysis with optimization recommendations

These functions work together to support operations that need to:
- Handle large datasets within memory constraints
- Validate data integrity before processing
- Process data in batches efficiently
- Understand data characteristics and optimize accordingly
