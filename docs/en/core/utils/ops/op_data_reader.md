# DataReader Documentation

**Module:** `pamola_core.utils.ops.op_data_reader`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Dependencies](#dependencies)
5. [Core Classes](#core-classes)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)
10. [Summary](#summary)

## Overview

The `DataReader` module provides a specialized, unified interface for reading data from various sources with optimal integration to `pamola_core.utils.io`. It ensures consistent handling of progress tracking, error reporting, memory management, and transparent encryption/decryption across all supported file formats.

This module is essential for operations that need to load data efficiently while maintaining framework standards for logging, error handling, and resource management.

## Key Features

- **Unified Reading Interface**: Single `read_dataframe()` method for all supported formats (CSV, JSON, Excel, Parquet)
- **Transparent Encryption Support**: Automatic detection and decryption of encrypted files
- **Memory-Optimized Loading**: Pre-flight checks, automatic format detection, and chunked processing
- **Automatic Parameter Inference**: Detects CSV dialects, file formats, and appropriate read options
- **Comprehensive Progress Tracking**: Integrates with `HierarchicalProgressTracker` for status visibility
- **Multi-File Dataset Handling**: Processes multiple files efficiently with memory-aware strategies
- **Dask Integration**: Automatic switching to Dask for out-of-core processing when needed

## Architecture

The `DataReader` class orchestrates multiple utilities from `pamola_core.utils.io`:

```
DataReader
  ├── Pre-flight Checks
  │   ├── estimate_file_memory_list()
  │   ├── get_system_memory()
  │   └── validate_file_format()
  │
  ├── Format Detection
  │   ├── detect_csv_dialect()
  │   ├── is_encrypted_file()
  │   └── get_file_metadata()
  │
  ├── Reading Strategy
  │   ├── read_full_csv() (single file, fits in memory)
  │   ├── read_csv_in_chunks() (single file, chunked)
  │   ├── read_multi_csv() (multiple CSV files)
  │   ├── read_similar_files() (similar files)
  │   ├── read_multi_file_dask() (Dask distributed)
  │   ├── read_parquet()
  │   ├── read_excel()
  │   ├── read_json()
  │   └── read_dataframe() (unified dispatcher)
  │
  ├── Post-Read Optimization
  │   └── optimize_dataframe_memory()
  │
  └── Error Handling & Logging
      └── Comprehensive exception reporting
```

### Data Flow

```
Input Source(s)
      ↓
Pre-flight Memory Check
      ↓
Format Detection & Parameter Inference
      ↓
Strategy Selection (Full/Chunked/Dask/Multi-file)
      ↓
Read with Progress Tracking
      ↓
Optional Memory Optimization
      ↓
Return (DataFrame, error_info)
```

## Dependencies

| Module | Purpose |
|--------|---------|
| `pamola_core.utils.io` | Core reading functions, format detection, memory estimation |
| `pamola_core.utils.progress` | `HierarchicalProgressTracker` for progress visualization |
| `pamola_core.errors.exceptions` | `ValidationError` for error handling |
| `pandas` | DataFrame operations |
| `dask.dataframe` | Distributed processing for large files |
| `pamola_core.utils.logging` | Logging infrastructure |

## Core Classes

### DataReader

Main class providing unified data reading interface.

#### Constructor

```python
def __init__(self, logger=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logger` | `logging.Logger` | None | Optional logger instance; creates default if not provided |

#### Methods

##### read_dataframe()

```python
def read_dataframe(
    source: Union[PathType, Dict[str, PathType], Dict[str, List[PathType]]],
    file_format: Optional[str] = None,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
    quotechar: str = '"',
    sheet_name: Optional[Union[str, int]] = 0,
    use_dask: bool = False,
    memory_limit: Optional[float] = None,
    encryption_key: Optional[str] = None,
    auto_optimize: bool = True,
    show_progress: bool = True,
    detect_parameters: bool = True,
    use_encryption: bool = False,
    encryption_mode: Optional[str] = None,
    **kwargs,
) -> ResultWithError
```

**Purpose:** Unified interface for reading data from various sources with comprehensive options.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | Path, str, or Dict | Required | File path(s) or dictionary of paths |
| `file_format` | str | None | Override auto-detected format (csv, parquet, json, excel) |
| `columns` | List[str] | None | Read only specific columns (reduces memory) |
| `nrows` | int | None | Read maximum number of rows |
| `skiprows` | int or List[int] | None | Skip specified rows |
| `encoding` | str | "utf-8" | Text encoding for text-based formats |
| `delimiter` | str | "," | Field delimiter for CSV files |
| `quotechar` | str | '"' | Text qualifier character for CSV |
| `sheet_name` | str or int | 0 | Sheet name/index for Excel files |
| `use_dask` | bool | False | Use Dask for distributed processing |
| `memory_limit` | float | None | Memory limit in GB for auto-switching to Dask |
| `encryption_key` | str | None | Key for decrypting encrypted files |
| `auto_optimize` | bool | True | Optimize memory usage after loading |
| `show_progress` | bool | True | Display progress bars |
| `detect_parameters` | bool | True | Auto-detect format and parameters |
| `use_encryption` | bool | False | Enable encryption support |
| `encryption_mode` | str | None | Encryption mode override |
| `**kwargs` | Any | - | Additional arguments to underlying reader |

**Returns:**

| Type | Description |
|------|-------------|
| Tuple[Optional[Union[pd.DataFrame, dd.DataFrame]], Optional[Dict[str, Any]]] | (DataFrame or None, error_info dict or None) |

**Error Handling:**

The method returns error information in the second tuple element:

```python
df, error_info = reader.read_dataframe("data.csv")
if error_info is not None:
    print(f"Error: {error_info['message']}")
    print(f"Error type: {error_info['type']}")
```

**Behavior:**

1. **Pre-flight Memory Check**: Estimates file sizes and available system memory
2. **Format Detection**: Automatically detects file format from extension or content
3. **Parameter Inference**: Detects CSV dialects and optimal read parameters
4. **Strategy Selection**:
   - Single small file → `read_full_csv()`
   - Single large file → `read_csv_in_chunks()`
   - Multiple files → `read_multi_csv()` or `read_multi_file_dask()`
   - Parquet/Excel/JSON → Format-specific readers
5. **Memory Optimization**: Applies `optimize_dataframe_memory()` if `auto_optimize=True`
6. **Progress Tracking**: Shows progress bars when `show_progress=True`

## Usage Examples

### Example 1: Read a Single CSV File

```python
from pamola_core.utils.ops.op_data_reader import DataReader

reader = DataReader()

# Read a CSV file with auto-detection
df, error = reader.read_dataframe("data/input.csv")

if error is None:
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(df.head())
else:
    print(f"Failed to read file: {error['message']}")
```

### Example 2: Read Large File with Memory Optimization

```python
# Read a large CSV with Dask if needed
df, error = reader.read_dataframe(
    "data/large_dataset.csv",
    columns=["customer_id", "email", "phone"],  # Read only needed columns
    memory_limit=2.0,  # Use Dask if > 2GB
    auto_optimize=True  # Optimize memory after loading
)

if error is None and df is not None:
    print(f"Successfully loaded {len(df)} rows")
```

### Example 3: Read Multiple CSV Files

```python
# Read multiple similar CSV files
source_dict = {
    "jan": "data/2025_01_sales.csv",
    "feb": "data/2025_02_sales.csv",
    "mar": "data/2025_03_sales.csv"
}

df, error = reader.read_dataframe(source_dict)

if error is None:
    print(f"Combined dataset: {len(df)} rows")
```

### Example 4: Read Excel File with Specific Sheet

```python
df, error = reader.read_dataframe(
    "reports/financial_data.xlsx",
    sheet_name="Q1_Results",  # Read specific sheet
    nrows=1000  # Limit to first 1000 rows
)

if error is None:
    print(df.info())
```

### Example 5: Read Encrypted CSV File

```python
df, error = reader.read_dataframe(
    "data/sensitive.csv.enc",
    encryption_key="my_secret_key",
    use_encryption=True
)

if error is None:
    print("Successfully decrypted and loaded data")
```

### Example 6: Read with Custom Parameters

```python
# Read CSV with custom dialect
df, error = reader.read_dataframe(
    "data/custom_dialect.csv",
    delimiter=";",  # Semicolon delimiter
    quotechar="'",  # Single quote for text
    encoding="latin-1",  # Different encoding
    skiprows=[0, 1]  # Skip header rows
)
```

### Example 7: Integration with Operations

```python
from pamola_core.utils.ops.op_base import DataFrameOperation
from pamola_core.utils.ops.op_data_reader import DataReader

class CustomAnalysisOperation(DataFrameOperation):
    def __init__(self, **kwargs):
        super().__init__(name="CustomAnalysis", **kwargs)
        self.reader = DataReader(logger=self.logger)

    def execute(self, data_source, task_dir, reporter, **kwargs):
        # Use DataReader to load additional data
        df, error = self.reader.read_dataframe(
            data_source.get_file_path("main"),
            auto_optimize=True,
            show_progress=True
        )

        if error is not None:
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=f"Failed to read data: {error['message']}"
            )

        # Process the DataFrame
        # ... your operation logic ...

        return OperationResult(status=OperationStatus.SUCCESS)
```

## Best Practices

1. **Always Check Error Information**
   - Never assume successful reads; always check the error_info tuple element
   - Log errors for debugging and audit trails

2. **Use Column Filtering for Large Datasets**
   - Read only required columns to reduce memory usage
   - Especially important for datasets with hundreds of columns

3. **Leverage Auto-Detection**
   - Set `detect_parameters=True` for CSV files to auto-detect dialects
   - Let the reader determine optimal read strategy automatically

4. **Configure Memory Limits**
   - Set `memory_limit` based on available system memory
   - Use Dask (`use_dask=True`) for datasets larger than available RAM

5. **Optimize for Repeated Reads**
   - Cache DataReader instances to avoid re-initialization
   - Reuse same reader for multiple files in a batch operation

6. **Progress Tracking**
   - Enable `show_progress=True` for long-running reads
   - Monitor progress in user-facing operations

7. **Encryption Security**
   - Never hardcode encryption keys; use environment variables or secure vaults
   - Validate file encryption status before reading

8. **Post-Read Optimization**
   - Enable `auto_optimize=True` to reduce memory footprint
   - Particularly useful when DataFrame stays in memory for processing

## Troubleshooting

### Issue: Out of Memory Error When Reading Large File

**Solution:**
```python
# Enable Dask for out-of-core processing
df, error = reader.read_dataframe(
    "huge_file.csv",
    use_dask=True,  # Use distributed processing
    memory_limit=2.0  # Set memory limit in GB
)
```

### Issue: Unrecognized CSV Dialect

**Solution:**
```python
# Specify format parameters explicitly
df, error = reader.read_dataframe(
    "unusual_file.csv",
    delimiter="|",
    quotechar="'",
    encoding="latin-1",
    detect_parameters=False  # Skip auto-detection
)
```

### Issue: Encrypted File Not Recognized

**Solution:**
```python
# Explicitly enable encryption handling
df, error = reader.read_dataframe(
    "data.csv.enc",
    use_encryption=True,
    encryption_key="your_key",
    encryption_mode="AES-256-CBC"
)

if error:
    print(f"Encryption error: {error}")
```

### Issue: Memory Optimization Failed

**Solution:**
```python
# Disable auto-optimization and handle manually
df, error = reader.read_dataframe(
    "data.csv",
    auto_optimize=False  # Skip optimization
)

if df is not None and error is None:
    # Manually optimize if needed
    df, info = optimize_dataframe_memory(df)
```

### Issue: Very Slow Read Performance

**Solution:**
```python
# Reduce dataset size or use chunking
df, error = reader.read_dataframe(
    "data.csv",
    columns=["col1", "col2", "col3"],  # Read fewer columns
    nrows=100000,  # Limit rows
    show_progress=True  # Monitor progress
)
```

## Related Components

| Component | Purpose | Usage |
|-----------|---------|-------|
| `pamola_core.utils.io` | Low-level reading functions | Used internally by DataReader |
| `DataSource` | Data source management | Container for DataReader output |
| `BaseOperation` | Operation framework | Parent class for operations using DataReader |
| `OperationResult` | Operation results | Returns DataReader errors wrapped in results |
| `HierarchicalProgressTracker` | Progress tracking | Integrated with DataReader for status |
| `HierarchicalProgressTracker` | Nested progress | Used for multi-stage reads |

## Summary

The `DataReader` module is a critical utility for operations that need flexible, robust data loading capabilities. By providing a unified interface with intelligent strategy selection, it abstracts complexity while maintaining framework standards for error handling, progress tracking, and resource management.

Key strengths:
- Single interface for all data formats
- Automatic optimization and strategy selection
- Comprehensive error reporting
- Tight integration with io.py
- Memory-aware processing

Use DataReader whenever your operation needs to:
- Load data from user-provided sources
- Handle multiple file formats transparently
- Process large files efficiently
- Maintain consistent error handling patterns
- Track operation progress
