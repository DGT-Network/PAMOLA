# IO Helpers Module Documentation

**Package:** `pamola_core.utils.io_helpers`
**Version:** 1.0
**Last Updated:** 2026-03-23
**Type:** Internal (Non-Public API)

## Overview

The `io_helpers` package provides a collection of specialized utility functions for I/O operations within PAMOLA.CORE. These helpers support encryption/decryption, CSV processing, memory management, file handling, and other I/O infrastructure used by higher-level components like `DataReader` and `DataSource`.

## Architecture

```
pamola_core.utils.io_helpers/
├── Core Utilities
│   ├── crypto_utils.py - Encryption/decryption operations
│   ├── crypto_router.py - Encryption mode detection
│   └── directory_utils.py - Directory and temp file management
│
├── Format-Specific Helpers
│   ├── csv_utils.py - CSV reading/writing utilities
│   ├── json_utils.py - JSON operations
│   ├── dask_utils.py - Dask distributed processing
│   ├── readers.py - Multi-format reading
│   └── format_utils.py - Format detection and conversion
│
├── Memory Management
│   ├── memory_utils.py - Memory monitoring and optimization
│   ├── multi_file_utils.py - Multi-file memory management
│   └── file_utils.py - File size and metadata
│
├── Error Handling
│   ├── error_utils.py - Error categorization and handling
│   └── temp_files.py - Temporary file management
│
├── Advanced Features
│   ├── image_utils.py - Image file handling
│   ├── provider_interface.py - Abstract provider interface
│   └── config_utils.py - Configuration utilities
└── __init__.py - Public API exports
```

## Component Files

| File | Purpose | Key Functions |
|------|---------|---|
| `crypto_utils.py` | Encryption/decryption | `encrypt_file()`, `decrypt_file()`, `decrypt_data()` |
| `crypto_router.py` | Encryption detection | `detect_encryption_mode()` |
| `csv_utils.py` | CSV operations | CSV reading/writing with dialect detection |
| `json_utils.py` | JSON operations | JSON reading/writing with compression |
| `dask_utils.py` | Dask integration | Distributed DataFrame processing |
| `memory_utils.py` | Memory management | Memory monitoring and optimization |
| `file_utils.py` | File operations | File size, metadata, existence checks |
| `format_utils.py` | Format handling | Format detection and validation |
| `readers.py` | Multi-format reading | Unified reading interface |
| `directory_utils.py` | Directory management | Temp file cleanup, directory operations |
| `error_utils.py` | Error handling | Exception categorization |
| `multi_file_utils.py` | Multi-file support | Processing multiple files efficiently |
| `image_utils.py` | Image handling | Image file operations |
| `temp_files.py` | Temp files | Temporary file lifecycle |
| `provider_interface.py` | Provider abstraction | Abstract interfaces for providers |

## Public API

The following functions are exposed in the public API via `__init__.py`:

```python
# Encryption/Decryption
from pamola_core.utils.io_helpers import decrypt_file
from pamola_core.utils.io_helpers import decrypt_data
from pamola_core.utils.io_helpers import encrypt_file

# Directory Management
from pamola_core.utils.io_helpers import safe_remove_temp_file

# Encryption Detection
from pamola_core.utils.io_helpers import detect_encryption_mode
```

## Key Concepts

### Encryption Support

The `io_helpers` package provides transparent encryption/decryption for data files:

- **detect_encryption_mode()**: Identifies encryption method from file
- **encrypt_file()**: Encrypts file with specified algorithm
- **decrypt_file()**: Decrypts encrypted files
- **decrypt_data()**: Decrypts in-memory data

Supported encryption methods:
- AES (Advanced Encryption Standard)
- Age encryption format
- Custom provider implementations

### Memory Management

Comprehensive memory monitoring and optimization:

- **Memory monitoring**: Track system and process memory usage
- **Memory optimization**: Suggest type conversions and storage improvements
- **Chunk processing**: Process large files in memory-efficient chunks
- **Dask integration**: Automatic switch to distributed processing for large datasets

### Format Detection

Automatic detection and validation of file formats:

- CSV with dialect detection (delimiters, quoting, escaping)
- JSON with compression detection
- Parquet, Excel, and other tabular formats
- Custom format detection via provider interface

### Error Handling

Specialized error handling for I/O operations:

- File not found errors
- Permission errors
- Format validation errors
- Encryption/decryption errors
- Memory errors

## Usage Patterns

### Pattern 1: Encrypted File Reading

```python
from pamola_core.utils.io_helpers import detect_encryption_mode, decrypt_file

# Detect encryption
mode = detect_encryption_mode("data.csv.enc")

# Decrypt if needed
if mode:
    decrypt_file("data.csv.enc", "data.csv", encryption_key="secret")
```

### Pattern 2: CSV Processing with Dialect Detection

```python
from pamola_core.utils.io_helpers.csv_utils import prepare_csv_reader_options

# Get optimal CSV reading options
options = prepare_csv_reader_options(
    encoding="utf-8",
    delimiter=",",
    quotechar='"'
)

# Use with pandas
df = pd.read_csv("data.csv", **options)
```

### Pattern 3: Large File Processing with Dask

```python
from pamola_core.utils.io_helpers.dask_utils import read_large_csv

# Use Dask for large files
ddf = read_large_csv("huge_file.csv", blocksize="64MB")

# Process in chunks
result = ddf.map_partitions(process_partition).compute()
```

### Pattern 4: Memory-Aware Processing

```python
from pamola_core.utils.io_helpers.memory_utils import estimate_memory_usage
from pamola_core.utils.io_helpers.file_utils import get_file_size

# Check if file fits in memory
file_size = get_file_size("data.csv")
available_memory = get_available_memory()

if file_size < available_memory * 0.8:
    df = pd.read_csv("data.csv")  # Load all
else:
    # Process in chunks
    process_in_chunks("data.csv")
```

## Best Practices

1. **Always Check Encryption**
   - Use `detect_encryption_mode()` before reading encrypted files
   - Provide encryption keys securely
   - Validate decrypted data integrity

2. **Handle Large Files Carefully**
   - Use memory estimation before loading
   - Switch to Dask for files larger than available RAM
   - Process in chunks when possible

3. **Validate Formats**
   - Detect CSV dialects automatically
   - Validate file format before processing
   - Handle format-specific options

4. **Clean Up Temporary Files**
   - Use `safe_remove_temp_file()` for cleanup
   - Implement proper error handling
   - Verify file deletion

5. **Monitor Memory Usage**
   - Check available system memory before operations
   - Track process memory during processing
   - Implement memory thresholds

## Error Handling

The `io_helpers` package provides specialized error handling:

```python
from pamola_core.errors import PamolaFileNotFoundError, ValidationError

try:
    data = read_encrypted_file("data.csv.enc", key="secret")
except PamolaFileNotFoundError:
    print("File not found")
except ValidationError:
    print("File format invalid")
```

## Integration Points

`io_helpers` is used by:

- **DataReader**: Unified reading interface using helpers
- **DataSource**: Data source management with encryption support
- **BaseOperation**: Operations using I/O helpers for data access
- **pamola_core.utils.io**: High-level I/O functions built on helpers

## Related Documentation

- [DataReader](../ops/op_data_reader.md) - Uses io_helpers for reading
- [DataSource](../ops/op_data_source.md) - Uses io_helpers for data management
- [BaseOperation](../ops/op_base.md) - Uses io_helpers for I/O operations
- [Encryption Guide](../../../encryption-guide.md) - Detailed encryption documentation

## Summary

The `io_helpers` package provides essential infrastructure for file I/O, encryption, memory management, and format handling. These utilities abstract complexity while maintaining performance and security standards.

Key strengths:
- Transparent encryption/decryption support
- Automatic format detection
- Memory-aware processing
- Error categorization
- Provider-based extensibility

The helpers support the framework's core data processing capabilities and enable robust, efficient handling of diverse data sources and formats.
