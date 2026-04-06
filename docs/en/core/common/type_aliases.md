# Type Aliases & Configuration Classes

**Module:** `pamola_core.common.type_aliases`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Type Aliases](#type-aliases)
3. [CryptoConfig Class](#cryptoconfig-class)
4. [FileCryptoConfig Class](#filecryptoconfig-class)
5. [Utility Functions](#utility-functions)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Related Components](#related-components)

## Overview

The `type_aliases` module provides custom type definitions for consistency across the codebase and configuration classes for managing encryption settings. This module defines flexible type signatures for file paths, dataframes, and crypto configurations used throughout PAMOLA.CORE.

**Content Type:** Type Definitions & Configuration Classes
**Primary Purpose:** Type safety and encryption management
**Dependencies:** pandas, dask, pathlib, typing

## Type Aliases

### PathLike

```python
PathLike = Union[str, Path]
```

**Purpose:** Accept both string and Path objects

**Usage:**
```python
from pamola_core.common.type_aliases import PathLike
from pathlib import Path

def load_data(filepath: PathLike) -> pd.DataFrame:
    """Load data from filepath (string or Path)."""
    filepath = Path(filepath)  # Normalize to Path
    return pd.read_csv(filepath)

# Both work
load_data("/path/to/file.csv")
load_data(Path("/path/to/file.csv"))
```

### PathLikeOrList

```python
PathLikeOrList = Union[PathLike, List[PathLike]]
```

**Purpose:** Accept single path or list of paths

**Usage:**
```python
from pamola_core.common.type_aliases import PathLikeOrList
from pathlib import Path

def process_files(files: PathLikeOrList) -> list:
    """Process one or more files."""
    if isinstance(files, (str, Path)):
        files = [files]

    results = []
    for file in files:
        results.append(process_file(file))
    return results

# Both work
process_files("file.csv")
process_files(["file1.csv", "file2.csv"])
```

### DataFrameType

```python
DataFrameType = Union[pd.DataFrame, dd.DataFrame]
```

**Purpose:** Accept both pandas and dask DataFrames

**Usage:**
```python
from pamola_core.common.type_aliases import DataFrameType
import pandas as pd
import dask.dataframe as dd

def analyze_data(df: DataFrameType) -> dict:
    """Analyze pandas or dask DataFrame."""
    if isinstance(df, dd.DataFrame):
        # Dask-specific operations
        return df.compute().describe().to_dict()
    else:
        # Pandas operations
        return df.describe().to_dict()

# Both work
pandas_df = pd.read_csv("data.csv")
dask_df = dd.read_csv("data-*.csv")

analyze_data(pandas_df)
analyze_data(dask_df)
```

## CryptoConfig Class

### Overview

`CryptoConfig` manages encryption configuration with validation and serialization support.

**Purpose:** Centralized encryption settings management
**Validation:** Mode and algorithm validation
**Features:** Dictionary serialization, validation, default values

### Constructor

```python
CryptoConfig(
    mode: Optional[str] = None,
    algorithm: Optional[str] = None,
    key: Optional[str] = None,
    key_path: Optional[PathLike] = None
)
```

**Parameters:**
- `mode` (str, optional): Encryption mode (e.g., "simple", "age")
- `algorithm` (str, optional): Encryption algorithm (e.g., "AES")
- `key` (str, optional): Encryption key as string
- `key_path` (PathLike, optional): Path to key file

**Validation:**
- Validates mode against `allowed_modes` (extensible)
- Validates algorithm against `allowed_algorithms` (extensible)
- Raises `ValidationError` on invalid input

### Methods

#### `__repr__()`
Returns string representation of configuration

```python
config = CryptoConfig(mode="simple", algorithm="AES")
print(config)
# CryptoConfig(mode=simple, algorithm=AES, key=None, key_path=None)
```

#### `to_dict() -> Dict`
Convert configuration to dictionary

```python
config = CryptoConfig(
    mode="simple",
    algorithm="AES",
    key="secret-key",
    key_path="/path/to/keyfile"
)

config_dict = config.to_dict()
# {
#     "mode": "simple",
#     "algorithm": "AES",
#     "key": "secret-key",
#     "key_path": "/path/to/keyfile"
# }
```

#### `from_dict(data: Optional[dict]) -> CryptoConfig` (classmethod)
Create instance from dictionary

```python
config_dict = {
    "mode": "age",
    "algorithm": "AES-256",
    "key": "secret-key"
}

config = CryptoConfig.from_dict(config_dict)
# CryptoConfig(mode=age, algorithm=AES-256, key=secret-key, key_path=None)

# Empty dict or None uses defaults
config = CryptoConfig.from_dict()
# CryptoConfig(mode=None, algorithm=None, key=None, key_path=None)
```

#### `_validate()`
Validate configuration (called in constructor)

Raises `ValidationError` if:
- `mode` is set but not in `allowed_modes`
- `algorithm` is set but not in `allowed_algorithms`

### Class Attributes

```python
class CryptoConfig:
    mode: Optional[str] = None
    algorithm: Optional[str] = None
    key: Optional[str] = None
    key_path: Optional[PathLike] = None
```

### Usage Examples

```python
from pamola_core.common.type_aliases import CryptoConfig

# Create basic config
config = CryptoConfig(mode="simple")

# Create with all fields
config = CryptoConfig(
    mode="age",
    algorithm="AES-256",
    key="my-encryption-key",
    key_path="/etc/crypto/key.pem"
)

# Serialize/deserialize
config_dict = config.to_dict()
config_restored = CryptoConfig.from_dict(config_dict)

# Default values
config_default = CryptoConfig()  # All None
```

## FileCryptoConfig Class

### Overview

`FileCryptoConfig` associates files with their encryption settings.

**Purpose:** File-specific encryption configuration
**Usage:** Task framework, file encryption pipelines
**Features:** Path normalization, nested config serialization

### Constructor

```python
FileCryptoConfig(
    file_paths: Optional[PathLikeOrList] = None,
    crypto_config: Optional[CryptoConfig] = None
)
```

**Parameters:**
- `file_paths` (PathLikeOrList, optional): Single file or list of files
- `crypto_config` (CryptoConfig, optional): Associated encryption config

### Methods

#### `__repr__()`
String representation

```python
config = FileCryptoConfig(
    file_paths="/data/file.csv",
    crypto_config=CryptoConfig(mode="simple")
)
print(config)
# FileCryptoConfig(file_paths=/data/file.csv, crypto_config=CryptoConfig(...))
```

#### `to_dict() -> Dict`
Convert to dictionary (handles nested CryptoConfig)

```python
file_config = FileCryptoConfig(
    file_paths=["/data/file1.csv", "/data/file2.csv"],
    crypto_config=CryptoConfig(
        mode="age",
        algorithm="AES",
        key="secret"
    )
)

config_dict = file_config.to_dict()
# {
#     "file_paths": ["/data/file1.csv", "/data/file2.csv"],
#     "crypto_config": {
#         "mode": "age",
#         "algorithm": "AES",
#         "key": "secret",
#         "key_path": None
#     }
# }
```

#### `from_dict(data: Optional[dict]) -> FileCryptoConfig` (classmethod)
Create from dictionary (recreates nested CryptoConfig)

```python
config_dict = {
    "file_paths": "/data/file.csv",
    "crypto_config": {
        "mode": "simple",
        "algorithm": "AES"
    }
}

file_config = FileCryptoConfig.from_dict(config_dict)
# FileCryptoConfig with CryptoConfig reconstructed from nested dict
```

### Class Attributes

```python
class FileCryptoConfig:
    file_paths: Optional[PathLikeOrList] = None
    crypto_config: Optional[CryptoConfig] = None
```

### Usage Examples

```python
from pamola_core.common.type_aliases import FileCryptoConfig, CryptoConfig

# Single file configuration
config = FileCryptoConfig(
    file_paths="/data/sensitive.csv",
    crypto_config=CryptoConfig(mode="age")
)

# Multiple files
config = FileCryptoConfig(
    file_paths=["/data/file1.csv", "/data/file2.csv"],
    crypto_config=CryptoConfig(mode="simple", algorithm="AES")
)

# Serialize for storage
config_dict = config.to_dict()
save_config(config_dict)

# Deserialize
loaded_dict = load_config()
config = FileCryptoConfig.from_dict(loaded_dict)
```

## Utility Functions

### convert_to_flatten_dict()

Flattens nested dictionaries into single-level structure.

```python
def convert_to_flatten_dict(
    data: dict,
    prefix: str = "",
    separator: str = "_",
    append_key: bool = True
) -> dict
```

**Parameters:**
- `data` (dict): Dictionary to flatten
- `prefix` (str): Prefix for keys (usually empty at start)
- `separator` (str): Separator between prefix and key
- `append_key` (bool): Whether to append separator to prefix

**Returns:** Flattened dictionary

**Example:**
```python
from pamola_core.common.type_aliases import convert_to_flatten_dict

nested = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "credentials": {
            "username": "admin",
            "password": "secret"
        }
    },
    "app": {
        "name": "PAMOLA",
        "version": "1.0"
    }
}

flat = convert_to_flatten_dict(nested)
# {
#     "database_host": "localhost",
#     "database_port": 5432,
#     "database_credentials_username": "admin",
#     "database_credentials_password": "secret",
#     "app_name": "PAMOLA",
#     "app_version": "1.0"
# }

# Custom separator
flat_hyphen = convert_to_flatten_dict(nested, separator="-")
# {"database-host": "localhost", ...}
```

**Use Cases:**
- Environment variable conversion
- Configuration flattening
- Database schema flattening

## Usage Examples

### Type-Safe Function Signatures

```python
from pamola_core.common.type_aliases import PathLike, DataFrameType
import pandas as pd

def process_dataset(
    input_path: PathLike,
    df: DataFrameType,
    output_path: PathLike
) -> DataFrameType:
    """
    Process dataset with flexible input/output types.

    Parameters:
        input_path: Can be string or Path object
        df: Can be pandas or dask DataFrame
        output_path: Can be string or Path object

    Returns:
        Processed DataFrame (same type as input)
    """
    # Process DataFrame
    if isinstance(df, dd.DataFrame):
        result = df.map_partitions(apply_transformation).compute()
    else:
        result = df.apply(apply_transformation)

    # Handle PathLike
    from pathlib import Path
    output_path = Path(output_path)
    result.to_csv(output_path)

    return result
```

### Encryption Configuration Workflow

```python
from pamola_core.common.type_aliases import CryptoConfig, FileCryptoConfig
import json

# Create configuration
crypto_config = CryptoConfig(
    mode="age",
    algorithm="AES-256",
    key_path="/secure/key.pem"
)

file_config = FileCryptoConfig(
    file_paths=["/data/sensitive1.csv", "/data/sensitive2.csv"],
    crypto_config=crypto_config
)

# Save to JSON
config_dict = file_config.to_dict()
with open("encryption_config.json", "w") as f:
    json.dump(config_dict, f)

# Load from JSON
with open("encryption_config.json") as f:
    loaded_dict = json.load(f)

loaded_config = FileCryptoConfig.from_dict(loaded_dict)

# Use in encryption task
for file_path in loaded_config.file_paths:
    encrypt_file(file_path, loaded_config.crypto_config)
```

### Configuration Flattening for Environment Variables

```python
from pamola_core.common.type_aliases import convert_to_flatten_dict
import os

config = {
    "database": {
        "primary": {
            "host": "db1.example.com",
            "port": 5432
        },
        "replica": {
            "host": "db2.example.com",
            "port": 5432
        }
    },
    "encryption": {
        "mode": "age",
        "algorithm": "AES-256"
    }
}

flat_config = convert_to_flatten_dict(config)

# Set environment variables
for key, value in flat_config.items():
    env_var = f"APP_{key.upper()}"
    os.environ[env_var] = str(value)

# Now accessible as:
# APP_DATABASE_PRIMARY_HOST, APP_DATABASE_REPLICA_HOST, etc.
```

## Best Practices

1. **Use Type Aliases for Consistency**
   ```python
   # Good - clear intent
   def load_file(path: PathLike) -> pd.DataFrame:
       pass

   # Less clear
   def load_file(path) -> pd.DataFrame:
       pass
   ```

2. **Always Validate CryptoConfig**
   ```python
   # Good - validation happens in constructor
   config = CryptoConfig(mode="age")  # Validates mode

   # Unsafe
   config = {"mode": "invalid"}  # No validation
   ```

3. **Use Serialization for Storage**
   ```python
   # Good - serializable format
   config_dict = crypto_config.to_dict()
   save_json(config_dict)

   # Avoid
   pickle.dump(crypto_config, file)  # Not portable
   ```

4. **Normalize Paths When Needed**
   ```python
   from pathlib import Path

   def process_path(path: PathLike) -> Path:
       """Normalize PathLike to Path object."""
       return Path(path)
   ```

5. **Document CryptoConfig Validation Rules**
   ```python
   """
   CryptoConfig validation:
   - mode: Must be in ['simple', 'age'] if specified
   - algorithm: Must be in ['AES', 'AES-256'] if specified
   - key: Optional, use either key or key_path
   - key_path: Optional, path must exist
   """
   ```

## Related Components

- **EncryptionMode** (`pamola_core.common.enum.encryption_mode`) - Uses mode values
- **Task Configuration** (`pamola_core.utils.tasks`) - Uses CryptoConfig
- **I/O Operations** (`pamola_core.io`) - Uses PathLike for file handling
- **Constants** (`pamola_core.common.constants`) - Related configuration

## Implementation Notes

- Type aliases use Python's `typing` module for compatibility
- CryptoConfig uses class attributes for flexibility
- FileCryptoConfig handles nested serialization automatically
- `convert_to_flatten_dict` is recursive and handles arbitrary nesting
- All classes support None defaults for optional fields

## Summary

| Component | Purpose | Key Usage |
|-----------|---------|-----------|
| `PathLike` | Flexible file path type | Function parameters |
| `PathLikeOrList` | Single/multiple paths | Batch operations |
| `DataFrameType` | Pandas/Dask support | Distributed computing |
| `CryptoConfig` | Encryption settings | Secure data handling |
| `FileCryptoConfig` | File+encryption pairing | Task configuration |
| `convert_to_flatten_dict` | Config flattening | Environment variables |
