# Configs Module Documentation

**Module:** `pamola_core.configs`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Configuration Sources](#configuration-sources)
6. [Usage Examples](#usage-examples)
7. [Environment Variables](#environment-variables)
8. [Best Practices](#best-practices)
9. [Related Components](#related-components)
10. [Summary Analysis](#summary-analysis)

## Overview

The `configs` module provides centralized configuration management for the PAMOLA.CORE ecosystem. It supports multiple configuration sources (environment variables, JSON config files), validation, and dynamic override capabilities while maintaining sensible defaults.

### Purpose

The configs module:
- Provides unified configuration access across PAMOLA components
- Supports development and production environments with different settings
- Loads configuration from JSON files with automatic discovery
- Allows environment variable overrides for deployment flexibility
- Validates configuration parameters before use
- Manages data repository detection and directory structure configuration
- Handles performance, logging, and privacy-specific settings

### Module Structure

```
pamola_core/configs/
├── __init__.py              # Empty (internal API)
├── config_variables.py      # L-Diversity specific config
├── field_definitions.py     # Field metadata and privacy categories
└── settings.py              # Core configuration management
```

### Key Design Principles

- **Sensible Defaults**: Works out-of-box with reasonable defaults
- **Override Hierarchy**: Environment > Config File > Defaults
- **Single Responsibility**: Each submodule handles specific configuration domain
- **Lazy Loading**: Configuration loaded on-demand, not at import time
- **Non-Intrusive**: No side effects at import; explicit loading required

## Key Features

### 1. **Multi-Source Configuration Loading**
- **JSON Config Files**: Load from `prj_config.json`, `.pamola_core/config.json`
- **Environment Variables**: `PAMOLA_*` prefixed variables override defaults
- **Project Root Detection**: Auto-discover project structure
- **User Home Directory**: Support for user-specific config in `~/.pamola_core/`

### 2. **Configuration Validation**
- **L-Diversity Parameters**: Validate diversity types, l-values, c-values
- **Type Checking**: Ensure configuration values match expected types
- **Range Validation**: Enforce valid parameter ranges
- **Clear Error Messages**: Describe validation failures with actionable guidance

### 3. **Directory Structure Management**
- **Flexible Organization**: Configure raw/processed/logs/configs subdirectories
- **Nested Dictionaries**: Support hierarchical configuration
- **Path Resolution**: Automatic directory creation on-demand

### 4. **Data Repository Management**
- **Auto-Detection**: Discover data repository in project structure
- **Environment Override**: `PAMOLA_DATA_REPOSITORY` environment variable
- **Explicit Setting**: Programmatic configuration via `set_data_repository()`

### 5. **Performance Settings**
- **Chunk Size**: Configure batch processing sizes (default: 100,000 records)
- **Memory Limits**: Set memory allocation constraints
- **Encoding**: Configure default text encoding
- **Delimiters**: Set default CSV delimiters

### 6. **Logging Configuration**
- **Log Levels**: Configure via config or environment
- **File Output**: Log to specified file
- **Format Strings**: Customize log message format
- **Dynamic Level Setting**: Apply configuration at runtime

## Architecture

### Configuration Hierarchy

```
┌─────────────────────────────────────────┐
│     get_config() / load_config()        │
├─────────────────────────────────────────┤
│  Configuration Loading Pipeline:        │
├─────────────────────────────────────────┤
│  1. Load DEFAULT_CONFIG                 │
│  2. Override with JSON file (if found)  │
│  3. Override with env variables         │
│  4. Auto-detect missing values          │
│  5. Validate final configuration        │
├─────────────────────────────────────────┤
│  _config (singleton pattern)            │
└─────────────────────────────────────────┘
```

### File Search Order

Configuration files searched in this order:

1. Explicit `config_path` parameter (if provided)
2. `PAMOLA_CONFIG_PATH` environment variable
3. `{project_root}/configs/prj_config.json`
4. `~/.pamola_core/config.json` (user home)
5. `{project_root}/prj_config.json` (fallback)

### Data Repository Discovery

```
1. Check PAMOLA_DATA_REPOSITORY env var
2. Scan {project_root}/data/
3. Scan {project_root}/DATA/
4. Fall back to project root
```

## Core Components

### settings.py

Central configuration management for general PAMOLA settings.

| Function | Purpose |
|----------|---------|
| `load_config(config_path)` | Load configuration from file or defaults |
| `get_config()` | Get current configuration (singleton) |
| `get_data_repository()` | Get configured data repository path |
| `set_data_repository(path)` | Explicitly set data repository |
| `get_directory_structure()` | Get directory hierarchy configuration |
| `get_performance_settings()` | Get performance-related settings |
| `get_logging_settings()` | Get logging configuration |
| `save_config(config_path)` | Save current config to JSON file |
| `update_nested_dict(d1, d2)` | Deep merge dictionaries |
| `get_config_file_paths()` | Get candidate config file paths |

### config_variables.py

L-Diversity anonymization-specific configuration.

| Function | Purpose |
|----------|---------|
| `get_l_diversity_config(overrides)` | Get L-Diversity config with optional overrides |
| `validate_l_diversity_config(config)` | Validate L-Diversity parameter values |

**Default L-Diversity Parameters:**
- `l`: L-diversity level (default: 3)
- `diversity_type`: Type of diversity ("distinct", "entropy", "recursive"; default: "distinct")
- `c_value`: Recursive diversity constant (default: 1.0)
- `k`: K-anonymity baseline (default: 2)
- `use_dask`: Enable Dask processing (default: False)
- `mask_value`: Default mask string (default: "MASKED")
- And more (see [Environment Variables](#environment-variables))

### field_definitions.py

Metadata definitions for data fields including privacy categories and anonymization strategies.

**Key Classes:**
- `FieldType` enum: Data types (SHORT_TEXT, LONG_TEXT, DOUBLE, LONG, DATE)
- `PrivacyCategory` enum: Privacy levels (DIRECT_IDENTIFIER, QUASI_IDENTIFIER, etc.)
- `AnonymizationStrategy` enum: Anonymization methods (PSEUDONYMIZATION, GENERALIZATION, etc.)
- `ProfilingTask` enum: Data profiling operations (COMPLETENESS, UNIQUENESS, etc.)

**Constants:**
- `TABLES`: Mapping of table names to field lists
- `FIELD_DEFINITIONS`: Comprehensive field metadata for resume dataset

## Configuration Sources

### Default Configuration (settings.py)

Built-in defaults for general PAMOLA settings:

```python
DEFAULT_CONFIG = {
    "data_repository": None,  # Auto-detected
    "directory_structure": {
        "raw": "raw",
        "processed": "processed",
        "logs": "logs",
        "configs": "configs"
    },
    "logging": {
        "level": "INFO",
        "file": "pamola_processing.log",
        "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    },
    "performance": {
        "chunk_size": 100000,
        "default_encoding": "utf-8",
        "default_delimiter": ",",
        "memory_limit_mb": 1000
    }
}
```

### JSON Configuration Files

Load from files in JSON format:

```json
{
  "data_repository": "/data/resume_dataset",
  "directory_structure": {
    "raw": "input",
    "processed": "output",
    "logs": "logs"
  },
  "logging": {
    "level": "DEBUG",
    "file": "/var/log/pamola.log"
  },
  "performance": {
    "chunk_size": 50000,
    "memory_limit_mb": 2000
  }
}
```

### Environment Variables

Override defaults via `PAMOLA_*` prefixed variables (see [Environment Variables](#environment-variables) section).

## Usage Examples

### Example 1: Load Configuration with Auto-Detection

```python
from pamola_core.configs.settings import get_config

# Load config (tries standard locations, auto-detects data repo)
config = get_config()

print(f"Data repository: {config['data_repository']}")
print(f"Chunk size: {config['performance']['chunk_size']}")
print(f"Log level: {config['logging']['level']}")
```

### Example 2: Load Specific Config File

```python
from pamola_core.configs.settings import load_config
from pathlib import Path

# Load from specific path
config = load_config(config_path=Path("configs/custom_config.json"))

print(f"Config loaded from: {Path('configs/custom_config.json')}")
```

### Example 3: Get L-Diversity Configuration

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Get L-Diversity defaults
config = get_l_diversity_config()

# Or with overrides
custom_config = get_l_diversity_config({
    "l": 5,
    "diversity_type": "entropy",
    "npartitions": 8
})

print(f"L-Diversity level: {custom_config['l']}")
print(f"Diversity type: {custom_config['diversity_type']}")
```

### Example 4: Manage Data Repository

```python
from pamola_core.configs.settings import (
    get_data_repository,
    set_data_repository
)

# Get current repository
repo = get_data_repository()
print(f"Current repo: {repo}")

# Change repository
set_data_repository("/new/data/path")

# Verify change
new_repo = get_data_repository()
print(f"New repo: {new_repo}")
```

### Example 5: Access Field Definitions

```python
from pamola_core.configs.field_definitions import (
    FIELD_DEFINITIONS,
    FieldType,
    PrivacyCategory
)

# Get field metadata
email_def = FIELD_DEFINITIONS['email']
print(f"Field type: {email_def['type']}")
print(f"Privacy category: {email_def['category']}")
print(f"Strategy: {email_def['strategy']}")

# Filter fields by type
text_fields = {
    name: defn for name, defn in FIELD_DEFINITIONS.items()
    if defn['type'] == FieldType.SHORT_TEXT
}
print(f"Text fields: {list(text_fields.keys())}")
```

### Example 6: Save Modified Configuration

```python
from pamola_core.configs.settings import (
    get_config,
    set_data_repository,
    save_config
)

# Modify config
set_data_repository("/updated/data/path")

# Save to file
config_path = save_config(config_path="configs/updated_config.json")
print(f"Config saved to: {config_path}")
```

## Environment Variables

All PAMOLA configuration can be overridden via environment variables:

### General Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PAMOLA_CONFIG_PATH` | (none) | Path to configuration JSON file |
| `PAMOLA_DATA_REPOSITORY` | (auto-detect) | Path to data repository |

### L-Diversity Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PAMOLA_L_DIVERSITY_L` | 3 | L-diversity level |
| `PAMOLA_L_DIVERSITY_TYPE` | "distinct" | Diversity type: "distinct", "entropy", "recursive" |
| `PAMOLA_L_DIVERSITY_C_VALUE` | 1.0 | Recursive diversity constant |
| `PAMOLA_L_DIVERSITY_K` | 2 | K-anonymity baseline |
| `PAMOLA_L_DIVERSITY_USE_DASK` | "False" | Enable Dask processing |
| `PAMOLA_L_DIVERSITY_MASK_VALUE` | "MASKED" | Mask string |
| `PAMOLA_L_DIVERSITY_SUPPRESSION` | "True" | Enable suppression |
| `PAMOLA_L_DIVERSITY_NPARTITIONS` | 4 | Dask partition count |
| `PAMOLA_L_DIVERSITY_OPTIMIZE_MEMORY` | "True" | Memory optimization |
| `PAMOLA_L_DIVERSITY_LOG_LEVEL` | "INFO" | Logging level |
| `PAMOLA_L_DIVERSITY_HIST_BINS` | 20 | Histogram bin count |
| `PAMOLA_L_DIVERSITY_SAVE_FORMAT` | "png" | Visualization format |
| `PAMOLA_L_DIVERSITY_RISK_THRESHOLD` | 0.5 | Privacy risk threshold |

### Boolean Conversion

String values converted to booleans:
- True: "true" (case-insensitive)
- False: any other value

```python
# Environment variable
os.environ['PAMOLA_L_DIVERSITY_USE_DASK'] = 'True'

# Read by config
config = get_l_diversity_config()
assert config['use_dask'] is True  # Converted to Python bool
```

## Best Practices

### 1. **Load Configuration Once**

```python
# Good: Load once at startup
config = get_config()

# Pass config to functions/classes
def process_data(config):
    repo = config['data_repository']
    chunk_size = config['performance']['chunk_size']
```

### 2. **Validate Configuration Early**

```python
from pamola_core.configs.config_variables import validate_l_diversity_config

config = get_l_diversity_config()

if not validate_l_diversity_config(config):
    raise ValueError("Invalid L-Diversity configuration")
```

### 3. **Use Environment Variables for Deployment**

```python
# In production, set environment variables instead of config files:
export PAMOLA_DATA_REPOSITORY=/prod/data
export PAMOLA_L_DIVERSITY_L=5
export PAMOLA_LOG_LEVEL=WARNING

# Python code reads from environment automatically
config = get_config()
```

### 4. **Handle Missing Configurations Gracefully**

```python
from pamola_core.configs.settings import get_config

config = get_config()

# Access with defaults
chunk_size = config.get('performance', {}).get('chunk_size', 100000)
```

### 5. **Keep Config Files in Version Control**

```
configs/
├── prj_config.json        # Committed: defaults/dev settings
├── prj_config.prod.json   # Not committed: production values
└── .env                   # Not committed: secrets
```

## Related Components

- **`pamola_core.utils.paths`**: Project root detection
- **`pamola_core.errors`**: Configuration validation errors
- **Anonymization operations**: Consume L-Diversity config
- **Data I/O modules**: Use performance and encoding settings
- **Logging system**: Configured via config module

## Summary Analysis

The configs module provides flexible, multi-source configuration management:

- **Hierarchy Support**: Environment > File > Defaults maintains predictable override behavior
- **Lazy Loading**: No side effects; explicit loading required
- **Validation**: Type and range checking prevents runtime errors
- **Auto-Detection**: Smart discovery of project structure and data locations
- **Extensibility**: Easy to add new configuration domains
- **Environment-Friendly**: Deployment via env variables supported
- **Backward Compatible**: Sensible defaults ensure existing code works unchanged
