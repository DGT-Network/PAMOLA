# Common Module Documentation

**Module:** `pamola_core.common`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Module Structure](#module-structure)
5. [Dependencies](#dependencies)
6. [Core Components](#core-components)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Related Components](#related-components)

## Overview

The `pamola_core.common` module serves as the foundation layer for PAMOLA.CORE, providing shared utilities, constants, type definitions, enumerations, and helper functions used across all operations. This module is classified as **Internal (Non-Public API)** and is designed to support the framework's operation-first architecture.

The common module centralizes:
- Enumeration types for operations and configurations
- Global constants and standard values
- Type aliases and crypto configurations
- Helper functions for data manipulation and validation
- Logging and regex pattern utilities

## Key Features

- **Enumerations**: Comprehensive enum types for privacy metrics, anonymization strategies, distance metrics, and more
- **Constants Management**: Centralized storage of operation names, date formats, pandas dtype mappings, and artifact categories
- **Type Aliases**: Custom type definitions for file paths, dataframes, and crypto configurations
- **Logging Support**: Privacy-aware logging with JSON serialization and change tracking
- **Helper Functions**: Data validation, transformation, and analysis utilities
- **Pattern Matching**: Regex patterns for email, phone, dates, and financial data
- **Configuration Classes**: CryptoConfig and FileCryptoConfig for encryption management

## Architecture

```
pamola_core/common/
├── __init__.py                 # Package marker (non-public)
├── constants.py                # Global constants and standards
├── type_aliases.py             # Type definitions and crypto configs
├── enum/                       # Enumeration types
│   ├── __init__.py
│   ├── encryption_mode.py      # Encryption modes
│   ├── mask_strategy_enum.py   # Masking strategies
│   ├── distance_metric_type.py # Distance metrics
│   ├── fidelity_metrics.py     # Fidelity metrics
│   ├── privacy_metrics_type.py # Privacy metrics
│   ├── utility_metrics_type.py # Utility metrics
│   ├── datetime_generalization.py # DateTime methods
│   ├── numeric_generalization.py  # Numeric methods
│   ├── analysis_mode_enum.py   # Analysis modes
│   ├── language_enum.py        # Language support
│   ├── model_type.py           # Model types
│   ├── statistical_method.py   # Statistical methods
│   ├── relationship_type.py    # Relationship types
│   ├── custom_components.py    # UI components
│   ├── custom_functions.py     # Custom functions
│   ├── form_groups.py          # Form field grouping
│   └── fidelity_metrics_type.py # Fidelity metrics type
├── helpers/                    # Helper functions
│   ├── __init__.py
│   ├── data_helper.py          # Data processing utilities
│   ├── dataset_helper.py       # Dataset splitting
│   ├── data_profiler.py        # Data profiling
│   └── math_helper.py          # Math utilities
├── logging/                    # Logging functionality
│   ├── __init__.py
│   └── privacy_logging.py      # Privacy-aware logging
├── validation/                 # Validation utilities
│   ├── __init__.py
│   └── check_column.py         # Column validation
└── regex/                      # Regex patterns
    ├── __init__.py
    └── patterns.py             # Common patterns
```

## Module Structure

### Core Modules

**constants.py** - Global constants including operation names, date formats, pandas dtype maps, and artifact categories

**type_aliases.py** - Type definitions and configuration classes for crypto operations

**Enumerations** - 17+ enum types covering operations, metrics, methods, and UI components

**Helpers** - Utilities for data validation, transformation, profiling, and math operations

**Logging** - Privacy-aware logging with JSON serialization

**Validation** - Column and field validation utilities

**Regex** - Common patterns for emails, phones, dates, and financial data

## Dependencies

Internal:
- `pamola_core.errors.exceptions` - Custom exception classes

External:
- pandas (data manipulation)
- numpy (numerical operations)
- dask (distributed dataframes)
- scikit-learn (ML utilities)

## Core Components

### 1. Enumerations

**EncryptionMode**
- `NONE`, `SIMPLE`, `AGE`
- Used for task encryption configuration

**MaskStrategyEnum**
- `FIXED`, `PATTERN`, `RANDOM`, `WORDS`
- Defines masking strategies for anonymization

**Distance Metrics**
- `EUCLIDEAN`, `MANHATTAN`, `COSINE`, `MAHALANOBIS`
- Used in privacy metrics calculations

**Fidelity Metrics**
- `KS`, `KL`, `JS`, `WASSERSTEIN`
- Measure statistical similarity between datasets

**Privacy Metrics**
- `DCR`, `NNDR`, `UNIQUENESS`, `K_ANONYMITY`, `L_DIVERSITY`, `T_CLOSENESS`
- Evaluate privacy preservation

**Utility Metrics**
- `R2`, `MSE`, `MAE`, `AUROC`, `ACCURACY`, `F1`, `PRECISION`, `RECALL`
- Measure data utility and model performance

### 2. Helper Classes

**DataHelper** - Comprehensive utilities for data type detection, range processing, and privacy transformations

**dataset_helper** - Train-test splitting functionality

**math_helper** - Float normalization and statistical operations

### 3. Configuration Classes

**CryptoConfig** - Manages encryption mode, algorithm, and key configuration

**FileCryptoConfig** - Associates file paths with crypto settings

### 4. Constants

- `OPERATION_NAMES` - Standard operation identifiers
- `COMMON_DATE_FORMATS` - 40+ date format patterns
- `FREQ_MAP` - Frequency mapping for time periods
- `PANDAS_DTYPE_MAP` - Type conversions for pandas
- `DISTRIBUTION_LABELS` - Metric display labels
- `Artifact_Category_*` - Artifact classification constants

## Usage Examples

### Import Enumerations

```python
from pamola_core.common.enum import (
    EncryptionMode,
    MaskStrategyEnum,
    PrivacyMetricsType,
    UtilityMetricsType
)

# Use encryption mode
mode = EncryptionMode.SIMPLE
print(mode.value)  # "simple"

# Use metric types
privacy_metric = PrivacyMetricsType.K_ANONYMITY
utility_metric = UtilityMetricsType.AUROC
```

### Use DataHelper

```python
import pandas as pd
from pamola_core.common.helpers.data_helper import DataHelper

df = pd.DataFrame({"age": [25.0, 30.0, 35.5]})

# Check if numeric
is_numeric = DataHelper.is_non_numeric(df["age"])

# Bin numeric values
binned = DataHelper.bin_numeric(df["age"], num_bins=3)

# Convert to datetime
dates = DataHelper.convert_to_datetime(df["date_col"])
```

### Use CryptoConfig

```python
from pamola_core.common.type_aliases import CryptoConfig

config = CryptoConfig(
    mode="simple",
    algorithm="AES",
    key="my-secret-key"
)

# Convert to dict
config_dict = config.to_dict()

# Recreate from dict
config2 = CryptoConfig.from_dict(config_dict)
```

### Use Logging

```python
from pamola_core.common.logging.privacy_logging import save_privacy_logging

change_log = {
    "anonymization": {
        "fields": ["name", "email"],
        "rows_affected": 1000
    }
}

save_privacy_logging(
    change_log=change_log,
    log_str="anonymization_log.json",
    track_changes=True
)
```

## Best Practices

1. **Use Enums Instead of Strings**: Always use enum types for configuration options to ensure type safety
   ```python
   # Good
   mode = EncryptionMode.SIMPLE

   # Avoid
   mode = "simple"  # String is error-prone
   ```

2. **Leverage DataHelper Methods**: Use provided utilities instead of custom implementations
   ```python
   # Good
   result = DataHelper.convert_to_datetime(series)

   # Avoid
   result = pd.to_datetime(series, errors="coerce")  # Less robust
   ```

3. **Validate Data Before Processing**: Use validation helpers
   ```python
   from pamola_core.common.validation.check_column import check_columns_exist

   check_columns_exist(df, ["required_col"])
   ```

4. **Use Constants for Standard Values**: Reference Constants class instead of hardcoding
   ```python
   from pamola_core.common.constants import Constants

   date_formats = Constants.COMMON_DATE_FORMATS
   safe_globals = Constants.SAFE_GLOBALS
   ```

5. **Handle Crypto Configuration Properly**: Use CryptoConfig for secure storage
   ```python
   config = CryptoConfig.from_dict(user_input)
   # Config validates mode and algorithm
   ```

## Related Components

- **pamola_core.errors** - Exception handling for common errors
- **pamola_core.anonymization** - Uses enums and helpers for anonymization
- **pamola_core.profiling** - Leverages DataHelper for data analysis
- **pamola_core.metrics** - Uses metric enumerations and constants
- **pamola_core.transformations** - Depends on validation and helper utilities
- **pamola_core.io** - Uses constants for data I/O configuration

## Quick Reference

| Component | Purpose | Key Classes/Functions |
|-----------|---------|----------------------|
| Enumerations | Type-safe configuration options | EncryptionMode, MaskStrategyEnum, *MetricsType |
| Constants | Global standard values | Constants, GROUP_TITLES, OPERATION_CONFIG_GROUPS |
| Type Aliases | Custom type definitions | CryptoConfig, FileCryptoConfig, PathLike, DataFrameType |
| Helpers | Data utilities | DataHelper, split_dataset, replace_special_floats |
| Logging | Change tracking | save_privacy_logging, log_privacy_transformations |
| Validation | Field checks | check_columns_exist |
| Patterns | Regex utilities | CommonPatterns, FinancialPatterns, PhonePatterns |

For detailed documentation on individual components, see the specific module documentation files.
