# L-Diversity Configuration Variables

**Module:** `pamola_core.configs.config_variables`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Configuration Parameters](#configuration-parameters)
3. [Validation Rules](#validation-rules)
4. [Usage Examples](#usage-examples)
5. [Environment Variables](#environment-variables)
6. [Configuration Overrides](#configuration-overrides)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Technical Summary](#technical-summary)

## Overview

The `config_variables.py` module manages L-Diversity anonymization-specific configuration. It provides centralized parameter storage, environment variable binding, and validation for L-Diversity operations used in privacy-preserving data transformations.

### Purpose

This module:
- Defines L-Diversity parameters with sensible defaults
- Supports environment variable overrides for deployment flexibility
- Validates configuration before use to prevent runtime errors
- Manages privacy model parameters (l, diversity_type, c_value)
- Configures processing strategies (Dask, memory optimization)
- Handles visualization and compliance settings

### Module Structure

```python
pamola_core/configs/config_variables.py
├── L_DIVERSITY_DEFAULTS     # Dict of default values
├── validate_l_diversity_config()    # Validation function
└── get_l_diversity_config()         # Configuration getter with overrides
```

## Configuration Parameters

### Default Configuration Dictionary

The `L_DIVERSITY_DEFAULTS` contains all L-Diversity settings:

```python
L_DIVERSITY_DEFAULTS = {
    # Core privacy parameters
    "l": 3,                           # L-diversity level
    "diversity_type": "distinct",     # Type of diversity
    "c_value": 1.0,                   # Recursive diversity constant
    "k": 2,                           # K-anonymity baseline

    # Processing strategies
    "use_dask": False,                # Enable Dask processing
    "mask_value": "MASKED",           # Mask string for masking operations
    "suppression": True,              # Enable suppression strategy

    # Performance settings
    "npartitions": 4,                 # Dask partition count
    "optimize_memory": True,          # Memory optimization flag

    # Logging
    "log_level": "INFO",              # Logging level

    # Visualization
    "visualization": {
        "hist_bins": 20,              # Histogram bins
        "save_format": "png",         # Output format
    },

    # Compliance
    "compliance": {
        "risk_threshold": 0.5,        # Privacy risk threshold
        "supported_regulations": ["GDPR", "HIPAA", "CCPA"],
    }
}
```

### Core Privacy Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `l` | int | 3 | `PAMOLA_L_DIVERSITY_L` | Minimum diversity level (≥1) |
| `diversity_type` | str | "distinct" | `PAMOLA_L_DIVERSITY_TYPE` | Type: "distinct", "entropy", "recursive" |
| `c_value` | float | 1.0 | `PAMOLA_L_DIVERSITY_C_VALUE` | Recursive diversity constant (>0) |
| `k` | int | 2 | `PAMOLA_L_DIVERSITY_K` | K-anonymity baseline (≥1) |

### Processing Strategy Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `use_dask` | bool | False | `PAMOLA_L_DIVERSITY_USE_DASK` | Enable Dask for distributed processing |
| `mask_value` | str | "MASKED" | `PAMOLA_L_DIVERSITY_MASK_VALUE` | Mask string for full masking |
| `suppression` | bool | True | `PAMOLA_L_DIVERSITY_SUPPRESSION` | Enable suppression of sensitive fields |

### Performance Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `npartitions` | int | 4 | `PAMOLA_L_DIVERSITY_NPARTITIONS` | Number of Dask partitions (if use_dask=True) |
| `optimize_memory` | bool | True | `PAMOLA_L_DIVERSITY_OPTIMIZE_MEMORY` | Enable memory optimization techniques |

### Logging Parameter

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `log_level` | str | "INFO" | `PAMOLA_L_DIVERSITY_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Visualization Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `visualization.hist_bins` | int | 20 | `PAMOLA_L_DIVERSITY_HIST_BINS` | Number of bins for histogram visualizations |
| `visualization.save_format` | str | "png" | `PAMOLA_L_DIVERSITY_SAVE_FORMAT` | Image format (png, jpg, pdf, svg) |

### Compliance Parameters

| Parameter | Type | Default | Environment Variable | Description |
|-----------|------|---------|----------------------|-------------|
| `compliance.risk_threshold` | float | 0.5 | `PAMOLA_L_DIVERSITY_RISK_THRESHOLD` | Privacy risk threshold (0.0-1.0) |
| `compliance.supported_regulations` | list | ["GDPR", "HIPAA", "CCPA"] | (not environment-configurable) | Applicable regulations |

## Validation Rules

The `validate_l_diversity_config()` function enforces these rules:

### L-Diversity Level Validation

```python
# Rule: l must be a positive integer
l_value = config.get("l", 3)
if not isinstance(l_value, int) or l_value < 1:
    raise InvalidParameterError(
        param_name="l",
        param_value=l_value,
        reason=f"l must be a positive integer, got {l_value}"
    )
```

**Valid:** `1, 2, 3, 5, 10` (any positive integer)
**Invalid:** `0, -1, 3.5, "3"` (non-integer or ≤0)

### Diversity Type Validation

```python
# Rule: diversity_type in ["distinct", "entropy", "recursive"]
diversity_type = config.get("diversity_type", "distinct")
valid_types = ["distinct", "entropy", "recursive"]
if diversity_type not in valid_types:
    raise InvalidParameterError(
        param_name="diversity_type",
        param_value=diversity_type,
        reason=f"diversity_type must be one of {valid_types}"
    )
```

**Valid:** `"distinct"`, `"entropy"`, `"recursive"`
**Invalid:** `"combined"`, `"partial"`, `"div_type"`

### C-Value Validation (Recursive Diversity Only)

```python
# Rule: If diversity_type == "recursive", c_value must be > 0
if diversity_type == "recursive":
    c_value = config.get("c_value", 1.0)
    if not isinstance(c_value, (int, float)) or c_value <= 0:
        raise InvalidParameterError(
            param_name="c_value",
            param_value=c_value,
            reason=f"c_value must be a positive number, got {c_value}"
        )
```

**Valid (when diversity_type='recursive'):** `0.5, 1.0, 2.5, 10`
**Invalid (when diversity_type='recursive'):** `0, -1.0, "1.0"`
**Ignored (when diversity_type!='recursive'):** c_value not validated

### K-Value Validation

```python
# Rule: k must be a positive integer
k_value = config.get("k", 2)
if not isinstance(k_value, int) or k_value < 1:
    raise InvalidParameterError(
        param_name="k_value",
        param_value=k_value,
        reason=f"k must be a positive integer, got {k_value}"
    )
```

**Valid:** `1, 2, 5, 10`
**Invalid:** `0, -1, 2.5, "5"`

### Validation Return Value

```python
def validate_l_diversity_config(config: Dict[str, Any]) -> bool:
    """
    Returns True if all validations pass.
    Returns False on validation error (does not raise).
    """
```

**Behavior:**
- Returns `True`: All parameters valid
- Returns `False`: One or more validation errors (error printed to stdout)
- Does not raise exceptions (uses print for error reporting)

## Usage Examples

### Example 1: Load Default Configuration

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Get defaults
config = get_l_diversity_config()

print(f"L-Diversity level: {config['l']}")
print(f"Diversity type: {config['diversity_type']}")
print(f"Mask value: {config['mask_value']}")
```

**Output:**
```
L-Diversity level: 3
Diversity type: distinct
Mask value: MASKED
```

### Example 2: Override Specific Parameters

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Override defaults with custom values
custom_config = get_l_diversity_config({
    "l": 5,                      # Change l-diversity level
    "diversity_type": "entropy",  # Change diversity model
    "npartitions": 8              # Increase Dask partitions
})

print(f"L: {custom_config['l']}")
print(f"Type: {custom_config['diversity_type']}")
print(f"Partitions: {custom_config['npartitions']}")
```

**Output:**
```
L: 5
Type: entropy
Partitions: 8
```

### Example 3: Validate Configuration

```python
from pamola_core.configs.config_variables import (
    validate_l_diversity_config,
    get_l_diversity_config
)

config = get_l_diversity_config()

if validate_l_diversity_config(config):
    print("Configuration is valid")
else:
    print("Configuration validation failed")
```

### Example 4: Recursive Diversity with c_value

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Configure recursive diversity
recursive_config = get_l_diversity_config({
    "diversity_type": "recursive",
    "l": 2,
    "c_value": 2.5  # Required for recursive
})

print(f"Diversity model: {recursive_config['diversity_type']}")
print(f"C value: {recursive_config['c_value']}")
```

### Example 5: Performance Tuning

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Configure for large-scale processing
perf_config = get_l_diversity_config({
    "use_dask": True,           # Enable distributed processing
    "npartitions": 16,          # Increase parallelism
    "optimize_memory": True,    # Enable memory optimization
    "log_level": "DEBUG"        # Detailed logging for troubleshooting
})

print(f"Dask enabled: {perf_config['use_dask']}")
print(f"Partitions: {perf_config['npartitions']}")
```

## Environment Variables

All L-Diversity parameters can be set via environment variables with `PAMOLA_` prefix.

### Reading from Environment

```python
import os

# Set environment variables
os.environ['PAMOLA_L_DIVERSITY_L'] = '5'
os.environ['PAMOLA_L_DIVERSITY_TYPE'] = 'entropy'
os.environ['PAMOLA_L_DIVERSITY_USE_DASK'] = 'True'

# Load configuration (reads from environment)
from pamola_core.configs.config_variables import get_l_diversity_config
config = get_l_diversity_config()

print(f"L: {config['l']}")              # 5
print(f"Type: {config['diversity_type']}")  # 'entropy'
print(f"Use Dask: {config['use_dask']}")    # True
```

### Environment Variable Format

```bash
# Integer values: numeric strings
export PAMOLA_L_DIVERSITY_L=5
export PAMOLA_L_DIVERSITY_K=3
export PAMOLA_L_DIVERSITY_NPARTITIONS=8

# Float values: numeric strings
export PAMOLA_L_DIVERSITY_C_VALUE=2.5
export PAMOLA_L_DIVERSITY_RISK_THRESHOLD=0.7

# String values: quoted strings
export PAMOLA_L_DIVERSITY_TYPE=entropy
export PAMOLA_L_DIVERSITY_MASK_VALUE="***"

# Boolean values: "true"/"false" (case-insensitive)
export PAMOLA_L_DIVERSITY_USE_DASK=true
export PAMOLA_L_DIVERSITY_OPTIMIZE_MEMORY=false
```

### Boolean Conversion Logic

```python
# From settings.py
os.getenv("PAMOLA_L_DIVERSITY_USE_DASK", "False").lower() == "true"

# Examples:
# "true"    -> True
# "True"    -> True
# "TRUE"    -> True
# "false"   -> False
# "False"   -> False
# "1"       -> False (not "true")
# "yes"     -> False (not "true")
```

## Configuration Overrides

### Override Precedence

```
1. Explicit function parameter (overrides parameter)
2. Environment variable (PAMOLA_* prefixed)
3. Default value (L_DIVERSITY_DEFAULTS)
```

### Deep Update Behavior

The `get_l_diversity_config()` function performs deep dictionary merge:

```python
# Defaults (complete)
defaults = {
    "l": 3,
    "visualization": {
        "hist_bins": 20,
        "save_format": "png"
    }
}

# Override (partial)
overrides = {
    "l": 5,
    "visualization": {
        "hist_bins": 50
    }
}

# Result (recursive merge)
result = get_l_diversity_config(overrides)
# {
#     "l": 5,                     # Updated from override
#     "visualization": {
#         "hist_bins": 50,        # Updated from override
#         "save_format": "png"    # Preserved from default
#     }
# }
```

### Validation After Override

Configuration is validated after overrides applied:

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Invalid override
config = get_l_diversity_config({
    "l": -1  # Invalid: must be >= 1
})

# Result: Returns defaults with warning message
# "Warning: L-Diversity configuration validation failed. Using default settings."
```

## Error Handling

### Validation Errors

Validation errors are reported via print statements (not exceptions):

```python
from pamola_core.configs.config_variables import validate_l_diversity_config

config = {
    "l": "not_an_integer",
    "diversity_type": "invalid_type"
}

result = validate_l_diversity_config(config)
# Prints: "L-Diversity configuration validation error: ..."
# Returns: False
```

### InvalidParameterError

The validation function raises `InvalidParameterError` internally:

```python
from pamola_core.errors.exceptions import InvalidParameterError

# Raised during validation for invalid parameters
raise InvalidParameterError(
    param_name="l",
    param_value=-1,
    reason="l must be a positive integer, got -1"
)
```

### Fallback to Defaults

If validation fails, defaults are returned:

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# Invalid overrides
bad_config = get_l_diversity_config({
    "l": -5,                          # Invalid
    "diversity_type": "unknown_type"  # Invalid
})

# get_l_diversity_config() returns L_DIVERSITY_DEFAULTS
# (invalid parameters rejected, defaults used)
```

## Best Practices

### 1. **Validate After Load**

```python
from pamola_core.configs.config_variables import (
    get_l_diversity_config,
    validate_l_diversity_config
)

config = get_l_diversity_config()

if not validate_l_diversity_config(config):
    raise ValueError("Configuration validation failed")

# Safe to use config
use_dask = config['use_dask']
```

### 2. **Use Sensible Defaults**

```python
# Good: Accept defaults unless user overrides
user_overrides = {}  # Empty if no user config
config = get_l_diversity_config(user_overrides)

# Not recommended: Rebuilding from scratch
config = {
    "l": 3,
    "diversity_type": "distinct",
    # ... rebuilding all parameters
}
```

### 3. **Environment for Deployment**

```bash
# In Docker/Kubernetes, use env vars instead of config files
export PAMOLA_L_DIVERSITY_L=5
export PAMOLA_L_DIVERSITY_USE_DASK=true
export PAMOLA_L_DIVERSITY_NPARTITIONS=16

# Python code reads from environment automatically
config = get_l_diversity_config()
```

### 4. **Performance Tuning**

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# For large datasets
large_dataset_config = get_l_diversity_config({
    "use_dask": True,           # Distributed processing
    "npartitions": 32,          # More parallelism
    "optimize_memory": True,    # Memory constraints
})
```

### 5. **Compliance Configuration**

```python
from pamola_core.configs.config_variables import get_l_diversity_config

# For strict privacy requirements
privacy_config = get_l_diversity_config({
    "l": 5,                                    # Higher diversity level
    "k": 5,                                    # Higher k-anonymity
    "compliance": {
        "risk_threshold": 0.1,                 # Stricter threshold
        "supported_regulations": ["GDPR", "CCPA"]
    }
})
```

## Technical Summary

The `config_variables.py` module provides:

- **Centralized L-Diversity Configuration**: All parameters in one place
- **Environment Integration**: Direct environment variable binding
- **Validation Framework**: Type and range checking before use
- **Nested Structures**: Support for hierarchical parameters (visualization, compliance)
- **Deep Merge**: Proper handling of partial overrides
- **Error Resilience**: Graceful fallback to defaults on validation failure
- **Flexible Override**: Environment > Parameter > Default precedence

The module enables deployers to configure L-Diversity anonymization without code changes, supporting both traditional config file deployments and modern containerized environments.
