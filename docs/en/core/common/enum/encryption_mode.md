# EncryptionMode Enumeration

**Module:** `pamola_core.common.enum.encryption_mode`
**Class:** `EncryptionMode`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Enum Members](#enum-members)
3. [Class Methods](#class-methods)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Related Components](#related-components)

## Overview

`EncryptionMode` is a string-based enumeration that defines the encryption modes supported by the PAMOLA.CORE task framework. It specifies how data or files should be encrypted during task execution and in temporary storage. This enum is used throughout the framework for secure data handling configuration.

**Parent Class:** `Enum`
**Type:** String Enum
**Scope:** Task framework configuration

## Enum Members

| Member | Value | Description |
|--------|-------|-------------|
| `NONE` | `"none"` | No encryption applied. Data is stored as-is without encryption. Fastest option but least secure. |
| `SIMPLE` | `"simple"` | Simple symmetric encryption. Uses basic symmetric algorithms for data protection. Balanced security and performance. |
| `AGE` | `"age"` | Age encryption (more secure, supports key rotation). More robust encryption with support for key rotation and modern security practices. |

## Class Methods

### `from_string(value: str) -> EncryptionMode`

Converts a string value to its corresponding `EncryptionMode` enum member.

**Parameters:**
- `value` (str): The string representation of the encryption mode. Case-insensitive.

**Returns:**
- `EncryptionMode`: The corresponding enum member

**Behavior:**
- Converts input to lowercase for case-insensitive matching
- Returns `SIMPLE` as default fallback for invalid inputs
- Catches `ValidationError`, `ValueError`, and `AttributeError`

**Raises:**
- Does not raise exceptions; defaults to `SIMPLE` on error

**Example:**
```python
from pamola_core.common.enum.encryption_mode import EncryptionMode

mode = EncryptionMode.from_string("AGE")
# Returns: EncryptionMode.AGE

mode = EncryptionMode.from_string("simple")
# Returns: EncryptionMode.SIMPLE

mode = EncryptionMode.from_string("invalid")
# Returns: EncryptionMode.SIMPLE (default)
```

## Usage Examples

### Basic Enum Usage

```python
from pamola_core.common.enum.encryption_mode import EncryptionMode

# Access enum members
mode1 = EncryptionMode.NONE
mode2 = EncryptionMode.SIMPLE
mode3 = EncryptionMode.AGE

# Get string value
print(mode1.value)  # "none"
print(mode2.value)  # "simple"
print(mode3.value)  # "age"

# Compare enum members
if mode1 == EncryptionMode.NONE:
    print("No encryption")
```

### Convert from String

```python
from pamola_core.common.enum.encryption_mode import EncryptionMode

# Convert string to enum
mode = EncryptionMode.from_string("age")
print(mode)  # EncryptionMode.AGE

# Case insensitive
mode = EncryptionMode.from_string("AGE")
print(mode)  # EncryptionMode.AGE

# Invalid input defaults to SIMPLE
mode = EncryptionMode.from_string("invalid_mode")
print(mode)  # EncryptionMode.SIMPLE
```

### Use in Task Configuration

```python
from pamola_core.common.enum.encryption_mode import EncryptionMode
from pamola_core.utils.tasks.base_task import TaskConfig

# Create task configuration with encryption
config = TaskConfig(
    encryption_mode=EncryptionMode.AGE,
    # ... other config
)

# Or use string and convert
mode_str = "age"
config = TaskConfig(
    encryption_mode=EncryptionMode.from_string(mode_str),
    # ... other config
)
```

### Conditional Logic Based on Encryption

```python
from pamola_core.common.enum.encryption_mode import EncryptionMode

def process_data(encryption_mode: EncryptionMode):
    if encryption_mode == EncryptionMode.NONE:
        print("Processing without encryption")
        # Fast path for non-encrypted data
    elif encryption_mode == EncryptionMode.SIMPLE:
        print("Applying simple encryption")
        # Standard encryption
    elif encryption_mode == EncryptionMode.AGE:
        print("Applying AGE encryption with key rotation")
        # Advanced encryption with rotation support
    else:
        print("Unknown encryption mode")

process_data(EncryptionMode.AGE)
```

## Best Practices

1. **Always Use Enum Members, Never Hardcoded Strings**
   ```python
   # Good
   mode = EncryptionMode.AGE

   # Avoid
   mode = "age"  # String is error-prone
   ```

2. **Use `from_string()` for User Input Conversion**
   ```python
   # Good - safely converts user input
   user_input = "AGE"
   mode = EncryptionMode.from_string(user_input)

   # Less safe - direct string comparison
   mode = user_input.lower()
   ```

3. **Check Enum Value Before Using**
   ```python
   from pamola_core.common.enum.encryption_mode import EncryptionMode

   mode = EncryptionMode.from_string(user_input)
   if mode == EncryptionMode.NONE:
       # Handle no encryption case
       pass
   ```

4. **Type Hints for Function Parameters**
   ```python
   from pamola_core.common.enum.encryption_mode import EncryptionMode

   def setup_encryption(mode: EncryptionMode) -> None:
       """Setup encryption based on mode."""
       if mode == EncryptionMode.SIMPLE:
           # Setup simple encryption
           pass
   ```

5. **Use in Enumerations When Listing Supported Modes**
   ```python
   from pamola_core.common.enum.encryption_mode import EncryptionMode

   supported_modes = [EncryptionMode.NONE, EncryptionMode.SIMPLE, EncryptionMode.AGE]
   for mode in supported_modes:
       print(f"Supported: {mode.value}")
   ```

## Related Components

- **TaskConfig** (`pamola_core.utils.tasks.base_task`) - Uses `EncryptionMode` for task configuration
- **FileCryptoConfig** (`pamola_core.common.type_aliases`) - Associates files with encryption settings
- **CryptoConfig** (`pamola_core.common.type_aliases`) - Stores encryption algorithm and key with mode
- **Task Framework** (`pamola_core.utils.tasks`) - Implements encryption based on mode

## Security Considerations

1. **NONE Mode**: Use only for non-sensitive data or development environments
2. **SIMPLE Mode**: Suitable for standard privacy-preserving operations
3. **AGE Mode**: Recommended for sensitive data requiring key rotation and modern security practices
4. **Default Fallback**: The `from_string()` method defaults to `SIMPLE` for invalid inputs to ensure some protection

## Implementation Notes

- The enum uses Python's standard `Enum` class with string values
- The `from_string()` method provides lenient conversion with a safe default
- All enum values are lowercase for consistency in configuration files and APIs
