# MaskStrategyEnum Enumeration

**Module:** `pamola_core.common.enum.mask_strategy_enum`
**Class:** `MaskStrategyEnum`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Enum Members](#enum-members)
3. [Strategy Details](#strategy-details)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Related Components](#related-components)

## Overview

`MaskStrategyEnum` is a string-based enumeration defining the different strategies for applying character-level masking in data anonymization. These strategies determine how specific parts of text values (such as names, email addresses, or identification numbers) are obfuscated while maintaining some data structure or utility.

**Parent Class:** `str, Enum`
**Type:** String Enum
**Scope:** Anonymization operations
**Used By:** Partial masking and full masking operations

## Enum Members

| Member | Value | Description |
|--------|-------|-------------|
| `FIXED` | `"fixed"` | Apply mask using fixed start/end positions. Masks characters at specific indices. |
| `PATTERN` | `"pattern"` | Apply mask using format patterns or regular expressions. Masks based on matched patterns. |
| `RANDOM` | `"random"` | Randomly mask characters in the string. Unpredictable masking positions. |
| `WORDS` | `"words"` | Word-by-word masking strategy. Masks entire words while preserving word boundaries. |

## Strategy Details

### FIXED Strategy

**Use Case:** When you know exact positions to mask (e.g., last 4 digits of SSN)

```
Original: "123456789"
Masked:   "12345****" (positions 5-8 masked)
```

**Characteristics:**
- Requires explicit start and end positions
- Deterministic and reproducible
- Best for structured data with consistent format
- Preserves masked value length awareness

**Example Data:**
- Credit card numbers (mask last 4 digits)
- Social security numbers (mask middle digits)
- Phone numbers (mask area code or exchange)

### PATTERN Strategy

**Use Case:** When masking should follow text patterns (regex/format)

```
Original: "john.smith@example.com"
Masked:   "j***.s****@example.com" (based on email pattern)
```

**Characteristics:**
- Uses regular expressions or format patterns
- Flexible for varying length inputs
- Masks identified pattern components
- Works with multiple patterns

**Example Data:**
- Email addresses (mask local part)
- Phone numbers (mask specific digits)
- URLs (mask domain parts)
- Formatted identifiers

### RANDOM Strategy

**Use Case:** When maximum unpredictability is desired

```
Original: "SensitiveData123"
Masked:   "Se**iti***Da**123" (random characters masked)
```

**Characteristics:**
- Each masking operation produces different results
- Highest anonymization certainty
- Reduces pattern recognition
- Prevents positional re-identification

**Example Data:**
- Generic text fields
- Descriptions and comments
- Free-form text requiring strong anonymization

### WORDS Strategy

**Use Case:** When full words should be masked while preserving word structure

```
Original: "John Michael Smith"
Masked:   "**** ******* *****" (each word as unit)
```

**Characteristics:**
- Masks complete words
- Preserves word boundaries and spacing
- Maintains sentence structure for length analysis
- Useful for multi-word values

**Example Data:**
- Names and full names
- Organization names
- Addresses and place names
- Multi-word titles

## Usage Examples

### Basic Enum Usage

```python
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum

# Access enum members
strategy1 = MaskStrategyEnum.FIXED
strategy2 = MaskStrategyEnum.PATTERN
strategy3 = MaskStrategyEnum.RANDOM
strategy4 = MaskStrategyEnum.WORDS

# Get string value
print(strategy1.value)  # "fixed"
print(strategy2.value)  # "pattern"

# Compare enum members
if strategy1 == MaskStrategyEnum.FIXED:
    print("Using fixed position masking")
```

### Use in Masking Operation Configuration

```python
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.anonymization import PartialMaskingConfig

# Create partial masking with specific strategy
config = PartialMaskingConfig(
    masking_strategy=MaskStrategyEnum.FIXED,
    start_index=0,
    end_index=4,
    mask_character="*"
)

# Mask email using pattern
email_config = PartialMaskingConfig(
    masking_strategy=MaskStrategyEnum.PATTERN,
    pattern=r"(.+)@(.+)",  # Match email structure
    mask_character="*"
)

# Mask names word-by-word
name_config = PartialMaskingConfig(
    masking_strategy=MaskStrategyEnum.WORDS,
    mask_character="*"
)
```

### Select Strategy Based on Data Type

```python
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
import pandas as pd

def select_masking_strategy(column_name: str, dtype) -> MaskStrategyEnum:
    """Select appropriate masking strategy based on column characteristics."""
    if "phone" in column_name.lower():
        return MaskStrategyEnum.FIXED  # Fixed digit positions
    elif "email" in column_name.lower():
        return MaskStrategyEnum.PATTERN  # Email pattern
    elif "name" in column_name.lower():
        return MaskStrategyEnum.WORDS  # Word-level masking
    else:
        return MaskStrategyEnum.RANDOM  # Default to random

# Usage
df = pd.DataFrame({
    "customer_name": ["John Smith"],
    "email": ["john@example.com"],
    "phone": ["555-1234"]
})

for col in df.columns:
    strategy = select_masking_strategy(col, df[col].dtype)
    print(f"{col}: {strategy.value}")
```

### Conditional Masking Based on Strategy

```python
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum

def apply_masking(value: str, strategy: MaskStrategyEnum) -> str:
    """Apply masking based on selected strategy."""
    if strategy == MaskStrategyEnum.FIXED:
        return value[:3] + "*" * (len(value) - 3)
    elif strategy == MaskStrategyEnum.PATTERN:
        import re
        return re.sub(r'\w', '*', value, count=len(value)//2)
    elif strategy == MaskStrategyEnum.RANDOM:
        import random
        chars = list(value)
        positions = random.sample(range(len(chars)), len(chars)//2)
        for pos in positions:
            chars[pos] = '*'
        return ''.join(chars)
    elif strategy == MaskStrategyEnum.WORDS:
        words = value.split()
        return ' '.join(['*' * len(w) for w in words])
    else:
        return value

# Test
print(apply_masking("SensitiveData", MaskStrategyEnum.FIXED))     # "Sen*********"
print(apply_masking("SensitiveData", MaskStrategyEnum.PATTERN))   # Mixed masking
print(apply_masking("John Smith", MaskStrategyEnum.WORDS))        # "**** *****"
```

## Best Practices

1. **Match Strategy to Data Type**
   ```python
   # Good - strategy matches data structure
   ssn_strategy = MaskStrategyEnum.FIXED  # SSN has fixed positions
   name_strategy = MaskStrategyEnum.WORDS  # Names have word structure

   # Avoid - mismatched strategy
   ssn_strategy = MaskStrategyEnum.RANDOM  # Too unpredictable for structured data
   ```

2. **Use Enum Members in Configuration**
   ```python
   # Good - type-safe
   config = {"strategy": MaskStrategyEnum.PATTERN}

   # Avoid - string is error-prone
   config = {"strategy": "pattern"}
   ```

3. **Document Strategy Selection Rationale**
   ```python
   def create_masking_config(field_name: str) -> PartialMaskingConfig:
       """
       Create masking config for field.

       Phone numbers use FIXED strategy because they have
       consistent format (XXX-XXX-XXXX).
       """
       if "phone" in field_name:
           return PartialMaskingConfig(
               masking_strategy=MaskStrategyEnum.FIXED,
               start_index=0,
               end_index=3
           )
   ```

4. **Consider Privacy Impact**
   ```python
   # Highest privacy: RANDOM
   strategy_high_privacy = MaskStrategyEnum.RANDOM

   # Moderate privacy: PATTERN or FIXED (structure preserved)
   strategy_moderate_privacy = MaskStrategyEnum.FIXED

   # Lower privacy: WORDS (word structure preserved)
   strategy_lower_privacy = MaskStrategyEnum.WORDS
   ```

5. **Combine with Other Parameters**
   ```python
   from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum

   # Complete configuration
   masking_config = {
       "strategy": MaskStrategyEnum.FIXED,
       "start_index": 0,
       "end_index": 4,
       "mask_character": "*",
       "preserve_length": True
   }
   ```

## Related Components

- **PartialMaskingConfig** (`pamola_core.anonymization.masking`) - Configures partial masking with strategy
- **FullMaskingConfig** (`pamola_core.anonymization.masking`) - Configures full field masking
- **PartialMaskingOperation** (`pamola_core.anonymization.masking.partial_masking_op`) - Implements masking
- **Masking Presets** (`pamola_core.anonymization.commons.masking_presets`) - Predefined masking patterns
- **Masking Patterns** (`pamola_core.anonymization.commons.masking_patterns`) - Pattern definitions

## Privacy Considerations

1. **RANDOM Strategy**: Provides highest privacy protection, best for sensitive identifiers
2. **PATTERN Strategy**: Moderate protection, suitable for semi-structured data
3. **FIXED Strategy**: Depends on what is masked; good for known sensitive positions
4. **WORDS Strategy**: Lower privacy impact, useful for maintaining some utility

## Performance Notes

- **FIXED**: Very fast (simple indexing)
- **PATTERN**: Medium speed (regex matching required)
- **RANDOM**: Medium speed (random number generation)
- **WORDS**: Fast (simple string operations)

## Implementation Notes

- The enum inherits from both `str` and `Enum`, making members directly comparable to strings
- All values are lowercase for consistency with configuration conventions
- Strategy selection should be documented in operation configurations
