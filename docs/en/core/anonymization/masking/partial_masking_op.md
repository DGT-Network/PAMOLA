# Partial Masking Operation

**Module:** `pamola_core.anonymization.masking.partial_masking_op`
**Version:** 2.0.0
**Last Updated:** 2025-03-27
**Status:** Stable

## Overview

The `PartialMaskingOperation` selectively masks portions of data while preserving specified parts for utility. It supports multiple masking strategies (fixed, pattern-based, random, word-based) with position-based and pattern-based control, enabling flexible privacy-utility tradeoffs.

## Constructor Signature

```python
def __init__(
    self,
    field_name: str,
    # ==== Masking Basics ====
    mask_char: str = "*",
    mask_strategy: str = "fixed",  # fixed, pattern, random, words
    mask_percentage: Optional[float] = None,  # Random % to mask
    # ==== Position-based Masking ====
    unmasked_prefix: int = 0,
    unmasked_suffix: int = 0,
    unmasked_positions: Optional[List[int]] = None,
    # ==== Pattern-based Masking ====
    pattern_type: Optional[str] = None,
    mask_pattern: Optional[str] = None,
    preserve_pattern: Optional[str] = None,
    # ==== Format & Word Preservation ====
    preserve_separators: bool = True,
    preserve_word_boundaries: bool = False,
    # ==== Advanced Masking Behavior ====
    case_sensitive: bool = True,
    random_mask: bool = False,
    mask_char_pool: Optional[str] = None,
    # ==== Preset / Templates ====
    preset_type: Optional[str] = None,
    preset_name: Optional[str] = None,
    # ==== Multi-field Consistency ====
    consistency_fields: Optional[List[str]] = None,
    **kwargs,
):
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Name of the field to be masked |
| `mask_char` | str | "*" | Character used to mask sensitive content |
| `mask_strategy` | str | "fixed" | Masking strategy: "fixed", "pattern", "random", or "words" |
| `mask_percentage` | Optional[float] | None | Percentage (0–100) of characters to mask randomly |
| `unmasked_prefix` | int | 0 | Number of characters at start to remain visible |
| `unmasked_suffix` | int | 0 | Number of characters at end to remain visible |
| `unmasked_positions` | Optional[List[int]] | None | Specific index positions to remain unmasked |
| `pattern_type` | Optional[str] | None | Predefined pattern type (e.g., 'email', 'phone', 'ipv4') |
| `mask_pattern` | Optional[str] | None | Custom regex pattern for masking if pattern-based strategy selected |
| `preserve_pattern` | Optional[str] | None | Regex pattern to preserve (everything else masked) |
| `preserve_separators` | bool | True | Whether to keep separators (e.g., '-', '_', '.') unchanged |
| `preserve_word_boundaries` | bool | False | Whether to avoid masking across word boundaries |
| `case_sensitive` | bool | True | Whether matching is case-sensitive |
| `random_mask` | bool | False | Whether to use random characters from pool instead of fixed mask_char |
| `mask_char_pool` | Optional[str] | None | Pool of characters for random masking (e.g., "ABC123") |
| `preset_type` | Optional[str] | None | Preset category for reusable masking templates |
| `preset_name` | Optional[str] | None | Name of specific preset configuration to apply |
| `consistency_fields` | Optional[List[str]] | None | Other fields to mask consistently with main field |
| `**kwargs` | dict | - | Additional keyword arguments passed to `AnonymizationOperation` |

## Key Masking Strategies

### Fixed Strategy
Masks specified positions with a fixed character.
```python
op = PartialMaskingOperation(
    field_name="ssn",
    mask_strategy="fixed",
    mask_char="*",
    unmasked_prefix=3,
    unmasked_suffix=4
)
# Input: "123456789" → Output: "123****789"
```

### Pattern Strategy
Applies predefined or custom regex patterns to mask specific data types.
```python
op = PartialMaskingOperation(
    field_name="email",
    mask_strategy="pattern",
    pattern_type="email"
)
# Masks domain while preserving first character of username
```

### Random Strategy
Masks a random percentage of characters.
```python
op = PartialMaskingOperation(
    field_name="text",
    mask_strategy="random",
    mask_percentage=50,
    random_mask=True,
    mask_char_pool="0123456789"
)
```

### Word Strategy
Masks entire words or tokens.
```python
op = PartialMaskingOperation(
    field_name="address",
    mask_strategy="words",
    mask_char="***",
    preserve_separators=True
)
```

## Key Methods

### execute()
Executes the partial masking operation on the input data source.

```python
def execute(
    self,
    data_source: DataSource,
    task_dir: Path,
    reporter: Any,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> OperationResult:
```

**Features:**
- Full integration with PAMOLA framework
- Support for both pandas and Dask DataFrames
- Conditional and k-anonymity-based processing
- Comprehensive metrics and visualization
- Error handling and progress tracking

## Usage Examples

### Email Partial Masking
```python
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource

op = PartialMaskingOperation(
    field_name="email",
    mask_strategy="pattern",
    pattern_type="email"
)

data_source = DataSource.from_file_path("users.csv", name="main")
result = op.execute(
    data_source=data_source,
    task_dir=Path("output/task_001"),
    reporter=None
)
```

### SSN Prefix/Suffix Masking
```python
op = PartialMaskingOperation(
    field_name="ssn",
    mask_strategy="fixed",
    mask_char="*",
    unmasked_prefix=3,
    unmasked_suffix=4
)
# Input: "123-45-6789" → Output: "123-**-6789"
```

### Credit Card Tokenization
```python
op = PartialMaskingOperation(
    field_name="cc_number",
    mask_strategy="fixed",
    mask_char="X",
    unmasked_suffix=4,
    preserve_separators=True
)
# Input: "4532-1111-5678-9010" → Output: "XXXX-XXXX-XXXX-9010"
```

## Integration with Base Class

**Reference:** This operation inherits from the abstract base class documented in [base_anonymization_op.md](../base_anonymization_op.md). See that file for all shared parameters, methods, conditional processing, and k-anonymity integration details.

## Metrics

The operation collects:
- **Suppression Rate:** Percentage of characters masked
- **Anonymization Effectiveness:** Information loss ratio
- **Disclosure Risk:** Based on unmasked portions
- **Performance Metrics:** Processing time and throughput

## Related Components

- `AnonymizationOperation` (base class)
- `MaskingPatterns` (pattern utilities)
- `PartialMaskingConfig` (configuration schema)

## Changelog

**v2.0.0 (2025-06-15)**
- Added pattern-based masking with predefined types (email, phone, ipv4)
- Added word-level masking strategy
- Added random character pool masking
- Added preset templates for reusable configurations
- Added multi-field consistency support
- Enhanced word and separator boundary preservation

**v1.0.0 (2025-01-20)**
- Initial implementation with position-based masking
