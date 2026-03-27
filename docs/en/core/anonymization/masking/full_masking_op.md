# Full Masking Operation

**Module:** `pamola_core.anonymization.masking.full_masking_op`
**Version:** 4.0.0
**Last Updated:** 2025-03-27
**Status:** Stable

## Overview

The `FullMaskingOperation` replaces all characters in a field value with a configurable mask character, providing complete obfuscation of sensitive data. This operation supports both fixed and random masking strategies, optional format preservation, and comprehensive metrics collection.

## Constructor Signature

```python
def __init__(
    self,
    field_name: str,
    # ==== Masking configuration ====
    mask_char: str = "*",
    preserve_length: bool = True,
    fixed_length: Optional[int] = None,
    random_mask: bool = False,
    mask_char_pool: Optional[str] = None,
    # Format handling
    preserve_format: bool = False,
    format_patterns: Optional[Dict[str, str]] = None,
    # Type-specific handling
    numeric_output: str = "string",  # string, numeric, preserve
    date_format: Optional[str] = None,
    **kwargs,
):
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `field_name` | str | Required | Name of the field to apply masking |
| `mask_char` | str | "*" | Character used for masking field values |
| `preserve_length` | bool | True | Whether to preserve original string length of masked values |
| `fixed_length` | Optional[int] | None | Fixed output length for all masked values; if None, uses input length |
| `random_mask` | bool | False | Whether to use random characters from a pool instead of fixed mask_char |
| `mask_char_pool` | Optional[str] | None | Pool of characters to randomly sample from if `random_mask=True` |
| `preserve_format` | bool | False | Whether to preserve data format or structure (e.g., keep dashes or parentheses) |
| `format_patterns` | Optional[Dict[str, str]] | None | Custom regex patterns for identifying and preserving data formats |
| `numeric_output` | str | "string" | Defines output type for numeric fields: "string", "numeric", or "preserve" |
| `date_format` | Optional[str] | None | Date format string to use when masking datetime fields |
| `**kwargs` | dict | - | Additional keyword arguments passed to `AnonymizationOperation` |

## Key Methods

### execute()
Executes the full masking operation on the input data source.

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
- Full integration with PAMOLA framework (caching, progress tracking, metrics collection)
- Support for both pandas and Dask DataFrames with automatic engine switching
- Conditional processing based on field values
- Comprehensive metrics collection (suppression rate, effectiveness, disclosure risk)
- Visualization generation for before/after comparisons
- Error handling with detailed logging

### process_batch()
Processes a batch of data rows.

**Parameters:**
- `batch : pd.DataFrame` - DataFrame batch to process

**Returns:**
- `pd.DataFrame` - Processed batch with masking applied

### process_value()
Processes a single value (abstract method - implemented by subclass).

## Usage Examples

### Basic Full Masking
```python
from pamola_core.anonymization.masking.full_masking_op import FullMaskingOperation
from pamola_core.utils.ops.op_data_source import DataSource

# Create operation
op = FullMaskingOperation(
    field_name="ssn",
    mask_char="*",
    preserve_length=True
)

# Execute
data_source = DataSource.from_file_path("data.csv", name="main")
result = op.execute(
    data_source=data_source,
    task_dir=Path("output/task_001"),
    reporter=None
)
```

### Random Masking
```python
op = FullMaskingOperation(
    field_name="credit_card",
    random_mask=True,
    mask_char_pool="0123456789",
    fixed_length=16
)
```

### Format-Preserving Masking
```python
op = FullMaskingOperation(
    field_name="phone",
    mask_char="X",
    preserve_format=True,  # Preserves dashes and parentheses
    format_patterns={
        "phone": r"(\d{3})-(\d{3})-(\d{4})"
    }
)
```

## Integration with Base Class

**Reference:** This operation inherits from the abstract base class documented in [base_anonymization_op.md](../base_anonymization_op.md). See that file for all shared parameters, methods, conditional processing, and k-anonymity integration details.

## Metrics

The operation collects:
- **Suppression Rate:** Percentage of values that were masked
- **Anonymization Effectiveness:** Information loss ratio
- **Disclosure Risk:** Simple disclosure risk assessment
- **Performance Metrics:** Processing time and throughput

## Related Components

- `AnonymizationOperation` (base class)
- `MaskingPatterns` (pattern utilities)
- `FullMaskingConfig` (configuration schema)

## Changelog

**v4.0.0 (2025-06-15)**
- Full support for random masking with character pools
- Format preservation using custom regex patterns
- Type-specific handling for numeric and datetime fields
- Enhanced metrics collection
- Full Dask support for large datasets

**v3.0.0 (2025-03-15)**
- Initial stable release with core masking functionality
