# AnalysisMode Enumeration

**Module:** `pamola_core.common.enum.analysis_mode_enum`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

AnalysisMode defines the operational modes for k-anonymity analysis. It controls whether the operation generates metrics and reports, enriches the DataFrame with k-values, or performs both operations simultaneously.

## Members

| Member | Value | Description |
|--------|-------|-------------|
| `ANALYZE` | `"ANALYZE"` | Generate privacy metrics and reports from the analyzed dataset |
| `ENRICH` | `"ENRICH"` | Add computed k-values as new columns to the DataFrame |
| `BOTH` | `"BOTH"` | Perform both analysis (generate metrics) and enrichment (add columns) |

## Usage

### Basic Enumeration Access

```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

# Access members
mode = AnalysisMode.ANALYZE
print(mode.value)  # Output: "ANALYZE"
print(mode.name)   # Output: "ANALYZE"
```

### Conditional Logic

```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

def configure_analysis(mode: AnalysisMode) -> dict:
    if mode == AnalysisMode.ANALYZE:
        return {"compute_metrics": True, "enrich_df": False}
    elif mode == AnalysisMode.ENRICH:
        return {"compute_metrics": False, "enrich_df": True}
    else:  # BOTH
        return {"compute_metrics": True, "enrich_df": True}

config = configure_analysis(AnalysisMode.BOTH)
```

### Iterating Over Members

```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

for mode in AnalysisMode:
    print(f"Mode: {mode.name} ({mode.value})")
```

## Member Descriptions

### ANALYZE
**Value:** `"ANALYZE"`

Computes k-anonymity metrics and generates privacy reports without modifying the input DataFrame structure. Results are returned as separate metric dictionaries or report objects.

**Use case:** When you need privacy evaluation metrics without altering the original data structure.

### ENRICH
**Value:** `"ENRICH"`

Calculates k-anonymity values and adds them as new columns to the DataFrame. The original data columns are preserved with k-values appended.

**Use case:** When you want to annotate each row with its computed k-anonymity value for downstream analysis.

### BOTH
**Value:** `"BOTH"`

Performs both operations: generates privacy metrics AND enriches the DataFrame with k-values. Provides complete analysis output.

**Use case:** When comprehensive analysis is needed—both quantified metrics and annotated data.

## Related Components

- **K-Anonymity Analysis:** Used in k-anonymity operations to control output format
- **Privacy Metrics:** Connected to `PrivacyMetricsType` for metric specifications
- **DataFrame Operations:** Works with data enrichment and transformation workflows

## Common Patterns

### Select Mode by Use Case

```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

# For reporting only
reporting_mode = AnalysisMode.ANALYZE

# For data enhancement
enrichment_mode = AnalysisMode.ENRICH

# For comprehensive analysis
comprehensive_mode = AnalysisMode.BOTH
```

### Convert String to Enum (if supported)

```python
from pamola_core.common.enum.analysis_mode_enum import AnalysisMode

mode_str = "ANALYZE"
mode = AnalysisMode[mode_str]  # Use name lookup
# or
mode = next((m for m in AnalysisMode if m.value == mode_str), None)
```

## Best Practices

1. **Type Safety:** Always use enum members instead of hardcoded strings
   ```python
   # Good
   mode = AnalysisMode.BOTH

   # Avoid
   mode = "BOTH"  # Error-prone
   ```

2. **Configuration Documentation:** Clearly document which mode is chosen and why
   ```python
   # Good - explains reasoning
   analysis_mode = AnalysisMode.ENRICH  # Need k-values for filtering
   ```

3. **Consistent Usage:** Use the same mode throughout a workflow
   ```python
   mode = AnalysisMode.BOTH
   results = analyze_k_anonymity(df, mode)
   # Continue using 'mode' for consistency
   ```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Privacy Metrics Type](./privacy_metrics_type.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- Covers all three analysis modes
- Usage examples and best practices
