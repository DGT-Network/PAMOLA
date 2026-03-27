# LDiversityMetricsCalculator Documentation

**Module:** `pamola_core.privacy_models.l_diversity.metrics`
**Class:** `LDiversityMetricsCalculator`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

`LDiversityMetricsCalculator` provides comprehensive metrics calculation for l-Diversity datasets. It computes privacy, utility, and fidelity metrics with caching and performance optimization.

**Location:** `pamola_core/privacy_models/l_diversity/metrics.py`

## Core Methods

### __init__(processor=None)

**Signature:**
```python
def __init__(self, processor=None):
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `processor` | object | LDiversityCalculator instance for advanced calculations |

### calculate_metrics(data, quasi_identifiers, sensitive_attributes, **kwargs)

**Purpose:** Calculate comprehensive metrics for l-Diversity dataset.

**Signature:**
```python
def calculate_metrics(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
    **kwargs
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'privacy_metrics': {
        'l_value': int,
        'min_diversity': float,
        'max_diversity': float,
        'avg_diversity': float,
        'entropy': float
    },
    'utility_metrics': {
        'information_loss': float,
        'data_utility_score': float,
        'suppression_rate': float
    },
    'fidelity_metrics': {
        'completeness': float,
        'accuracy': float,
        'preservation_score': float
    }
}
```

## Usage Examples

### Example 1: Basic Metrics

```python
from pamola_core.privacy_models import LDiversityCalculator
from pamola_core.privacy_models.l_diversity.metrics import LDiversityMetricsCalculator

processor = LDiversityCalculator(l=3, diversity_type='distinct')
metrics_calc = LDiversityMetricsCalculator(processor=processor)

# Calculate metrics
metrics = metrics_calc.calculate_metrics(
    df,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)

print(f"Min Diversity: {metrics['privacy_metrics']['min_diversity']}")
print(f"Data Utility: {metrics['utility_metrics']['data_utility_score']}%")
print(f"Information Loss: {metrics['utility_metrics']['information_loss']:.1f}%")
```

### Example 2: Multi-Attribute Analysis

```python
metrics_calc = LDiversityMetricsCalculator()

# Compare metrics for different sensitive attributes
for sensitive_attr in ['diagnosis', 'treatment', 'procedure']:
    metrics = metrics_calc.calculate_metrics(
        df,
        quasi_identifiers=['age', 'zip_code'],
        sensitive_attributes=[sensitive_attr]
    )
    print(f"\n{sensitive_attr}:")
    print(f"  Privacy: {metrics['privacy_metrics']['avg_diversity']:.1f}")
    print(f"  Utility: {metrics['utility_metrics']['data_utility_score']:.1f}%")
```

## Metric Details

### Privacy Metrics
- **l_value**: Achieved l parameter
- **min_diversity**: Lowest diversity in any group
- **max_diversity**: Highest diversity in any group
- **avg_diversity**: Average diversity across groups
- **entropy**: Shannon entropy of sensitive attribute distribution

### Utility Metrics
- **information_loss**: % information lost through anonymization
- **data_utility_score**: Overall utility (100% = no loss)
- **suppression_rate**: % of records suppressed

### Fidelity Metrics
- **completeness**: % of data retained (100% = no suppression)
- **accuracy**: % of attribute values unchanged
- **preservation_score**: Overall preservation metric

## Best Practices

1. **Calculate After Transformation:**
   ```python
   anonymized = processor.apply_model(df, quasi_ids)
   metrics = metrics_calc.calculate_metrics(anonymized, quasi_ids, sens_attrs)
   ```

2. **Track Privacy-Utility Trade-off:**
   ```python
   # Compare different l values
   for l in [2, 3, 4, 5]:
       processor = LDiversityCalculator(l=l)
       anonymized = processor.apply_model(df, quasi_ids)
       metrics = metrics_calc.calculate_metrics(anonymized, quasi_ids, sens_attrs)
       print(f"l={l}: privacy={metrics['privacy_metrics']['min_diversity']}, "
             f"utility={metrics['utility_metrics']['data_utility_score']}")
   ```

3. **Use Processor for Caching:**
   ```python
   # Processor's cache speeds up repeated metric calculations
   processor = LDiversityCalculator(l=3)
   metrics_calc = LDiversityMetricsCalculator(processor=processor)
   ```

## Related Components

- [LDiversityCalculator](./l_diversity_calculator.md)
- [LDiversityReport](./l_diversity_report.md)
