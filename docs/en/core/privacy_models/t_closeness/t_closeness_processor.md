# TCloseness Documentation

**Module:** `pamola_core.privacy_models.t_closeness.calculation`
**Class:** `TCloseness`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Initialization](#initialization)
4. [Core Methods](#core-methods)
5. [Distance Metrics](#distance-metrics)
6. [Usage Examples](#usage-examples)
7. [Mathematical Foundation](#mathematical-foundation)
8. [Best Practices](#best-practices)
9. [Related Components](#related-components)

## Overview

`TCloseness` implements the t-Closeness privacy model, ensuring that the distribution of sensitive attributes in any k-anonymous group is close to the overall dataset distribution. It uses Wasserstein distance for distribution comparison.

**Key Guarantee:** For every quasi-identifier group, the sensitive attribute distribution distance ≤ t from the overall distribution.

**Location:** `pamola_core/privacy_models/t_closeness/calculation.py`

## Key Features

- **Distribution-Based Privacy:** Maintains statistical properties of data
- **Wasserstein Distance:** Robust distance metric for distributions
- **Sensitive Attribute Distribution:** Ensures distribution similarity
- **Suppression Support:** Remove non-compliant groups if needed
- **Utility Preservation:** Better data utility than suppression-only approaches
- **Flexible t Values:** Adjustable distance threshold

## Initialization

### Constructor Signature

```python
def __init__(
    self,
    quasi_identifiers: List[str],
    sensitive_column: str,
    t: float
):
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `quasi_identifiers` | List[str] | Columns to group data for t-closeness evaluation |
| `sensitive_column` | str | Column with sensitive attribute to protect |
| `t` | float | Maximum allowed distribution distance (0.0 to 1.0) |

### Usage Examples

```python
# Basic t-closeness (t=0.1, fairly strict)
processor = TCloseness(
    quasi_identifiers=['age', 'zip_code'],
    sensitive_column='salary',
    t=0.1
)

# Lenient t-closeness (t=0.3)
processor = TCloseness(
    quasi_identifiers=['age', 'zip_code'],
    sensitive_column='salary',
    t=0.3
)

# Strict t-closeness (t=0.05)
processor = TCloseness(
    quasi_identifiers=['age', 'zip_code'],
    sensitive_column='salary',
    t=0.05
)
```

## Core Methods

### evaluate_privacy(data, quasi_identifiers, **kwargs)

**Purpose:** Evaluate t-Closeness compliance without modification.

**Signature:**
```python
def evaluate_privacy(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    **kwargs
) -> Dict[str, Any]:
```

**Returns:**
```python
{
    'is_t_close': bool,
    'max_t_value': float,              # Largest group distance
    'min_t_value': float,              # Smallest group distance
    'avg_t_value': float,              # Average distance
    'non_compliant_groups': int,       # Groups with distance > t
    't_threshold': float,
    'global_distribution': {...},      # Overall attribute distribution
    'group_distributions': {...}       # Per-group distributions
}
```

**Example:**
```python
processor = TCloseness(
    quasi_identifiers=['age', 'zip_code'],
    sensitive_column='salary',
    t=0.1
)

evaluation = processor.evaluate_privacy(df, quasi_identifiers=['age', 'zip_code'])

if evaluation['is_t_close']:
    print("✓ Dataset is t-close")
else:
    print(f"✗ Dataset violates t-closeness")
    print(f"  Max distance: {evaluation['max_t_value']:.3f} (threshold: {processor.t})")
```

### apply_model(data, quasi_identifiers, suppression=True, **kwargs)

**Purpose:** Apply t-Closeness transformation to dataset.

**Signature:**
```python
def apply_model(
    self,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    suppression: bool = True,
    **kwargs
) -> pd.DataFrame:
```

**Example:**
```python
processor = TCloseness(['age', 'zip_code'], 'salary', t=0.1)

# Apply with suppression
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'zip_code'],
    suppression=True
)
```

### process(data)

**Purpose:** Generic processing interface (passes through data).

**Signature:**
```python
def process(self, data):
```

## Distance Metrics

### Wasserstein Distance

t-Closeness uses Wasserstein distance (also called Earth Mover's Distance):

**For categorical attributes:** Earth Mover distance with category dissimilarity matrix

**For numeric attributes:** L1 norm of distribution difference

**Range:** 0.0 (identical) to 1.0 (completely different)

**Interpretation:**
- t < 0.05 → Very similar distributions (strict privacy)
- t < 0.10 → Similar distributions (strong privacy)
- t < 0.20 → Moderately similar (balanced privacy)
- t < 0.40 → Loosely similar (utility-focused)

## Usage Examples

### Example 1: Basic Evaluation

```python
from pamola_core.privacy_models import TCloseness
import pandas as pd

# Load data
df = pd.read_csv('employee_data.csv')

# Create processor
processor = TCloseness(
    quasi_identifiers=['age', 'department'],
    sensitive_column='salary',
    t=0.15
)

# Evaluate
evaluation = processor.evaluate_privacy(df, ['age', 'department'])

if evaluation['is_t_close']:
    print("✓ Dataset is t-close")
    print(f"  Max distance: {evaluation['max_t_value']:.3f}")
else:
    print("✗ Dataset is not t-close")
    print(f"  Non-compliant groups: {evaluation['non_compliant_groups']}")
```

### Example 2: Transform to t-Closeness

```python
processor = TCloseness(['age', 'location'], 'salary', t=0.1)

# Evaluate before
eval_before = processor.evaluate_privacy(df, ['age', 'location'])
print(f"Before: is_t_close={eval_before['is_t_close']}")

# Transform
anonymized = processor.apply_model(df, ['age', 'location'], suppression=True)
print(f"Original records: {len(df)}")
print(f"Anonymized records: {len(anonymized)}")

# Verify
eval_after = processor.evaluate_privacy(anonymized, ['age', 'location'])
assert eval_after['is_t_close'], "Transformation failed"
```

### Example 3: Find Optimal t Value

```python
# Try different t values to find acceptable threshold
processor_base = TCloseness(['age', 'zip_code'], 'salary', t=0.0)

for t_value in [0.05, 0.10, 0.15, 0.20, 0.25]:
    processor = TCloseness(['age', 'zip_code'], 'salary', t=t_value)
    eval_result = processor.evaluate_privacy(df, ['age', 'zip_code'])

    print(f"t={t_value:.2f}: is_t_close={eval_result['is_t_close']}, "
          f"non_compliant={eval_result['non_compliant_groups']}")
```

### Example 4: Compare with k-Anonymity

```python
from pamola_core.privacy_models import KAnonymityProcessor, TCloseness

# k-Anonymity approach
k_processor = KAnonymityProcessor(k=5)
k_anonymized = k_processor.apply_model(df, quasi_ids)

# t-Closeness approach
t_processor = TCloseness(quasi_ids, 'salary', t=0.1)
t_anonymized = t_processor.apply_model(df, quasi_ids)

print(f"k-Anonymity records: {len(k_anonymized)}")
print(f"t-Closeness records: {len(t_anonymized)}")

# t-Closeness typically retains more utility
```

## Mathematical Foundation

### t-Closeness Definition

A quasi-identifier group G is t-close if:

```
D(P_G, P) ≤ t
```

Where:
- P_G = distribution of sensitive attribute in group G
- P = overall distribution of sensitive attribute
- D = Wasserstein distance metric

### Wasserstein Distance for Categorical Data

```
W(P, Q) = min over all couplings C of sum(|x-y| * C(x,y))
```

Measures minimum effort needed to transform one distribution to another.

## Best Practices

1. **Choose Appropriate t:**
   ```python
   # Start with t=0.15 (reasonable balance)
   processor = TCloseness(quasi_ids, 'sensitive_attr', t=0.15)

   # Evaluate and adjust if needed
   eval_result = processor.evaluate_privacy(df, quasi_ids)
   ```

2. **Combine with k-Anonymity:**
   ```python
   # t-Closeness alone may not prevent identity disclosure
   # Combine: apply k-anonymity first, then verify t-closeness
   k_processor = KAnonymityProcessor(k=3)
   kAnon_df = k_processor.apply_model(df, quasi_ids)

   t_processor = TCloseness(quasi_ids, 'sensitive_attr', t=0.15)
   eval_t = t_processor.evaluate_privacy(kAnon_df, quasi_ids)
   ```

3. **Monitor Distribution Preservation:**
   ```python
   eval_before = processor.evaluate_privacy(df, quasi_ids)
   eval_after = processor.evaluate_privacy(anonymized, quasi_ids)

   # Check if distributions are similar
   print(f"Global distribution preserved: {eval_before['is_t_close']}")
   ```

4. **Document t Value Choice:**
   ```python
   report_data = {
       't_closeness_config': {
           't': 0.15,
           'justification': 'Balances privacy and utility while maintaining distribution'
       }
   }
   ```

## Troubleshooting

### Issue: Dataset Not t-Close

**Solutions:**
- Increase t (more lenient)
- Reduce quasi-identifiers
- Apply generalization before t-closeness check

### Issue: Too Much Data Suppressed

**Solutions:**
- Increase t threshold
- Use partial masking instead of suppression
- Combine with k-Anonymity for efficiency

## Related Components

- [KAnonymityProcessor](../k_anonymity/k_anonymity_processor.md)
- [LDiversityCalculator](../l_diversity/l_diversity_calculator.md)
- [BasePrivacyModelProcessor](../base_privacy_model.md)

## Summary

`TCloseness` provides distribution-preserving privacy that:
- Maintains statistical properties
- Better data utility than suppression alone
- Complements k-Anonymity and l-Diversity
- Uses mathematically robust distance metrics

Use t-closeness (t=0.1-0.2) for applications where preserving distributions is important, combined with k-Anonymity for identity protection.
