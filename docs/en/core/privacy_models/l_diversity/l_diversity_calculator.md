# LDiversityCalculator Documentation

**Module:** `pamola_core.privacy_models.l_diversity.calculation`
**Class:** `LDiversityCalculator`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Initialization](#initialization)
4. [Diversity Types](#diversity-types)
5. [Core Methods](#core-methods)
6. [Usage Examples](#usage-examples)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Related Components](#related-components)

## Overview

`LDiversityCalculator` implements l-Diversity, extending k-Anonymity by ensuring diverse sensitive attribute values within each quasi-identifier group. It provides comprehensive privacy evaluation, transformation, risk assessment, and reporting.

**Key Guarantee:** Every group satisfying k-Anonymity also contains at least l distinct (or sufficiently diverse) values of sensitive attributes.

**Location:** `pamola_core/privacy_models/l_diversity/calculation.py`

## Key Features

- **Multiple Diversity Types:** Distinct, entropy, and recursive (c,l)-diversity
- **Advanced Metrics:** Comprehensive privacy, utility, and fidelity calculations
- **Risk Assessment:** Attribute disclosure risk with cache optimization
- **Caching System:** Centralized results cache avoids redundant calculations
- **Flexible Strategies:** Suppression, full masking, partial masking
- **Visualization Tools:** Distribution plots and risk heatmaps
- **Reporting:** Compliance and technical reports
- **Large Dataset Support:** Dask integration for parallel processing
- **Adaptive l-values:** Group-specific anonymization levels

## Initialization

### Constructor Signature

```python
def __init__(
    self,
    l: int = 3,
    diversity_type: str = "distinct",
    c_value: float = 1.0,
    k: int = 2,
    config_override: Optional[Dict[str, Any]] = None,
    use_dask: bool = False,
    log_level: str = "INFO",
    adaptive_l: Optional[Dict[Tuple, int]] = None,
):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `l` | int | 3 | Minimum attribute diversity |
| `diversity_type` | str | "distinct" | "distinct", "entropy", or "recursive" |
| `c_value` | float | 1.0 | Parameter for recursive (c,l)-diversity |
| `k` | int | 2 | Minimum group size (k-anonymity component) |
| `config_override` | Dict | None | Override default configuration |
| `use_dask` | bool | False | Enable Dask for large datasets |
| `log_level` | str | "INFO" | Logging verbosity |
| `adaptive_l` | Dict[Tuple, int] | None | Custom l values for specific groups |

### Usage Examples

```python
# Distinct diversity (l=3 distinct values)
processor = LDiversityCalculator(
    l=3,
    diversity_type='distinct',
    k=2
)

# Entropy diversity (more flexible)
processor = LDiversityCalculator(
    l=3,
    diversity_type='entropy',
    k=2
)

# Recursive (c,l)-diversity (stronger privacy)
processor = LDiversityCalculator(
    l=3,
    diversity_type='recursive',
    c_value=2.0,
    k=2
)

# Large dataset with Dask
processor = LDiversityCalculator(
    l=4,
    diversity_type='distinct',
    use_dask=True
)

# Adaptive l values
adaptive_l = {
    ('adult', 'high_income'): 5,
    ('adult', 'low_income'): 3,
    ('senior',): 4
}
processor = LDiversityCalculator(
    l=3,
    adaptive_l=adaptive_l
)
```

## Diversity Types

### 1. Distinct Diversity

**Definition:** Each group contains at least l distinct values of the sensitive attribute.

**Pros:**
- Simplest to understand and implement
- Fast computation
- Strong practical privacy

**Cons:**
- Doesn't account for value frequency distribution
- May be weak for skewed distributions

**Example:**
```python
processor = LDiversityCalculator(l=3, diversity_type='distinct')

# Requirement: each group must have ≥3 different values
# {A, A, B} → NOT diverse
# {A, B, C} → diverse
# {A, A, A, B, C, D} → diverse
```

### 2. Entropy Diversity

**Definition:** The entropy of sensitive attribute distribution in each group meets threshold.

**Formula:** H = -Σ(p_i * log(p_i))

**Pros:**
- Accounts for value frequency
- More flexible than distinct
- Good for continuous distributions

**Cons:**
- More complex to compute
- Requires threshold calibration
- May be computationally expensive

**Example:**
```python
processor = LDiversityCalculator(l=3, diversity_type='entropy')

# Entropy threshold depends on number of values
# High entropy = values well-distributed
# Low entropy = values skewed/dominated by one value
```

### 3. Recursive (c,l)-Diversity

**Definition:** r₁ < c(r_{l})** where r_i is frequency of i-th most frequent value.

**Pros:**
- Strongest privacy guarantee
- Addresses frequency attacks
- Flexible parameter (c_value)

**Cons:**
- Most computationally expensive
- Requires c_value tuning
- Hardest to satisfy

**Example:**
```python
processor = LDiversityCalculator(
    l=3,
    diversity_type='recursive',
    c_value=2.0
)

# Requirement: r₁ < 2.0 * r₃
# {A(50%), B(25%), C(25%)} → frequency ratio = 50/25 = 2.0 (borderline)
# {A(40%), B(30%), C(30%)} → frequency ratio = 40/30 = 1.33 (diverse)
```

## Core Methods

### 1. process(data)

**Purpose:** Generic data processing interface.

**Signature:**
```python
def process(self, data: Any) -> Any:
```

### 2. evaluate_privacy(data, quasi_identifiers, **kwargs)

**Purpose:** Assess l-Diversity compliance without modification.

**Signature:**
```python
def evaluate_privacy(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    **kwargs
) -> dict
```

**Returns:**
```python
{
    'is_l_diverse': bool,
    'l_value': int,
    'min_diversity': float,
    'max_diversity': float,
    'avg_diversity': float,
    'non_diverse_groups': int,
    'diversity_type': str,
    'dataset_info': {...}
}
```

**Example:**
```python
processor = LDiversityCalculator(l=3, diversity_type='distinct')
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)

if evaluation['is_l_diverse']:
    print(f"✓ Dataset is {evaluation['l_value']}-diverse")
else:
    print(f"✗ Not diverse: min={evaluation['min_diversity']}, need {processor.l}")
```

### 3. apply_model(data, quasi_identifiers, suppression=True, **kwargs)

**Purpose:** Transform data to achieve l-Diversity.

**Signature:**
```python
def apply_model(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    suppression: bool = True,
    **kwargs
) -> pd.DataFrame
```

**Example:**
```python
processor = LDiversityCalculator(l=3, diversity_type='distinct')
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'zip_code'],
    suppression=True
)
```

### 4. evaluate_attribute_disclosure_risk()

**Purpose:** Assess risk of sensitive attribute disclosure.

**Signature:**
```python
def evaluate_attribute_disclosure_risk(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_attribute: str,
    **kwargs
) -> dict
```

**Returns:**
```python
{
    'attribute': str,
    'overall_risk': float,
    'high_risk_groups': int,
    'risk_distribution': {...},
    'recommendations': [...]
}
```

### 5. generate_report(save_path=None, include_visualizations=True)

**Purpose:** Generate comprehensive l-Diversity report.

**Signature:**
```python
def generate_report(
    self,
    save_path: Optional[str] = None,
    include_visualizations: bool = True
) -> dict
```

**Returns:** Complete report dictionary with metadata, metrics, and visualization paths

## Usage Examples

### Example 1: Basic l-Diversity Evaluation

```python
from pamola_core.privacy_models import LDiversityCalculator
import pandas as pd

# Load data
df = pd.read_csv('patient_data.csv')

# Create processor
processor = LDiversityCalculator(
    l=3,
    diversity_type='distinct',
    k=2
)

# Evaluate
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code'],
    sensitive_attributes=['diagnosis']
)

# Check results
if evaluation['is_l_diverse']:
    print("✓ Dataset meets l-diversity requirement")
else:
    print(f"✗ Dataset fails: min_diversity={evaluation['min_diversity']}")
    print(f"  Non-diverse groups: {evaluation['non_diverse_groups']}")
```

### Example 2: Transform to l-Diversity

```python
processor = LDiversityCalculator(
    l=4,
    diversity_type='entropy',
    k=3
)

# Check before
eval_before = processor.evaluate_privacy(df, quasi_ids, sens_attrs)
print(f"Before: is_l_diverse={eval_before['is_l_diverse']}")

# Transform
anonymized = processor.apply_model(df, quasi_ids, suppression=True)
print(f"Original records: {len(df)}")
print(f"Anonymized records: {len(anonymized)}")

# Verify after
eval_after = processor.evaluate_privacy(anonymized, quasi_ids, sens_attrs)
assert eval_after['is_l_diverse'], "Transformation failed"
print(f"After: is_l_diverse={eval_after['is_l_diverse']}")
```

### Example 3: Multi-Attribute Risk Assessment

```python
processor = LDiversityCalculator(l=3, diversity_type='distinct')

# Evaluate each sensitive attribute
sensitive_attrs = ['diagnosis', 'treatment', 'marital_status']

for attr in sensitive_attrs:
    risk = processor.evaluate_attribute_disclosure_risk(
        df,
        quasi_identifiers=['age', 'zip_code'],
        sensitive_attribute=attr
    )
    print(f"{attr}: risk={risk['overall_risk']:.1f}%")
```

### Example 4: Adaptive l-Diversity

```python
# Different privacy for different demographics
adaptive_l = {
    ('senior', 'rural'): 5,        # Higher privacy needed
    ('senior', 'urban'): 4,
    ('adult', 'rural'): 3,         # Lower privacy needed
    ('adult', 'urban'): 3,
}

processor = LDiversityCalculator(
    l=3,
    diversity_type='distinct',
    adaptive_l=adaptive_l
)

anonymized = processor.apply_model(df, ['age_group', 'location'])
```

### Example 5: Comprehensive Reporting

```python
from pamola_core.privacy_models import LDiversityCalculator
import json

processor = LDiversityCalculator(
    l=4,
    diversity_type='entropy',
    k=3
)

# Process
anonymized = processor.apply_model(df, quasi_ids, suppression=True)

# Generate full report
report = processor.generate_report(
    save_path='reports/',
    include_visualizations=True
)

# Save report
with open('ldiversity_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"Report saved: {report['metadata']['report_id']}")
```

## Performance Optimization

### Use Caching

The processor includes centralized caching:

```python
processor = LDiversityCalculator(l=3)

# First evaluation (computed)
eval1 = processor.evaluate_privacy(df, quasi_ids)

# Second evaluation (uses cache - much faster)
eval2 = processor.evaluate_privacy(df, quasi_ids)
```

### Enable Dask for Large Datasets

```python
# For datasets > 1GB
processor = LDiversityCalculator(l=3, use_dask=True)

# Processing uses parallel Dask DataFrames
anonymized = processor.apply_model(df, quasi_ids)
```

### Batch Processing

```python
# Process large dataset in batches
batch_size = 100000
all_anonymized = []

for batch in pd.read_csv('large_file.csv', chunksize=batch_size):
    anonymized_batch = processor.apply_model(batch, quasi_ids)
    all_anonymized.append(anonymized_batch)

result = pd.concat(all_anonymized, ignore_index=True)
```

## Best Practices

1. **Choose Appropriate Diversity Type:**
   - Start with `'distinct'` for simplicity
   - Use `'entropy'` for better distribution handling
   - Use `'recursive'` for strongest privacy

2. **Validate Quasi-Identifiers:**
   ```python
   quasi_ids = ['age', 'zip_code', 'gender']
   # Ensure these can be linked to external data
   ```

3. **Monitor Sensitive Attributes:**
   ```python
   # Evaluate each sensitive attribute separately
   for attr in ['diagnosis', 'treatment']:
       risk = processor.evaluate_attribute_disclosure_risk(
           df, quasi_ids, attr
       )
   ```

4. **Test Parameters:**
   ```python
   # Start with l=3, adjust based on results
   processor = LDiversityCalculator(l=3, diversity_type='entropy')
   eval = processor.evaluate_privacy(df, quasi_ids)

   if not eval['is_l_diverse']:
       # Increase l or use recursive diversity
   ```

5. **Document Decisions:**
   ```python
   report = processor.generate_report()
   # Report includes full configuration for audit trail
   ```

## Troubleshooting

### Issue: Dataset Not l-Diverse

**Solutions:**
- Increase l parameter
- Change diversity_type to 'entropy' or 'recursive'
- Remove irrelevant quasi-identifiers
- Aggregate sensitive attribute values

### Issue: High Information Loss

**Solutions:**
- Reduce l parameter
- Use masking instead of suppression
- Use generalization strategies
- Combine with t-Closeness for better utility

### Issue: Performance Degradation

**Solutions:**
- Enable Dask: `use_dask=True`
- Sample data for testing
- Use fewer quasi-identifiers
- Enable caching (automatic)

## Related Components

- [LDiversityMetricsCalculator](./l_diversity_metrics.md)
- [LDiversityPrivacyRiskAssessor](./privacy_risk_assessor.md)
- [AttributeDisclosureRiskAnalyzer](./attribute_disclosure.md)
- [LDiversityReport](./l_diversity_report.md)

## Summary

`LDiversityCalculator` provides comprehensive l-Diversity implementation with:
- Three diversity types for different privacy needs
- Advanced risk assessment and caching
- Flexible transformation strategies
- Integrated visualization and reporting

Start with distinct diversity and l=3, then adjust based on privacy evaluation results. Combine with attribute disclosure risk assessment for comprehensive protection.
