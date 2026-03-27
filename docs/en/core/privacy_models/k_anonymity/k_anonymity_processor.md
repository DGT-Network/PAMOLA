# KAnonymityProcessor Documentation

**Module:** `pamola_core.privacy_models.k_anonymity.calculation`
**Class:** `KAnonymityProcessor`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Initialization](#initialization)
5. [Core Methods](#core-methods)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)

## Overview

`KAnonymityProcessor` implements the k-Anonymity privacy model, ensuring that each record is indistinguishable from at least k-1 others based on quasi-identifiers. It provides evaluation, transformation, and analysis capabilities for k-Anonymity compliance.

**Key Guarantee:** Every group formed by quasi-identifier values contains at least k records.

**Location:** `pamola_core/privacy_models/k_anonymity/calculation.py`

## Key Features

- **Efficient Grouping:** Uses pandas `groupby()` for fast group computation
- **Adaptive k-levels:** Define different k values for different groups
- **Progress Tracking:** Built-in `tqdm` progress bars for large datasets
- **Parallel Processing:** Dask integration for very large datasets
- **Integrated Metrics:** Privacy, utility, and fidelity assessment
- **Visualization Tools:** k-distribution and risk heatmaps
- **Comprehensive Reporting:** Compliance and audit reports
- **Memory Optimization:** Efficient handling of large datasets

## Architecture

### Inheritance

```
BasePrivacyModelProcessor
    ↓
KAnonymityProcessor (ABC, implements abstract methods)
    ↓
Methods: process(), evaluate_privacy(), apply_model()
        + visualization_methods()
        + reporting_methods()
```

### Key Components

```python
KAnonymityProcessor
├── __init__()                          # Configuration
├── process()                           # Data processing
├── evaluate_privacy()                  # Privacy assessment
├── apply_model()                       # Anonymization
├── enrich_with_k_values()             # Add k column
├── calculate_risk()                    # Re-identification risk
├── calculate_metrics()                 # Privacy/utility metrics
├── generate_report()                   # Create report
└── visualization methods               # Plot generation
```

## Initialization

### Constructor Signature

```python
def __init__(
    self,
    k: int = 3,
    adaptive_k: Optional[Dict[Tuple, int]] = None,
    suppression: bool = True,
    mask_value: str = "MASKED",
    use_dask: bool = False,
    log_level: str = "INFO",
    progress_tracking: bool = True,
    config_override: Optional[Dict[str, Any]] = None,
):
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 3 | Minimum group size requirement |
| `adaptive_k` | Dict[Tuple, int] | None | Custom k values for specific groups |
| `suppression` | bool | True | Suppress vs mask non-compliant records |
| `mask_value` | str | "MASKED" | String to use for masking |
| `use_dask` | bool | False | Enable Dask for large datasets |
| `log_level` | str | "INFO" | Logging verbosity |
| `progress_tracking` | bool | True | Show progress bar |
| `config_override` | Dict | None | Override default configuration |

### Usage Examples

```python
# Basic initialization (k=3)
processor = KAnonymityProcessor()

# Custom k value
processor = KAnonymityProcessor(k=5)

# Adaptive k for different groups
adaptive_k = {
    ('adult', 'urban'): 5,
    ('adult', 'rural'): 3,
    ('senior', 'urban'): 4,
}
processor = KAnonymityProcessor(adaptive_k=adaptive_k)

# Large dataset with Dask
processor = KAnonymityProcessor(k=5, use_dask=True)

# Custom configuration
processor = KAnonymityProcessor(
    k=3,
    mask_value="***",
    log_level="DEBUG",
    progress_tracking=False
)
```

## Core Methods

### 1. evaluate_privacy(data, quasi_identifiers, **kwargs)

**Purpose:** Assess k-Anonymity compliance without modifying data.

**Signature:**
```python
def evaluate_privacy(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str],
    **kwargs
) -> dict
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | pd.DataFrame | Yes | Dataset to evaluate |
| `quasi_identifiers` | list[str] | Yes | Quasi-identifier columns |
| `**kwargs` | dict | No | Additional parameters |

**Returns:**

```python
{
    'is_k_anonymous': bool,           # Compliance status
    'min_k': int,                     # Smallest group size
    'max_k': int,                     # Largest group size
    'avg_k': float,                   # Average group size
    'records_at_risk': int,           # Non-compliant records
    're_identification_risk': float,   # Risk percentage
    'dataset_info': {
        'total_records': int,
        'total_groups': int
    }
}
```

**Example:**
```python
processor = KAnonymityProcessor(k=5)
evaluation = processor.evaluate_privacy(
    df,
    quasi_identifiers=['age', 'zip_code', 'gender']
)

print(f"k-Anonymous: {evaluation['is_k_anonymous']}")
print(f"Minimum group size: {evaluation['min_k']}")
print(f"Records at risk: {evaluation['records_at_risk']}")
```

### 2. apply_model(data, quasi_identifiers, suppression=True, **kwargs)

**Purpose:** Apply k-Anonymity transformation to dataset.

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

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | — | Input dataset |
| `quasi_identifiers` | list[str] | — | Quasi-identifier columns |
| `suppression` | bool | True | Remove vs mask non-compliant records |
| `**kwargs` | dict | — | Additional options |

**Returns:** Anonymized DataFrame

**Example:**
```python
processor = KAnonymityProcessor(k=3)

# Apply with suppression (removes non-compliant rows)
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'city'],
    suppression=True
)

# Apply with masking (retains non-compliant rows, masks values)
anonymized = processor.apply_model(
    df,
    quasi_identifiers=['age', 'city'],
    suppression=False,
    mask_value="***"
)
```

### 3. enrich_with_k_values(data, quasi_identifiers)

**Purpose:** Add k-value column to dataset for traceability.

**Signature:**
```python
def enrich_with_k_values(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str]
) -> pd.DataFrame
```

**Returns:** DataFrame with 'k_value' column added

**Example:**
```python
processor = KAnonymityProcessor(k=3)
enriched = processor.enrich_with_k_values(df, ['age', 'zip_code'])

# Each row now has k_value column showing its group size
print(enriched[['age', 'zip_code', 'k_value']].head())
#    age zip_code  k_value
# 0   25    12345        5
# 1   25    12345        5  (same group)
# 2   30    67890        3
```

### 4. calculate_risk(data, quasi_identifiers)

**Purpose:** Calculate re-identification risk for each record.

**Signature:**
```python
def calculate_risk(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str]
) -> pd.DataFrame
```

**Returns:** DataFrame with risk metrics (risk_score, is_at_risk columns)

**Example:**
```python
processor = KAnonymityProcessor(k=5)
df_with_risk = processor.calculate_risk(df, ['age', 'zip_code'])

# Identify high-risk records
high_risk = df_with_risk[df_with_risk['is_at_risk']]
print(f"High-risk records: {len(high_risk)}")
```

### 5. calculate_metrics(data, quasi_identifiers)

**Purpose:** Calculate comprehensive privacy, utility, and fidelity metrics.

**Signature:**
```python
def calculate_metrics(
    self,
    data: pd.DataFrame,
    quasi_identifiers: list[str]
) -> dict
```

**Returns:**

```python
{
    'privacy_metrics': {
        'min_k': int,
        'max_k': int,
        'avg_k': float,
        'entropy': float
    },
    'utility_metrics': {
        'information_loss': float,
        'data_utility_score': float
    },
    'fidelity_metrics': {
        'completeness': float,
        'accuracy': float
    }
}
```

### 6. generate_report(include_visualizations=True)

**Purpose:** Generate comprehensive k-Anonymity report.

**Signature:**
```python
def generate_report(self, include_visualizations: bool = True) -> dict
```

**Returns:** Report with metadata, metrics, and visualization paths

## Usage Examples

### Example 1: Basic Privacy Evaluation

```python
from pamola_core.privacy_models import KAnonymityProcessor
import pandas as pd

df = pd.read_csv('customer_data.csv')
processor = KAnonymityProcessor(k=5)

# Evaluate
evaluation = processor.evaluate_privacy(df, ['age', 'zip_code'])

if evaluation['is_k_anonymous']:
    print("✓ Dataset meets k-anonymity requirement")
else:
    print(f"✗ Dataset fails k-anonymity")
    print(f"  Min group size: {evaluation['min_k']} (need {processor.k})")
    print(f"  At-risk records: {evaluation['records_at_risk']}")
```

### Example 2: Transform with Evaluation

```python
processor = KAnonymityProcessor(k=3)

# Check before
eval_before = processor.evaluate_privacy(df, quasi_ids)
print(f"Before: {len(df)} records, min_k={eval_before['min_k']}")

# Transform
anonymized = processor.apply_model(df, quasi_ids, suppression=True)
print(f"After: {len(anonymized)} records")

# Verify
eval_after = processor.evaluate_privacy(anonymized, quasi_ids)
assert eval_after['is_k_anonymous'], "Transformation failed"
```

### Example 3: Adaptive k Values

```python
# Different k for different groups
adaptive_k = {
    ('25-40', 'urban'): 5,
    ('25-40', 'rural'): 3,
    ('40-65', 'urban'): 4,
}

processor = KAnonymityProcessor(adaptive_k=adaptive_k)
anonymized = processor.apply_model(df, ['age_group', 'location'])
```

### Example 4: Traceability with k Values

```python
processor = KAnonymityProcessor(k=3)

# Add k column for tracking
df_traced = processor.enrich_with_k_values(df, ['age', 'zip_code'])

# Analyze group distribution
k_distribution = df_traced['k_value'].value_counts().sort_index()
print("Group size distribution:")
print(k_distribution)

# Find all records in size-3 groups
small_groups = df_traced[df_traced['k_value'] == 3]
```

### Example 5: Risk Analysis

```python
processor = KAnonymityProcessor(k=5)

# Calculate risks
df_risk = processor.calculate_risk(df, quasi_ids)

# High-risk records
at_risk = df_risk[df_risk['is_at_risk']]
print(f"Records at risk: {len(at_risk)} ({len(at_risk)/len(df)*100:.1f}%)")

# Risk distribution
print(df_risk['risk_score'].describe())
```

## Best Practices

1. **Choose Appropriate k:**
   - k=3-5: Basic privacy
   - k=10+: Moderate privacy
   - k=20+: Strong privacy

2. **Select Relevant Quasi-Identifiers:**
   ```python
   # Include columns that could be linked to external data
   quasi_ids = ['age', 'zip_code', 'gender', 'occupation']
   ```

3. **Evaluate Before Transformation:**
   ```python
   eval = processor.evaluate_privacy(df, quasi_ids)
   if not eval['is_k_anonymous']:
       # Adjust k or quasi-identifiers before applying
   ```

4. **Monitor Information Loss:**
   ```python
   metrics = processor.calculate_metrics(df, quasi_ids)
   print(f"Information loss: {metrics['utility_metrics']['information_loss']}")
   ```

5. **Use Adaptive k When Appropriate:**
   ```python
   # Different privacy requirements for different demographics
   adaptive_k = {
       ('high_income',): 5,
       ('low_income',): 3
   }
   ```

## Troubleshooting

### Issue: Dataset Not k-Anonymous

**Problem:** `is_k_anonymous: False` with low min_k

**Solutions:**
- Increase k value (may increase information loss)
- Remove less relevant quasi-identifiers
- Generalize quasi-identifier values (e.g., binning ages)
- Use l-Diversity for attribute-level privacy

### Issue: Too Many Records Suppressed

**Problem:** apply_model removes most records

**Solutions:**
- Reduce k parameter
- Use suppression=False for masking instead
- Use generalization strategies
- Consider different privacy model (l-Diversity, t-Closeness)

### Issue: Performance on Large Datasets

**Problem:** Processing too slow

**Solutions:**
- Enable Dask: `use_dask=True`
- Sample data for initial testing
- Use fewer quasi-identifiers
- Increase log_level to reduce verbosity

## Related Components

- [KAnonymityReport](./k_anonymity_report.md) — Report generation
- [KAnonymityVisualization](./k_anonymity_visualization.md) — Visualization functions
- [Privacy Models Overview](../privacy_models_overview.md) — Model comparison

## Summary

`KAnonymityProcessor` provides a complete k-Anonymity implementation with:
- Fast evaluation and transformation
- Adaptive parameters support
- Integrated metrics and visualization
- Large dataset support via Dask
- Comprehensive reporting

Use it for basic privacy requirements and as a foundation for stronger models like l-Diversity. Start with k=3-5 and adjust based on evaluation results.
