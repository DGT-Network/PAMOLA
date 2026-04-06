# Privacy Risk Assessment

**Module:** `pamola_core.analysis.privacy_risk`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Overview

The `privacy_risk` module implements formal privacy models (k-anonymity, l-diversity, t-closeness) combined with simulated attack-based risks (re-identification, attribute disclosure, membership inference) to provide a comprehensive privacy assessment score for tabular datasets.

This module is designed for governance workflows, pre/post-anonymization audits, and continuous privacy monitoring. All metrics are normalized to [0, 1] ranges and aggregated into a single 0–100 risk score reflecting organizational risk exposure.

## Key Features

- **Formal Privacy Models**: k-anonymity, l-diversity, t-closeness with per-record analysis
- **Simulated Attack Metrics**: Linkage attack (re-identification), attribute disclosure, membership inference
- **Weighted Risk Aggregation**: Configurable weights for each metric component
- **JSON-Serializable Output**: All results as dictionaries for downstream workflows
- **Production-Ready**: Safe defaults, detailed logging, comprehensive error handling

## Architecture

```
calculate_full_risk(df, quasi_identifiers, sensitive_attributes, weights) → Dict

Helper functions:
├── _calculate_k_anonymity(df, quasi_ids) → Dict
├── _calculate_l_diversity(df, quasi_ids, sensitive_attrs) → Tuple[Dict, float]
├── _calculate_t_closeness(df, quasi_ids, sensitive_attrs) → Dict
├── _simulate_linkage_attack(df, quasi_ids, sensitive_attrs) → float
├── _simulate_attribute_inference(df, quasi_ids, sensitive_attrs) → float
└── _simulate_membership_inference(df, quasi_ids, sensitive_attrs) → float
```

## Core Functions

### `calculate_full_risk()`

Comprehensive privacy risk assessment combining formal models and attack simulation.

**Signature:**
```python
def calculate_full_risk(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | — | Dataset to assess |
| `quasi_identifiers` | List[str] | — | Columns forming equivalence classes (e.g., age, zip_code, gender) |
| `sensitive_attributes` | List[str] | — | Sensitive columns to protect (e.g., disease, salary) |
| `weights` | Dict \| None | Default | Custom weights for risk components; see below |

**Weight Dictionary** (if not provided, uses defaults):
```python
{
    "k_anonymity": 0.40,                    # Strength of k-anonymity (larger k = lower risk)
    "l_diversity": 0.10,                    # Diversity of sensitive values within groups
    "attribute_disclosure_risk": 0.30,      # Deterministic inference of sensitive attrs
    "membership_inference_risk": 0.20       # Detectability of individuals in dataset
}
```
All weights must sum to 1.0; raises `ValidationError` otherwise.

**Returns:**
```python
{
    "reidentification_risk": float,         # [0, 1] Proportion of unique records
    "attribute_disclosure_risk": float,     # [0, 1] Proportion of deterministic groups
    "membership_inference_risk": float,     # [0, 1] Proportion of singleton groups
    "k_anonymity": {                        # k-anonymity details
        "k": int,                           # Minimum equivalence class size
        "quasi_identifiers": List[str],     # QI columns used
        "equivalence_classes": int,         # Total distinct QI combinations
        "smallest_class": int,              # Size of smallest group (same as k)
        "records_in_smallest_classes": int  # Count of records in minimum-size groups
    },
    "l_diversity": {                        # l-diversity details
        "l": int,                           # Minimum distinct sensitive values per group
        "sensitive_attributes": List[str],  # Attributes used
        "entropy": float                    # Average entropy across groups
    },
    "risk_assessment": int                  # Final score [0, 100]
}
```

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `ValidationError` | Weights don't sum to 1.0 or required inputs missing |
| `ColumnNotFoundError` | Quasi-identifier or sensitive attribute column not found |

### Helper Functions

#### `_calculate_k_anonymity()`

Computes k-anonymity: minimum group size under quasi-identifiers.

```python
def _calculate_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> Dict[str, Any]
```

**Returns:**
```python
{
    "k": int,                              # Minimum equivalence class size (0 if df empty)
    "quasi_identifiers": List[str],        # Columns used
    "equivalence_classes": int,            # Distinct QI combinations
    "smallest_class": int,                 # Size of smallest group
    "records_in_smallest_classes": int     # Total records in minimum-size groups
}
```

**Interpretation:**
- `k=1`: All records unique (worst privacy)
- `k=5`: At least 5 indistinguishable records per group
- `k=100+`: Excellent anonymity (typical threshold)

#### `_calculate_l_diversity()`

Measures sensitive value diversity within equivalence classes using entropy.

```python
def _calculate_l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str]
) -> Tuple[Dict[str, Any], float]
```

**Returns:**
```python
(
    {
        "l": int,                          # Minimum distinct values per group
        "sensitive_attributes": List[str], # Attributes analyzed
        "entropy": float                   # Average Shannon entropy [0, log2(n_values)]
    },
    max_entropy: float                     # Maximum possible entropy (used in risk calc)
)
```

**Interpretation:**
- `entropy=0`: No diversity (all groups have single value)
- `entropy=2.0`: Good diversity for binary attribute
- `entropy=log2(n)`: Perfect diversity (all values equally likely)

#### `_simulate_linkage_attack()`

Re-identification risk: proportion of records in unique QI combinations.

```python
def _simulate_linkage_attack(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str]
) -> float
```

**Returns:** Risk score in [0, 1].

**Logic:** Groups records by QI (and optionally sensitive attributes), identifies singleton groups (size=1), and returns `singleton_count / total_records`.

#### `_simulate_attribute_inference()`

Attribute disclosure risk: deterministic inference of sensitive values from QIs.

```python
def _simulate_attribute_inference(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str]
) -> float
```

**Returns:** Risk score in [0, 1].

**Logic:** For each QI group, check if ANY sensitive attribute has exactly one unique value. Risk = proportion of such deterministic groups.

#### `_simulate_membership_inference()`

Membership inference risk: detectability of individuals in dataset.

```python
def _simulate_membership_inference(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str]
) -> float
```

**Returns:** Risk score in [0, 1].

**Logic:** Groups by QI + sensitive attributes, counts singleton groups, returns `singleton_count / total_records`.

## Usage Examples

### Basic Risk Assessment
```python
import pandas as pd
from pamola_core.analysis import calculate_full_risk

df = pd.read_csv("health_data.csv")
quasi_ids = ["age", "zip_code", "gender"]
sensitive = ["disease", "treatment"]

risk = calculate_full_risk(df, quasi_ids, sensitive)

print(f"Risk Score: {risk['risk_assessment']}%")
print(f"k-anonymity: {risk['k_anonymity']['k']}")
print(f"Re-id risk: {risk['reidentification_risk']:.1%}")
```

### Custom Weights (Privacy-Conservative)
```python
# Emphasize k-anonymity and membership inference
weights = {
    "k_anonymity": 0.50,
    "l_diversity": 0.05,
    "attribute_disclosure_risk": 0.20,
    "membership_inference_risk": 0.25
}

risk = calculate_full_risk(df, quasi_ids, sensitive, weights=weights)
```

### Pre/Post-Anonymization Comparison
```python
risk_before = calculate_full_risk(df_original, quasi_ids, sensitive)
risk_after = calculate_full_risk(df_anonymized, quasi_ids, sensitive)

improvement = risk_before['risk_assessment'] - risk_after['risk_assessment']
print(f"Risk reduced by {improvement}%")
```

### Risk Monitoring Pipeline
```python
def assess_batch(df_batch, quasi_ids, sensitive_attrs):
    """Monitor risk across multiple datasets."""
    result = calculate_full_risk(df_batch, quasi_ids, sensitive_attrs)

    if result['risk_assessment'] > 50:
        raise RuntimeError(f"Risk {result['risk_assessment']}% exceeds threshold")

    return result

# In production
for batch in data_batches:
    try:
        metrics = assess_batch(batch, QI_COLS, SENSITIVE_COLS)
        log_metrics(metrics)
    except RuntimeError as e:
        alert_team(str(e))
```

## Interpretation Guide

### Risk Score (0–100)

| Score | Interpretation | Action |
|-------|-----------------|--------|
| 0–20 | Minimal risk | Good privacy; monitor regularly |
| 21–50 | Moderate risk | Apply targeted anonymization |
| 51–80 | High risk | Significant anonymization needed |
| 81–100 | Critical risk | Restrict access; anonymize aggressively |

### k-Anonymity Thresholds

| k Value | Privacy Level | Recommendation |
|---------|---------------|-----------------|
| 1–5 | Very weak | Unacceptable for sensitive data |
| 5–10 | Weak | Inadequate for health/financial data |
| 10–20 | Moderate | Acceptable with additional safeguards |
| 20+ | Good | Suitable for most applications |
| 100+ | Excellent | Suitable for public release |

### Entropy Levels (l-diversity)

| Entropy | Diversity | Interpretation |
|---------|-----------|-----------------|
| 0.0 | None | All records in group have same sensitive value |
| 0.5–1.0 | Low | Some variation in sensitive values |
| 1.5+ | Good | Strong diversity within groups |
| log2(n) | Perfect | All sensitive values equally likely |

## Best Practices

1. **Choose Realistic Quasi-Identifiers**: Include only attributes attackers could know externally (age, gender, ZIP) not unique identifiers (SSN, email).

2. **Combine Multiple Metrics**: Never rely on k-anonymity alone. Assess k-anonymity + l-diversity + attack risks together.

3. **Validate on Holdout Data**: Test anonymization on representative sample before applying to entire dataset.

4. **Set Organizational Risk Threshold**: Define acceptable risk score based on data sensitivity and use case. Adjust weights accordingly.

5. **Monitor Post-Release**: After data release, periodically re-assess risk as new quasi-identifiers or linking data become available.

6. **Document Assumptions**: Record which columns are treated as quasi-identifiers and sensitive attributes for audit trails.

## Limitations

- **Heuristic attacks**: Simulated attacks are simplified; real attacks may be more sophisticated
- **Static assessment**: Assumes quasi-identifier knowledge is fixed; doesn't account for emerging linkage data
- **Assumes independence**: Models don't capture temporal patterns or hierarchical relationships
- **Deterministic assumptions**: Doesn't model probabilistic inference attacks (e.g., Bayesian re-identification)

## Related Components

- [`analyze_dataset_summary()`](./dataset_summary.md) - Field types and outliers before risk assessment
- [`analyze_descriptive_stats()`](./descriptive_stats.md) - Statistical basis for risk evaluation
- [`anonymization`](../anonymization/) - Apply operations to reduce risk
- [`metrics`](../metrics/) - Post-anonymization utility evaluation

## Summary Analysis

**Purpose**: Quantify privacy risk using formal models + attack simulation; track risk across data transformations.

**Inputs**: DataFrame, quasi-identifier columns, sensitive attribute columns.

**Output**: Detailed risk breakdown with k-anonymity, l-diversity, attack metrics, and weighted score.

**Strengths**:
- Comprehensive: Combines multiple privacy models
- Configurable: Customizable weights for organizational priorities
- Practical: Includes simulated attack metrics alongside formal models
- Tracked: Output format suitable for governance dashboards and audits

**Typical Workflow**: Assess before anonymization → Apply transformations → Re-assess → Monitor post-release.
