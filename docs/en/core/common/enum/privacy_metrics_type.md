# PrivacyMetricsType Enumeration

**Module:** `pamola_core.common.enum.privacy_metrics_type`
**Class:** `PrivacyMetricsType`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Enum Members](#enum-members)
3. [Privacy Metrics Categories](#privacy-metrics-categories)
4. [Metric Descriptions](#metric-descriptions)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)

## Overview

`PrivacyMetricsType` is a string-based enumeration defining privacy metrics used to evaluate privacy preservation in anonymized datasets. These metrics measure the risk of re-identification and the effectiveness of privacy-preserving transformations.

**Parent Class:** `str, Enum`
**Type:** String Enum
**Scope:** Privacy evaluation, anonymization assessment
**Used By:** Privacy analysis operations, risk assessment

## Enum Members

### Distance-Based Privacy Metrics

| Member | Value | Description |
|--------|-------|-------------|
| `DCR` | `"dcr"` | Distance to Closest Record - Euclidean distance to nearest neighbor |
| `NNDR` | `"nndr"` | Nearest Neighbor Distance Ratio - Ratio of distances to 1st and 2nd nearest neighbors |

### Uniqueness-Based Privacy Metrics

| Member | Value | Description |
|--------|-------|-------------|
| `UNIQUENESS` | `"uniqueness"` | Risk of re-identification from record uniqueness |
| `K_ANONYMITY` | `"k_anonymity"` | k-Anonymity - Each record indistinguishable from k-1 others |
| `L_DIVERSITY` | `"l_diversity"` | l-Diversity - Sensitive attributes diverse within equivalence classes |
| `T_CLOSENESS` | `"t_closeness"` | t-Closeness - Distribution of sensitive attributes close to overall |

## Privacy Metrics Categories

### Distance-Based Metrics

These metrics measure the distance between a record and its closest match in the dataset, quantifying how uniquely identifiable a record is.

**DCR (Distance to Closest Record)**

Purpose: Measure vulnerability to linkage attacks

- **Definition**: Minimum Euclidean distance from a record to any other record
- **Interpretation**: Lower values = higher privacy risk (easier to link)
- **Range**: [0, ∞)
- **Risk Level**:
  - DCR < 0.1: Very high risk
  - DCR 0.1-0.5: High risk
  - DCR 0.5-2.0: Medium risk
  - DCR > 2.0: Low risk

**Use Cases:**
- Assess linkage attack vulnerability
- Evaluate record-level privacy
- Set disclosure controls

**NNDR (Nearest Neighbor Distance Ratio)**

Purpose: Measure record distinguishability

- **Definition**: Ratio of distance to 2nd NN / distance to 1st NN
- **Interpretation**: Higher values = more privacy (less distinguishable)
- **Range**: [1, ∞)
- **Interpretation**:
  - NNDR ≈ 1: Record is very distinctive
  - NNDR 1-2: Moderate distinguishability
  - NNDR > 2: Low distinguishability (more private)

**Use Cases:**
- Identify distinctive records
- Assess anonymization effectiveness
- Find records needing additional protection

### Uniqueness-Based Metrics

These metrics evaluate whether records can be distinguished based on combinations of quasi-identifier values.

**UNIQUENESS**

Purpose: Measure re-identification risk

- **Definition**: Proportion of records that are unique in the dataset
- **Interpretation**: Higher = higher privacy risk
- **Range**: [0, 1] (0-100%)
- **Risk Assessment**:
  - 0% unique: Perfect k-anonymity (k=n/groups)
  - <5% unique: Good privacy
  - 5-20% unique: Moderate privacy
  - >20% unique: Poor privacy

**Use Cases:**
- Quick privacy assessment
- Identify unique records
- Set minimum k for k-anonymity

**K_ANONYMITY**

Purpose: Ensure indistinguishability

- **Definition**: Each combination of quasi-identifiers appears ≥k times
- **Interpretation**: Higher k = stronger privacy guarantee
- **Common k values**:
  - k=2: Very weak (pair indistinguishability)
  - k=5: Minimum reasonable
  - k=10: Standard recommendation
  - k=100+: Strong privacy (for sensitive data)

**Guarantees:** An adversary cannot single out an individual with probability >1/k

**Use Cases:**
- Regulatory compliance (GDPR, HIPAA often require k=5)
- Standard anonymization requirement
- Group-level privacy guarantee

**L_DIVERSITY**

Purpose: Prevent sensitive attribute inference

- **Definition**: Each equivalence class (group) has ≥l distinct sensitive values
- **Interpretation**: Higher l = more diverse sensitive attributes
- **Common l values**:
  - l=2: Two distinct values (basic)
  - l=3-5: Reasonable diversity
  - l=10+: Strong diversity

**Prevents:** Linkage attacks even when quasi-identifiers are known

**Use Cases:**
- Protect against attribute inference
- Handle sensitive attributes (medical, financial)
- Stronger than k-anonymity

**T_CLOSENESS**

Purpose: Hide overall distribution of sensitive attributes

- **Definition**: Maximum difference between overall and group distributions ≤t
- **Interpretation**: Lower t = closer distributions (more privacy)
- **Common t values**:
  - t=0.05: Tight (5% difference)
  - t=0.1: Standard (10% difference)
  - t=0.2: Loose (20% difference)

**Prevents:** Inference attacks based on group statistics

**Use Cases:**
- Strongest privacy guarantee
- Sensitive numeric attributes
- Prevent statistical inference

## Metric Descriptions

### Privacy Guarantee Strength

```
Weakest              Medium              Strongest
├─ Uniqueness
├─ K-Anonymity (k=2)
├─ K-Anonymity (k=10)
├─ L-Diversity (l=2)
├─ L-Diversity (l=5)
├─ DCR + NNDR
└─ T-Closeness (t=0.05)
```

### Computational Complexity

| Metric | Time | Space | Notes |
|--------|------|-------|-------|
| DCR | O(n²) | O(n) | All pairwise distances |
| NNDR | O(n²) | O(n) | Find nearest neighbors |
| UNIQUENESS | O(n) | O(n) | Count unique combinations |
| K_ANONYMITY | O(n) | O(n) | Equivalence class grouping |
| L_DIVERSITY | O(n) | O(n) | Diversity within groups |
| T_CLOSENESS | O(n) | O(n) | Distribution comparison |

## Usage Examples

### Basic Enum Usage

```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

# Access members
metric_dcr = PrivacyMetricsType.DCR
metric_k_anon = PrivacyMetricsType.K_ANONYMITY

# Get string value
print(metric_dcr.value)  # "dcr"
print(metric_k_anon.value)  # "k_anonymity"

# Compare
if metric_k_anon == PrivacyMetricsType.K_ANONYMITY:
    print("Using k-anonymity assessment")
```

### Privacy Assessment Workflow

```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

def assess_privacy_preservation(
    original_df,
    anonymized_df,
    quasi_identifiers: list,
    sensitive_attrs: list
) -> dict:
    """
    Assess privacy with multiple metrics.
    """
    assessment = {}

    # Distance-based metrics
    assessment[PrivacyMetricsType.DCR.value] = compute_dcr(
        anonymized_df,
        quasi_identifiers
    )
    assessment[PrivacyMetricsType.NNDR.value] = compute_nndr(
        anonymized_df,
        quasi_identifiers
    )

    # Uniqueness-based metrics
    assessment[PrivacyMetricsType.UNIQUENESS.value] = compute_uniqueness(
        anonymized_df,
        quasi_identifiers
    )
    assessment[PrivacyMetricsType.K_ANONYMITY.value] = compute_k_anonymity(
        anonymized_df,
        quasi_identifiers
    )
    assessment[PrivacyMetricsType.L_DIVERSITY.value] = compute_l_diversity(
        anonymized_df,
        quasi_identifiers,
        sensitive_attrs
    )
    assessment[PrivacyMetricsType.T_CLOSENESS.value] = compute_t_closeness(
        original_df,
        anonymized_df,
        quasi_identifiers,
        sensitive_attrs
    )

    return assessment

# Usage
results = assess_privacy_preservation(
    original_df,
    anonymized_df,
    quasi_identifiers=["age", "zipcode", "gender"],
    sensitive_attrs=["salary", "disease"]
)

print("Privacy Assessment:")
for metric, value in results.items():
    print(f"  {metric}: {value}")
```

### Select Metric by Risk Profile

```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

def select_privacy_metrics(risk_level: str) -> list:
    """Select appropriate metrics based on risk level."""

    selections = {
        "low_risk": [
            PrivacyMetricsType.UNIQUENESS
        ],
        "medium_risk": [
            PrivacyMetricsType.K_ANONYMITY,
            PrivacyMetricsType.DCR
        ],
        "high_risk": [
            PrivacyMetricsType.K_ANONYMITY,
            PrivacyMetricsType.L_DIVERSITY,
            PrivacyMetricsType.DCR,
            PrivacyMetricsType.NNDR
        ],
        "maximum_privacy": [
            PrivacyMetricsType.K_ANONYMITY,
            PrivacyMetricsType.L_DIVERSITY,
            PrivacyMetricsType.T_CLOSENESS,
            PrivacyMetricsType.DCR,
            PrivacyMetricsType.NNDR,
            PrivacyMetricsType.UNIQUENESS
        ]
    }

    return selections.get(risk_level, [PrivacyMetricsType.K_ANONYMITY])

# Usage
metrics = select_privacy_metrics("high_risk")
print(f"Selected metrics: {[m.value for m in metrics]}")
```

### Compliance-Based Metric Selection

```python
from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

def get_compliance_metrics(regulation: str) -> dict:
    """Get required metrics for regulatory compliance."""

    requirements = {
        "GDPR": {
            "required": [PrivacyMetricsType.K_ANONYMITY],
            "k_minimum": 5,
            "recommended": [PrivacyMetricsType.L_DIVERSITY]
        },
        "HIPAA": {
            "required": [PrivacyMetricsType.K_ANONYMITY],
            "k_minimum": 5,
            "recommended": [
                PrivacyMetricsType.L_DIVERSITY,
                PrivacyMetricsType.T_CLOSENESS
            ]
        },
        "CCPA": {
            "required": [PrivacyMetricsType.UNIQUENESS],
            "uniqueness_threshold": 0.05,
            "recommended": [PrivacyMetricsType.K_ANONYMITY]
        }
    }

    return requirements.get(regulation, {})

# Usage
gdpr_reqs = get_compliance_metrics("GDPR")
print(f"GDPR requires: {gdpr_reqs['required']}")
print(f"Minimum k: {gdpr_reqs['k_minimum']}")
```

## Best Practices

1. **Use Multiple Metrics for Complete Picture**
   ```python
   # Good - comprehensive assessment
   metrics = [
       PrivacyMetricsType.K_ANONYMITY,
       PrivacyMetricsType.L_DIVERSITY,
       PrivacyMetricsType.DCR
   ]

   # Avoid - single metric may miss risks
   metrics = [PrivacyMetricsType.K_ANONYMITY]
   ```

2. **Set Realistic Privacy Thresholds**
   ```python
   # Good - documented requirements
   requirements = {
       PrivacyMetricsType.K_ANONYMITY.value: {
           "minimum_k": 5,
           "recommended_k": 10
       },
       PrivacyMetricsType.L_DIVERSITY.value: {
           "minimum_l": 2
       }
   }

   # Avoid - arbitrary thresholds
   ```

3. **Match Metrics to Data Sensitivity**
   ```python
   # Good - sensitive data gets stronger metrics
   if sensitive_data:
       metrics = [
           PrivacyMetricsType.K_ANONYMITY,
           PrivacyMetricsType.L_DIVERSITY,
           PrivacyMetricsType.T_CLOSENESS
       ]
   ```

4. **Document Metric Selection Rationale**
   ```python
   """
   Privacy Metrics for Health Data:
   - K-Anonymity (k≥10): Indistinguishability requirement
   - L-Diversity: Prevent disease inference
   - DCR: Assess linkage vulnerability
   """
   ```

5. **Use Type Hints for Functions**
   ```python
   from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType

   def compute_privacy_metric(
       df,
       metric: PrivacyMetricsType
   ) -> float:
       """Compute specified privacy metric."""
       if metric == PrivacyMetricsType.K_ANONYMITY:
           return compute_k_anonymity(df)
       # ... handle other metrics ...
   ```

## Related Components

- **DistanceMetricType** (`pamola_core.common.enum.distance_metric_type`) - Distance calculations
- **UtilityMetricsType** (`pamola_core.common.enum.utility_metrics_type`) - Utility metrics
- **Metrics Module** (`pamola_core.metrics.privacy`) - Implements privacy calculations
- **Analysis** (`pamola_core.analysis`) - Privacy evaluation

## Comparison with Fidelity Metrics

| Aspect | Privacy Metrics | Fidelity Metrics |
|--------|-----------------|------------------|
| **Measures** | Risk of re-identification | Data utility preservation |
| **Goal** | Minimize privacy risk | Maximize data utility |
| **Tradeoff** | Privacy vs. Utility | Utility vs. Privacy |
| **Examples** | k-anonymity, DCR | KS test, KL divergence |

## Privacy-Utility Tradeoff

```
         Privacy Risk
              ↑
              |
        ╔═════╬═════╗
        ║   Trade    ║
        ║    off     ║
        ╚═════╬═════╝
              |
    Data Utility →
```

- Strong privacy guarantees → Low utility
- Weak privacy → High utility
- Goal: Balance both dimensions

## Summary

`PrivacyMetricsType` provides enumeration for 6 privacy metrics:

| Group | Metrics | Use |
|-------|---------|-----|
| **Distance-Based** | DCR, NNDR | Individual record privacy |
| **Uniqueness-Based** | Uniqueness, k-anonymity, l-diversity, t-closeness | Group-level privacy |

Choose metrics based on:
- Data sensitivity (health → stronger metrics)
- Regulatory requirements (GDPR, HIPAA)
- Risk tolerance
- Computational constraints
