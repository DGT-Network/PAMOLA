# Attacks Module Documentation

**Module:** `pamola_core.attacks`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Attack Types](#attack-types)
3. [Quick Comparison](#quick-comparison)
4. [Module Architecture](#module-architecture)
5. [Common Workflow](#common-workflow)
6. [Key Concepts](#key-concepts)
7. [Related Components](#related-components)

## Overview

The `attacks` module provides privacy risk simulation tools for testing anonymized data robustness. It implements multiple privacy breach scenarios to assess whether data anonymization is effective.

**Purpose:** Evaluate privacy risk by simulating real-world attacks that attempt to:
- Re-identify individuals (linkage attacks)
- Determine if a person was in the training dataset (membership inference)
- Infer hidden sensitive attributes (attribute inference)
- Find similarity between original and anonymized datasets

**Use Case:** After anonymization, run attacks to verify the privacy level achieved:
```python
from pamola_core.attacks import (
    LinkageAttack, MembershipInference, AttributeInference,
    DistanceToClosestRecord, NearestNeighborDistanceRatio, AttackMetrics
)

# Test if linkage attack can re-identify people
linkage = LinkageAttack()
matches = linkage.record_linkage_attack(original_data, anonymized_data, keys)

# Test membership inference
mia = MembershipInference()
predictions = mia.membership_inference_attack_dcr(training_data, test_data)
metrics = AttackMetrics().attack_metrics(y_true, predictions)
```

## Attack Types

| Attack | Purpose | Input | Output | Difficulty |
|--------|---------|-------|--------|------------|
| **Linkage** | Re-identify records by matching common attributes | 2 datasets + keys | Match pairs (DataFrame) | Medium |
| **Membership Inference** | Determine if someone was in training set | 2 datasets | Binary predictions (array) | Low-Medium |
| **Attribute Inference** | Infer missing sensitive attribute | 2 datasets + target field | Predicted values (Series) | Low |
| **Distance to Closest Record (DCR)** | Measure dataset dissimilarity using nearest neighbor distance | 2 datasets | Distances (array) | Low |
| **Nearest Neighbor Distance Ratio (NNDR)** | Ratio of 1st to 2nd nearest neighbor distance | 2 datasets | Ratios (array) | Low |

## Quick Comparison

### When to Use Each Attack

**Linkage Attack**
- Goal: Can attacker re-identify individuals?
- When: You have external datasets with overlapping attributes
- Risk: High if quasi-identifiers not removed

**Membership Inference (DCR)**
- Goal: Can attacker tell if someone was in training data?
- When: You suspect similarity between original and anonymized datasets
- Risk: Medium - DCR is a direct distance metric

**Membership Inference (NNDR)**
- Goal: Can attacker infer membership using neighborhood patterns?
- When: You want confidence-based membership detection
- Risk: Low-Medium - relies on nearest neighbor ratios

**Membership Inference (Model)**
- Goal: Can a trained model infer membership from feature patterns?
- When: You have clustered training data and want ML-based inference
- Risk: Low - confidence-based, not distance-based

**Attribute Inference**
- Goal: Can attacker guess missing sensitive attributes?
- When: Some attributes are partially visible in anonymized data
- Risk: Medium - uses entropy-based feature selection

**DCR / NNDR Metrics**
- Goal: Direct privacy metrics without binary prediction
- When: You want numerical risk scores
- Risk: Low - purely computational metrics

## Module Architecture

```
pamola_core.attacks/
├── base.py                          # AttackInitialization (abstract)
├── preprocess_data.py               # PreprocessData (TF-IDF + scaling)
├── linkage_attack.py                # LinkageAttack (3 variants)
├── membership_inference.py          # MembershipInference (DCR/NNDR/Model)
├── attribute_inference.py           # AttributeInference (entropy-based)
├── distance_to_closest_record.py    # DistanceToClosestRecord (DCR metric)
├── nearest_neighbor_distance_ratio.py # NearestNeighborDistanceRatio (NNDR metric)
├── attack_metrics.py                # AttackMetrics (evaluation metrics)
└── __init__.py                      # Public API exports
```

### Class Hierarchy
```
AttackInitialization (ABC)
  └─ PreprocessData
       ├─ LinkageAttack
       ├─ MembershipInference
       ├─ AttributeInference
       ├─ DistanceToClosestRecord
       ├─ NearestNeighborDistanceRatio
       └─ AttackMetrics
```

All attack classes inherit from `PreprocessData`, which provides TF-IDF vectorization and numeric scaling.

## Common Workflow

### 1. Setup
```python
from pamola_core.attacks import LinkageAttack, MembershipInference, AttackMetrics
import pandas as pd

# Load datasets
original = pd.read_csv("original.csv")
anonymized = pd.read_csv("anonymized.csv")
```

### 2. Run Attacks
```python
# Linkage: direct record matching
linkage = LinkageAttack()
matches = linkage.record_linkage_attack(original, anonymized, keys=["name", "dob"])

# Membership Inference: DCR variant
mia = MembershipInference()
predictions = mia.membership_inference_attack_dcr(original, anonymized)
```

### 3. Evaluate
```python
# Create ground truth (1 = member, 0 = non-member)
y_true = np.ones(len(original))  # All are members (in original)

# Compute metrics
metrics_obj = AttackMetrics()
results = metrics_obj.attack_metrics(y_true, predictions)
asr = metrics_obj.attack_success_rate(y_true, predictions)
rrs = metrics_obj.residual_risk_score(y_true, predictions)

print(f"Attack Accuracy: {results['Attack Accuracy']}")
print(f"Attack Success Rate: {asr}")
print(f"Residual Risk Score: {rrs}")
```

### 4. Interpret Results
- **High Attack Accuracy (>0.7)** → Anonymization is weak
- **Low Attack Accuracy (<0.55)** → Anonymization is strong
- **ASR > 0.5** → Many training members identified (risky)
- **RRS > 0.3** → Significant privacy risk

## Key Concepts

### Data Preprocessing
All attacks use `PreprocessData.preprocess_data()`:
- **Categorical columns** → TF-IDF vectorization (max 5000 features)
- **Numeric columns** → Standard scaling (zero mean, unit variance)
- **Result:** Combined numeric feature matrix for similarity/distance computation

### Distance Metrics
- **Euclidean:** Default for KDTree (straight-line distance)
- **Custom metric:** Supported via cdist method (cosine, manhattan, etc.)

### Thresholds
Most attacks use adaptive thresholds (median of metric values) if not specified:
```python
# Explicit threshold
mia = MembershipInference(dcr_threshold=0.5)

# Adaptive threshold (automatic)
mia = MembershipInference()  # Uses median(dcr_values)
```

### Feature Importance in Attacks
- **High-entropy features** → Less predictive (removed by AttributeInference)
- **Low-entropy features** → More predictive (used in AttributeInference)
- Entropy = -Σ(p * log2(p)) where p = probability of each value

## Related Components

- **[Profiling Module](../profiling/)** — Analyze dataset characteristics before anonymization
- **[Metrics Module](../metrics/)** — Compute privacy metrics (k-anonymity, l-diversity)
- **[Privacy Models](../privacy_models/)** — Model privacy risks at the data level
- **[Anonymization Module](../anonymization/)** — Apply transformations (masking, generalization)

## Export Reference

All public classes available via:
```python
from pamola_core.attacks import (
    # Base
    AttackInitialization,
    BaseAttack,

    # Core attacks
    LinkageAttack,
    MembershipInference,
    MembershipInferenceAttack,
    AttributeInference,
    AttributeInferenceAttack,
    DistanceToClosestRecord,
    DistanceToClosestRecordAttack,
    NearestNeighborDistanceRatio,
    NearestNeighborDistanceRatioAttack,

    # Metrics
    AttackMetrics,
)
```

## Next Steps

- **[Base Attack](./base_attack.md)** — Abstract interfaces
- **[Linkage Attack](./linkage_attack.md)** — Re-identification techniques
- **[Membership Inference](./membership_inference.md)** — Training set membership
- **[Attribute Inference](./attribute_inference.md)** — Hidden attribute inference
- **[Distance Metrics](./distance_to_closest_record.md)** — DCR computation
- **[Attack Metrics](./attack_metrics.md)** — Evaluation and scoring
