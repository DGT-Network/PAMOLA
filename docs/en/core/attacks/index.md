# Attacks Module Index

**Module:** `pamola_core.attacks`
**Location:** `d:/AIShowRoom/DGT-Network/PAMOLA/docs/en/core/attacks/`
**Status:** Complete Documentation (8 files)

## Quick Navigation

### Start Here
**→ [Attacks Overview](./attacks_overview.md)** — Module entry point with attack comparison table, architecture, and workflow

### Attack Classes

#### Re-Identification Attacks
- **[Linkage Attack](./linkage_attack.md)** — 3 record linkage methods
  - `record_linkage_attack()` — Exact matching
  - `probabilistic_linkage_attack()` — Fuzzy matching (Fellegi-Sunter)
  - `cluster_vector_linkage_attack()` — PCA + cosine similarity

#### Membership Inference Attacks
- **[Membership Inference](./membership_inference.md)** — 3 membership detection methods
  - `membership_inference_attack_dcr()` — Distance-based
  - `membership_inference_attack_nndr()` — Ratio-based
  - `membership_inference_attack_model()` — Confidence-based

#### Attribute Attacks
- **[Attribute Inference](./attribute_inference.md)** — Sensitive attribute guessing
  - `attribute_inference_attack()` — Entropy-based feature selection

### Metric Classes

#### Distance Metrics
- **[Distance to Closest Record (DCR)](./distance_to_closest_record.md)** — Dataset dissimilarity metric
  - `calculate_dcr()` — Nearest neighbor distance computation

- **[Nearest Neighbor Distance Ratio (NNDR)](./nearest_neighbor_distance_ratio.md)** — Confidence metric
  - `calculate_nndr()` — Ratio-based nearest neighbor analysis

#### Evaluation
- **[Attack Metrics](./attack_metrics.md)** — Attack effectiveness evaluation
  - `attack_metrics()` — Comprehensive metrics (accuracy, precision, recall, F1, AUC, advantage)
  - `attack_success_rate()` — ASR metric
  - `residual_risk_score()` — RRS metric

### Foundation
- **[Base Attack](./base_attack.md)** — Abstract interfaces
  - `AttackInitialization` class
  - `PreprocessData` helper class

## By Task

### I need to test privacy of anonymized data
1. Start with [Attacks Overview](./attacks_overview.md) — understand threat model
2. Choose attack type based on your data:
   - Exact/deterministic attributes → [Linkage Attack](./linkage_attack.md) → `record_linkage_attack()`
   - Noisy/real-world attributes → [Linkage Attack](./linkage_attack.md) → `probabilistic_linkage_attack()`
   - Multi-dimensional data → [Linkage Attack](./linkage_attack.md) → `cluster_vector_linkage_attack()`
3. Measure risk with metrics:
   - Baseline similarity → [DCR](./distance_to_closest_record.md)
   - Confidence in matches → [NNDR](./nearest_neighbor_distance_ratio.md)
4. Evaluate success → [Attack Metrics](./attack_metrics.md)

### I need to detect membership inference risk
1. Read [Membership Inference](./membership_inference.md) overview
2. Choose method:
   - Simple distance-based → `membership_inference_attack_dcr()`
   - Confidence-based → `membership_inference_attack_nndr()`
   - ML-based → `membership_inference_attack_model()`
3. Get ground truth (know who should be members)
4. Evaluate with [Attack Metrics](./attack_metrics.md):
   - `attack_metrics()` — Overall performance
   - `attack_success_rate()` — Can you find members?
   - `residual_risk_score()` — How much better than random?

### I need to infer hidden attributes
1. Read [Attribute Inference](./attribute_inference.md) — understand entropy-based selection
2. Run `attribute_inference_attack()` with:
   - Training data (complete attributes)
   - Test data (partial attributes)
   - Target attribute name (to infer)
3. Evaluate prediction accuracy with standard ML metrics

### I'm implementing a custom attack
1. Read [Base Attack](./base_attack.md) — understand interface
2. Inherit from `PreprocessData` (not `AttackInitialization`)
3. Implement required methods:
   - `process()` — your attack logic
   - `preprocess_data()` — inherited from parent
4. Use [Attack Metrics](./attack_metrics.md) to evaluate results

### I need to understand a metric
- **DCR** → [Distance to Closest Record](./distance_to_closest_record.md) — "How far is the test record from nearest training record?"
- **NNDR** → [Nearest Neighbor Distance Ratio](./nearest_neighbor_distance_ratio.md) — "Is this record unique (1st/2nd neighbor ratio)?"
- **Accuracy** → [Attack Metrics](./attack_metrics.md) → "Overall correctness of attack predictions"
- **ASR** → [Attack Metrics](./attack_metrics.md) → "What % of actual members were identified?"
- **RRS** → [Attack Metrics](./attack_metrics.md) → "Is attack better than random guessing?"

## Attack Selection Guide

### Choose Attack Type

| If You Have | Best Attack | Alternative |
|------------|-------------|-------------|
| Original + anonymized datasets | Linkage | DCR/NNDR metrics |
| Clear quasi-identifiers | Record Linkage | Probabilistic Linkage |
| Noisy/misspelled data | Probabilistic Linkage | Record Linkage |
| High-dimensional data | CVPLA Linkage | NNDR metric |
| Training + test sets | Membership Inference | DCR/NNDR |
| Ground truth membership | Model-based MIA | DCR/NNDR |
| Partial sensitive attributes | Attribute Inference | None (use together) |
| Need baseline metric | DCR | NNDR |
| Need confidence scores | NNDR | DCR |
| Need to compare methods | Attack Metrics | All metrics |

### Risk Level Assessment

**HIGH RISK** (Need stronger anonymization):
- Linkage attack finds many matches
- Membership inference ASR > 0.7
- Attribute inference accuracy > 0.8
- DCR mean < 1.0
- NNDR mean < 0.5
- RRS > 0.3

**MEDIUM RISK** (Consider improvements):
- Linkage attack finds some matches
- Membership inference ASR 0.5-0.7
- Attribute inference accuracy 0.6-0.8
- DCR mean 1.0-3.0
- NNDR mean 0.5-0.8
- RRS 0.1-0.3

**LOW RISK** (Good anonymization):
- Linkage attack finds few/no matches
- Membership inference ASR < 0.5
- Attribute inference accuracy < 0.6
- DCR mean > 3.0
- NNDR mean > 0.8
- RRS < 0.1

## Code Examples By Topic

### Basic Usage
```python
from pamola_core.attacks import (
    LinkageAttack, MembershipInference, AttributeInference,
    DistanceToClosestRecord, AttackMetrics
)
import pandas as pd

# Load data
original = pd.read_csv('original.csv')
anonymized = pd.read_csv('anonymized.csv')

# Run attacks
linkage = LinkageAttack()
matches = linkage.record_linkage_attack(original, anonymized, ['age', 'city'])

mia = MembershipInference()
predictions = mia.membership_inference_attack_dcr(original, anonymized)

# Evaluate
metrics = AttackMetrics()
results = metrics.attack_metrics(y_true, predictions)
print(f"Attack Accuracy: {results['Attack Accuracy']}")
```

### Complete Workflow
See [Membership Inference](./membership_inference.md#complete-membership-inference-workflow) for 40+ line example with:
- Dataset creation
- Multiple attack methods
- Evaluation with all metrics
- Risk classification

### Threshold Optimization
See [Attack Metrics](./attack_metrics.md#threshold-optimization) for example showing:
- Try multiple threshold values
- Compare accuracy across thresholds
- Find optimal for your privacy goals

## Import Reference

```python
# Base classes
from pamola_core.attacks import AttackInitialization, BaseAttack

# Core attacks
from pamola_core.attacks import LinkageAttack
from pamola_core.attacks import MembershipInference, MembershipInferenceAttack
from pamola_core.attacks import AttributeInference, AttributeInferenceAttack

# Metrics
from pamola_core.attacks import DistanceToClosestRecord, DistanceToClosestRecordAttack
from pamola_core.attacks import NearestNeighborDistanceRatio, NearestNeighborDistanceRatioAttack
from pamola_core.attacks import AttackMetrics

# All at once
from pamola_core.attacks import (
    AttackInitialization,
    LinkageAttack,
    MembershipInference,
    AttributeInference,
    DistanceToClosestRecord,
    NearestNeighborDistanceRatio,
    AttackMetrics,
)
```

## Common Workflows

### Privacy Risk Assessment (Complete)
1. **Baseline Metrics** → [DCR](./distance_to_closest_record.md)
2. **Re-identification Test** → [Linkage Attack](./linkage_attack.md)
3. **Membership Inference** → [Membership Inference](./membership_inference.md)
4. **Attribute Risk** → [Attribute Inference](./attribute_inference.md)
5. **Comprehensive Evaluation** → [Attack Metrics](./attack_metrics.md)

### Quick Privacy Check (5 minutes)
1. Compute [DCR](./distance_to_closest_record.md) → Check mean distance
2. Run [Membership Inference](./membership_inference.md#quick-usage) → Check ASR
3. Calculate [RRS](./attack_metrics.md#residual-risk-score) → Assess risk

### Detailed Privacy Analysis (30 minutes)
1. Profile data with [Profiling Module](../profiling/)
2. Run all attacks from [Attacks Overview](./attacks_overview.md)
3. Create comparison table (see [Membership Inference](./membership_inference.md#multi-method-comparison))
4. Generate risk report with metrics from [Attack Metrics](./attack_metrics.md)

### Anonymization Validation
1. Apply anonymization technique
2. Run baseline attacks (Linkage, DCR)
3. Check metrics improve
4. Iterate until risk acceptable

## Troubleshooting Quick Links

- **"No matches found"** → [Linkage Attack Troubleshooting](./linkage_attack.md#troubleshooting)
- **"All predictions are 0"** → [Membership Inference Troubleshooting](./membership_inference.md#troubleshooting)
- **"Threshold issues"** → [Attack Metrics Troubleshooting](./attack_metrics.md#troubleshooting)
- **"Empty array returned"** → [DCR Troubleshooting](./distance_to_closest_record.md#troubleshooting)
- **"Performance too slow"** → [Linkage Attack Performance](./linkage_attack.md#troubleshooting) or [DCR Performance](./distance_to_closest_record.md#troubleshooting)

## Related Documentation

- **[Profiling Module](../profiling/)** — Analyze data before anonymization
- **[Metrics Module](../metrics/)** — Privacy metrics (k-anonymity, l-diversity)
- **[Anonymization Module](../anonymization/)** — Apply protective transformations
- **[Privacy Models](../privacy_models/)** — Theoretical privacy foundations

## Version Information

- **Module Version:** 1.0
- **Documentation Version:** 1.0
- **Last Updated:** 2026-03-23
- **Python Support:** 3.10+
- **Dependencies:** numpy, pandas, scipy, scikit-learn, recordlinkage

## File Summary

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| attacks_overview.md | ~300 | Module entry point | All classes overview |
| base_attack.md | ~200 | Abstract interfaces | AttackInitialization |
| linkage_attack.md | ~400 | Re-identification | LinkageAttack (3 methods) |
| membership_inference.md | ~500 | Membership detection | MembershipInference (3 methods) |
| attribute_inference.md | ~350 | Attribute inference | AttributeInference |
| distance_to_closest_record.md | ~450 | Distance metric | DistanceToClosestRecord |
| nearest_neighbor_distance_ratio.md | ~500 | Confidence metric | NearestNeighborDistanceRatio |
| attack_metrics.md | ~500 | Evaluation | AttackMetrics (3 methods) |
| **TOTAL** | **~3,200** | **Complete module** | **8 classes, 15+ methods** |

---

**Questions?** Check relevant documentation file's [Troubleshooting](#troubleshooting) or [Best Practices](#best-practices) sections.

**Getting Started?** Start with [Attacks Overview](./attacks_overview.md).

**Need Examples?** Each documentation file includes 4-5+ detailed, runnable examples.
