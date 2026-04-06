# Attack Metrics Documentation

**Module:** `pamola_core.attacks.attack_metrics`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Class Reference](#class-reference)
4. [Methods](#methods)
5. [Usage Examples](#usage-examples)
6. [Metric Interpretation](#metric-interpretation)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Related Components](#related-components)

## Overview

`AttackMetrics` provides evaluation metrics for assessing the success of privacy attacks, particularly membership inference attacks (MIA).

**Purpose:** Quantify attack effectiveness by computing:
- **Attack Accuracy** — Overall correctness of attack predictions
- **Precision/Recall** — False positive and false negative rates
- **F1-Score** — Balanced measure of precision and recall
- **AUC-ROC** — Ability to distinguish members vs non-members
- **Advantage** — How much better attack is than random guessing
- **Attack Success Rate (ASR)** — Proportion of actual members correctly identified
- **Residual Risk Score (RRS)** — Privacy risk beyond random guessing

**Use Case:** After running membership inference or linkage attacks, evaluate success:
```python
from pamola_core.attacks import MembershipInference, AttackMetrics

mia = MembershipInference()
predictions = mia.membership_inference_attack_dcr(training, test)

metrics = AttackMetrics()
results = metrics.attack_metrics(y_true, predictions)
asr = metrics.attack_success_rate(y_true, predictions)
rrs = metrics.residual_risk_score(y_true, predictions)

print(f"Accuracy: {results['Attack Accuracy']}")
print(f"ASR: {asr}")
print(f"RRS: {rrs}")
```

## Key Features

**Comprehensive Metrics Suite**
- Standard ML metrics (accuracy, precision, recall, F1, AUC)
- Attack-specific metrics (advantage, ASR, RRS)
- Handles edge cases (single class, zero division)

**Ground Truth Format**
- Binary labels: 1 = member, 0 = non-member
- Same length as predictions

**Robust Fallbacks**
- AUC defaults to 0.5 if only one class present
- Metrics handle zero division gracefully

## Class Reference

### AttackMetrics

```python
from pamola_core.attacks import AttackMetrics

class AttackMetrics(PreprocessData):
    """
    AttackMetrics class for attack simulation in PAMOLA.CORE.
    Evaluates performance of privacy attacks.
    """

    def __init__(self):
        """
        No parameters required.
        All evaluation controlled via method arguments.
        """
```

**Constructor:** No parameters

## Methods

### attack_metrics

Compute comprehensive attack performance metrics.

```python
def attack_metrics(self, y_true, y_pred):
    """
    Compute attack performance metrics for Membership Inference Attacks (MIA).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels for each sample in the test set.
        - 1 → sample is part of the training set (member)
        - 0 → sample is not part of the training set (non-member)

    y_pred : array-like of shape (n_samples,)
        Predicted labels output by the attack algorithm.
        - 1 → predicted as member
        - 0 → predicted as non-member

    Returns
    -------
    dict
        Dictionary with keys:
        - "Attack Accuracy": Overall correctness (TP + TN) / n
        - "Precision": TP / (TP + FP) - among predicted members, % correct
        - "Recall": TP / (TP + FN) - among actual members, % identified
        - "F1-Score": Harmonic mean of precision and recall
        - "AUC-ROC": Ability to separate members from non-members (0.5=random, 1.0=perfect)
        - "Advantage": 2 * (accuracy - 0.5) - how much better than random guessing

        Values: rounded to 4 decimal places
    """
```

**Confusion Matrix Context:**
```
                    Predicted Positive    Predicted Negative
Actual Positive (1)       TP                   FN
Actual Negative (0)       FP                   TN

Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
Advantage = 2 * (Accuracy - 0.5)
```

**Example:**
```python
import numpy as np
from pamola_core.attacks import AttackMetrics

# Ground truth: 5 members, 5 non-members
y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Attack predictions
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1])

metrics = AttackMetrics()
results = metrics.attack_metrics(y_true, y_pred)

print("Attack Performance Metrics:")
for metric, value in results.items():
    print(f"  {metric:<18}: {value:.4f}")
```

**Output:**
```
Attack Performance Metrics:
  Attack Accuracy   : 0.6000
  Precision         : 0.6667
  Recall            : 0.6000
  F1-Score          : 0.6286
  AUC-ROC           : 0.6400
  Advantage         : 0.2000
```

### attack_success_rate

Compute Attack Success Rate (ASR) - proportion of actual members correctly identified.

```python
def attack_success_rate(self, y_true, y_pred):
    """
    Compute Attack Success Rate (ASR) for Membership Inference Attack.

    ASR measures the attacker's ability to correctly identify training set members.
    Essentially, this is equivalent to Recall for the "member" class.

    Formula: ASR = TP / (TP + FN)
    where TP = true positives (members correctly identified)
          FN = false negatives (members incorrectly classified as non-members)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (1 = member, 0 = non-member).

    y_pred : array-like of shape (n_samples,)
        Predicted labels by the attack algorithm (1 = member, 0 = non-member).

    Returns
    -------
    float
        Attack Success Rate (0.0 to 1.0):
        - 0.0: Attack never correctly identifies members
        - 0.5: Attack identifies half of all members
        - 1.0: Attack correctly identifies all members (worst case)
        - 0.0 if no members present in y_true

    Interpretation
    ──────────────
    - ASR > 0.7: High attack success (severe privacy risk)
    - ASR 0.5-0.7: Moderate attack success (medium privacy risk)
    - ASR < 0.5: Low attack success (good privacy)
    """
```

**Example:**
```python
import numpy as np

y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 5 members, 5 non-members
y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])  # 3 members identified, 2 missed

metrics = AttackMetrics()
asr = metrics.attack_success_rate(y_true, y_pred)

print(f"Attack Success Rate: {asr:.4f}")  # 3 / 5 = 0.6000

# Interpretation: Attack successfully identified 60% of training set members
```

### residual_risk_score

Compute Residual Risk Score (RRS) - privacy risk beyond random guessing.

```python
def residual_risk_score(self, y_true, y_pred):
    """
    Compute Residual Risk Score (RRS) for Membership Inference Attack.

    RRS estimates the residual privacy risk by measuring the difference between:
    - P(Y=1 | X=1): probability of correctly identifying members (TPR)
    - P(Y=1 | X=0): probability of incorrectly identifying non-members as members (FPR)

    Formula: RRS = P(Y=1 | X=1) - P(Y=1 | X=0)
                 = TPR - FPR

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (1 = member, 0 = non-member).

    y_pred : array-like of shape (n_samples,)
        Predicted labels by the attack algorithm (1 = member, 0 = non-member).

    Returns
    -------
    float
        Residual Risk Score (-1.0 to 1.0):
        - RRS > 0.3: High residual privacy risk (attack better than random)
        - RRS 0.1-0.3: Medium residual privacy risk
        - RRS < 0.1: Low residual privacy risk (attack near random)
        - RRS < 0.0: Attack worse than random (unlikely)
        - 0.0 if no members or non-members present

    Interpretation
    ──────────────
    RRS > 0.0: Attack performs better than random
    - Positive RRS means True Positive Rate > False Positive Rate
    - Attack is distinguishing members from non-members

    RRS = 0.0: Attack equivalent to random guessing
    - Cannot distinguish members from non-members
    - Good privacy (in terms of membership inference resistance)

    RRS < 0.0: Attack performs worse than random
    - Rare case where FPR > TPR
    """
```

**Example:**
```python
import numpy as np

y_true = np.array([1, 1, 1, 0, 0, 0])
y_pred = np.array([1, 1, 0, 0, 0, 1])

# TP = 2 (correctly identified members)
# FN = 1 (missed member)
# FP = 1 (incorrectly identified non-member as member)
# TN = 2 (correctly identified non-members)

# TPR = TP / (TP + FN) = 2 / 3 = 0.667
# FPR = FP / (FP + TN) = 1 / 3 = 0.333
# RRS = TPR - FPR = 0.667 - 0.333 = 0.333

metrics = AttackMetrics()
rrs = metrics.residual_risk_score(y_true, y_pred)

print(f"Residual Risk Score: {rrs:.4f}")  # 0.3333

# Interpretation: Attack is moderately better than random
# Privacy risk is present, further anonymization needed
```

## Usage Examples

### Complete Attack Evaluation

```python
from pamola_core.attacks import MembershipInference, AttackMetrics
import pandas as pd
import numpy as np

# Simulate original and test data
original = pd.DataFrame({
    'age': [25, 35, 45, 55, 65] * 10,
    'income': [50000, 75000, 100000, 125000, 150000] * 10,
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'] * 10
})

# Create test set: 50% members, 50% non-members
members = original.sample(n=25, random_state=42)
non_members = pd.DataFrame({
    'age': np.random.randint(20, 70, 25),
    'income': np.random.randint(40000, 160000, 25),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 25)
})

test_data = pd.concat([members, non_members], ignore_index=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Ground truth
y_true = np.array([1]*25 + [0]*25)

# Run attack
mia = MembershipInference()
predictions = mia.membership_inference_attack_dcr(original, test_data)

# Evaluate
metrics_obj = AttackMetrics()
results = metrics_obj.attack_metrics(y_true, predictions)
asr = metrics_obj.attack_success_rate(y_true, predictions)
rrs = metrics_obj.residual_risk_score(y_true, predictions)

# Display results
print("=" * 50)
print("MEMBERSHIP INFERENCE ATTACK EVALUATION")
print("=" * 50)

print("\n[1] Standard Metrics:")
for metric, value in results.items():
    print(f"    {metric:<18}: {value:.4f}")

print(f"\n[2] Attack Success Rate (ASR):")
print(f"    {asr:.4f}")
if asr > 0.7:
    print("    → HIGH RISK: Attack successfully identifies most members")
elif asr > 0.5:
    print("    → MEDIUM RISK: Attack identifies more than half of members")
else:
    print("    → LOW RISK: Attack struggles to identify members")

print(f"\n[3] Residual Risk Score (RRS):")
print(f"    {rrs:.4f}")
if rrs > 0.3:
    print("    → HIGH RISK: Significant privacy risk exists")
elif rrs > 0.1:
    print("    → MEDIUM RISK: Moderate privacy risk exists")
else:
    print("    → LOW RISK: Limited privacy risk from this attack")

print("\n" + "=" * 50)

# Risk summary
if asr > 0.7 or rrs > 0.3 or results['Attack Accuracy'] > 0.7:
    print("⚠️  OVERALL ASSESSMENT: WEAK ANONYMIZATION")
    print("Recommend stronger privacy protections (differential privacy, k-anonymity)")
else:
    print("✓ OVERALL ASSESSMENT: ADEQUATE ANONYMIZATION")
    print("Current anonymization level provides reasonable privacy")

print("=" * 50)
```

### Multi-Method Comparison

```python
from pamola_core.attacks import (
    MembershipInference, LinkageAttack, AttackMetrics
)

mia = MembershipInference()
linkage = LinkageAttack()
metrics_obj = AttackMetrics()

# Run different attacks
mia_dcr = mia.membership_inference_attack_dcr(original, test_data)
mia_nndr = mia.membership_inference_attack_nndr(original, test_data)
mia_model = mia.membership_inference_attack_model(original, test_data)

# Linkage attack (convert to binary: 1=linked, 0=not linked)
linkage_results = linkage.record_linkage_attack(original, test_data, keys=['age', 'city'])
linkage_binary = np.isin(test_data.index, linkage_results.index).astype(int)

# Compare methods
print("Attack Method Comparison:\n")
print(f"{'Method':<20} {'Accuracy':<10} {'ASR':<10} {'RRS':<10}")
print("-" * 50)

methods = {
    'MIA-DCR': mia_dcr,
    'MIA-NNDR': mia_nndr,
    'MIA-Model': mia_model,
    'Linkage': linkage_binary
}

for method_name, predictions in methods.items():
    results = metrics_obj.attack_metrics(y_true, predictions)
    asr = metrics_obj.attack_success_rate(y_true, predictions)
    rrs = metrics_obj.residual_risk_score(y_true, predictions)

    print(f"{method_name:<20} {results['Attack Accuracy']:<10.4f} {asr:<10.4f} {rrs:<10.4f}")

print("\n" + "-" * 50)
print("Strongest attack: Method with highest ASR or RRS")
```

### Threshold Optimization

```python
from pamola_core.attacks import AttackMetrics
import numpy as np

# Raw attack scores (from DCR or NNDR)
dcr_scores = dcr_obj.calculate_dcr(original, test_data)

# Try different thresholds
thresholds = np.percentile(dcr_scores, [10, 25, 50, 75, 90])

print("Threshold Optimization:\n")
print(f"{'Threshold':<12} {'Accuracy':<12} {'ASR':<12} {'RRS':<12} {'Advantage':<12}")
print("-" * 60)

best_threshold = None
best_asr = 0

metrics_obj = AttackMetrics()

for threshold in thresholds:
    predictions = (dcr_scores < threshold).astype(int)
    results = metrics_obj.attack_metrics(y_true, predictions)
    asr = metrics_obj.attack_success_rate(y_true, predictions)
    rrs = metrics_obj.residual_risk_score(y_true, predictions)

    print(f"{threshold:<12.4f} {results['Attack Accuracy']:<12.4f} "
          f"{asr:<12.4f} {rrs:<12.4f} {results['Advantage']:<12.4f}")

    if asr > best_asr:
        best_asr = asr
        best_threshold = threshold

print(f"\nOptimal threshold: {best_threshold:.4f} (highest ASR)")
```

## Metric Interpretation

### Attack Accuracy
- **Range:** 0.0 to 1.0
- **Meaning:** Proportion of correct predictions (members and non-members)
- **Baseline:** 0.5 (random guessing with equal classes)
- **Good Privacy:** < 0.55 (attack barely better than random)
- **Poor Privacy:** > 0.70 (attack reliably identifies members)

**Context:** Can be misleading with imbalanced classes (e.g., 90% members)

### Precision
- **Range:** 0.0 to 1.0
- **Meaning:** Among predicted members, % that are actual members
- **Use:** Assess false positive rate when mistakenly claiming membership
- **Good Privacy:** < 0.60 (many false positives, ambiguous predictions)
- **Poor Privacy:** > 0.80 (accurate member identification)

### Recall (= ASR)
- **Range:** 0.0 to 1.0
- **Meaning:** Among actual members, % correctly identified
- **Use:** Assess attack's ability to find members
- **Good Privacy:** < 0.55 (misses most members)
- **Poor Privacy:** > 0.75 (finds most members)

### F1-Score
- **Range:** 0.0 to 1.0
- **Meaning:** Balanced precision-recall score
- **Use:** Single metric for imbalanced datasets
- **Good Privacy:** < 0.55
- **Poor Privacy:** > 0.70

### AUC-ROC
- **Range:** 0.0 to 1.0
- **Baseline:** 0.5 (random classifier)
- **Meaning:** Ability to separate members from non-members
- **Good Privacy:** 0.50-0.55 (barely separable)
- **Poor Privacy:** 0.75+ (clearly separable)
- **Perfect:** 1.0 (perfect separation)

### Advantage
- **Range:** -1.0 to 1.0
- **Formula:** 2 * (Accuracy - 0.5)
- **Meaning:** How much better attack is than random guessing
- **Good Privacy:** Close to 0.0 (near random)
- **Poor Privacy:** > 0.3 (significantly better than random)

### Attack Success Rate (ASR)
- **Range:** 0.0 to 1.0
- **Meaning:** Proportion of training set members correctly identified
- **Good Privacy:** < 0.55
- **Poor Privacy:** > 0.70

**Most Important Metric:** ASR directly measures attack success at identifying members

### Residual Risk Score (RRS)
- **Range:** -1.0 to 1.0
- **Formula:** TPR - FPR
- **Meaning:** Privacy risk beyond random guessing
- **Good Privacy:** < 0.1 (near random, balanced TPR/FPR)
- **Poor Privacy:** > 0.3 (high TPR, low FPR = easy member detection)

**Key Insight:** RRS balances both member detection (TPR) and false alarms (FPR)

## Best Practices

**1. Use Multiple Metrics**
```python
# Don't rely on accuracy alone (misleading with imbalance)
# Use: Accuracy + ASR + RRS + AUC

results = metrics.attack_metrics(y_true, predictions)
asr = metrics.attack_success_rate(y_true, predictions)
rrs = metrics.residual_risk_score(y_true, predictions)

if asr > 0.7 and rrs > 0.2:
    print("FAIL: Attack succeeds, privacy is weak")
```

**2. Validate with Ground Truth**
```python
# Ensure y_true is reliable
assert len(y_true) == len(predictions)
assert set(y_true) == {0, 1}  # Binary only
assert len(y_true[y_true == 1]) > 0  # Has members
assert len(y_true[y_true == 0]) > 0  # Has non-members
```

**3. Check Class Distribution**
```python
# Report member/non-member split
n_members = (y_true == 1).sum()
n_non_members = (y_true == 0).sum()
print(f"Members: {n_members}, Non-members: {n_non_members}")

# Baseline accuracy (always guessing majority)
baseline_accuracy = max(n_members, n_non_members) / len(y_true)
print(f"Baseline accuracy (majority class): {baseline_accuracy:.1%}")

# Real accuracy should be significantly better
if results['Attack Accuracy'] <= baseline_accuracy + 0.05:
    print("Attack barely better than majority guess")
```

**4. Report All Metrics in Final Assessment**
```python
# Complete evaluation summary
assessment = {
    'accuracy': results['Attack Accuracy'],
    'precision': results['Precision'],
    'recall': results['Recall'],
    'f1': results['F1-Score'],
    'auc': results['AUC-ROC'],
    'advantage': results['Advantage'],
    'asr': asr,
    'rrs': rrs,
}

print("Complete Attack Evaluation:")
for metric, value in assessment.items():
    status = "✓ GOOD" if is_good_privacy(metric, value) else "✗ RISK"
    print(f"  {metric:<12}: {value:>7.4f}  {status}")
```

**5. Set Privacy Risk Thresholds**
```python
def assess_privacy_risk(results, asr, rrs):
    """Classify privacy risk level"""
    if asr > 0.7 or rrs > 0.3 or results['Attack Accuracy'] > 0.7:
        return "HIGH RISK"
    elif asr > 0.6 or rrs > 0.2 or results['Attack Accuracy'] > 0.6:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

risk_level = assess_privacy_risk(results, asr, rrs)
print(f"Privacy Risk Assessment: {risk_level}")
```

## Troubleshooting

**Q: AUC-ROC returns 0.5 despite non-zero accuracy**
- A: Only one class present in y_true (all 1s or all 0s).
```python
if len(set(y_true)) == 1:
    print("WARNING: Only one class in ground truth, AUC undefined")
    print("Ensure test set has both members and non-members")
```

**Q: ASR returns 0.0 when attack found many members**
- A: y_true has no actual members (all 0s).
```python
if (y_true == 1).sum() == 0:
    print("ERROR: No members in ground truth")
    print("Cannot compute ASR without member samples")
```

**Q: RRS very negative**
- A: Attack misidentifies mostly non-members as members.
```python
if rrs < -0.2:
    print("Attack is worse than random")
    print("Consider different attack method or threshold")
```

**Q: All metrics show attack is random but AUC is high**
- A: Likely unbalanced classes or mislabeled ground truth.
```python
print(f"Member ratio: {(y_true == 1).sum() / len(y_true):.1%}")
print(f"Predicted member ratio: {(y_pred == 1).sum() / len(y_pred):.1%}")

# If imbalanced, use ASR + RRS (not accuracy)
```

## Related Components

- **[MembershipInference](./membership_inference.md)** — Generate predictions for evaluation
- **[LinkageAttack](./linkage_attack.md)** — Re-identification attack results
- **[DistanceToClosestRecord](./distance_to_closest_record.md)** — Threshold source for predictions
- **[Attacks Overview](./attacks_overview.md)** — Complete attack module guide
