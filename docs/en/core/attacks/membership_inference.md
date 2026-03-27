# Membership Inference Documentation

**Module:** `pamola_core.attacks.membership_inference`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Class Reference](#class-reference)
4. [Attack Methods](#attack-methods)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Components](#related-components)

## Overview

`MembershipInference` implements attacks that determine whether a specific record was in the training dataset used to create anonymized data.

**Purpose:** Test whether attacker can infer membership by:
- Measuring distance to training data (DCR method)
- Analyzing nearest neighbor patterns (NNDR method)
- Training a confidence-based classifier (Model method)

**Threat Model:**
```
Attacker's Knowledge              Test Record             Question
================================================================================
Original training data     +      Test data record   →   Was this record used
Anonymized output data                                   to create the
                                                         anonymized dataset?
```

**Risk:** High — if attacker can definitively determine membership, they've partially de-anonymized the dataset.

## Key Features

| Method | Metric | Complexity | Best For |
|--------|--------|-----------|----------|
| **DCR** | Distance to closest training record | Low | Dataset-level similarity testing |
| **NNDR** | Ratio of 1st to 2nd nearest neighbor | Low-Medium | Confidence-based membership |
| **Model** | Clustering + Random Forest confidence | Medium | Complex feature relationships |

## Class Reference

### MembershipInference

```python
from pamola_core.attacks import MembershipInference

class MembershipInference(PreprocessData):
    """
    MembershipInference class for attack simulation in PAMOLA.CORE.
    Implements three membership inference attack (MIA) variants.
    """

    def __init__(self, dcr_threshold=None, nndr_threshold=None, m_threshold=None):
        """
        Parameters
        -----------
        dcr_threshold : float, optional (default=None)
            Distance threshold for DCR method (median of dcr_values if None).
            Test record predicted as member if distance < threshold.

        nndr_threshold : float, optional (default=None)
            Distance ratio threshold for NNDR method (median of nndr_values if None).
            Test record predicted as member if ratio < threshold.

        m_threshold : float, optional (default=None)
            Confidence threshold for Model method (median of confidence if None).
            Test record predicted as member if confidence > threshold.
        """
        self.dcr_threshold = dcr_threshold
        self.nndr_threshold = nndr_threshold
        self.m_threshold = m_threshold
        self.dcr = DistanceToClosestRecord()
        self.nndr = NearestNeighborDistanceRatio()
```

### Constructor Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `dcr_threshold` | float | None (→median) | Membership decision boundary for DCR |
| `nndr_threshold` | float | None (→median) | Membership decision boundary for NNDR |
| `m_threshold` | float | None (→median) | Membership decision boundary for Model |

## Attack Methods

### 1. membership_inference_attack_dcr

Determines membership using Distance to Closest Record.

```python
def membership_inference_attack_dcr(
    self, data_train: pd.DataFrame, data_test: pd.DataFrame
) -> np.ndarray:
    """
    Membership Inference Attack using DCR (Distance to Closest Record).

    Logic: Test records close to training data are likely members.
    Implementation: 1 if DCR < threshold, 0 otherwise.

    Parameters
    ----------
    data_train : pd.DataFrame
        Training dataset (used to fit the "reference" distribution).
        All records in training set are members (ground truth = 1).

    data_test : pd.DataFrame
        Test dataset (queries for membership).
        Attack predicts whether each record was in training set.

    Returns
    -------
    np.ndarray
        Binary predictions of shape (n_test,):
        - 1 = Inferred as member of data_train
        - 0 = Inferred as non-member
    """
```

**Internal Workflow:**
```
1. Preprocess both datasets (TF-IDF + scaling)
2. Compute DCR for each test record (distance to nearest training record)
3. Determine threshold:
   - If dcr_threshold set: use it
   - Else: use median(positive_dcr_values) or median(all_dcr_values)
4. Predict: 1 if distance < threshold, else 0
```

**Threshold Logic:**
```python
# Step 1: Compute DCR
dcr_values = self.dcr.calculate_dcr(data_train, data_test, method="cdist")

# Step 2: Determine threshold
if self.dcr_threshold is not None:
    threshold = self.dcr_threshold
else:
    positive_vals = dcr_values[dcr_values > 0]
    if len(positive_vals) > 0:
        threshold = np.median(positive_vals)  # Prefer positive values
    else:
        threshold = np.median(dcr_values)

# Step 3: Predict
predictions = (dcr_values < threshold).astype(int)
```

**Example:**
```python
from pamola_core.attacks import MembershipInference
import pandas as pd
import numpy as np

# Training data (5 members)
data_train = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'salary': [50000, 70000, 90000, 100000, 120000],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
})

# Test data (mix of members + non-members)
data_test = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],           # 25, 35, 45 are in training
    'salary': [50000, 60000, 70000, 75000, 90000],
    'city': ['NYC', 'Denver', 'LA', 'Boston', 'Chicago']
})

# Run DCR attack
mia = MembershipInference(dcr_threshold=0.5)
predictions = mia.membership_inference_attack_dcr(data_train, data_test)

# Ground truth: first 3 records are members
y_true = np.array([1, 0, 1, 0, 1])

print("Predictions:", predictions)
print("Accuracy:", np.mean(predictions == y_true))
```

**Returns:**
```python
array([1, 0, 1, 0, 1])  # Predictions
```

### 2. membership_inference_attack_nndr

Determines membership using Nearest Neighbor Distance Ratio.

```python
def membership_inference_attack_nndr(
    self, data_train: pd.DataFrame, data_test: pd.DataFrame
) -> np.ndarray:
    """
    Membership Inference Attack using NNDR (Nearest Neighbor Distance Ratio).

    Logic: NNDR = d1 / d2 where d1 = distance to 1st NN, d2 = distance to 2nd NN.
    Members: small NNDR (close to single training record).
    Non-members: large NNDR (equidistant from multiple training records).

    Parameters
    ----------
    data_train : pd.DataFrame
        Training dataset (reference set).

    data_test : pd.DataFrame
        Test dataset (membership queries).

    Returns
    -------
    np.ndarray
        Binary predictions of shape (n_test,):
        - 1 = Inferred as member (low NNDR)
        - 0 = Inferred as non-member (high NNDR)
    """
```

**Internal Workflow:**
```
1. Preprocess both datasets (TF-IDF + scaling)
2. Compute NNDR for each test record (d1 / d2)
3. Determine threshold (median if not provided)
4. Predict: 1 if ratio < threshold, else 0
```

**NNDR Intuition:**
- **NNDR ≈ 0.1** → Record very close to one training point (likely member)
- **NNDR ≈ 0.5** → Record equidistant from two points (ambiguous)
- **NNDR ≈ 0.9** → Record far from nearest neighbor (likely non-member)

**Example:**
```python
from pamola_core.attacks import MembershipInference
import pandas as pd

data_train = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500]
})

data_test = pd.DataFrame({
    'feature1': [1.05, 2.5, 3.0, 10.0],     # First similar to 1.0, third is exact match
    'feature2': [10, 25, 30, 100],
    'feature3': [100, 250, 300, 1000]
})

mia = MembershipInference(nndr_threshold=0.4)
predictions = mia.membership_inference_attack_nndr(data_train, data_test)

print("Predictions:", predictions)
# Expected: [1, 0, 1, 0] (first and third close, others far)
```

### 3. membership_inference_attack_model

Determines membership using clustering and classifier confidence.

```python
def membership_inference_attack_model(
    self, data_train, data_test
) -> np.ndarray:
    """
    Membership Inference Attack using clustering + Random Forest confidence.

    Logic:
    1. Cluster training data (K-means, k=5)
    2. Train classifier to predict cluster labels (Random Forest)
    3. Get classifier confidence on test data
    4. Members: high confidence predictions
    5. Non-members: low confidence predictions

    Parameters
    ----------
    data_train : pd.DataFrame
        Training dataset.

    data_test : pd.DataFrame
        Test dataset.

    Returns
    -------
    np.ndarray
        Binary predictions of shape (n_test,):
        - 1 = High confidence in cluster prediction (inferred member)
        - 0 = Low confidence in cluster prediction (inferred non-member)
    """
```

**Internal Workflow:**
```
1. Preprocess both datasets (TF-IDF + scaling)
2. Cluster training data (K-means, k=min(5, len(data_train)))
3. Train Random Forest classifier on training clusters
4. Get prediction confidence on test data (max probability)
5. Determine threshold (median confidence if not provided)
6. Predict: 1 if confidence > threshold, else 0
```

**Example:**
```python
from pamola_core.attacks import MembershipInference
import pandas as pd
import numpy as np

# Larger dataset with clear clusters
np.random.seed(42)
data_train = pd.DataFrame({
    'x': np.concatenate([np.random.normal(0, 1, 50),
                        np.random.normal(5, 1, 50)]),
    'y': np.concatenate([np.random.normal(0, 1, 50),
                        np.random.normal(5, 1, 50)]),
    'z': np.concatenate([np.random.normal(0, 1, 50),
                        np.random.normal(5, 1, 50)])
})

# Test: some from cluster centers, some far away
data_test = pd.DataFrame({
    'x': [0.1, 5.2, 10.0],  # Near cluster 1, near cluster 2, far
    'y': [0.2, 5.1, 10.1],
    'z': [0.3, 5.0, 10.2]
})

mia = MembershipInference(m_threshold=0.7)
predictions = mia.membership_inference_attack_model(data_train, data_test)

print("Predictions:", predictions)
# Expected: [1, 1, 0] (first two near clusters, third is far)
```

## Usage Examples

### Complete Membership Inference Workflow

```python
from pamola_core.attacks import MembershipInference, AttackMetrics
import pandas as pd
import numpy as np

# Simulate original and test datasets
original_train = pd.DataFrame({
    'age': [25, 35, 45, 55, 65, 28, 38, 48, 58],
    'income': [50000, 70000, 90000, 110000, 130000, 60000, 80000, 100000, 120000],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Boston', 'Seattle', 'Miami', 'Denver']
})

# Create test: 50% from original, 50% new records
original_members = original_train.sample(n=5, random_state=42)
new_non_members = pd.DataFrame({
    'age': [22, 32, 42, 52, 62],
    'income': [40000, 75000, 95000, 115000, 140000],
    'city': ['Austin', 'San Diego', 'Portland', 'DC', 'Atlanta']
})

test_data = pd.concat([original_members, new_non_members], ignore_index=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Ground truth: first 5 are members, last 5 are non-members
y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Run all three MIA variants
mia = MembershipInference()
predictions_dcr = mia.membership_inference_attack_dcr(original_train, test_data)
predictions_nndr = mia.membership_inference_attack_nndr(original_train, test_data)
predictions_model = mia.membership_inference_attack_model(original_train, test_data)

# Evaluate with AttackMetrics
metrics = AttackMetrics()

print("=== DCR Method ===")
results_dcr = metrics.attack_metrics(y_true, predictions_dcr)
print(f"Accuracy: {results_dcr['Attack Accuracy']}")
print(f"ASR: {metrics.attack_success_rate(y_true, predictions_dcr)}")

print("\n=== NNDR Method ===")
results_nndr = metrics.attack_metrics(y_true, predictions_nndr)
print(f"Accuracy: {results_nndr['Attack Accuracy']}")
print(f"ASR: {metrics.attack_success_rate(y_true, predictions_nndr)}")

print("\n=== Model Method ===")
results_model = metrics.attack_metrics(y_true, predictions_model)
print(f"Accuracy: {results_model['Attack Accuracy']}")
print(f"ASR: {metrics.attack_success_rate(y_true, predictions_model)}")

# Compare methods
print("\n=== Method Comparison ===")
print(f"Best method: {'DCR' if results_dcr['Attack Accuracy'] > max(results_nndr['Attack Accuracy'], results_model['Attack Accuracy']) else 'NNDR' if results_nndr['Attack Accuracy'] > results_model['Attack Accuracy'] else 'Model'}")
```

### Threshold Tuning Example

```python
from pamola_core.attacks import MembershipInference
import numpy as np

# Find optimal threshold for DCR
dcr_values = mia.dcr.calculate_dcr(data_train, data_test, method="cdist")

# Try different percentiles
thresholds = [10, 25, 50, 75, 90]
for percentile in thresholds:
    threshold = np.percentile(dcr_values, percentile)
    predictions = (dcr_values < threshold).astype(int)
    accuracy = np.mean(predictions == y_true)
    print(f"Percentile {percentile}: threshold={threshold:.4f}, accuracy={accuracy:.3f}")

# Use optimal threshold
optimal_threshold = np.percentile(dcr_values, 50)
mia_tuned = MembershipInference(dcr_threshold=optimal_threshold)
```

## Best Practices

**1. Use Multiple Methods**
- DCR: baseline dataset similarity
- NNDR: confidence-based membership
- Model: complex feature relationships

**2. Validate with Ground Truth**
```python
# Create test set with known members/non-members
known_members = original_data
known_non_members = synthetic_data  # Generated, not in original

test_combined = pd.concat([known_members, known_non_members])
y_true = np.array([1]*len(known_members) + [0]*len(known_non_members))

# Evaluate
predictions = mia.membership_inference_attack_dcr(original_data, test_combined)
accuracy = np.mean(predictions == y_true)
```

**3. Adjust Thresholds Based on Data**
```python
# Start with adaptive thresholds (median)
mia = MembershipInference()

# After evaluation, tune if needed
mia = MembershipInference(dcr_threshold=0.75)  # Stricter
```

**4. Check for Class Imbalance**
```python
# If test set is 90% members, 10% non-members
# Accuracy can be misleading. Use ASR and RRS instead.

asr = metrics.attack_success_rate(y_true, predictions)
rrs = metrics.residual_risk_score(y_true, predictions)

# RRS > 0.2 indicates real privacy risk despite imbalance
```

## Troubleshooting

**Q: All predictions are 0 (no members inferred)**
- A: Threshold too high, test data dissimilar from training.
```python
# Check threshold value
print(f"Threshold: {mia.dcr_threshold}")
print(f"Min DCR value: {dcr_values.min()}")
print(f"Median DCR value: {dcr_values.median()}")

# Lower threshold or set explicitly
mia = MembershipInference(dcr_threshold=np.percentile(dcr_values, 25))
```

**Q: All predictions are 1 (everyone is member)**
- A: Threshold too low, test data too similar to training.
```python
# Raise threshold
mia = MembershipInference(dcr_threshold=np.percentile(dcr_values, 75))
```

**Q: Model method fails with "n_clusters > len(data_train)"**
- A: Training set too small. Code handles this (min 5, len(data)).
```python
# If training has <5 records, use smaller k
if len(data_train) < 5:
    print("Warning: Training set very small, model may be unreliable")
```

**Q: NNDR method returns NaN or inf values**
- A: Division by zero in ratio. Code protects against this (1e-10 floor).
```python
# Code: nndr_values = distances[:, 0] / np.maximum(distances[:, 1], 1e-10)
# Already handled in library
```

## Related Components

- **[DistanceToClosestRecord](./distance_to_closest_record.md)** — DCR metric implementation
- **[NearestNeighborDistanceRatio](./nearest_neighbor_distance_ratio.md)** — NNDR metric implementation
- **[AttackMetrics](./attack_metrics.md)** — Evaluate attack success
- **[Linkage Attack](./linkage_attack.md)** — Re-identification attacks
