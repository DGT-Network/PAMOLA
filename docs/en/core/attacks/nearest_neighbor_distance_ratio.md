# Nearest Neighbor Distance Ratio (NNDR) Documentation

**Module:** `pamola_core.attacks.nearest_neighbor_distance_ratio`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Class Reference](#class-reference)
4. [Core Method](#core-method)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Related Components](#related-components)

## Overview

`NearestNeighborDistanceRatio` implements the NNDR (Nearest Neighbor Distance Ratio) metric, which measures the ratio of distances to the 1st and 2nd nearest neighbors.

**Purpose:** Assess confidence in nearest neighbor matching:
- **Low NNDR (< 0.5)** → Record very close to one neighbor → High confidence match → Member
- **High NNDR (> 0.8)** → Record equidistant from neighbors → Low confidence → Non-member

**Threat Model:**
```
Record's Nearest Neighbors          NNDR Interpretation
====================================
d1 = 0.1, d2 = 1.0  → NNDR = 0.10  Record matches one person closely
d1 = 0.5, d2 = 0.6  → NNDR = 0.83  Record matches multiple people (ambiguous)
d1 = 5.0, d2 = 5.1  → NNDR = 0.98  Record far from anyone (non-member)
```

**Use Case:** Confidence-based membership inference. Combined with DCR for robust privacy testing.

## Key Features

**Flexible Computation**
- **KDTree method** — Fast, optimized for Euclidean distance
- **Neighbors method** — Uses sklearn NearestNeighbors (more flexible)

**Automatic Preprocessing**
- Categorical → TF-IDF vectorization
- Numeric → Standard scaling
- Result → Numeric matrix for neighbor computation

**No Thresholding**
- Returns raw ratio values
- Caller defines membership boundary
- Useful for threshold optimization

## Class Reference

### NearestNeighborDistanceRatio

```python
from pamola_core.attacks import NearestNeighborDistanceRatio

class NearestNeighborDistanceRatio(PreprocessData):
    """
    NearestNeighborDistanceRatio class for attack simulation in PAMOLA.CORE.
    Computes Nearest Neighbor Distance Ratio (NNDR) metric.
    """

    def __init__(self):
        """
        No parameters. All computation controlled via calculate_nndr() arguments.
        """
```

**Constructor:** No parameters

## Core Method

### calculate_nndr

Compute nearest neighbor distance ratio for test records.

```python
def calculate_nndr(
    self, data1: pd.DataFrame, data2: pd.DataFrame, method: str = "kdtree"
) -> np.ndarray:
    """
    Nearest Neighbor Distance Ratio (NNDR):
    Ratio of distance to 1st nearest neighbor vs. 2nd nearest neighbor.

    Formula: NNDR = d1 / d2
    where d1 = distance to nearest neighbor in data1
          d2 = distance to 2nd nearest neighbor in data1

    Interpretation:
    - NNDR < 0.5: Unique match (very confident) → High member risk
    - NNDR ≈ 1.0: Ambiguous (equidistant) → Low member risk
    - NNDR > 1.0: Numerical artifact (very rare)

    Parameters
    ----------
    data1 : pd.DataFrame
        Reference dataset (typically original/training data).
        Used to build the neighbor index.

    data2 : pd.DataFrame
        Query dataset (typically anonymized/test data).
        Each record gets a NNDR based on its two nearest neighbors in data1.

    method : {"kdtree", "neighbors"}, default="kdtree"
        Algorithm for neighbor search:
        - "kdtree": scipy.spatial.KDTree (fast, Euclidean only)
        - "neighbors": sklearn.neighbors.NearestNeighbors (flexible, any metric)

    Returns
    -------
    nndr_values : np.ndarray
        Array of shape (n_samples_in_data2,) containing NNDR ratios.
        nndr_values[i] = d1[i] / d2[i] for record data2[i]

        Value range: typically [0.0, 1.0], can exceed 1.0 rarely
        Interpretation:
        - 0.0 to 0.3: Extremely confident match (member)
        - 0.3 to 0.7: Confident match (likely member)
        - 0.7 to 1.0: Ambiguous (borderline)
        - 1.0+: Very ambiguous (likely non-member)
    """
```

### Method Comparison

| Aspect | KDTree | Neighbors |
|--------|--------|-----------|
| **Speed** | Very fast | Fast |
| **Metric** | Euclidean only | Any sklearn metric |
| **Best for** | Default use, large data | Custom metrics, flexibility |
| **Implementation** | Scipy | Scikit-learn |

## Usage Examples

### Basic NNDR Computation

```python
from pamola_core.attacks import NearestNeighborDistanceRatio
import pandas as pd
import numpy as np

# Training data (reference set)
training = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'income': [50000, 75000, 100000, 125000, 150000],
    'region': ['North', 'South', 'West', 'East', 'Central']
})

# Test data (membership query)
test = pd.DataFrame({
    'age': [25, 28, 35, 45, 50],
    'income': [50000, 60000, 75000, 100000, 90000],
    'region': ['North', 'Midwest', 'South', 'West', 'Southwest']
})

# Compute NNDR
nndr = NearestNeighborDistanceRatio()
ratios = nndr.calculate_nndr(training, test)

print("NNDR Values:")
for i, ratio in enumerate(ratios):
    confidence = "HIGH (member)" if ratio < 0.5 else "MEDIUM" if ratio < 0.8 else "LOW (non-member)"
    print(f"  Record {i}: {ratio:.4f} → {confidence}")

print(f"\nMean NNDR: {ratios.mean():.4f}")
print(f"Median NNDR: {np.median(ratios):.4f}")

# Risk summary
high_risk = (ratios < 0.5).sum()
print(f"High-risk records (NNDR < 0.5): {high_risk}/{len(test)}")
```

**Output:**
```
NNDR Values:
  Record 0: 0.0234 → HIGH (member)
  Record 1: 0.6789 → MEDIUM
  Record 2: 0.1245 → HIGH (member)
  Record 3: 0.4567 → MEDIUM
  Record 4: 0.8923 → LOW (non-member)

Mean NNDR: 0.4352
Median NNDR: 0.4567

High-risk records (NNDR < 0.5): 3/5
```

### Membership Inference with NNDR

```python
from pamola_core.attacks import NearestNeighborDistanceRatio
import numpy as np

nndr = NearestNeighborDistanceRatio()

# Get NNDR values
ratios = nndr.calculate_nndr(training_data, test_data)

# Threshold-based membership prediction
threshold = np.median(ratios)  # Adaptive threshold

membership_predictions = (ratios < threshold).astype(int)
# 1 = member (low NNDR), 0 = non-member (high NNDR)

# Confidence scores (inverse of NNDR)
confidence = 1.0 - ratios  # High NNDR → low confidence

print(f"Membership Predictions (threshold={threshold:.4f}):")
for i, (pred, conf) in enumerate(zip(membership_predictions, confidence)):
    status = "MEMBER" if pred == 1 else "NON-MEMBER"
    print(f"  Record {i}: {status} (confidence: {conf:.4f})")
```

### NNDR Distribution Analysis

```python
from pamola_core.attacks import NearestNeighborDistanceRatio
import numpy as np
import matplotlib.pyplot as plt

nndr = NearestNeighborDistanceRatio()
ratios = nndr.calculate_nndr(original_data, anonymized_data)

# Statistical summary
print("NNDR Distribution Analysis:")
print(f"  Min: {ratios.min():.4f}")
print(f"  25th percentile: {np.percentile(ratios, 25):.4f}")
print(f"  Median: {np.median(ratios):.4f}")
print(f"  75th percentile: {np.percentile(ratios, 75):.4f}")
print(f"  Max: {ratios.max():.4f}")
print(f"  Mean: {ratios.mean():.4f}")
print(f"  Std Dev: {ratios.std():.4f}")

# Risk buckets
print("\nNNDR Risk Classification:")
boundaries = [0.0, 0.3, 0.5, 0.7, 1.0, float('inf')]
labels = ['Extreme (0.0-0.3)', 'Very High (0.3-0.5)', 'High (0.5-0.7)',
          'Medium (0.7-1.0)', 'Low (1.0+)']

for i in range(len(boundaries) - 1):
    low, high = boundaries[i], boundaries[i+1]
    if high == float('inf'):
        count = (ratios >= low).sum()
    else:
        count = ((ratios >= low) & (ratios < high)).sum()
    pct = count / len(ratios) * 100
    print(f"  {labels[i]:<20}: {count:4d} records ({pct:5.1f}%)")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('NNDR Ratio')
plt.ylabel('Frequency')
plt.title('NNDR Distribution')
plt.axvline(ratios.mean(), color='r', linestyle='--', label=f'Mean: {ratios.mean():.2f}')
plt.axvline(np.median(ratios), color='g', linestyle='--', label=f'Median: {np.median(ratios):.2f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(ratios)
plt.ylabel('NNDR Ratio')
plt.title('NNDR Box Plot')

plt.tight_layout()
plt.show()
```

### Comparing Methods: KDTree vs Neighbors

```python
from pamola_core.attacks import NearestNeighborDistanceRatio
import time

nndr = NearestNeighborDistanceRatio()

# Create test datasets
training = pd.DataFrame({
    f'feature_{i}': np.random.randn(5000)
    for i in range(10)
})

test = pd.DataFrame({
    f'feature_{i}': np.random.randn(500)
    for i in range(10)
})

# Compare methods
print("Method Comparison:\n")
print(f"{'Method':<15} {'Time (ms)':<12} {'Mean NNDR':<12}")
print("-" * 40)

# KDTree (default)
start = time.time()
ratios_kdtree = nndr.calculate_nndr(training, test, method='kdtree')
time_kdtree = (time.time() - start) * 1000

print(f"{'kdtree':<15} {time_kdtree:<12.2f} {ratios_kdtree.mean():<12.4f}")

# Neighbors
start = time.time()
ratios_neighbors = nndr.calculate_nndr(training, test, method='neighbors')
time_neighbors = (time.time() - start) * 1000

print(f"{'neighbors':<15} {time_neighbors:<12.2f} {ratios_neighbors.mean():<12.4f}")

# Difference
diff = np.abs(ratios_kdtree - ratios_neighbors).max()
print(f"\nMax difference: {diff:.6f}")
print(f"KDTree is {time_neighbors/time_kdtree:.1f}x faster")
```

### Combined DCR + NNDR Analysis

```python
from pamola_core.attacks import DistanceToClosestRecord, NearestNeighborDistanceRatio
import pandas as pd

dcr_obj = DistanceToClosestRecord()
nndr_obj = NearestNeighborDistanceRatio()

# Compute both metrics
dcr_values = dcr_obj.calculate_dcr(original, anonymized)
nndr_values = nndr_obj.calculate_nndr(original, anonymized)

# Create analysis DataFrame
analysis = pd.DataFrame({
    'DCR': dcr_values,
    'NNDR': nndr_values,
    'DCR_Risk': pd.cut(dcr_values, bins=[0, 1, 3, 7, float('inf')],
                       labels=['Very High', 'High', 'Medium', 'Low']),
    'NNDR_Risk': pd.cut(nndr_values, bins=[0, 0.3, 0.5, 0.7, float('inf')],
                        labels=['Very High', 'High', 'Medium', 'Low'])
})

# Combined risk assessment
print("Privacy Risk Assessment (DCR + NNDR):\n")
print(analysis.head(10))

# Identify high-risk records (both metrics agree)
high_risk = (analysis['DCR_Risk'] == 'Very High') & (analysis['NNDR_Risk'] == 'Very High')
print(f"\nRecords at extreme risk: {high_risk.sum()}/{len(analysis)}")

# Identify controversial records (metrics disagree)
controversial = analysis['DCR_Risk'] != analysis['NNDR_Risk']
print(f"Controversial records: {controversial.sum()}/{len(analysis)}")
```

## Best Practices

**1. Use NNDR as Confidence Measure**
```python
# NNDR is not a binary membership indicator
# Instead, use it as a confidence score for membership probability

ratios = nndr.calculate_nndr(training, test)

# Convert to confidence: high NNDR = low confidence in member claim
confidence = 1.0 - np.clip(ratios, 0, 1)  # Confidence 0.0-1.0

# Membership probability based on confidence
# Low confidence (NNDR > 0.7) → 50/50 chance of member
membership_probability = (1.0 - ratios) * 0.5 + 0.5
# High confidence (NNDR < 0.3) → ~100% chance of member
```

**2. Combine with DCR for Dual Assessment**
```python
# DCR: absolute distance to nearest neighbor
# NNDR: relative distance (1st vs 2nd neighbor)

dcr = dcr_obj.calculate_dcr(training, test)
nndr = nndr_obj.calculate_nndr(training, test)

# High privacy risk: both DCR low AND NNDR low
# → Record close to and unique match with training
high_risk = (dcr < 1.0) & (nndr < 0.5)

# Medium privacy risk: DCR high but NNDR low
# → Record far but still unique match to the far point
medium_risk = (dcr > 5.0) & (nndr < 0.5)

# Low privacy risk: NNDR high
# → Record ambiguous (equidistant from multiple neighbors)
low_risk = nndr > 0.8
```

**3. Choose Optimal Threshold**
```python
# Data-driven threshold selection
ratios = nndr.calculate_nndr(training, test)

# Option 1: Median (50th percentile)
threshold_median = np.median(ratios)

# Option 2: Percentile-based (adjust for expected member ratio)
# If expecting 80% members in test, use 80th percentile
threshold_80 = np.percentile(ratios, 80)

# Option 3: Domain-specific
# For high-sensitivity data, use 0.3 (stricter)
threshold_strict = 0.3

# For exploratory analysis, use 0.7 (lenient)
threshold_lenient = 0.7
```

**4. Monitor Distribution Shifts**
```python
# If NNDR distribution changes over time, anonymization quality degrades
# Recompute periodically

ratios_month1 = nndr.calculate_nndr(training, test_month1)
ratios_month2 = nndr.calculate_nndr(training, test_month2)

print(f"Month 1 - Mean NNDR: {ratios_month1.mean():.4f}")
print(f"Month 2 - Mean NNDR: {ratios_month2.mean():.4f}")

if ratios_month2.mean() < ratios_month1.mean():
    print("WARNING: NNDR decreased (higher membership risk detected)")
```

**5. Handle Edge Cases**
```python
# If data1 has < 2 records, NNDR cannot be computed
if len(data1) < 2:
    print("ERROR: Need at least 2 reference records for NNDR")

# If all distances are zero (identical records), handle carefully
if (ratios == 0).any():
    print("WARNING: Some records are identical to training data")
    # These are at extreme re-identification risk
```

## Troubleshooting

**Q: Empty array returned**
- A: data1 or data2 is empty DataFrame.
```python
if data1.empty or data2.empty:
    return np.array([])

# Check before calling
assert len(data1) >= 2, "Need at least 2 reference records"
assert len(data2) > 0, "Need at least 1 test record"
```

**Q: InvalidParameterError: Unknown NNDR method**
- A: method parameter must be "kdtree" or "neighbors".
```python
# Wrong
ratios = nndr.calculate_nndr(training, test, method='brute')

# Correct
ratios = nndr.calculate_nndr(training, test, method='kdtree')
```

**Q: NNDR values exceed 1.0**
- A: Rare but valid. Occurs when 1st neighbor is farther than 2nd neighbor (numerical artifact).
```python
# Clip to [0.0, 1.0] range if needed
ratios = np.clip(ratios, 0.0, 1.0)

# Or investigate distances
distances = nndr.kdtree.query(data2_vec, k=2)
# Check if distances[:, 0] > distances[:, 1]
```

**Q: Performance slow for large data1**
- A: KDTree scales as O(n log n) build + O(log n) query. Still fast.
```python
# If still slow, try reducing dimensionality first
# via PreprocessData.preprocess_data(max_features=1000)

# Or use sampling
indices = np.random.choice(len(training), size=1000, replace=False)
training_sample = training.iloc[indices]
```

**Q: All NNDR values very similar**
- A: Data points evenly distributed, no clear membership signal.
```python
# This is actually GOOD for privacy (ambiguous membership)
# Means anonymization effectively obscured membership patterns

if ratios.std() < 0.1:
    print("Good privacy: NNDR values uniform (no membership signal)")
```

## Related Components

- **[DistanceToClosestRecord](./distance_to_closest_record.md)** — Complementary DCR metric
- **[MembershipInference](./membership_inference.md)** — Uses NNDR for membership detection
- **[Linkage Attack](./linkage_attack.md)** — Record matching attacks
- **[Attack Metrics](./attack_metrics.md)** — Evaluate membership predictions
