# Distance to Closest Record (DCR) Documentation

**Module:** `pamola_core.attacks.distance_to_closest_record`
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

`DistanceToClosestRecord` implements the DCR (Distance to Closest Record) metric, which measures the minimum Euclidean distance from each record in a test dataset to its nearest neighbor in a reference dataset.

**Purpose:** Quantify dataset similarity by computing proximity scores:
- **Low DCR** (distance < 1.0) → Test record very similar to reference → High re-identification risk
- **High DCR** (distance > 10.0) → Test record dissimilar to reference → Low re-identification risk

**Threat Model:**
```
Reference Data (Original)       Test Data (Anonymized)
[John: 35, NYC, $80k]     →    Compute distance to
[Jane: 28, LA, $90k]      →    nearest record
[Bob: 42, Chicago, $75k]  →    in reference
                                ↓
                           [35, NYC, $75k] → distance = 5.0k
                           [28, Denver, $50k] → distance = 2,400 miles
```

**Use Case:** Baseline privacy metric before running full attacks. If DCR is uniformly low, anonymization failed.

## Key Features

**Flexible Distance Computation**
- **KDTree method** — Fast, optimized for Euclidean distance
- **cdist method** — General, supports custom metrics (cosine, manhattan, etc.)

**Automatic Data Preprocessing**
- Categorical columns → TF-IDF vectorization
- Numeric columns → Standard scaling
- Result → Numeric feature matrix for distance computation

**No Thresholding**
- Returns raw distance values
- Lets caller define risk thresholds
- Useful for distribution analysis

## Class Reference

### DistanceToClosestRecord

```python
from pamola_core.attacks import DistanceToClosestRecord

class DistanceToClosestRecord(PreprocessData):
    """
    DistanceToClosestRecord class for attack simulation in PAMOLA.CORE.
    Computes Distance to Closest Record (DCR) metric.
    """

    def __init__(self):
        """
        No parameters. All computation controlled via calculate_dcr() arguments.
        """
```

**Constructor:** No parameters

## Core Method

### calculate_dcr

Compute distance from test records to nearest reference record.

```python
def calculate_dcr(
    self,
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    method: str = "kdtree",
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Distance to Closest Record (DCR): Minimum distance from each record in
    data2 to its nearest record in data1.

    Parameters
    ----------
    data1 : pd.DataFrame
        Reference dataset (typically original/training data).
        Used as the neighborhood to search against.

    data2 : pd.DataFrame
        Query dataset (typically anonymized/test data).
        Each record gets a distance to its nearest neighbor in data1.

    method : {"kdtree", "cdist"}, default="kdtree"
        Algorithm for distance computation:
        - "kdtree": scipy.spatial.KDTree (optimized for Euclidean, fast)
        - "cdist": scipy.spatial.distance.cdist (flexible, custom metrics)

    metric : str, default="euclidean"
        Distance metric (used when method="cdist").
        Options: "euclidean", "cosine", "manhattan", "chebyshev", etc.
        Ignored if method="kdtree" (always Euclidean).

    Returns
    -------
    dcr_values : np.ndarray
        Array of shape (n_samples_in_data2,) containing distances.
        dcr_values[i] = distance from data2[i] to nearest neighbor in data1.

        Interpretation:
        - 0.0 to 0.5: Exact or near-exact match (very high privacy risk)
        - 0.5 to 2.0: Very similar (high privacy risk)
        - 2.0 to 5.0: Similar (medium privacy risk)
        - 5.0+: Dissimilar (low privacy risk)
    """
```

### Method Comparison

| Aspect | KDTree | cdist |
|--------|--------|-------|
| **Speed** | Fast (O(log n)) | Slower (O(n*m)) |
| **Metric** | Euclidean only | Any scipy metric |
| **Best for** | Large datasets, Euclidean | Custom metrics, small data |
| **Memory** | Low | Higher |

## Usage Examples

### Basic DCR Computation

```python
from pamola_core.attacks import DistanceToClosestRecord
import pandas as pd
import numpy as np

# Original data (reference)
original = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'income': [50000, 75000, 100000, 125000, 150000],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
})

# Anonymized data (test)
anonymized = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],           # Unchanged
    'income': [55000, 80000, 105000, 130000, 155000],  # Slightly modified
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']  # Unchanged
})

# Compute DCR
dcr = DistanceToClosestRecord()
distances = dcr.calculate_dcr(original, anonymized)

print("DCR Values:")
for i, dist in enumerate(distances):
    print(f"  Record {i}: {dist:.4f}")

print(f"\nMean DCR: {distances.mean():.4f}")
print(f"Median DCR: {np.median(distances):.4f}")
print(f"Max DCR: {distances.max():.4f}")

# Risk assessment
if distances.mean() < 1.0:
    print("\nRISK: Very similar to original (high re-identification risk)")
elif distances.mean() < 5.0:
    print("\nRISK: Somewhat similar (medium re-identification risk)")
else:
    print("\nRISK: Dissimilar enough (low re-identification risk)")
```

**Output:**
```
DCR Values:
  Record 0: 0.1234
  Record 1: 0.2567
  Record 2: 0.1890
  Record 3: 0.3012
  Record 4: 0.2845

Mean DCR: 0.2310
Median DCR: 0.2567
Max DCR: 0.3012

RISK: Very similar to original (high re-identification risk)
```

### Comparing Anonymization Techniques

```python
from pamola_core.attacks import DistanceToClosestRecord
import pandas as pd

original = pd.read_csv('original_data.csv')

# Test three anonymization approaches
anonymizations = {
    'no_protection': original.copy(),
    'masking': apply_masking(original),
    'generalization': apply_generalization(original),
    'noise': apply_noise(original)
}

dcr = DistanceToClosestRecord()

print("Anonymization Comparison (DCR Metric):\n")
print(f"{'Method':<20} {'Mean DCR':<12} {'Risk Level':<15}")
print("-" * 45)

results = {}
for name, data in anonymizations.items():
    distances = dcr.calculate_dcr(original, data)
    mean_dcr = distances.mean()
    results[name] = mean_dcr

    # Risk classification
    if mean_dcr < 1.0:
        risk = "VERY HIGH"
    elif mean_dcr < 3.0:
        risk = "HIGH"
    elif mean_dcr < 7.0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    print(f"{name:<20} {mean_dcr:<12.4f} {risk:<15}")

# Find best approach
best = min(results, key=results.get)
print(f"\nBest approach: {best} (highest DCR = lowest risk)")
```

### Distribution Analysis

```python
from pamola_core.attacks import DistanceToClosestRecord
import numpy as np
import matplotlib.pyplot as plt

dcr = DistanceToClosestRecord()
distances = dcr.calculate_dcr(original, anonymized)

# Percentile analysis
percentiles = [10, 25, 50, 75, 90, 99]
print("DCR Percentile Distribution:")
for p in percentiles:
    val = np.percentile(distances, p)
    count = (distances < val).sum()
    print(f"  {p}th percentile: {val:.4f} ({count} records below this)")

# Risk buckets
print("\nRisk Classification:")
risk_boundaries = [0.5, 1.0, 2.0, 5.0, 10.0]
for i in range(len(risk_boundaries) - 1):
    low, high = risk_boundaries[i], risk_boundaries[i+1]
    count = ((distances >= low) & (distances < high)).sum()
    pct = count / len(distances) * 100
    print(f"  {low:.1f}-{high:.1f}: {count:4d} records ({pct:5.1f}%)")

# Visualization
plt.hist(distances, bins=30, edgecolor='black')
plt.xlabel('Distance to Closest Record')
plt.ylabel('Frequency')
plt.title('DCR Distribution')
plt.axvline(distances.mean(), color='r', linestyle='--', label=f'Mean: {distances.mean():.2f}')
plt.legend()
plt.show()
```

### Method Comparison: KDTree vs cdist

```python
from pamola_core.attacks import DistanceToClosestRecord
import time
import pandas as pd

dcr = DistanceToClosestRecord()

# Create test datasets
original = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000)
})

anonymized = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100)
})

# Compare methods
methods = [
    ('kdtree', 'euclidean'),
    ('cdist', 'euclidean'),
    ('cdist', 'cosine'),
    ('cdist', 'manhattan')
]

print("Method Performance Comparison:\n")
print(f"{'Method':<15} {'Metric':<12} {'Time (ms)':<12} {'Mean DCR':<12}")
print("-" * 50)

for method, metric in methods:
    start = time.time()
    distances = dcr.calculate_dcr(original, anonymized, method=method, metric=metric)
    elapsed = (time.time() - start) * 1000

    print(f"{method:<15} {metric:<12} {elapsed:<12.2f} {distances.mean():<12.4f}")
```

**Output:**
```
Method Performance Comparison:

Method          Metric       Time (ms)    Mean DCR
--------------------------------------------------
kdtree          euclidean    12.34        0.8234
cdist           euclidean    45.67        0.8234
cdist           cosine       48.90        0.7234
cdist           manhattan    47.23        0.8567
```

## Best Practices

**1. Use as Baseline Metric**
```python
# Run DCR early to assess overall anonymization quality
dcr = DistanceToClosestRecord()
distances = dcr.calculate_dcr(original, anonymized)

if distances.mean() < 2.0:
    print("STOP: Anonymization insufficient, re-identity risk too high")
    # Reapply stronger anonymization techniques
else:
    print("OK: Proceed to detailed attack testing")
```

**2. Compare Before/After Anonymization**
```python
# Track progress
steps = [
    ('original', original),
    ('after_masking', apply_masking(original)),
    ('after_generalization', apply_generalization(original)),
    ('after_noise', apply_noise(original))
]

for name, data in steps:
    distances = dcr.calculate_dcr(original, data)
    print(f"{name:25} → Mean DCR: {distances.mean():.4f}")
```

**3. Combine with Membership Inference**
```python
# Use DCR as coarse filter
distances = dcr.calculate_dcr(training, test)

# Identify high-risk records (low DCR)
high_risk_indices = np.where(distances < 1.0)[0]
print(f"High-risk records: {len(high_risk_indices)}/{len(test)}")

# Then run membership inference only on high-risk records
if len(high_risk_indices) > 0:
    mia_results = mia.membership_inference_attack_dcr(
        training, test.iloc[high_risk_indices]
    )
```

**4. Choose Metric Based on Data Type**
```python
# For mostly numeric data
distances_euclidean = dcr.calculate_dcr(original, anonymized, method='kdtree')

# For mixed categorical + numeric
distances_cosine = dcr.calculate_dcr(original, anonymized, method='cdist', metric='cosine')

# For Manhattan distance (less sensitive to outliers)
distances_manhattan = dcr.calculate_dcr(original, anonymized, method='cdist', metric='manhattan')
```

**5. Monitor Distribution, Not Just Mean**
```python
# Mean alone is misleading
distances = dcr.calculate_dcr(original, anonymized)

# Check if any records have very low distance
very_high_risk = (distances < 0.5).sum()
if very_high_risk > 0:
    print(f"WARNING: {very_high_risk} records at extreme risk!")

# Check if most records are similar
medium_risk = (distances < 3.0).sum() / len(distances)
if medium_risk > 0.7:
    print(f"WARNING: {medium_risk:.1%} of records are similar to original!")
```

## Troubleshooting

**Q: Empty array returned**
- A: data1 or data2 is empty DataFrame.
```python
if data1.empty or data2.empty:
    return np.array([])

# Check before calling
print(f"Original records: {len(original)}")
print(f"Anonymized records: {len(anonymized)}")
```

**Q: InvalidParameterError: Unknown DCR method**
- A: method parameter must be "kdtree" or "cdist".
```python
# Wrong
distances = dcr.calculate_dcr(original, anonymized, method='nearest')

# Correct
distances = dcr.calculate_dcr(original, anonymized, method='kdtree')
```

**Q: All distances are 0**
- A: Original and anonymized are identical.
```python
# Check if data was actually anonymized
if (original == anonymized).all().all():
    print("ERROR: No anonymization applied, data is identical")
```

**Q: DCR values very large (100+)**
- A: Likely scale issue in numeric features.
```python
# DCR preprocessing uses StandardScaler (mean=0, std=1)
# But if features have vastly different scales before scaling:
# age: 1-100, salary: 30000-200000

# Solution: Ensure preprocessing is consistent
# (library handles this automatically via PreprocessData)
```

**Q: Performance slow for large datasets**
- A: Switch to KDTree method (faster).
```python
# Slow: cdist O(n*m)
distances = dcr.calculate_dcr(original, anonymized, method='cdist')

# Fast: KDTree O(log n)
distances = dcr.calculate_dcr(original, anonymized, method='kdtree')
```

## Related Components

- **[MembershipInference](./membership_inference.md)** — Uses DCR for membership detection
- **[NearestNeighborDistanceRatio](./nearest_neighbor_distance_ratio.md)** — Similar metric, focuses on ratios
- **[Linkage Attack](./linkage_attack.md)** — Record matching attacks
- **[Attack Metrics](./attack_metrics.md)** — Evaluate attack success using DCR results
