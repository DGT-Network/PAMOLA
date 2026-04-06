# DistanceMetricType Enumeration

**Module:** `pamola_core.common.enum.distance_metric_type`
**Class:** `DistanceMetricType`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Enum Members](#enum-members)
3. [Metric Definitions](#metric-definitions)
4. [Use Cases](#use-cases)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Related Components](#related-components)

## Overview

`DistanceMetricType` is a string-based enumeration defining distance metrics used in privacy and utility calculations. These metrics measure the similarity or distance between original and anonymized datasets, or between records in clustering and k-anonymity operations.

**Parent Class:** `str, Enum`
**Type:** String Enum
**Scope:** Metrics calculations
**Used By:** Privacy analysis, utility measurement, clustering operations

## Enum Members

| Member | Value | Description |
|--------|-------|-------------|
| `EUCLIDEAN` | `"euclidean"` | Euclidean (L2) distance. Straight-line distance in multi-dimensional space. |
| `MANHATTAN` | `"manhattan"` | Manhattan (L1) distance. Sum of absolute differences across dimensions. |
| `COSINE` | `"cosine"` | Cosine distance. Measures angle between vectors (0 to 2). |
| `MAHALANOBIS` | `"mahalanobis"` | Mahalanobis distance. Accounts for correlation between variables. |

## Metric Definitions

### EUCLIDEAN Distance

**Formula:** `√(Σ(x_i - y_i)²)`

**Characteristics:**
- Most common distance metric
- Assumes uniform variable scales
- Sensitive to outliers
- Computationally efficient
- Range: [0, ∞)

**Use Cases:**
- General numerical comparisons
- Clustering analysis
- Multi-dimensional record similarity

**Example:**
```
Point A: (0, 0)
Point B: (3, 4)
Distance: √(3² + 4²) = √25 = 5
```

### MANHATTAN Distance

**Formula:** `Σ|x_i - y_i|`

**Characteristics:**
- Measures along grid/axes (taxi-cab metric)
- Less sensitive to outliers than Euclidean
- Often faster to compute
- Suitable for categorical data
- Range: [0, ∞)

**Use Cases:**
- City-block distances
- Integer-valued data
- Robust analysis with outliers

**Example:**
```
Point A: (0, 0)
Point B: (3, 4)
Distance: |3-0| + |4-0| = 7
```

### COSINE Distance

**Formula:** `1 - (A·B / (||A|| ||B||))`

**Characteristics:**
- Measures angle between vectors
- Independent of magnitude
- Range: [0, 2]
- Works well with normalized data
- Suitable for sparse/text data

**Use Cases:**
- Text/document similarity
- Sparse high-dimensional data
- Recommendation systems

**Example:**
```
Vector A: [1, 0, 1]
Vector B: [1, 1, 0]
Cosine similarity: 1/√6 ≈ 0.408
Cosine distance: 1 - 0.408 = 0.592
```

### MAHALANOBIS Distance

**Formula:** `√((x-y)ᵀ Σ⁻¹ (x-y))`

where Σ is the covariance matrix

**Characteristics:**
- Accounts for variable correlations
- Accounts for variable variance
- More computationally expensive
- Requires covariance matrix calculation
- Best with multivariate normal data
- Range: [0, ∞)

**Use Cases:**
- Anomaly detection
- Correlated variables
- Statistical outlier identification

**Properties:**
- Unitless measure
- Accounts for interdependencies
- More robust for correlated data

## Use Cases

### Privacy Metrics (Distance to Closest Record)

```python
from pamola_core.common.enum.distance_metric_type import DistanceMetricType

# Use Mahalanobis for privacy analysis with correlated attributes
metric = DistanceMetricType.MAHALANOBIS
# Accounts for age-income correlation in privacy calculation
```

### Utility Preservation Analysis

```python
# Use Euclidean for comparing numeric distributions
metric = DistanceMetricType.EUCLIDEAN
# Calculates similarity between original and generalized values
```

### Record Linkage and Clustering

```python
# Use Manhattan for robust clustering with outliers
metric = DistanceMetricType.MANHATTAN
# Less sensitive to extreme values in quasi-identifiers
```

### Text and Categorical Data

```python
# Use Cosine for comparing categorical profiles
metric = DistanceMetricType.COSINE
# Measures similarity of categorical distributions
```

## Usage Examples

### Basic Enum Usage

```python
from pamola_core.common.enum.distance_metric_type import DistanceMetricType

# Access enum members
metric1 = DistanceMetricType.EUCLIDEAN
metric2 = DistanceMetricType.MANHATTAN
metric3 = DistanceMetricType.COSINE
metric4 = DistanceMetricType.MAHALANOBIS

# Get string value
print(metric1.value)  # "euclidean"

# Compare enum members
if metric1 == DistanceMetricType.EUCLIDEAN:
    print("Using Euclidean distance")
```

### Use in Privacy Metrics Configuration

```python
from pamola_core.common.enum.distance_metric_type import DistanceMetricType
from pamola_core.metrics import PrivacyMetricsConfig

# Configure privacy metrics with specific distance metric
privacy_config = {
    "metric_type": DistanceMetricType.MAHALANOBIS,
    "quasi_identifiers": ["age", "zipcode", "gender"],
    "distance_calculation": "record_based"
}

# Distance to Closest Record (DCR) privacy metric
dcr_metric = {
    "name": "Distance to Closest Record",
    "distance_metric": DistanceMetricType.EUCLIDEAN,
    "threshold": 0.5
}
```

### Select Metric Based on Data Characteristics

```python
from pamola_core.common.enum.distance_metric_type import DistanceMetricType
import pandas as pd
import numpy as np

def select_distance_metric(df: pd.DataFrame, columns: list) -> DistanceMetricType:
    """Select appropriate distance metric based on data characteristics."""

    # Check for correlations
    corr_matrix = df[columns].corr()
    max_correlation = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()

    if max_correlation > 0.7:
        # High correlation - use Mahalanobis to account for it
        return DistanceMetricType.MAHALANOBIS

    # Check for outliers
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).sum().sum()

    if outliers > len(df) * 0.05:
        # >5% outliers - use Manhattan (robust)
        return DistanceMetricType.MANHATTAN

    # Default to Euclidean
    return DistanceMetricType.EUCLIDEAN

# Usage
df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "income": [30000, 35000, 40000, 45000, 50000],
    "score": [75, 80, 85, 90, 95]
})

metric = select_distance_metric(df, ["age", "income", "score"])
print(f"Selected metric: {metric.value}")
```

### Calculate Distance Between Records

```python
from pamola_core.common.enum.distance_metric_type import DistanceMetricType
import numpy as np
from scipy.spatial.distance import euclidean, manhattan, cosine, mahalanobis

def calculate_distance(record1, record2, metric: DistanceMetricType):
    """Calculate distance between two records."""

    if metric == DistanceMetricType.EUCLIDEAN:
        return euclidean(record1, record2)
    elif metric == DistanceMetricType.MANHATTAN:
        return manhattan(record1, record2)
    elif metric == DistanceMetricType.COSINE:
        return cosine(record1, record2)
    elif metric == DistanceMetricType.MAHALANOBIS:
        cov_matrix = np.cov(record1, record2)
        try:
            return mahalanobis(record1, record2, np.linalg.inv(cov_matrix))
        except:
            return euclidean(record1, record2)  # Fallback
    else:
        raise ValueError(f"Unknown metric: {metric}")

# Example
original = np.array([25, 35000, 75])
anonymized = np.array([30, 35000, 80])

for metric in [DistanceMetricType.EUCLIDEAN, DistanceMetricType.MANHATTAN]:
    dist = calculate_distance(original, anonymized, metric)
    print(f"{metric.value}: {dist:.2f}")
```

## Best Practices

1. **Choose Metric Based on Data Characteristics**
   ```python
   from pamola_core.common.enum.distance_metric_type import DistanceMetricType

   # Correlated data: Use Mahalanobis
   metric = DistanceMetricType.MAHALANOBIS

   # Sparse/high-dimensional: Use Cosine
   metric = DistanceMetricType.COSINE

   # General numeric: Use Euclidean
   metric = DistanceMetricType.EUCLIDEAN
   ```

2. **Document Metric Selection**
   ```python
   def privacy_analysis(quasi_identifiers: list) -> dict:
       """
       Analyze privacy preservation.

       Uses Mahalanobis distance because quasi-identifiers
       are often correlated (age-income, zipcode-demographics).
       """
       metric = DistanceMetricType.MAHALANOBIS
       return {"metric": metric, ...}
   ```

3. **Normalize Data Appropriately**
   ```python
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   from pamola_core.common.enum.distance_metric_type import DistanceMetricType

   # For Euclidean, normalize to equal scales
   scaler = StandardScaler()
   df_normalized = pd.DataFrame(
       scaler.fit_transform(df),
       columns=df.columns
   )

   metric = DistanceMetricType.EUCLIDEAN
   # Calculate distances on normalized data
   ```

4. **Use Enum in Type Hints**
   ```python
   from pamola_core.common.enum.distance_metric_type import DistanceMetricType

   def calculate_privacy_metrics(
       original_df,
       anonymized_df,
       metric: DistanceMetricType
   ) -> dict:
       """Calculate privacy metrics using specified distance metric."""
       pass
   ```

5. **Document Comparison Rationale**
   ```python
   # Good - clear documentation
   """
   EUCLIDEAN vs MANHATTAN:
   - EUCLIDEAN: Assumes equal dimension importance
   - MANHATTAN: More robust to outliers

   Choose based on data characteristics and analysis goals.
   """
   ```

## Related Components

- **PrivacyMetricsType** (`pamola_core.common.enum.privacy_metrics_type`) - Privacy metrics using distance
- **Distance to Closest Record (DCR)** (`pamola_core.metrics.privacy.distance`) - Privacy metric
- **Nearest Neighbor Distance Ratio (NNDR)** (`pamola_core.metrics.privacy.neighbor`) - Privacy metric
- **Privacy Analysis** (`pamola_core.analysis`) - Uses distance metrics for evaluation

## Computational Complexity

| Metric | Time Complexity | Space Complexity | Notes |
|--------|-----------------|------------------|-------|
| EUCLIDEAN | O(d) | O(1) | d = dimensions |
| MANHATTAN | O(d) | O(1) | Fastest for sparse data |
| COSINE | O(d) | O(1) | Requires normalization |
| MAHALANOBIS | O(d³) | O(d²) | Requires covariance matrix |

## Related Metrics

- **Fidelity Metrics** (`pamola_core.common.enum.fidelity_metrics`) - Statistical similarity
- **Privacy Metrics** (`pamola_core.common.enum.privacy_metrics_type`) - Privacy evaluation
- **Utility Metrics** (`pamola_core.common.enum.utility_metrics_type`) - Data utility

## Implementation Notes

- All distance metrics are symmetric: distance(A,B) = distance(B,A)
- Metrics should be normalized to [0,1] range for comparison across different scales
- Mahalanobis requires non-singular covariance matrix
- Cosine distance measures angle in vector space (not geometric distance)
