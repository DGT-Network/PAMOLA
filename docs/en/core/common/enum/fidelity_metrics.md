# Fidelity Metrics Enumeration

**Module:** `pamola_core.common.enum.fidelity_metrics` & `fidelity_metrics_type`
**Classes:** `FidelityMetrics`, `FidelityMetricsType`
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

`FidelityMetrics` and `FidelityMetricsType` are string-based enumerations that define statistical metrics for comparing the similarity between original and anonymized datasets. These metrics measure how well data utility is preserved after privacy-preserving transformations.

**Parent Class:** `str, Enum`
**Type:** String Enum
**Scope:** Metrics calculations, utility evaluation
**Used By:** Data utility analysis, privacy-utility tradeoff assessment

## Enum Members

| Member | Value | Description |
|--------|-------|-------------|
| `KS` | `"ks"` | Kolmogorov-Smirnov test. Compares distributions via max difference in cumulative distribution functions. |
| `KL` | `"kl"` | Kullback-Leibler divergence. Measures asymmetric divergence between probability distributions. |
| `JS` | `"js"` | Jensen-Shannon divergence. Symmetric version of KL divergence with bounded range. |
| `WASSERSTEIN` | `"wasserstein"` | Wasserstein distance. Measures minimum cost to transform one distribution to another. |

## Metric Definitions

### KS (Kolmogorov-Smirnov Test)

**Formula:** `D = max|F_orig(x) - F_anon(x)|`

where F is the cumulative distribution function

**Characteristics:**
- Tests whether two distributions are different
- Computes maximum vertical distance between CDFs
- Distribution-free (no assumptions about distribution shape)
- Sensitive to differences anywhere in distribution
- Range: [0, 1] where 0 = identical, 1 = completely different
- Single-valued test statistic

**Interpretation:**
- KS = 0: Distributions are identical
- KS = 0.05: Slight difference
- KS = 0.2: Moderate difference
- KS = 0.5: Large difference

**Advantages:**
- Non-parametric (no distribution assumptions)
- Easy to interpret
- Fast computation
- Good for continuous data

**Disadvantages:**
- Only considers univariate distributions
- May be overly sensitive to location vs. scale differences
- Doesn't account for probability mass

**Use Cases:**
- Quick distribution comparison
- Goodness-of-fit testing
- Comparing continuous numeric columns

**Example:**
```
Original: Normal distribution (μ=0, σ=1)
Anonymized: Uniform distribution (0-1 range)
KS = 0.45  (Moderate difference in distributions)
```

### KL (Kullback-Leibler Divergence)

**Formula:** `D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))`

**Characteristics:**
- Asymmetric measure (KL(P||Q) ≠ KL(Q||P))
- Information-theoretic divergence
- Measures information loss when using Q instead of P
- Range: [0, ∞) where 0 = identical
- Undefined if Q(x)=0 and P(x)≠0

**Interpretation:**
- KL = 0: Distributions identical
- KL = 0.01: Very similar
- KL = 0.1: Similar
- KL = 1.0: Significantly different
- KL > 5: Very different

**Advantages:**
- Mathematically elegant
- Theoretically well-founded
- Captures subtle distribution differences
- Useful for optimization

**Disadvantages:**
- Asymmetric (depends on direction)
- Can be infinite/undefined
- Less intuitive interpretation
- Computationally expensive for large data

**Use Cases:**
- Variational inference
- Generative model evaluation
- Distribution divergence measurement
- Optimization objectives

**Example:**
```
Original: {A: 0.5, B: 0.3, C: 0.2}
Anonymized: {A: 0.4, B: 0.4, C: 0.2}
KL(Original||Anonymized) = 0.057
KL(Anonymized||Original) = 0.051
(Different values show asymmetry)
```

### JS (Jensen-Shannon Divergence)

**Formula:** `D_JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)`

where M = 0.5 * (P + Q)

**Characteristics:**
- Symmetric version of KL divergence
- Square root is proper metric
- Range: [0, 1] for normalized distributions
- Always defined (finite)
- Balanced information loss metric

**Interpretation:**
- JS = 0: Distributions identical
- JS = 0.01: Very similar
- JS = 0.1: Similar
- JS = 0.5: Very different

**Advantages:**
- Symmetric (treats distributions equally)
- Always defined and bounded
- Square root is metric
- Theoretically sound

**Disadvantages:**
- More computationally expensive than KL
- Less established in literature
- Less commonly optimized for

**Use Cases:**
- Comparing generative models (fair comparison)
- Evaluating anonymization quality
- Symmetric divergence measurement
- Probability distribution comparison

**Example:**
```
Original distribution: {A: 0.5, B: 0.3, C: 0.2}
Anonymized distribution: {A: 0.4, B: 0.4, C: 0.2}
JS = 0.054
(Symmetric measure, same either direction)
```

### WASSERSTEIN (Wasserstein Distance)

**Formula:** `W(P,Q) = inf_γ E[|X-Y|]`

where γ is a coupling of P and Q, X~P, Y~Q

**Characteristics:**
- Optimal transport distance
- Considers "cost" of transforming one distribution to another
- Takes into account ground metric (e.g., Euclidean distance)
- Range: [0, ∞) depending on data scale
- Always defined and finite

**Interpretation:**
- Wasserstein = 0: Distributions identical
- Low values (< 0.1): Similar distributions
- High values (> 1.0): Different distributions
- Interpretation depends on data scale and units

**Advantages:**
- Geometrically interpretable
- Accounts for ground metric
- Handles multivariate data well
- Useful for generative models
- Robust to small shifts

**Disadvantages:**
- Computationally expensive (requires optimization)
- Interpretation depends on ground metric
- Less commonly used in privacy literature
- More complex to compute

**Use Cases:**
- Comparing complex/multivariate distributions
- Generative adversarial networks (GANs)
- Optimal transport problems
- Distribution alignment measurement

**Example:**
```
1D case:
Original: [1, 2, 3, 4, 5]
Anonymized: [1.1, 2.1, 3.1, 4.1, 5.1]
Wasserstein ≈ 0.1  (Small shift/noise)

Original: [1, 2, 3, 4, 5]
Anonymized: [5, 4, 3, 2, 1]  (Reversed)
Wasserstein ≈ 3.0  (Large transformation)
```

## Use Cases

### Privacy-Utility Tradeoff Analysis

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

# Evaluate different anonymization levels
results = {}

for anonymization_level in [1, 5, 10]:
    # Apply anonymization
    anon_data = anonymize(data, level=anonymization_level)

    # Measure fidelity with multiple metrics
    results[anonymization_level] = {
        FidelityMetrics.KS.value: compute_ks(data, anon_data),
        FidelityMetrics.KL.value: compute_kl(data, anon_data),
        FidelityMetrics.JS.value: compute_js(data, anon_data),
        FidelityMetrics.WASSERSTEIN.value: compute_wasserstein(data, anon_data)
    }

# Choose level balancing privacy and utility
```

### Generative Model Evaluation

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

# Compare synthetic (generated) data to original
metrics = {
    FidelityMetrics.KL: "For optimization",
    FidelityMetrics.WASSERSTEIN: "For distribution similarity",
    FidelityMetrics.JS: "For symmetric comparison"
}

for metric, purpose in metrics.items():
    score = compute_metric(original, generated, metric.value)
    print(f"{metric.value}: {score:.4f} ({purpose})")
```

### Column-Level Utility Measurement

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

# Measure utility preservation for each column
column_metrics = {}

for column in df.columns:
    column_metrics[column] = {
        "ks": compute_ks(original[column], anonymized[column]),
        "kl": compute_kl(original[column], anonymized[column]),
        "js": compute_js(original[column], anonymized[column])
    }

# Identify columns with poor utility
poor_utility = [
    col for col, metrics in column_metrics.items()
    if metrics["ks"] > 0.3
]
```

## Usage Examples

### Basic Enum Usage

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics
from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType

# Both enums have same members
metric1 = FidelityMetrics.KS
metric2 = FidelityMetricsType.KL

# Get string value
print(metric1.value)  # "ks"
print(metric2.value)  # "kl"

# Compare
if metric1 == FidelityMetrics.KS:
    print("Using Kolmogorov-Smirnov test")
```

### Use in Configuration

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

# Configure utility metrics
config = {
    "metrics_to_compute": [
        FidelityMetrics.KS,
        FidelityMetrics.KL,
        FidelityMetrics.JS
    ],
    "threshold": {
        FidelityMetrics.KS.value: 0.1,
        FidelityMetrics.KL.value: 0.05,
        FidelityMetrics.JS.value: 0.05
    }
}
```

### Metric Selection by Use Case

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

def select_fidelity_metrics(use_case: str) -> list:
    """Select appropriate fidelity metrics for use case."""

    selections = {
        "quick_check": [FidelityMetrics.KS],
        "privacy_analysis": [
            FidelityMetrics.KL,
            FidelityMetrics.JS
        ],
        "generative_model": [
            FidelityMetrics.WASSERSTEIN,
            FidelityMetrics.KL
        ],
        "comprehensive": [
            FidelityMetrics.KS,
            FidelityMetrics.KL,
            FidelityMetrics.JS,
            FidelityMetrics.WASSERSTEIN
        ]
    }

    return selections.get(use_case, [FidelityMetrics.KS])

# Usage
metrics = select_fidelity_metrics("privacy_analysis")
print(f"Selected metrics: {[m.value for m in metrics]}")
```

### Compute Multiple Metrics

```python
from pamola_core.common.enum.fidelity_metrics import FidelityMetrics
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import numpy as np

def evaluate_fidelity(original, anonymized):
    """
    Compute multiple fidelity metrics.

    Parameters:
        original: Original data distribution
        anonymized: Anonymized data distribution

    Returns:
        dict: Metric results
    """
    results = {}

    # KS test
    ks_stat, ks_pval = ks_2samp(original, anonymized)
    results[FidelityMetrics.KS.value] = ks_stat

    # JS divergence
    hist_orig, bins = np.histogram(original, bins=20)
    hist_anon, _ = np.histogram(anonymized, bins=bins)
    p = hist_orig / hist_orig.sum()
    q = hist_anon / hist_anon.sum()
    results[FidelityMetrics.JS.value] = jensenshannon(p, q)

    return results

# Usage
results = evaluate_fidelity(original_data, anonymized_data)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

## Best Practices

1. **Use Multiple Metrics for Complete Picture**
   ```python
   from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

   # Good - comprehensive evaluation
   metrics = [FidelityMetrics.KS, FidelityMetrics.JS, FidelityMetrics.WASSERSTEIN]

   # Avoid - single metric may be misleading
   metrics = [FidelityMetrics.KS]
   ```

2. **Match Metric to Data Type**
   ```python
   # Good - appropriate for data
   # Univariate numeric: KS or KL
   # Multivariate: Wasserstein
   # Symmetric comparison: JS

   # Avoid - using random metric
   ```

3. **Set Reasonable Thresholds**
   ```python
   # Good - documented thresholds
   thresholds = {
       FidelityMetrics.KS.value: 0.1,      # For univariate
       FidelityMetrics.JS.value: 0.05,     # For symmetric
       FidelityMetrics.WASSERSTEIN.value: 0.5  # For multivariate
   }

   # Avoid - no thresholds or arbitrary values
   ```

4. **Document Metric Interpretation**
   ```python
   """
   Fidelity Metrics:
   - KS < 0.1: Good data utility (minimal distribution change)
   - KL < 0.05: Very similar distributions
   - JS < 0.05: Symmetric similarity
   - Wasserstein depends on data scale
   """
   ```

5. **Use Enum in Type Hints**
   ```python
   from pamola_core.common.enum.fidelity_metrics import FidelityMetrics

   def compute_fidelity(
       original,
       anonymized,
       metric: FidelityMetrics
   ) -> float:
       """Compute specified fidelity metric."""
       pass
   ```

## Related Components

- **UtilityMetricsType** (`pamola_core.common.enum.utility_metrics_type`) - Utility measurements
- **DistanceMetricType** (`pamola_core.common.enum.distance_metric_type`) - Distance metrics
- **Metrics Module** (`pamola_core.metrics`) - Implements fidelity calculations
- **Analysis** (`pamola_core.analysis`) - Uses fidelity metrics for evaluation

## Metric Comparison

| Metric | Type | Symmetry | Boundedness | Complexity | Use Case |
|--------|------|----------|-------------|-----------|----------|
| **KS** | Statistical | No | [0,1] | O(n log n) | Univariate comparison |
| **KL** | Information-theoretic | No | [0,∞) | O(n) | Optimization, asymmetric |
| **JS** | Information-theoretic | Yes | [0,1] | O(n) | Symmetric comparison |
| **WASSERSTEIN** | Optimal transport | Yes | [0,∞) | O(n³) | Multivariate, geometric |

## Implementation Notes

- `FidelityMetrics` and `FidelityMetricsType` are identical implementations
- Both inherit from `str` and `Enum` for string compatibility
- All values are lowercase for consistency
- Implementations in `pamola_core.metrics.fidelity.*` modules

## Summary

Fidelity metrics measure how well data utility is preserved after anonymization:

| Metric | Best For | Key Property |
|--------|----------|--------------|
| **KS** | Quick distribution check | Non-parametric |
| **KL** | Optimization target | Information loss |
| **JS** | Fair comparison | Symmetric |
| **WASSERSTEIN** | Complex distributions | Geometric distance |

Choose metrics based on your data characteristics and analysis goals.
