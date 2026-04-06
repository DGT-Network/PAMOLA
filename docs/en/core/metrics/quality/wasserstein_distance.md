# Wasserstein Distance Quality Metric
**Module:** pamola_core.metrics.quality.wasserstein_distance
**Version:** 1.0.0
**Status:** Stable
**Last Updated:** March 2025

## Table of Contents
1. [Module Overview](#1-module-overview)
2. [Source Code Hierarchy](#2-source-code-hierarchy)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Main Functionalities & Features](#4-main-functionalities--features)
5. [API Reference & Key Methods](#5-api-reference--key-methods)
6. [Usage Examples](#6-usage-examples)
7. [Troubleshooting & Investigation Guide](#7-troubleshooting--investigation-guide)
8. [Summary Analysis](#8-summary-analysis)
9. [Challenges, Limitations & Enhancement Opportunities](#9-challenges-limitations--enhancement-opportunities)
10. [Related Components & References](#10-related-components--references)
11. [Change Log & Contributors](#11-change-log--contributors)

## 1. Module Overview
Computes Wasserstein Distance (also called Earth Mover's Distance or optimal transport distance) for measuring the minimum cost to transform one probability distribution into another. Wasserstein distance provides a geometrically meaningful metric for comparing distributions, particularly useful for assessing quality of synthetic and anonymized data.

## 2. Source Code Hierarchy
- pamola_core/metrics/quality/wasserstein_distance.py
  - class WassersteinDistance
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original and transformed/synthetic/anonymized data samples
- Processing: Compute optimal transport between empirical distributions
- Output: Wasserstein distance (non-negative, bounded by max value difference)

## 4. Main Functionalities & Features
- Calculate 1D Wasserstein distance for univariate distributions
- Compute sliced Wasserstein distance for multivariate data
- Geometrically interpretable metric: minimum cost transport
- Robust to outliers compared to KL divergence
- Support for continuous and discrete data
- Column-level and dataset-level distance calculation

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize Wasserstein distance metric |
| `calculate_metric(original_df, transformed_df)` | Compute Wasserstein distance |

## 6. Usage Examples
```python
from pamola_core.metrics.quality.wasserstein_distance import WassersteinDistance
import pandas as pd
import numpy as np

# Original and synthetic data
np.random.seed(42)
original = pd.DataFrame({
    'age': np.random.normal(40, 10, 1000),
    'income': np.random.exponential(75000, 1000)
})

# Synthetic data with slight distribution shift
synthetic = pd.DataFrame({
    'age': np.random.normal(41, 10.5, 1000),
    'income': np.random.exponential(76000, 1000)
})

# Calculate Wasserstein distance
ws_metric = WassersteinDistance()
result = ws_metric.calculate_metric(original, synthetic)

print(result['wasserstein_distance'])  # Total Wasserstein distance
print(result['age_distance'])          # Per-column distance for 'age'
print(result['income_distance'])       # Per-column distance for 'income'
# Lower values indicate better distribution preservation
```

## 7. Troubleshooting & Investigation Guide
- Wasserstein distance is bounded by maximum value difference in each dimension
- For continuous data with large value ranges, normalize before comparison
- Sliced Wasserstein allows efficient multivariate computation
- Distance is in original data units; interpret relative to data range

## 8. Summary Analysis
- Geometrically meaningful distribution distance metric
- More robust to outliers than KL divergence
- Production-ready for synthetic data quality assessment

## 9. Challenges, Limitations & Enhancement Opportunities
- Computationally expensive for very large datasets
- Sensitive to marginal distributions, not joint distribution structure
- Sliced version is approximation; exact computation slower
- Future: add Sinkhorn divergence and optimal transport plans

## 10. Related Components & References
- Part of quality metrics in operations
- Complements KL divergence and KS test for comprehensive distribution assessment
- Used for evaluating synthetic data utility and fidelity

## 11. Change Log & Contributors
- v1.0.0: Wasserstein distance quality metric implementation (2025-03)
- Contributors: Metrics team
