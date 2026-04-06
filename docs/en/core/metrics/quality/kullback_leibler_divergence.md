# Kullback-Leibler Divergence Quality Metric
**Module:** pamola_core.metrics.quality.kullback_leibler_divergence
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
Computes Kullback-Leibler (KL) Divergence for measuring information loss in distribution change from original to transformed datasets. KL Divergence, also called relative entropy, quantifies how one probability distribution diverges from a reference distribution. Lower KL values indicate better distribution preservation and higher quality anonymized data.

## 2. Source Code Hierarchy
- pamola_core/metrics/quality/kullback_leibler_divergence.py
  - class KullbackLeiblerDivergence
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original and transformed/anonymized data (discrete or discretized continuous)
- Processing: Estimate probability distributions, calculate KL divergence
- Output: KL divergence in nats or bits (non-negative, unbounded)

## 4. Main Functionalities & Features
- Calculate KL divergence from reference to transformed distribution
- Support for discrete and continuous (discretized) data
- Handles zero-probability values with smoothing
- Column-level and dataset-level divergence calculation
- Asymmetric metric: KL(P||Q) ≠ KL(Q||P)
- Interpretable entropy-based metric

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize KL Divergence metric |
| `calculate_metric(original_df, transformed_df)` | Compute KL divergence |

## 6. Usage Examples
```python
from pamola_core.metrics.quality.kullback_leibler_divergence import KullbackLeiblerDivergence
import pandas as pd
import numpy as np

# Original and anonymized data
original = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'East'],
    'category': ['A', 'B', 'A', 'B', 'A']
})

transformed = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'East'],
    'category': ['A', 'B', 'A', 'A', 'B']
})

# Calculate KL divergence
kl_metric = KullbackLeiblerDivergence()
result = kl_metric.calculate_metric(original, transformed)

print(result['kl_divergence'])  # Total KL divergence (nats)
print(result['region_kl'])      # Per-column KL for 'region'
print(result['category_kl'])    # Per-column KL for 'category'
# Lower values indicate better distribution preservation
```

## 7. Troubleshooting & Investigation Guide
- Asymmetric: KL(original||transformed) vs KL(transformed||original) give different results
- Zero-probability handling: smoothing prevents infinity values
- Works best with discrete data; continuous data should be discretized
- Very large KL values indicate significant distribution change

## 8. Summary Analysis
- Information-theoretic approach to distribution quality assessment
- Production-ready for fidelity and quality evaluation
- Widely used in machine learning and information theory

## 9. Challenges, Limitations & Enhancement Opportunities
- Asymmetric; choice of direction matters
- Not normalized; values can be very large
- Sensitive to discretization for continuous data
- Future: add JS divergence and Wasserstein distance

## 10. Related Components & References
- Part of quality metrics in operations
- Complements kolmogorov_smirnov_test.py for distribution assessment
- Mirrors fidelity/distribution/kl_divergence.py for quality measurement

## 11. Change Log & Contributors
- v1.0.0: KL Divergence quality metric implementation (2025-03)
- Contributors: Metrics team
