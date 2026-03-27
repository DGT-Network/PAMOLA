# Kolmogorov-Smirnov Test Metric
**Module:** pamola_core.metrics.quality.kolmogorov_smirnov_test
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
Implements the Kolmogorov-Smirnov (KS) test for comparing distributions between original and transformed datasets. KS test is a non-parametric statistical test that measures the maximum difference between two cumulative distribution functions, providing a distribution-free method for quality assessment.

## 2. Source Code Hierarchy
- pamola_core/metrics/quality/kolmogorov_smirnov_test.py
  - class KolmogorovSmirnovTest
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original and transformed/anonymized numerical data
- Processing: Compute cumulative distributions, calculate maximum difference
- Output: KS statistic (0-1), p-value for significance testing

## 4. Main Functionalities & Features
- Non-parametric distribution comparison
- Computes KS statistic (maximum difference between CDFs)
- Calculate p-value for statistical significance
- Column-level and dataset-level testing
- Suitable for numerical and categorical data (after encoding)
- Interpretation: small KS value indicates similar distributions

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize KS test metric |
| `calculate_metric(original_df, transformed_df)` | Compute KS test statistics |

## 6. Usage Examples
```python
from pamola_core.metrics.quality.kolmogorov_smirnov_test import KolmogorovSmirnovTest
import pandas as pd
import numpy as np

# Original and anonymized/synthetic data
original = pd.DataFrame({
    'age': np.random.normal(40, 10, 1000),
    'income': np.random.normal(75000, 20000, 1000)
})

transformed = pd.DataFrame({
    'age': np.random.normal(40.5, 10.5, 1000),
    'income': np.random.normal(74500, 21000, 1000)
})

# Calculate KS test
ks_test = KolmogorovSmirnovTest()
result = ks_test.calculate_metric(original, transformed)

print(result['ks_statistic'])  # KS statistic (0-1, lower is better)
print(result['p_value'])       # P-value for significance test
print(result['age_ks'])        # Per-column KS for 'age'
print(result['income_ks'])     # Per-column KS for 'income'
```

## 7. Troubleshooting & Investigation Guide
- KS test assumes independent samples
- Small p-value suggests significant distribution difference
- More sensitive to changes in distribution center than tails
- Works best with continuous numerical data
- Large sample sizes may flag small, inconsequential differences

## 8. Summary Analysis
- Classical non-parametric test for distribution comparison
- Distribution-free; no assumptions about data distribution
- Production-ready for quality assessment

## 9. Challenges, Limitations & Enhancement Opportunities
- Univariate only; multivariate KS test not available
- Sensitive to sample size; may give false positives with large N
- May not detect differences in tails effectively
- Future: add Wasserstein distance and multivariate tests

## 10. Related Components & References
- Part of quality metrics in operations
- Complements kullback_leibler_divergence.py for distribution assessment
- Used in fidelity evaluation for dataset quality

## 11. Change Log & Contributors
- v1.0.0: Kolmogorov-Smirnov test implementation (2025-03)
- Contributors: Metrics team
