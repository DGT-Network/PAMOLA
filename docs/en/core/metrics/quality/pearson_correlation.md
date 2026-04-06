# Pearson Correlation Quality Metric
**Module:** pamola_core.metrics.quality.pearson_correlation
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
Computes Pearson Correlation Coefficient to assess preservation of linear relationships between variables in original and anonymized datasets. Pearson correlation measures the strength and direction of linear association between two continuous variables, providing insight into whether variable relationships are maintained after anonymization.

## 2. Source Code Hierarchy
- pamola_core/metrics/quality/pearson_correlation.py
  - class PearsonCorrelation
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original and anonymized numerical DataFrames
- Processing: Compute correlation matrices, compare correlation structures
- Output: Correlation preservation metrics (-1 to 1 range)

## 4. Main Functionalities & Features
- Calculate Pearson correlation coefficients
- Compare correlation matrices between original and anonymized data
- Measure correlation structure preservation
- Column-pair level correlation analysis
- Support for numerical data only
- Interpretable metric (-1 to 1 scale)

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize Pearson correlation metric |
| `calculate_metric(original_df, anonymized_df)` | Compute correlation preservation |

## 6. Usage Examples
```python
from pamola_core.metrics.quality.pearson_correlation import PearsonCorrelation
import pandas as pd
import numpy as np

# Original data with relationships
np.random.seed(42)
age = np.random.normal(40, 10, 100)
income = age * 2000 + np.random.normal(0, 10000, 100)  # Strong correlation
experience = age * 0.8 + np.random.normal(0, 5, 100)   # Strong correlation

original = pd.DataFrame({
    'age': age,
    'income': income,
    'experience': experience
})

# Anonymized with slight perturbation
anonymized = pd.DataFrame({
    'age': age + np.random.normal(0, 2, 100),
    'income': income + np.random.normal(0, 5000, 100),
    'experience': experience + np.random.normal(0, 1, 100)
})

# Calculate Pearson correlation preservation
corr_metric = PearsonCorrelation()
result = corr_metric.calculate_metric(original, anonymized)

print(result['correlation_preservation'])  # Overall preservation (0-1)
print(result['age_income_corr'])  # Correlation for age-income pair
print(result['age_experience_corr'])  # Correlation for age-experience pair
```

## 7. Troubleshooting & Investigation Guide
- Pearson correlation requires numerical data; encode categorical data first
- Sensitive to outliers; consider robust alternatives (Spearman)
- Requires minimum 2 numerical columns for pairwise correlation
- Strong correlations may indicate utility preservation but also potential information leakage

## 8. Summary Analysis
- Fundamental metric for assessing statistical relationship preservation
- Widely used in statistical quality assessment
- Production-ready for data utility evaluation

## 9. Challenges, Limitations & Enhancement Opportunities
- Linear relationships only; non-linear correlations not captured
- Sensitive to outliers
- Not suitable for categorical data
- Future: add Spearman, Kendall, and mutual information metrics

## 10. Related Components & References
- Part of quality metrics in operations
- Complements statistical_fidelity.py in fidelity assessment
- Used for evaluating preservation of variable relationships

## 11. Change Log & Contributors
- v1.0.0: Pearson correlation quality metric implementation (2025-03)
- Contributors: Metrics team
