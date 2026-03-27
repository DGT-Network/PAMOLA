# Statistical Fidelity Metric
**Module:** pamola_core.metrics.fidelity.statistical_fidelity
**Version:** 4.0.0
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
Assesses the statistical fidelity of anonymized or synthetic datasets by measuring how well statistical properties and relationships are preserved during privacy transformations. Evaluates mean preservation, variance preservation, and correlation preservation across numerical and categorical data, supporting both column-level and dataset-level analysis.

## 2. Source Code Hierarchy
- pamola_core/metrics/fidelity/statistical_fidelity.py
  - class StatisticalFidelityMetric
    - __init__
    - calculate_metric
    - _calculate_mean_preservation
    - _calculate_variance_preservation
    - _calculate_correlation_preservation

## 3. Architecture & Data Flow
- Input: original DataFrame and anonymized/synthetic DataFrame
- Processing: Compare statistical properties (means, variances, correlations)
- Output: Weighted composite fidelity score with component breakdown (0-100)

## 4. Main Functionalities & Features
- Calculate mean preservation for numerical columns
- Calculate variance preservation accuracy
- Calculate correlation preservation between column pairs
- Support for numerical and categorical data types
- Weighted aggregation of fidelity components (mean, variance, correlation)
- Column-level and dataset-level fidelity analysis
- Percentage-based scoring for interpretability

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(mean_weight, variance_weight, correlation_weight)` | Initialize fidelity metric with component weights |
| `calculate_metric(original_df, transformed_df)` | Compute overall statistical fidelity score |

## 6. Usage Examples
```python
from pamola_core.metrics.fidelity.statistical_fidelity import StatisticalFidelityMetric
import pandas as pd

# Original and anonymized data
original = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'income': [50000, 75000, 100000, 125000, 150000],
    'years_employed': [2, 5, 10, 15, 20]
})

anonymized = pd.DataFrame({
    'age': [26, 34, 46, 54, 66],
    'income': [51000, 74000, 101000, 124000, 151000],
    'years_employed': [2, 5, 10, 15, 20]
})

# Calculate statistical fidelity
fidelity = StatisticalFidelityMetric(
    mean_weight=0.4,
    variance_weight=0.3,
    correlation_weight=0.3
)

result = fidelity.calculate_metric(original, anonymized)
print(result['fidelity_score'])        # Overall fidelity (0-100)
print(result['mean_preservation'])     # Mean preservation component
print(result['variance_preservation']) # Variance preservation component
print(result['correlation_preservation']) # Correlation preservation component
```

## 7. Troubleshooting & Investigation Guide
- Ensure both DataFrames have same columns
- Numerical columns required for variance/correlation computation
- Categorical columns converted via one-hot encoding for correlation
- Weights must sum to 1.0; initialization validates this

## 8. Summary Analysis
- Comprehensive statistical property assessment
- Production-ready for fidelity evaluation in data anonymization
- Supports interpretable weighted scoring

## 9. Challenges, Limitations & Enhancement Opportunities
- One-hot encoding of categorical data increases dimensionality
- Correlation requires minimum 2 numerical columns
- Distribution similarity (beyond variance) not yet included
- Future: add distribution distance metrics (Wasserstein, KL divergence)

## 10. Related Components & References
- Part of fidelity metrics suite in operations/fidelity_ops.py
- Complements KL divergence and KS test metrics
- Used in FidelityOperation for comprehensive data utility assessment

## 11. Change Log & Contributors
- v4.0.0: Statistical fidelity metric implementation (2025-03)
- Contributors: Metrics team
