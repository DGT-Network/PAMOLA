# L-Diversity Loss Metric
**Module:** pamola_core.metrics.utility.ldiversity_loss
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
Evaluates l-diversity loss as a utility metric for anonymized datasets. L-diversity measures the diversity of sensitive values within equivalence classes, with l-diversity loss quantifying the reduction in diversity after anonymization. Essential for assessing privacy-utility trade-offs in datasets with sensitive attributes protected through l-diversity.

## 2. Source Code Hierarchy
- pamola_core/metrics/utility/ldiversity_loss.py
  - class LDiversityLossMetric
    - __init__
    - calculate_metric
    - _calculate_diversity

## 3. Architecture & Data Flow
- Input: Original and anonymized DataFrames with sensitive attributes
- Processing: Group by quasi-identifiers, measure diversity reduction in sensitive values
- Output: L-diversity loss score (0-1) representing utility reduction

## 4. Main Functionalities & Features
- Calculate l-diversity at multiple levels (distinct count, entropy)
- Compare l-diversity before and after anonymization
- Quantify diversity loss as utility metric
- Support for categorical sensitive attributes
- Interpretable loss score for privacy-utility analysis

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize l-diversity loss metric |
| `calculate_metric(original_df, anonymized_df, quasi_identifiers, sensitive_attribute)` | Compute l-diversity loss |

## 6. Usage Examples
```python
from pamola_core.metrics.utility.ldiversity_loss import LDiversityLossMetric
import pandas as pd

# Original data with sensitive attribute
original = pd.DataFrame({
    'age': [25, 25, 35, 35, 45],
    'gender': ['M', 'M', 'F', 'F', 'M'],
    'disease': ['Diabetes', 'Heart', 'Asthma', 'Diabetes', 'Cancer']
})

# Anonymized with age generalization
anonymized = pd.DataFrame({
    'age': ['20-30', '20-30', '30-40', '30-40', '40-50'],
    'gender': ['M', 'M', 'F', 'F', 'M'],
    'disease': ['Diabetes', 'Heart', 'Asthma', 'Diabetes', 'Cancer']
})

# Calculate l-diversity loss
ldiv_loss = LDiversityLossMetric()
result = ldiv_loss.calculate_metric(
    original_df=original,
    anonymized_df=anonymized,
    quasi_identifiers=['age', 'gender'],
    sensitive_attribute='disease'
)

print(result['ldiversity_loss'])  # Loss due to diversity reduction
print(result['original_diversity']) # Original diversity level
print(result['anonymized_diversity']) # Anonymized diversity level
```

## 7. Troubleshooting & Investigation Guide
- Ensure quasi-identifiers correctly define equivalence classes
- Sensitive attribute must be categorical
- Empty equivalence classes handled gracefully
- Low diversity loss indicates better privacy-utility balance

## 8. Summary Analysis
- Specialized metric for l-diversity-protected datasets
- Bridges privacy and utility assessment frameworks
- Production-ready for l-diversity policy validation

## 9. Challenges, Limitations & Enhancement Opportunities
- L-diversity alone doesn't guarantee strong privacy
- Entropy-based diversity may be skewed by rare values
- Doesn't account for attribute generalization impact
- Future: add t-closeness and differential privacy loss metrics

## 10. Related Components & References
- Part of utility metrics suite in operations/utility_ops.py
- Complements privacy/disclosure_risk.py for complete assessment
- Used for validating l-diversity policy compliance

## 11. Change Log & Contributors
- v1.0.0: L-diversity loss metric implementation (2025-03)
- Contributors: Metrics team
