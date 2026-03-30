# Information Loss Metrics
**Module:** pamola_core.metrics.utility.information_loss
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
Evaluates information loss during data anonymization using multiple strategies: generalization loss (granularity of generalized values), suppression loss (percentage of suppressed records), and overall utility reduction. Essential for assessing utility trade-offs in anonymized datasets and determining optimal anonymization parameters.

## 2. Source Code Hierarchy
- pamola_core/metrics/utility/information_loss.py
  - class InformationLossMetric
    - __init__
    - calculate_metric
  - class GeneralizationLossMetric
    - __init__
    - calculate_metric
  - class SuppressionLossMetric
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original DataFrame and anonymized DataFrame with generalization/suppression records
- Processing: Compare distribution/granularity, count suppressed records, calculate loss
- Output: Information loss scores (0-1 or 0-100) for different loss types

## 4. Main Functionalities & Features
- InformationLossMetric: Overall information loss from anonymization
- GeneralizationLossMetric: Loss due to value generalization (coarser granularity)
- SuppressionLossMetric: Loss from record/value suppression
- Support for numerical and categorical columns
- Weighted aggregation of different loss components
- Interpretable loss scores for decision-making

## 5. API Reference & Key Methods
| Class/Method | Description |
|---|---|
| `InformationLossMetric` | Overall information loss assessment |
| `calculate_metric(original_df, anonymized_df)` | Compute total information loss |
| `GeneralizationLossMetric` | Generalization-specific loss |
| `calculate_metric(original_df, anonymized_df)` | Compute generalization loss |
| `SuppressionLossMetric` | Suppression-specific loss |
| `calculate_metric(original_df, anonymized_df)` | Compute suppression loss |

## 6. Usage Examples
```python
from pamola_core.metrics.utility.information_loss import (
    InformationLossMetric,
    GeneralizationLossMetric,
    SuppressionLossMetric
)
import pandas as pd

# Original and generalized data
original = pd.DataFrame({
    'age': [25, 35, 45, 55, 65],
    'income': [50000, 75000, 100000, 125000, 150000],
    'zip': ['10001', '10002', '90001', '90002', '90003']
})

anonymized = pd.DataFrame({
    'age': ['20-30', '30-40', '40-50', '50-60', '60-70'],
    'income': ['<75K', '75-125K', '75-125K', '125-175K', '>150K'],
    'zip': ['100**', '100**', '900**', '900**', '900**']
})

# Calculate information loss
info_loss = InformationLossMetric()
result = info_loss.calculate_metric(original, anonymized)
print(result['information_loss'])  # Total loss (0-1)

# Calculate specific loss types
gen_loss = GeneralizationLossMetric()
gen_result = gen_loss.calculate_metric(original, anonymized)
print(gen_result['generalization_loss'])  # Loss from generalization

sup_loss = SuppressionLossMetric()
sup_result = sup_loss.calculate_metric(original, anonymized)
print(sup_result['suppression_loss'])  # Loss from suppression
```

## 7. Troubleshooting & Investigation Guide
- Ensure original and anonymized DataFrames have same columns
- Generalization loss calculation requires detecting hierarchy levels
- Suppression loss counts null/missing values as suppressed
- Compare loss scores with utility metrics for trade-off analysis

## 8. Summary Analysis
- Comprehensive information loss assessment framework
- Multiple loss types support fine-grained analysis
- Production-ready for anonymization parameter tuning

## 9. Challenges, Limitations & Enhancement Opportunities
- Generalization hierarchy detection is heuristic-based
- Suppression handling for missing vs explicitly suppressed values
- Value-level suppression (vs record-level) requires tracking
- Future: add Hellinger distance, mutual information loss metrics

## 10. Related Components & References
- Part of utility metrics suite in operations/utility_ops.py
- Complements utility/classification.py and utility/regression.py
- Used for assessing utility degradation in anonymization

## 11. Change Log & Contributors
- v1.0.0: Information loss metrics implementation (2025-03)
- Contributors: Metrics team
