# Disclosure Risk Metric
**Module:** pamola_core.metrics.privacy.disclosure_risk
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
Evaluates disclosure risk in anonymized datasets using multiple adversarial models. Implements prosecutor risk (attacker knows target is in dataset), journalist risk (attacker unsure if target is present), and marketer risk (focuses on unique records). Provides record-level and dataset-level risk calculations with configurable risk thresholds.

## 2. Source Code Hierarchy
- pamola_core/metrics/privacy/disclosure_risk.py
  - class DisclosureRiskMetric
    - __init__
    - calculate_metric
    - _calculate_prosecutor_risk
    - _calculate_journalist_risk
    - _calculate_marketer_risk
    - _assess_risk_level

## 3. Architecture & Data Flow
- Input: DataFrame with quasi-identifiers
- Processing: Group records by quasi-identifier combinations, calculate risk models
- Output: Risk scores (0-100) for each model and dataset-level aggregates

## 4. Main Functionalities & Features
- Prosecutor risk model: assumes target is in dataset
- Journalist risk model: assumes target may or may not be present
- Marketer risk model: focuses on proportion of unique records
- Record-level and dataset-level risk calculation
- Configurable risk threshold (default: 5%)
- Integration with k-anonymity and group size metrics
- Interpretable percentage-based risk scores

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(risk_threshold)` | Initialize disclosure risk metric with risk threshold |
| `calculate_metric(data, quasi_identifiers, k_column)` | Compute disclosure risk for dataset |

## 6. Usage Examples
```python
from pamola_core.metrics.privacy.disclosure_risk import DisclosureRiskMetric
import pandas as pd

data = pd.DataFrame({
    'age': [25, 25, 35, 35, 45],
    'gender': ['M', 'M', 'F', 'F', 'M'],
    'city': ['NYC', 'NYC', 'LA', 'LA', 'Chicago'],
    'salary': [50000, 51000, 75000, 76000, 100000]
})

# Calculate disclosure risk
risk_metric = DisclosureRiskMetric(risk_threshold=5.0)
result = risk_metric.calculate_metric(
    data=data,
    quasi_identifiers=['age', 'gender', 'city']
)

print(result['prosecutor_risk'])  # Risk if attacker knows target is in dataset
print(result['journalist_risk'])  # Risk if attacker unsure about presence
print(result['marketer_risk'])    # Proportion of unique records
print(result['overall_risk'])     # Aggregated risk level
print(result['records_at_risk'])  # Count of high-risk records
```

## 7. Troubleshooting & Investigation Guide
- Ensure quasi-identifiers are correctly specified
- Groups with single records (k=1) have highest risk
- Risk threshold (default 5%) can be adjusted for different privacy policies
- Check for missing values in quasi-identifier columns before calculation

## 8. Summary Analysis
- Enterprise-grade disclosure risk assessment
- Multiple adversarial models support different threat scenarios
- Production-ready for privacy impact assessments

## 9. Challenges, Limitations & Enhancement Opportunities
- Prosecutor risk assumes complete knowledge of quasi-identifiers
- Marketer risk calculation is simplified (unique record proportion)
- Does not account for external data availability
- Future: add t-closeness and differential privacy risk models

## 10. Related Components & References
- Part of privacy metrics suite in operations/privacy_ops.py
- Complements distance.py and identity.py for complete privacy assessment
- Integrates with k-anonymity group processing utilities

## 11. Change Log & Contributors
- v4.0.0: Disclosure risk metric implementation (2025-03)
- Contributors: Metrics team
