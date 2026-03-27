# R² Score Utility Metric
**Module:** pamola_core.metrics.utility.r2_score
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
Computes R² Score (coefficient of determination) for evaluating utility of anonymized datasets in regression tasks. R² measures the proportion of variance in the target variable explained by the regression model, providing an interpretable 0-1 metric for model performance. Essential for assessing how well regression models maintain predictive power on anonymized data.

## 2. Source Code Hierarchy
- pamola_core/metrics/utility/r2_score.py
  - class R2Score
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Actual and predicted regression values
- Processing: Calculate residual sum of squares, total sum of squares
- Output: R² score (0-1 range, where 1 is perfect, 0 is baseline model)

## 4. Main Functionalities & Features
- Calculate R² Score (coefficient of determination)
- Interpretation: proportion of variance explained by model
- Adjusted R² for multi-variable regression
- Comparison of R² between original and anonymized models
- Scale-independent metric suitable for cross-dataset comparison
- Intuitive interpretation (0-1 range)

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize R² Score metric |
| `calculate_metric(actual_values, predicted_values)` | Compute R² score |

## 6. Usage Examples
```python
from pamola_core.metrics.utility.r2_score import R2Score
import pandas as pd
import numpy as np

# Regression task: predicting salary from anonymized data
actual_salary = [50000, 75000, 100000, 125000, 150000]

# Predictions on original data
pred_original = [49000, 76000, 99000, 126000, 151000]

# Predictions on anonymized data
pred_anonymized = [48000, 78000, 98000, 128000, 149000]

# Calculate R² Score
r2_metric = R2Score()
result_original = r2_metric.calculate_metric(actual_salary, pred_original)
result_anon = r2_metric.calculate_metric(actual_salary, pred_anonymized)

print(result_original['r2_score'])    # R² on original data (0-1)
print(result_anon['r2_score'])        # R² on anonymized data
# Compare scores to assess utility degradation
print(f"Utility loss: {result_original['r2_score'] - result_anon['r2_score']:.2%}")
```

## 7. Troubleshooting & Investigation Guide
- Ensure actual and predicted values have same length
- R² can be negative if model performs worse than baseline
- R² not suitable for non-linear or misspecified models
- Compare R² across datasets for utility degradation assessment

## 8. Summary Analysis
- Standard metric for regression model evaluation
- Widely recognized and interpretable (0-1 scale)
- Scale-independent, suitable for cross-dataset comparison

## 9. Challenges, Limitations & Enhancement Opportunities
- R² assumes linear relationship; may not reflect true model quality
- Adjusted R² required for fair multi-variable comparison
- Can be misleading with high dimensionality
- Future: add correlation-based metrics and partial R²

## 10. Related Components & References
- Part of utility metrics suite in operations/utility_ops.py
- Complements mean_squared_error.py for regression evaluation
- Used for assessing utility of anonymized training data

## 11. Change Log & Contributors
- v1.0.0: R² Score metric implementation (2025-03)
- Contributors: Metrics team
