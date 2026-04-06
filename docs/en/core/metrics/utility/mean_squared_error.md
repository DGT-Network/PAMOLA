# Mean Squared Error Utility Metric
**Module:** pamola_core.metrics.utility.mean_squared_error
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
Computes Mean Squared Error (MSE) for evaluating utility of anonymized datasets in regression tasks. MSE measures the average squared difference between actual and predicted values, providing a quantitative assessment of how well regression models perform on anonymized data versus original data.

## 2. Source Code Hierarchy
- pamola_core/metrics/utility/mean_squared_error.py
  - class MeanSquaredError
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Input: Original and anonymized regression datasets with actual and predicted values
- Processing: Compute squared differences, calculate mean
- Output: MSE score (non-negative, unbounded), RMSE (root MSE)

## 4. Main Functionalities & Features
- Calculate Mean Squared Error for regression tasks
- Compute Root Mean Squared Error (RMSE) for interpretability
- Per-feature MSE calculation for detailed analysis
- Comparison of MSE between original and anonymized data
- Utility degradation assessment through MSE comparison
- Sensitive to large prediction errors

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__()` | Initialize Mean Squared Error metric |
| `calculate_metric(actual_values, predicted_values)` | Compute MSE and RMSE |

## 6. Usage Examples
```python
from pamola_core.metrics.utility.mean_squared_error import MeanSquaredError
import pandas as pd
import numpy as np

# Regression task: predicting income from anonymized data
actual_income = [50000, 75000, 100000, 125000, 150000]

# Predictions on original data
pred_original = [49000, 76000, 99000, 126000, 151000]

# Predictions on anonymized data
pred_anonymized = [48000, 78000, 98000, 128000, 149000]

# Calculate MSE
mse_metric = MeanSquaredError()
result_original = mse_metric.calculate_metric(actual_income, pred_original)
result_anon = mse_metric.calculate_metric(actual_income, pred_anonymized)

print(result_original['mse'])    # MSE on original data
print(result_original['rmse'])   # RMSE on original data
print(result_anon['mse'])        # MSE on anonymized data
print(result_anon['rmse'])       # RMSE on anonymized data
```

## 7. Troubleshooting & Investigation Guide
- Ensure actual and predicted values have same length
- MSE is scale-dependent; consider normalized MSE for comparison
- Large outliers can inflate MSE significantly
- RMSE in same units as target variable for better interpretability

## 8. Summary Analysis
- Standard metric for regression utility assessment
- Widely used in machine learning evaluation
- Production-ready for regression task evaluation

## 9. Challenges, Limitations & Enhancement Opportunities
- Sensitive to outliers and large errors
- Scale-dependent; not suitable for comparing different datasets
- Doesn't distinguish between bias and variance
- Future: add MAE, MAPE, and R-squared metrics

## 10. Related Components & References
- Part of utility metrics suite in operations/utility_ops.py
- Complements regression.py for comprehensive regression evaluation
- Used for assessing utility of anonymized training data

## 11. Change Log & Contributors
- v1.0.0: Mean Squared Error metric implementation (2025-03)
- Contributors: Metrics team
