# Regression Utility Metric
**Module:** pamola_core.metrics.utility.regression  
**Version:** 1.0.0  
**Status:** Stable  
**Last Updated:** July 23, 2025

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
Provides a unified interface for evaluating the utility of data transformations using standard regression models and metrics. Assesses how well transformed data preserves predictive power for regression tasks.

## 2. Source Code Hierarchy
- pamola_core/metrics/utility/regression.py
  - class RegressionUtility
    - __init__
    - calculate_metric

## 3. Architecture & Data Flow
- Used by operation wrappers to evaluate regression utility
- Leverages scikit-learn models and metrics

## 4. Main Functionalities & Features
- Supports multiple models (linear, rf, svr)
- Calculates r2, mae, mse, rmse, pmse
- Cross-validation, test split, grouped R2

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(models, metrics, cv_folds, test_size)` | Constructor |
| `calculate_metric(original_df, transformed_df, value_field, key_fields, aggregation)` | Computes utility metrics |

## 6. Usage Examples
```python
from pamola_core.metrics.utility.regression import RegressionUtility
ru = RegressionUtility(models=["linear"], metrics=["r2", "mae"])
```

## 7. Troubleshooting & Investigation Guide
- Ensure scikit-learn and pandas are installed
- Check for correct target column and data types

## 8. Summary Analysis
- Robust, flexible utility for regression metric evaluation
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Large datasets may require Dask or batch processing
- Future: add more model/metric support

## 10. Related Components & References
- pamola_core/metrics/utility/classification.py

## 11. Change Log & Contributors
- v4.0.0: Initial utility release (2025-07-22)
- Contributors: Metrics team
