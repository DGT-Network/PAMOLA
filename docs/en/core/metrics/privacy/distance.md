# Distance Privacy Metrics
**Module:** pamola_core.metrics.privacy.distance  
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
Implements Distance to Closest Record (DCR) privacy metric, which measures the minimum distance from synthetic records to real records. Lower DCR values indicate higher privacy risk. Supports multiple distance metrics (Euclidean, Manhattan, Cosine, Mahalanobis), feature normalization, and efficient nearest neighbor search using FAISS for large-scale datasets.

## 2. Source Code Hierarchy
- pamola_core/metrics/privacy/distance.py
  - class DistanceToClosestRecord
    - __init__
    - calculate_metric
    - _calculate_dcr_faiss
    - _interpret_dcr
    - _calculate_dcr_sklearn
    - _calculate_privacy_score

## 3. Architecture & Data Flow
- DCR class calculates minimum distances from synthetic to original records
- Supports optional feature normalization via StandardScaler
- FAISS integration for large-scale datasets
- Returns detailed risk assessment with percentile-based statistics

## 4. Main Functionalities & Features
- Calculate DCR (Distance to Closest Record) for privacy assessment
- Support for multiple distance metrics: euclidean, manhattan, cosine, mahalanobis
- Optional feature normalization using StandardScaler
- Multiple aggregation methods: min, mean_k (k-nearest neighbors), percentile-based
- FAISS support for efficient large-scale nearest neighbor search
- Percentile distribution calculation for risk assessment
- Human-readable risk interpretation and recommendations

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(normalize_features, distance_metric, n_neighbors, batch_size)` | Initialize DCR metric with configuration |
| `calculate_metric(original_df, transformed_df)` | Compute DCR between original and synthetic datasets |
| `_calculate_dcr_faiss(...)` | Calculate DCR using FAISS index for efficient nearest neighbors |
| `_interpret_dcr(dcr_stats)` | Interpret DCR statistics into risk level |
| `_calculate_dcr_sklearn(...)` | Calculate DCR using scikit-learn |
| `_calculate_privacy_score(...)` | Calculate privacy score based on DCR stats |

## 6. Usage Examples
```python
from pamola_core.metrics.privacy.distance import DistanceToClosestRecord
import pandas as pd

# Create DCR metric instance
dcr = DistanceToClosestRecord(
    distance_metric='euclidean',
    normalize_features=True,
    aggregation='min',
    percentiles=[5, 25, 50, 75, 95]
)

# Calculate DCR between original and synthetic data
original = pd.DataFrame({'age': [25, 35, 45], 'income': [50000, 75000, 100000]})
synthetic = pd.DataFrame({'age': [26, 34, 44], 'income': [51000, 74000, 99000]})
result = dcr.calculate_metric(original, synthetic)
print(result['dcr'])  # DCR score
print(result['risk_level'])  # Risk assessment
```

## 7. Troubleshooting & Investigation Guide
- Ensure original and synthetic DataFrames have matching columns
- For Mahalanobis metric, ensure covariance matrix is full-rank
- FAISS requires faiss installation (pip install faiss-cpu or faiss-gpu)
- Check data preprocessing for NaN/missing values before calculation

## 8. Summary Analysis
- Enterprise-grade DCR implementation with FAISS acceleration
- Comprehensive risk assessment with multiple aggregation strategies
- Production-ready for synthetic data validation and privacy assessment

## 9. Challenges, Limitations & Enhancement Opportunities
- FAISS integration requires faiss-cpu or faiss-gpu installation
- Mahalanobis metric requires full-rank covariance matrix
- Feature preprocessing impacts DCR interpretation
- Future: add GPU acceleration and batch processing improvements

## 10. Related Components & References
- pamola_core/metrics/privacy/identity.py
- pamola_core/metrics/privacy/neighbor.py

## 11. Change Log & Contributors
- v4.0.0: Initial metric release (2025-07-22)
- Contributors: Metrics team
