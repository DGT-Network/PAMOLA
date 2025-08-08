# Nearest Neighbor Distance Ratio (NNDR) Privacy Metric
**Module:** pamola_core.metrics.privacy.neighbor  
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
Implements the Nearest Neighbor Distance Ratio (NNDR) privacy metric, quantifying the ratio of distances from synthetic records to their nearest and second-nearest real records. Used for privacy risk assessment in synthetic data.

## 2. Source Code Hierarchy
- pamola_core/metrics/privacy/neighbor.py
  - class NearestNeighborDistanceRatio
    - __init__
    - calculate_metric
    - _interpret_nndr

## 3. Architecture & Data Flow
- Used by privacy metric operations and wrappers
- Class-based implementation for extensibility

## 4. Main Functionalities & Features
- Computes NNDR statistics and risk assessment
- Supports multiple distance metrics, normalization, and thresholds

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(distance_metric, n_neighbors, normalize_features, threshold, realistic_threshold, at_risk_threshold)` | Constructor |
| `calculate_metric(original_df, transformed_df)` | Computes NNDR statistics |
| `_interpret_nndr(nndr_stats, high_risk_count)` | Returns privacy risk interpretation |

## 6. Usage Examples
```python
from pamola_core.metrics.privacy.neighbor import NearestNeighborDistanceRatio
metric = NearestNeighborDistanceRatio(distance_metric="euclidean", normalize_features=True, threshold=0.5)
result = metric.calculate_metric(original_df, transformed_df)
```

## 7. Troubleshooting & Investigation Guide
- Ensure input DataFrames are compatible
- Check for correct distance metric and normalization settings

## 8. Summary Analysis
- Robust, extensible privacy metric implementation
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Large datasets may require optimization
- Future: add more privacy risk metrics and advanced thresholds

## 10. Related Components & References
- pamola_core/metrics/privacy/distance.py
- pamola_core/metrics/privacy/identity.py

## 11. Change Log & Contributors
- v4.0.0: Initial metric release (2025-07-22)
- Contributors: Metrics team
