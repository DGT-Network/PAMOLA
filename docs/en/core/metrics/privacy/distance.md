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
Suite of distance-based privacy metrics for evaluating similarity/dissimilarity between records, distributions, or datasets. Essential for privacy-preserving data analysis, synthetic data validation, and disclosure risk assessment.

## 2. Source Code Hierarchy
- pamola_core/metrics/privacy/distance.py
  - distance_metric
  - aggregate_distances
  - normalize_distance

## 3. Architecture & Data Flow
- Implements distance-based metrics as standalone functions
- Used by privacy metric operations and wrappers

## 4. Main Functionalities & Features
- Compute distance between records (Euclidean, Manhattan, Cosine, etc.)
- Aggregate and normalize distances
- Handles missing/NaN values gracefully

## 5. API Reference & Key Methods
| Function | Description |
|----------|-------------|
| `distance_metric(record1, record2, metric, normalize, ...)` | Computes distance between two records |
| `aggregate_distances(distances, method)` | Aggregates distance values |
| `normalize_distance(value, min_val, max_val)` | Normalizes a distance value |

## 6. Usage Examples
```python
from pamola_core.metrics.privacy.distance import distance_metric, aggregate_distances
score = distance_metric([1, 2, 3], [4, 5, 6], metric='euclidean')
agg = aggregate_distances([0.1, 0.2, 0.3], method='mean')
```

## 7. Troubleshooting & Investigation Guide
- Ensure input arrays are compatible and metric is supported
- Handles NaN/missing values by returning np.nan or skipping

## 8. Summary Analysis
- Robust, efficient for large arrays
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Custom metrics require user implementation
- Future: add more distance metrics and aggregation methods

## 10. Related Components & References
- pamola_core/metrics/privacy/identity.py
- pamola_core/metrics/privacy/neighbor.py

## 11. Change Log & Contributors
- v4.0.0: Initial metric release (2025-07-22)
- Contributors: Metrics team
