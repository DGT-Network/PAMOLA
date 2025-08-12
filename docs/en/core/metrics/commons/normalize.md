# Normalization Utilities
**Module:** pamola_core.metrics.commons.normalize  
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

# Normalization Utilities (`normalize.py`)

## 1. Module Overview
Provides utilities for normalizing metric values and distributions to standard scales, typically [0, 1]. Supports value and array normalization, pluggable methods, and inversion.

## 2. Source Code Hierarchy
- pamola_core/metrics/commons/normalize.py
  - normalize_metric_value
  - normalize_array_np
  - normalize_array_sklearn

## 3. Architecture & Data Flow
- Used by metric operations and wrappers for consistent scaling
- Supports both numpy and scikit-learn normalization

## 4. Main Functionalities & Features
- Normalize single values or arrays
- Support for minmax, zscore, probability normalization
- Handles edge cases (zero range, zero std, etc.)

## 5. API Reference & Key Methods
| Function | Description |
|----------|-------------|
| `normalize_metric_value(value, metric_range, target_range, higher_is_better)` | Normalize a value |
| `normalize_array_np(values, method)` | Normalize array (minmax/zscore) |
| `normalize_array_sklearn(data, method)` | Normalize array with sklearn |

## 6. Usage Examples
```python
from pamola_core.metrics.commons.normalize import normalize_metric_value
val = normalize_metric_value(0.8, (0, 1), (0, 1), True)
```

## 7. Troubleshooting & Investigation Guide
- Check for zero range or zero std in input data
- Ensure scikit-learn is installed for sklearn methods

## 8. Summary Analysis
- Fully tested, robust for all metric normalization needs

## 9. Challenges, Limitations & Enhancement Opportunities
- Probability normalization requires valid probability input
- Future: add more normalization strategies

## 10. Related Components & References
- Used by all metrics operation wrappers

## 11. Change Log & Contributors
- v4.0.0: Initial utility release (2025-07-22)
- Contributors: Metrics team
