# Identity Privacy Metrics (Uniqueness)
**Module:** pamola_core.metrics.privacy.identity  
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
Implements core privacy metrics for assessing uniqueness and re-identification risk in tabular datasets. Provides k-anonymity, l-diversity, and t-closeness calculations for privacy risk analysis.

## 2. Source Code Hierarchy
- pamola_core/metrics/privacy/identity.py
  - class Uniqueness
    - __init__
    - calculate_metric
    - _calculate_l_diversity
    - _calculate_t_closeness

## 3. Architecture & Data Flow
- Used by privacy metric operations and wrappers
- Class-based implementation for extensibility

## 4. Main Functionalities & Features
- Calculates k-anonymity, l-diversity, t-closeness
- Supports custom groupings and risk analysis

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__(quasi_identifiers, sensitives, k_values, l_diversity, t_closeness)` | Constructor |
| `calculate_metric(df)` | Computes all enabled privacy metrics |
| `_calculate_l_diversity(df)` | l-diversity statistics |
| `_calculate_t_closeness(df)` | t-closeness statistics |

## 6. Usage Examples
```python
from pamola_core.metrics.privacy.identity import Uniqueness
metric = Uniqueness(quasi_identifiers=["id"], sensitives=["sensitive"], k_values=[2, 5, 10])
result = metric.calculate_metric(df)
```

## 7. Troubleshooting & Investigation Guide
- Ensure DataFrame contains required columns
- Check for correct groupings and value types

## 8. Summary Analysis
- Robust, extensible privacy metric implementation
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Large datasets may require optimization
- Future: add more privacy risk metrics

## 10. Related Components & References
- pamola_core/metrics/privacy/distance.py
- pamola_core/metrics/privacy/neighbor.py

## 11. Change Log & Contributors
- v4.0.0: Initial metric release (2025-07-22)
- Contributors: Metrics team
