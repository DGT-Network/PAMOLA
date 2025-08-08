# Aggregation Utilities
**Module:** pamola_core.metrics.commons.aggregation  
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

# Aggregation Utilities (`aggregation.py`)

## 1. Module Overview
Provides utilities for aggregating column-level or metric-level scores into dataset-level or composite quality scores. Used for summarizing metric outputs across columns or metric types, supporting simple/weighted aggregation and composite index creation.

## 2. Source Code Hierarchy
- pamola_core/metrics/commons/aggregation.py
  - aggregate_column_metrics
  - create_composite_score
  - create_value_dictionary

## 3. Architecture & Data Flow
- Called by operation wrappers or metric calculation routines
- Aggregates results from multiple columns/metrics into a single score

## 4. Main Functionalities & Features
- Aggregate column-level metrics to dataset-level
- Create composite scores from multiple metrics
- Aggregate values using composite keys

## 5. API Reference & Key Methods
| Function | Description |
|----------|-------------|
| `aggregate_column_metrics(column_results, method, weights)` | Aggregates column metrics |
| `create_composite_score(metrics, weights, normalization)` | Composite weighted score |
| `create_value_dictionary(df, keys)` | Aggregates values by keys |

## 6. Usage Examples
```python
column_results = {"age": {"ks": 0.9}, "income": {"ks": 0.8}}
score = aggregate_column_metrics(column_results, method="mean")
```

## 7. Troubleshooting & Investigation Guide
- Ensure input dicts are correctly structured
- Check for missing weights if using weighted aggregation

## 8. Summary Analysis
- Simple, robust aggregation logic
- Fully covered by unit tests

## 9. Challenges, Limitations & Enhancement Opportunities
- Weighted aggregation requires all weights to be specified
- Future: add support for more aggregation methods

## 10. Related Components & References
- Used by all metrics operation wrappers

## 11. Change Log & Contributors
- v4.0.0: Initial utility release (2025-07-22)
- Contributors: Metrics team
