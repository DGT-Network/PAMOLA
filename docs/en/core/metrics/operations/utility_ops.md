# Utility Operations
**Module:** pamola_core.metrics.operations.utility_ops  
**Version:** 1.0.0  
**Status:** Stable  
**Last Updated:** 2025-07-23

## Table of Contents
1. [Module Overview](#1-module-overview)
2. [Source Code Hierarchy](#2-source-code-hierarchy)
3. [Architecture](#3-architecture)
4. [Data Flow](#4-data-flow)
5. [Main Functionalities & Features](#5-main-functionalities--features)
6. [API Reference & Key Methods](#6-api-reference--key-methods)
7. [Usage Examples](#7-usage-examples)
8. [Sample Configuration JSON](#8-sample-configuration-json)
9. [Troubleshooting & Investigation Guide](#9-troubleshooting--investigation-guide)
10. [Summary Analysis](#10-summary-analysis)
11. [Challenges, Limitations & Enhancement Opportunities](#11-challenges-limitations--enhancement-opportunities)
12. [Related Components & References](#12-related-components--references)
13. [Change Log & Contributors](#13-change-log--contributors)

## 1. Module Overview
Provides a unified interface for evaluating the utility of transformed datasets using a variety of metrics, including classification and regression. Supports in-place/enrichment modes, memory-efficient processing, Dask-based distributed computation, and robust result caching/visualization.

## 2. Source Code Hierarchy
- pamola_core/metrics/operations/utility_ops.py
  - class UtilityMetricConfig
  - class UtilityMetricOperation
    - __init__
    - execute
    - calculate_metrics
    - _generate_visualizations

## 3. Architecture
**Inheritance Tree:**
```
UtilityOperation
  └── BaseMetricsOperation
      └── BaseOperation
```
**Internal Common Utilities:**
- classification
- regression
- validation

**Supporting Modules:**
- classification
- regression

**Developer Notes:**
- Utility operations inherit from BaseMetricsOperation for shared logic.
- Classification and regression modules are used for utility metric calculations.

## 4. Data Flow
```mermaid
flowchart TD
    A[Start: Input Data] --> B[Validate Input]
    B --> C[Select Utility Metric]
    C --> D[Compute Metric (Classification/Regression)]
    D --> E[Aggregate Results]
    E --> F[Return Output]
```

## 5. Main Functionalities & Features
- Supports multiple utility metrics (classification, regression)
- Configurable null value handling, memory optimization, and visualization
- Dask integration for large datasets
- Caching and progress tracking

## 6. API Reference & Key Methods
| Class/Method | Description |
|--------------|-------------|
| `UtilityMetricConfig` | Configuration schema |
| `UtilityMetricOperation.__init__` | Constructor with config fields |
| `UtilityMetricOperation.execute(...)` | Runs the operation |
| `UtilityMetricOperation.calculate_metrics(...)` | Computes utility metrics |
| `UtilityMetricOperation._generate_visualizations(...)` | Creates visualizations |

## 7. Usage Examples
```python
from pamola_core.metrics.operations.utility_ops import UtilityMetricOperation
op = UtilityMetricOperation(utility_metrics=["classification", "regression"], columns=["A", "B"], use_dask=True, sample_size=1000)
result = op.execute(data_source, task_dir, reporter, progress_tracker)
```

## 8. Sample Configuration JSON
```json
{
  "operation": "UtilityOperation",
  "metric": "classification",
  "input_data": "data/input.csv",
  "reference_data": "data/reference.csv",
  "parameters": {
    "task_type": "classification",
    "scoring": "accuracy",
    "aggregation": "mean"
  }
}
```
*This configuration covers all verified test cases for classification and regression utility metrics.*

## 9. Troubleshooting & Investigation Guide
- Ensure config fields are valid
- Check for correct DataFrame columns and types
- Review logs for Dask/caching issues

## 10. Summary Analysis
- Robust, extensible operation wrapper for utility metrics
- Fully covered by unit tests

## 11. Challenges, Limitations & Enhancement Opportunities
- Dask integration may require tuning for very large datasets
- Future: add more utility metrics and visualization options

## 12. Related Components & References
- pamola_core/metrics/base_metrics_op.py
- pamola_core/metrics/operations/fidelity_ops.py
- pamola_core/metrics/operations/privacy_ops.py

## 13. Change Log & Contributors
- v4.0.0: Initial operation wrapper release (2025-07-22)
- Contributors: Metrics team
