# Privacy Operations
**Module:** pamola_core.metrics.operations.privacy_ops  
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
Base class for all privacy metric operations, defining common functionality, interface, and behavior for privacy risk metrics (e.g., DCR, NNDR, Uniqueness). Supports memory-efficient processing, Dask integration, progress tracking, and secure output.

## 2. Source Code Hierarchy
- pamola_core/metrics/operations/privacy_ops.py
  - class PrivacyMetricConfig
  - class PrivacyMetricOperation
    - __init__
    - execute
    - calculate_metrics
    - _generate_dcr_visualizations
    - _generate_nndr_visualizations
    - _generate_uniqueness_visualizations
    - _get_cache_parameters

## 3. Architecture
**Inheritance Tree:**
```text
PrivacyOperation
  └── BaseMetricsOperation
      └── BaseOperation
```

**Internal Common Utilities:**
- validation
- aggregation

**Supporting Modules:**
- distance
- identity
- neighbor

**Developer Notes:**
- Privacy operations extend BaseMetricsOperation for shared privacy logic.
- Utilities are used for validation and aggregation of privacy metrics.

## 4. Data Flow

The following flowchart illustrates the step-by-step data flow in the `execute()` method of `PrivacyMetricOperation`:

```mermaid
flowchart TD
    A[Start: Input Data] --> B[Validate Input]
    B --> C[Select Privacy Metric]
    C --> D[Compute Metric (Distance/Identity/Neighbor)]
    D --> E[Aggregate Results]
    E --> F[Return Output]
```

## 5. Main Functionalities & Features
- Supports multiple privacy risk metrics (DCR, NNDR, Uniqueness)
- Configurable columns, description, and visualization
- Dask integration for large datasets
- Caching and progress tracking

## 6. API Reference & Key Methods
| Class/Method | Description |
|--------------|-------------|
| `PrivacyMetricConfig` | Configuration schema |
| `PrivacyMetricOperation.__init__` | Constructor with config fields |
| `PrivacyMetricOperation.execute(...)` | Runs the operation |
| `PrivacyMetricOperation.calculate_metrics(...)` | Computes privacy metrics |
| `PrivacyMetricOperation._generate_dcr_visualizations(...)` | DCR visualizations |
| `PrivacyMetricOperation._generate_nndr_visualizations(...)` | NNDR visualizations |
| `PrivacyMetricOperation._generate_uniqueness_visualizations(...)` | Uniqueness visualizations |
| `PrivacyMetricOperation._get_cache_parameters()` | Returns cache parameters |

## 7. Usage Examples
```python
from pamola_core.metrics.operations.privacy_ops import PrivacyMetricOperation
op = PrivacyMetricOperation(privacy_metrics=["dcr", "nndr"], columns=["A", "B"])
result = op.calculate_metrics(df1, df2)
```

## 8. Sample Configuration JSON
```json
{
  "operation": "PrivacyOperation",
  "metric": "distance",
  "input_data": "data/input.csv",
  "reference_data": "data/reference.csv",
  "parameters": {
    "neighbor_k": 5,
    "identity_threshold": 0.1,
    "aggregation": "max"
  }
}
```
*This configuration covers all verified test cases for distance, identity, and neighbor privacy metrics.*

## 9. Troubleshooting & Investigation Guide
- Ensure config fields are valid
- Check for correct DataFrame columns and types
- Review logs for Dask/caching issues

## 10. Summary Analysis
- Robust, extensible operation wrapper for privacy metrics
- Fully covered by unit tests

## 11. Challenges, Limitations & Enhancement Opportunities
- Dask integration may require tuning for very large datasets
- Future: add more privacy metrics and visualization options

## 12. Related Components & References
- pamola_core/metrics/base_metrics_op.py
- pamola_core/metrics/operations/fidelity_ops.py
- pamola_core/metrics/operations/utility_ops.py

## 13. Change Log & Contributors
- v4.0.0: Initial operation wrapper release (2025-07-22)
- Contributors: Metrics team
