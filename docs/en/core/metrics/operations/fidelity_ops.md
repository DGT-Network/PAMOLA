# Fidelity Operations
**Module:** pamola_core.metrics.operations.fidelity_ops  
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
Base class for all fidelity metric operations, defining common functionality, interface, and behavior for distributional similarity metrics (e.g., KS, KL). Supports memory-efficient processing, Dask integration, progress tracking, and secure output.

## 2. Source Code Hierarchy
- pamola_core/metrics/operations/fidelity_ops.py
  - class FidelityConfig
  - class FidelityOperation
    - __init__
    - execute
    - calculate_metrics
    - _generate_visualizations
    - _get_cache_parameters

## 3. Architecture
**Inheritance Tree:**
```
FidelityOperation
  └── BaseMetricsOperation
      └── BaseOperation
```
**Internal Common Utilities:**
- aggregation
- normalize
- validation

**Supporting Modules:**
- kl_divergence
- ks_test

**Developer Notes:**
- All fidelity operations inherit from BaseMetricsOperation for shared logic.
- Utilities are imported from commons for data processing and validation.

## 4. Data Flow
```mermaid
flowchart TD
    A[Start: Input Data] --> B[Validate Input]
    B --> C[Select Fidelity Metric]
    C --> D[Compute Metric (KL/KS)]
    D --> E[Aggregate Results]
    E --> F[Return Output]
```

## 5. Main Functionalities & Features
- Supports multiple distributional similarity metrics (KS, KL)
- Configurable normalization, confidence, and visualization
- Dask integration for large datasets
- Caching and progress tracking

## 6. API Reference & Key Methods
| Class/Method | Description |
|--------------|-------------|
| `FidelityConfig` | Configuration schema |
| `FidelityOperation.__init__` | Constructor with config fields |
| `FidelityOperation.execute(...)` | Runs the operation |
| `FidelityOperation.calculate_metrics(...)` | Computes fidelity metrics |
| `FidelityOperation._generate_visualizations(...)` | Creates visualizations |
| `FidelityOperation._get_cache_parameters()` | Returns cache parameters |

## 7. Usage Examples
```python
from pamola_core.metrics.operations.fidelity_ops import FidelityOperation
op = FidelityOperation(fidelity_metrics=["ks", "kl"], columns=["A", "B"])
result = op.calculate_metrics(df1, df2)
```

## 8. Sample Configuration JSON
```json
{
  "operation": "FidelityOperation", // operation-specific
  "metric": "kl_divergence",        // operation-specific (choose: kl_divergence, ks_test, etc.)
  "input_data": "data/input.csv",   // framework-level (common to all operations)
  "reference_data": "data/reference.csv", // framework-level
  "parameters": {
    "normalize": true,               // package-level (metrics package)
    "aggregation": "mean",          // package-level (metrics package)
    "threshold": 0.05                // operation-specific (fidelity metric threshold)
  }
}
```
*Comments:*
- Fields marked as **framework-level** are common across multiple packages/operations (e.g., input_data, reference_data).
- Fields marked as **package-level** are common to all metrics operations (e.g., normalize, aggregation).
- Fields marked as **operation-specific** are unique to this operation (e.g., metric, threshold).
*This configuration covers all verified test cases for KL and KS metrics, including normalization and aggregation options.*

## 9. Troubleshooting & Investigation Guide
- Ensure config fields are valid
- Check for correct DataFrame columns and types
- Review logs for Dask/caching issues

## 10. Summary Analysis
- Robust, extensible operation wrapper for fidelity metrics
- Fully covered by unit tests

## 11. Challenges, Limitations & Enhancement Opportunities
- Dask integration may require tuning for very large datasets
- Future: add more distributional metrics and visualization options

## 12. Related Components & References
- pamola_core/metrics/base_metrics_op.py
- pamola_core/metrics/operations/privacy_ops.py
- pamola_core/metrics/operations/utility_ops.py

## 13. Change Log & Contributors
- v4.0.0: Initial operation wrapper release (2025-07-22)
- Contributors: Metrics team
