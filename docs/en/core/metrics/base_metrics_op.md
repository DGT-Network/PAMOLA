# Base Metrics Operation
**Module:** pamola_core.metrics.base_metrics_op  
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

# Base Metrics Operation (`base_metrics_op.py`)

## 1. Module Overview
Defines the `MetricsOperation` base class for all metrics operations in PAMOLA Core. Provides a unified interface for metrics calculation, result handling, distributed processing, and integration with the operation framework. Supports pandas and Dask backends, progress tracking, caching, and secure artifact management.

## 2. Source Code Hierarchy
- pamola_core/metrics/base_metrics_op.py
  - class MetricsOperation
    - __init__
    - execute
    - _validate_inputs
    - _get_dataset_by_name
    - _optimize_data
    - _sample_aligned
    - _calculate_metrics_with_config
    - calculate_metrics
    - _collect_basic_metrics
    - _handle_visualizations
    - _generate_visualizations
    - _cleanup_memory
    - _check_cache
    - _add_cached_metrics
    - _restore_cached_artifacts
    - _save_to_cache
    - _generate_cache_key

## 3. Architecture & Data Flow
- Inherited by all operation wrappers (fidelity, privacy, utility)
- Core operation lifecycle: validation → execution → result handling
- Data flows from data source → validation → metrics calculation → result/caching → visualization
- Supports both in-place (REPLACE) and enrichment (ENRICH) modes

## 4. Main Functionalities & Features
- Standardized operation lifecycle
- Configurable null value handling
- Memory-efficient processing for large datasets
- Caching and progress tracking
- Secure output generation with optional encryption
- Dask integration for distributed processing

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `__init__` | Constructor with all config options |
| `execute(data_source, task_dir, reporter, progress_tracker, **kwargs)` | Executes the operation |
| `_validate_inputs(...)` | Validates input dataframes |
| `_get_dataset_by_name(...)` | Retrieves datasets |
| `_optimize_data(df)` | Optimizes DataFrame memory |
| `_sample_aligned(...)` | Samples and aligns data |
| `_calculate_metrics_with_config(...)` | Calculates metrics |
| `calculate_metrics(...)` | Public metrics calculation |
| `_collect_basic_metrics(...)` | Collects statistics |
| `_handle_visualizations(...)` | Handles visualization |
| `_generate_visualizations(...)` | Generates visualizations |
| `_cleanup_memory(...)` | Cleans up memory |
| `_check_cache(...)` | Checks for cached results |
| `_add_cached_metrics(...)` | Adds cached metrics |
| `_restore_cached_artifacts(...)` | Restores cached artifacts |
| `_save_to_cache(...)` | Saves to cache |
| `_generate_cache_key(...)` | Generates cache key |

## 6. Usage Examples
```python
from pamola_core.metrics.base_metrics_op import MetricsOperation

# Example with all major config options (as tested)
op = MetricsOperation(
    name="test_metrics",
    mode="STANDALONE",
    columns=["A", "B"],
    column_mapping={"A": "A1"},
    normalize=False,
    confidence_level=0.9,
    description="desc",
    optimize_memory=False,
    sample_size=2,
    use_dask=True,
    npartitions=2,
    dask_partition_size="10MB",
    use_cache=False,
    use_encryption=True,
    encryption_mode="simple",
    encryption_key="key",
    visualization_theme="dark",
    visualization_backend="matplotlib",
    visualization_strict=True,
    visualization_timeout=60,
)

# Execute operation (with mocks or real objects as needed)
result = op.execute(data_source, task_dir, reporter, progress_tracker)
```

## 8. Sample Configuration JSON
```json
{
  "name": "test_metrics",              // framework-level
  "mode": "STANDALONE",                // framework-level
  "columns": ["A", "B"],               // package-level
  "column_mapping": {"A": "A1"},      // package-level
  "normalize": false,                    // package-level
  "confidence_level": 0.9,               // package-level
  "description": "desc",                // operation-specific
  "optimize_memory": false,              // package-level
  "sample_size": 2,                      // package-level
  "use_dask": true,                      // framework-level
  "npartitions": 2,                      // framework-level
  "dask_partition_size": "10MB",        // framework-level
  "use_cache": false,                    // framework-level
  "use_encryption": true,                 // framework-level
  "encryption_mode": "simple",          // framework-level
  "encryption_key": "key",              // framework-level
  "visualization_theme": "dark",        // package-level
  "visualization_backend": "matplotlib",// package-level
  "visualization_strict": true,          // package-level
  "visualization_timeout": 60            // package-level
}
```

## 7. Troubleshooting & Investigation Guide
- Ensure all required dependencies (pandas, numpy, dask) are installed
- Check for correct DataFrame structure and column mapping
- Use progress tracker for debugging long-running operations
- Review logs for cache and encryption issues

## 8. Summary Analysis
- High code quality and test coverage
- Robust for both small and large datasets
- Extensible for new metric types and wrappers

## 9. Challenges, Limitations & Enhancement Opportunities
- Dask integration may require tuning for very large datasets
- Encryption options depend on external key management
- Future: add more visualization backends and advanced caching

## 10. Related Components & References
- pamola_core/metrics/operations/fidelity_ops.py
- pamola_core/metrics/operations/privacy_ops.py
- pamola_core/metrics/operations/utility_ops.py

## 11. Change Log & Contributors
- v4.0.0: Unified base class finalized (2025-07-22)
- Contributors: Metrics team
