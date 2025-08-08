# Validation Utilities
**Module:** pamola_core.metrics.commons.validation  
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

# Validation Utilities (`validation.py`)

## 1. Module Overview
Validation utilities for comparing datasets and ensuring compatibility between original and transformed data. Used across fidelity, utility, and privacy metric operations.

## 2. Source Code Hierarchy
- pamola_core/metrics/commons/validation.py
  - ValidationResult (dataclass)
  - validate_dataset_compatibility
  - validate_metric_inputs
  - validate_confidence_level
  - validate_epsilon

## 3. Architecture & Data Flow
- Used by all metric operation wrappers to validate input dataframes and parameters

## 4. Main Functionalities & Features
- Structured validation results (success, errors, warnings)
- DataFrame compatibility checks
- Metric input validation
- Confidence/epsilon checks

## 5. API Reference & Key Methods
| Function/Class | Description |
|----------------|-------------|
| `ValidationResult` | Dataclass for validation results |
| `validate_dataset_compatibility(df1, df2, ...)` | Checks DataFrame compatibility |
| `validate_metric_inputs(original, transformed, columns, metric_type)` | Validates columns/types |
| `validate_confidence_level(level)` | Checks confidence level |
| `validate_epsilon(epsilon)` | Checks epsilon value |

## 6. Usage Examples
```python
from pamola_core.metrics.commons.validation import validate_dataset_compatibility
result = validate_dataset_compatibility(df1, df2)
if not result.success:
    print(result.errors)
```

## 7. Troubleshooting & Investigation Guide
- Check for column/type mismatches
- Review error/warning messages in ValidationResult

## 8. Summary Analysis
- Fully tested, robust for all validation needs

## 9. Challenges, Limitations & Enhancement Opportunities
- Strict validation may require tuning for edge cases
- Future: add more flexible validation options

## 10. Related Components & References
- Used by all metrics operation wrappers

## 11. Change Log & Contributors
- v4.0.0: Initial utility release (2025-07-22)
- Contributors: Metrics team
