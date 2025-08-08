# Data Utils
**Module:** pamola_core.anonymization.commons.data_utils  
**Version:** 1.0.0  
**Last Updated:** 2025-07-29
**Author:** PAMOLA Core Team

## Table of Contents
1. [Module Overview](#1-module-overview)
2. [Source Code Hierarchy](#2-source-code-hierarchy)
3. [Architecture & Data Flow](#3-architecture--data-flow)
4. [Main Functionalities & Features](#4-main-functionalities--features)
5. [API Reference & Key Methods](#5-api-reference--key-methods)
6. [Usage Examples](#6-usage-examples)
7. [Troubleshooting & Investigation Guide](#7-troubleshooting--investigation-guide)
8. [Unit Test Coverage](#unit-test-coverage)
9. [Summary Analysis](#8-summary-analysis)
10. [Challenges, Limitations & Enhancement Opportunities](#9-challenges-limitations--enhancement-opportunities)
11. [Related Components & References](#10-related-components--references)
12. [Change Log & Contributors](#11-change-log--contributors)

---

## 1. Module Overview

Privacy-specific data processing utilities for anonymization operations. Extends general-purpose utilities with privacy-aware null handling, risk-based filtering, and vulnerable record strategies.

## 2. Source Code Hierarchy

- pamola/pamola_core/anonymization/commons/data_utils.py

## 3. Architecture & Data Flow

- Used by anonymization operations for privacy-focused data processing
- Integrates with risk assessment and profiling modules
- Functions are called by masking, suppression, and risk-based anonymization routines

## 4. Main Functionalities & Features

- Privacy-aware null value handling (`process_nulls`)
- Risk-based record filtering (`filter_records_conditionally`)
- Vulnerable record handling (`handle_vulnerable_records`)
- Factory functions for risk/adaptive processors
- Integration with profiling results and risk assessments

## 5. API Reference & Key Methods

| Function | Description |
|----------|-------------|
| `process_nulls(series, strategy, anonymize_value)` | Handles nulls with privacy strategies |
| `filter_records_conditionally(df, risk_field, risk_threshold, ...)` | Filters records by risk |
| `handle_vulnerable_records(df, field_name, vulnerability_mask, strategy, replacement_value)` | Handles vulnerable records |
| `create_risk_based_processor(strategy, risk_threshold)` | Factory for risk-based processor |
| `create_privacy_level_processor(privacy_level)` | Factory for privacy level processor |
| `get_risk_statistics(df, risk_field, ...)` | Computes risk statistics |
| `get_privacy_recommendations(risk_stats)` | Provides privacy recommendations |

## 6. Usage Examples

```python
import pandas as pd
from pamola_core.anonymization.commons import data_utils

# Null processing
s = pd.Series([1, None, 3])
processed = data_utils.process_nulls(s, strategy="ANONYMIZE", anonymize_value="MASKED")

# Risk-based filtering
df = pd.DataFrame({"risk": [1, 6, 10], "val": [10, 20, 30]})
filtered, mask = data_utils.filter_records_conditionally(df, risk_field="risk", risk_threshold=5)

# Handle vulnerable records
result = data_utils.handle_vulnerable_records(df, "val", mask, strategy="suppress", replacement_value="SUPP")
```

## 7. Troubleshooting & Investigation Guide
- Ensure input DataFrames/Series are not empty and have expected columns
- Check for correct risk field names and valid threshold values
- Review error messages for unsupported strategies

## Unit Test Coverage

All tests for `commons/data_utils.py` now pass with no skips or failures. The test suite covers all major privacy-aware data utilities, including null handling, risk-based filtering, and vulnerable record strategies. No codebase issues remain. Documentation is complete and up to date.

## 8. Summary Analysis

The `data_utils` module is a core component of the anonymization pipeline, providing privacy-aware data processing utilities for null handling, risk-based filtering, and vulnerable record management. It is used by all major anonymization operations to ensure data quality and compliance with privacy requirements. All utilities are validated, production-ready, and fully integrated with the PAMOLA Core framework.

- Centralizes common data utility functions for anonymization workflows
- Handles type conversions, null value strategies, and data normalization
- Improves code maintainability and reduces duplication across modules
- Supports efficient, scalable data processing for large datasets
- Enables consistent, business-aligned data handling for privacy operations
- Facilitates onboarding and future automation by providing reusable utilities

## 9. Challenges, Limitations & Enhancement Opportunities
- Some strategies may require tuning for edge cases
- Future: add more adaptive and context-aware anonymization strategies

## 10. Related Components & References
- Used by anonymization operation wrappers and risk assessment modules
- Related: pamola_core/metrics/commons/aggregation.py

## 11. Change Log & Contributors

- 2025-07-25: Initial auto-generated documentation and test coverage (PAMOLA Core Team)