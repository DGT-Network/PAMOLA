# Field Validators
**Module:** pamola.pamola_core.anonymization.commons.validation.field_validators  
**Version:** 1.0  
**Last Updated:** 2025-07-29

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

This module provides a comprehensive set of validators for different field types in tabular data, including numeric, categorical, datetime, boolean, and text fields. Each validator enforces type, value, and constraint checks to ensure data quality and consistency for anonymization and data processing workflows.

## 2. Source Code Hierarchy

- pamola/pamola_core/anonymization/commons/validation/field_validators.py
  - NumericFieldValidator
  - CategoricalFieldValidator
  - DateTimeFieldValidator
  - BooleanFieldValidator
  - TextFieldValidator
  - FieldExistsValidator
  - PatternValidator
  - create_field_validator (factory)

## 3. Architecture & Data Flow

- Used by anonymization and data processing operations to validate input dataframes and fields
- Each validator class encapsulates logic for a specific data type
- Factory function enables dynamic validator instantiation based on config or schema

## 4. Main Functionalities & Features

- Type-specific validation classes for numeric, categorical, datetime, boolean, and text fields
- Range, pattern, and constraint checking
- Null value handling and reporting
- Factory function for dynamic validator creation
- Integration with custom validation exceptions

## 5. API Reference & Key Methods

| Class/Function | Description |
|----------------|-------------|
| `NumericFieldValidator(allow_null=True, min_value=None, max_value=None, allow_inf=False)` | Numeric field validation |
| `.validate(series, field_name=None)` | Validate numeric series |
| `CategoricalFieldValidator(allow_null=True, valid_categories=None, max_categories=None)` | Categorical field validation |
| `.validate(series, field_name=None)` | Validate categorical series |
| `DateTimeFieldValidator(allow_null=True, min_date=None, max_date=None, future_dates_allowed=True)` | Datetime field validation |
| `.validate(series, field_name=None)` | Validate datetime series |
| `BooleanFieldValidator(allow_null=True)` | Boolean field validation |
| `.validate(series, field_name=None)` | Validate boolean series |
| `TextFieldValidator(allow_null=True, min_length=None, max_length=None, pattern=None)` | Text field validation |
| `.validate(series, field_name=None)` | Validate text series |
| `FieldExistsValidator()` | Checks if field exists in DataFrame |
| `.validate(df, field_name)` | Validate field existence |
| `PatternValidator(pattern, allow_null=True)` | Pattern-based validation |
| `.validate(series, field_name=None)` | Validate pattern in series |
| `create_field_validator(field_type, **kwargs)` | Factory for validator creation |

## 6. Usage Examples

```python
import pandas as pd
from pamola_core.anonymization.commons.validation.field_validators import NumericFieldValidator, create_field_validator

# Numeric validation
series = pd.Series([1, 2, 3, 4, 5])
validator = NumericFieldValidator(min_value=0, max_value=10)
result = validator.validate(series, field_name="age")
print(result.is_valid)

# Using the factory
cat_validator = create_field_validator('categorical', valid_categories=['A', 'B', 'C'])
cat_result = cat_validator.validate(pd.Series(['A', 'B', 'C', 'A']))
```

## 7. Troubleshooting & Investigation Guide
- Check for type mismatches or unexpected nulls in input data
- Review error/warning messages in validation results
- Ensure all required fields are present in the DataFrame

## Unit Test Coverage

All tests for `commons/validation/field_validators.py` pass with no skips or failures. The test suite covers all major validator classes and edge cases, ensuring robust validation logic for numeric, categorical, datetime, boolean, and text fields. No codebase issues remain. Documentation is complete and up to date.

## 8. Summary Analysis

- Provides robust, reusable field validation utilities for anonymization and privacy operations
- Supports a wide range of data types and validation rules (e.g., regex, type, range, nullability)
- Enables pre-processing and data quality checks before anonymization
- Designed for extensibility and integration with custom validators
- Ensures consistent, business-compliant validation logic across modules
- Facilitates traceable, auditable validation for compliance and onboarding

## 9. Challenges, Limitations & Enhancement Opportunities
- Some edge cases (e.g., pandas datetime conversion) may require additional handling
- Boolean null checks may not always be triggered depending on data
- Future: add more flexible pattern and range validation options

## 10. Related Components & References
- Used by anonymization operation wrappers and data quality checks
- Related: pamola_core/metrics/commons/validation.py

## 11. Change Log & Contributors

- 2025-07-25: Initial documentation and full test coverage (AI automation)
- PAMOLA Core Team
