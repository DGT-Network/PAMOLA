# Partial Masking Operation
**Module:** pamola_core.anonymization.masking.partial_masking_op  
**Version:** 1.0.0  
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

## 1. Module Overview
Partial Masking Operation provides functionality to mask parts of sensitive data fields, preserving some original information while anonymizing the rest. Commonly used for data privacy in compliance scenarios.

## 2. Source Code Hierarchy
- pamola_core/anonymization/masking/partial_masking_op.py
  - class PartialMaskingOperation
    - mask_partial_value
    - ... (add other methods as needed)

## 3. Architecture & Data Flow
- Inherits from BaseAnonymizationOperation
- Utilizes internal masking utilities
- Supports configuration-driven masking logic

```mermaid
flowchart TD
    A[Input Data] --> B[PartialMaskingOperation.execute()]
    B --> C[mask_partial_value]
    C --> D[Output Masked Data]
```

## 4. Main Functionalities & Features
- Masks part of a string or value based on configuration
- Supports custom mask characters and positions
- Integrates with anonymization pipeline

## 5. API Reference & Key Methods
| Method | Description |
|--------|-------------|
| `mask_partial_value(value, config)` | Masks part of the input value according to config |
| `execute(data, config)` | Applies partial masking to input data |

## 6. Usage Examples
```python
from pamola_core.anonymization.masking.partial_masking_op import PartialMaskingOperation
op = PartialMaskingOperation(mask_char='*', start=2, end=6)
masked = op.mask_partial_value('SensitiveData', {'mask_char': '*', 'start': 2, 'end': 6})
# masked -> 'Se****iveData'
```

## 7. Troubleshooting & Investigation Guide
- Ensure configuration parameters (mask_char, start, end) are valid
- Check for input values shorter than mask range
- Review logs for pipeline integration errors

## Unit Test Coverage

All actionable tests validate output artifacts and business logic for prefix/suffix, pattern, and pool masking. No skips, xfails, or untestable tests remain. All tests pass and coverage is complete. File is marked as 'Full' per Metrics template.

## 8. Summary Analysis

- Provides partial masking for sensitive fields, balancing privacy and utility
- Supports flexible configuration of unmasked prefix/suffix and mask characters
- Integrates seamlessly with the base anonymization operation and metrics
- Enables business-driven masking strategies for diverse data types
- Facilitates compliance, traceability, and onboarding with clear usage patterns
- Reference implementation for partial masking in PAMOLA.CORE

## 9. Challenges, Limitations & Enhancement Opportunities
- Only supports string masking (future: add numeric/structured masking)
- Edge cases for very short strings
- Enhancement: support regex-based masking

## 10. Related Components & References
- pamola_core/anonymization/masking/base_anonymization_op.py
- pamola_core/anonymization/commons/masking_patterns.py

## 11. Change Log & Contributors
- 2025-07-25: Initial auto-generated documentation.
- 2025-07-28: Migrated to strict Metrics package template for compliance.

---

*This documentation is auto-generated as part of the PAMOLA Core Metrics package coverage process.*
