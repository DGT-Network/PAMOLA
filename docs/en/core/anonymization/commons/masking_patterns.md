# Masking Patterns
**Module:** pamola.pamola_core.anonymization.commons.masking_patterns  
**Version:** 1.0  
**Status:** ✅ Full  
**Last Updated:** 2025-07-28

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

---

## 1. Module Overview

The `masking_patterns.py` module provides a centralized library of regex-based masking patterns and utilities for anonymizing sensitive data types such as emails, phone numbers, SSNs, credit cards, IP addresses, account numbers, and more. It enables flexible, configurable, and testable masking logic for privacy-preserving data processing.

## 2. Source Code Hierarchy

- pamola/pamola_core/anonymization/commons/masking_patterns.py

## 3. Architecture & Data Flow

- Used by anonymization operations for pattern-based masking
- PatternConfig and MaskingPatterns provide centralized pattern management
- Pattern detection and masking utilities are called by masking operations and data processors

## 4. Main Functionalities & Features

- PatternConfig dataclass for pattern structure
- MaskingPatterns class for static pattern management
- Pattern detection and validation
- Pattern-based and format-preserving masking
- Mask character pools and security analysis

## 5. API Reference & Key Methods

| Function/Class | Description |
|----------------|-------------|
| `PatternConfig` | Dataclass for pattern configuration |
| `MaskingPatterns.get_pattern(pattern_type)` | Get pattern config by type |
| `MaskingPatterns.get_default_patterns()` | Get all default patterns |
| `MaskingPatterns.get_pattern_names()` | List all pattern names |
| `MaskingPatterns.validate_pattern_type(pattern_type)` | Validate pattern type |
| `MaskingPatterns.detect_pattern_type(value)` | Detect pattern type from value |
| `apply_pattern_mask(value, pattern_config, mask_char)` | Apply pattern-based mask |
| `get_format_preserving_mask(value, mask_char)` | Format-preserving mask |
| `create_random_mask(length, char_pool)` | Create random mask |
| `validate_mask_character(mask_char)` | Validate mask character |
| `analyze_pattern_security(pattern_config, test_values)` | Analyze pattern security |
| `generate_mask(mask_char, random_mask, mask_char_pool, length)` | Generate mask string |
| `generate_mask_char(mask_char, random_mask, mask_char_pool)` | Generate single mask char |
| `is_separator(char)` | Check if char is a separator |
| `preserve_pattern_mask(...)` | Preserve pattern mask |
| `get_mask_char_pool(pool_name)` | Get mask char pool |
| `set_mask_char_pool(pool_name, characters)` | Set mask char pool |
| `clear_mask_char_pools()` | Clear all mask char pools |

## 6. Usage Examples

```python
from pamola_core.anonymization.commons import masking_patterns as mp

# Mask an email address
pattern = mp.MaskingPatterns.get_pattern("email")
masked = mp.apply_pattern_mask("john.doe@example.com", pattern)
print(masked)  # Output: jo******@example.com

# Detect pattern type
ptype = mp.MaskingPatterns.detect_pattern_type("123-45-6789")
print(ptype)  # Output: ssn or ssn_middle

# Format-preserving mask
masked = mp.get_format_preserving_mask("123-45-6789")
print(masked)  # Output: ***-**-****

# Analyze pattern security
result = mp.analyze_pattern_security(pattern, ["john.doe@example.com"])
print(result["avg_visibility"])
```

## 7. Troubleshooting & Investigation Guide
- Ensure pattern type is supported and correctly specified
- Check for valid regex and mask character inputs
- Review error messages for unsupported or invalid patterns

## 8. Summary Analysis

- All major public methods and edge cases are covered by unit tests (≥90% line coverage).
- The module is extensible, testable, and supports a wide range of identifier types.
- Security analysis tools help assess the risk of each masking pattern.
- Masking logic is separated from pattern definitions for maintainability.

## 9. Challenges, Limitations & Enhancement Opportunities
- Some patterns may require tuning for international formats
- Future: add more pattern types and advanced masking strategies

## 10. Related Components & References
- Used by anonymization operation wrappers and masking utilities
- Related: pamola_core/metrics/commons/validation.py

## 11. Change Log & Contributors

- 2025-07-25: Full documentation and test coverage (Metrics template).
- 2025-07-28: Updated passing status, date, and coverage summary.
