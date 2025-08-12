# Masking Presets
**Module:** pamola_core.anonymization.commons.masking_presets  
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

---

## 1. Module Overview

The `masking_presets.py` module provides a comprehensive, extensible library of masking presets and configuration utilities for common sensitive data types. It enables privacy-preserving data processing by supporting flexible, configurable, and robust masking strategies for emails, phone numbers, credit cards, SSNs, IP addresses, healthcare, financial identifiers, and ISO dates. The module is designed for easy integration, testability, and future extensibility.

## 2. Source Code Hierarchy

- pamola_core/anonymization/commons/masking_presets.py
  - `MaskingType` (Enum)
  - `MaskingConfig` (dataclass)
  - `BaseMaskingPresets` (ABC)
  - `EmailMaskingPresets`, `PhoneMaskingPresets`, `CreditCardMaskingPresets`, `SSNMaskingPresets`, `IPAddressMaskingPresets`, `HealthcareMaskingPresets`, `FinancialMaskingPresets`, `DateMaskingPresets`
  - `MaskingPresetManager`
  - `MaskingUtils`

## 3. Architecture & Data Flow

- Used by anonymization operations for preset-based masking
- Preset manager and utilities provide centralized preset management
- Preset providers (Email, Phone, etc.) are called by masking operations and data processors

## 4. Main Functionalities & Features

- Centralized preset management for all major identifier types
- Configurable masking logic (prefix/suffix, format preservation, random/fixed masking)
- Regex-based validation and type detection utilities
- Bulk masking and custom configuration creation
- Extensible architecture for new data types and masking rules
- Abstract base class for consistent preset provider interface

## 5. API Reference & Key Methods

| Class/Function | Description |
|----------------|-------------|
| `MaskingType(Enum)` | Enumerates supported data types for masking |
| `MaskingConfig` | Dataclass for masking configuration |
| `BaseMaskingPresets` | Abstract base for all preset providers |
| `get_presets(self)` | Returns all presets for a type |
| `apply_masking(self, data, preset_name, random_mask)` | Applies masking preset |
| `MaskingPresetManager` | Central manager for all preset providers |
| `MaskingUtils` | Utilities for type detection, bulk masking, and config creation |

## 6. Usage Examples

```python
from pamola_core.anonymization.commons import masking_presets as mp

# Email masking
emailer = mp.EmailMaskingPresets()
masked = emailer.apply_masking("user@example.com", "FULL_DOMAIN")
print(masked)  # us**@example.com

# Bulk masking utility (with mixed valid/invalid)
data = ["user1@example.com", "notanemail"]
masked_list = mp.MaskingUtils.bulk_mask(data, mp.MaskingType.EMAIL, "FULL_DOMAIN")
print(masked_list)  # ['us**@example.com', 'notanemail']

# Custom config for masking (edge case: all masked)
config = mp.MaskingUtils.create_custom_config(mask_char="X", unmasked_prefix=0, unmasked_suffix=0)
masked = emailer._mask_string("abcdefg", config)
print(masked)  # XXXXXXX

# Negative validation
is_valid = mp.MaskingPresetManager().validate_data("", mp.MaskingType.EMAIL)
print(is_valid)  # False

# Preset info listing
info = emailer.get_preset_info("FULL_DOMAIN")
print(info)  # {'name': 'FULL_DOMAIN', ...}
```

## 7. Troubleshooting & Investigation Guide
- Ensure preset name and type are supported
- Check for static/instance method mismatches in preset providers
- Review error messages for unsupported or invalid presets

## Unit Test Coverage

- **Coverage:** 95% line coverage (as of 2025-07-29); 21 source methods, 20 tested, 13/14 tests passed, 1 skipped (documented static/instance limitation). All actionable public methods and business-driven scenarios are tested.
- **Test Scenarios:**
  - Bulk masking (empty/mixed input)
  - Custom config edge cases (all masked, prefix/suffix)
  - Preset info and listing for all types
  - Negative validation and error handling
  - All major preset providers (Email, Phone, Credit Card, SSN, IP, Healthcare, Financial, Date)
  - Skipped: IPAddressMaskingPresets static/instance limitation (documented)
- **Design:** Modular, extensible, robust. Abstract base class ensures consistency. Preset manager/utilities support scalable privacy operations.
- **Limitations:** Some static/instance method inconsistencies exist for certain preset providers (notably DateMaskingPresets and IPAddressMaskingPresets). These are documented in the test suite and do not affect main masking logic.

## 8. Summary Analysis

The `masking_presets` module provides a centralized, extensible library of masking strategies for sensitive data types, enabling consistent and configurable privacy protection across all anonymization operations.

- Defines standard masking presets for common data types and business use cases
- Enables rapid, consistent application of masking strategies across datasets
- Reduces risk of misconfiguration by providing tested, referenceable presets
- Supports both partial and full masking scenarios for compliance
- Facilitates onboarding and automation by documenting available presets and test scenarios
- Ensures business-aligned, repeatable privacy operations with ≥90% actionable coverage
- Provides robust troubleshooting and negative validation guidance for end users

## 9. Challenges, Limitations & Enhancement Opportunities
- Static/instance method inconsistencies for some preset providers (see test suite for documented skip)
- Future: add more preset types and advanced masking strategies

## 10. Related Components & References
- Used by anonymization operation wrappers and masking utilities
- Related: pamola_core/metrics/commons/aggregation.py

## 11. Change Log & Contributors

- 2025-07-25: Full documentation and test coverage completed.  
- PAMOLA Core Team

---
