"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of partial masking configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for fixed, pattern-based, random, and word-based masking strategies
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multiple masking strategies (fixed, pattern, random, words)
- Pattern-based masking with predefined and custom patterns
- Prefix/suffix/position-based unmasking controls
- Random character pool masking
- Preset configurations for common data types
- Conditional masking based on field values
- K-anonymity risk assessment integration

Changelog:
1.0.0 - 2025-11-18 - Initial creation of partial masking core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.anonymization.commons.masking_presets import MaskingType


class PartialMaskingConfig(OperationConfig):
    """
    Core configuration schema for PartialMasking backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Partial Masking Operation Core Configuration",
        "description": "Core schema for partial masking operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to apply partial masking.",
                    },
                    "mask_char": {
                        "type": "string",
                        "default": "*",
                        "title": "Mask Character",
                        "description": "Character used to mask sensitive content.",
                    },
                    "mask_strategy": {
                        "type": "string",
                        "default": MaskStrategyEnum.FIXED.value,
                        "title": "Masking Strategy",
                        "description": "Strategy for masking: fixed, pattern, random, or words.",
                        "oneOf": [
                            {
                                "const": MaskStrategyEnum.FIXED.value,
                                "description": MaskStrategyEnum.FIXED.value,
                            },
                            {
                                "const": MaskStrategyEnum.PATTERN.value,
                                "description": MaskStrategyEnum.PATTERN.value,
                            },
                            {
                                "const": MaskStrategyEnum.RANDOM.value,
                                "description": MaskStrategyEnum.RANDOM.value,
                            },
                            {
                                "const": MaskStrategyEnum.WORDS.value,
                                "description": MaskStrategyEnum.WORDS.value,
                            },
                        ],
                    },
                    "mask_percentage": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 100,
                        "title": "Mask Percentage",
                        "description": "Percentage of characters to mask randomly (for random strategy).",
                    },
                    "unmasked_prefix": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "title": "Unmasked Prefix",
                        "description": "Number of characters at the start to remain visible.",
                    },
                    "unmasked_suffix": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "title": "Unmasked Suffix",
                        "description": "Number of characters at the end to remain visible.",
                    },
                    "unmasked_positions": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "integer",
                            "oneOf": [
                                {"const": 0, "description": "0"},
                                {"const": 2, "description": "2"},
                                {"const": 4, "description": "4"},
                            ],
                        },
                        "title": "Unmasked Positions",
                        "description": "Specific index positions to remain unmasked.",
                        "default": None,
                    },
                    "pattern_type": {
                        "type": ["string", "null"],
                        "title": "Pattern Type",
                        "description": "Predefined pattern type (e.g., email, phone) for pattern-based masking.",
                        "default": None,
                        "oneOf": [
                            {"type": "null"},
                            {"const": "email", "description": "Email address"},
                            {"const": "email_domain", "description": "Email domain"},
                            {"const": "phone", "description": "Phone number"},
                            {
                                "const": "phone_international",
                                "description": "International phone number",
                            },
                            {
                                "const": "phone_us_formatted",
                                "description": "US phone number (formatted)",
                            },
                            {
                                "const": "phone_us_compact",
                                "description": "US phone number (compact)",
                            },
                            {
                                "const": "ssn",
                                "description": "Social Security Number (SSN)",
                            },
                            {
                                "const": "ssn_middle",
                                "description": "Middle part of SSN",
                            },
                            {
                                "const": "credit_card",
                                "description": "Credit card number",
                            },
                            {
                                "const": "credit_card_strict",
                                "description": "Strictly formatted credit card number",
                            },
                            {"const": "ip_address", "description": "IP address"},
                            {
                                "const": "ip_address_last_only",
                                "description": "Last segment of IP address",
                            },
                            {"const": "date_mdy", "description": "Date (MM/DD/YYYY)"},
                            {"const": "date_dmy", "description": "Date (DD/MM/YYYY)"},
                            {"const": "date_ymd", "description": "Date (YYYY/MM/DD)"},
                            {
                                "const": "date_iso",
                                "description": "ISO date (YYYY-MM-DD)",
                            },
                            {"const": "date_year_only", "description": "Year only"},
                            {"const": "birthdate", "description": "Birthdate"},
                            {
                                "const": "birthdate_dmy",
                                "description": "Birthdate (DD/MM/YYYY)",
                            },
                            {
                                "const": "date_month_year",
                                "description": "Date (Month/Year)",
                            },
                            {
                                "const": "date_dotted",
                                "description": "Date with dots (DD.MM.YYYY)",
                            },
                            {
                                "const": "account_number",
                                "description": "Account number",
                            },
                            {
                                "const": "account_number_last_only",
                                "description": "Last digits of account number",
                            },
                            {
                                "const": "license_plate",
                                "description": "License plate number",
                            },
                            {
                                "const": "driver_license",
                                "description": "Driver's license number",
                            },
                            {"const": "passport", "description": "Passport number"},
                            {
                                "const": "iban",
                                "description": "IBAN (International Bank Account Number)",
                            },
                            {"const": "url", "description": "Website URL"},
                            {"const": "username", "description": "Username"},
                            {
                                "const": "medical_record",
                                "description": "Medical record number",
                            },
                            {
                                "const": "health_insurance_number",
                                "description": "Health insurance number",
                            },
                            {
                                "const": "icd10_code",
                                "description": "ICD-10 medical code",
                            },
                            {"const": "patient_id", "description": "Patient ID"},
                        ],
                    },
                    "mask_pattern": {
                        "type": ["string", "null"],
                        "title": "Mask Pattern",
                        "description": "Custom regex pattern for masking (pattern strategy).",
                    },
                    "preserve_pattern": {
                        "type": ["string", "null"],
                        "title": "Preserve Pattern",
                        "description": "Regex pattern to preserve (mask all except matches).",
                    },
                    "preserve_separators": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Separators",
                        "description": "Whether to keep separators (e.g., '-', '_', '.') unchanged.",
                    },
                    "preserve_word_boundaries": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Word Boundaries",
                        "description": "Whether to avoid masking across word boundaries.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": True,
                        "title": "Case-Sensitive Matching",
                        "description": "Whether pattern matching is case-sensitive.",
                    },
                    "random_mask": {
                        "type": "boolean",
                        "default": False,
                        "title": "Random Mask",
                        "description": "Use random characters from a pool instead of a fixed mask_char.",
                    },
                    "mask_char_pool": {
                        "type": ["string", "null"],
                        "title": "Mask Character Pool",
                        "description": "Pool of characters to randomly sample from if random_mask is True.",
                    },
                    "preset_type": {
                        "type": ["string", "null"],
                        "title": "Preset Type",
                        "description": "Preset category for reusable masking templates.",
                        "default": None,
                        "oneOf": [
                            {"type": "null"},
                            {"const": MaskingType.EMAIL.value, "description": "Email"},
                            {"const": MaskingType.PHONE.value, "description": "Phone"},
                            {
                                "const": MaskingType.CREDIT_CARD.value,
                                "description": "Credit Card",
                            },
                            {"const": MaskingType.SSN.value, "description": "SSN"},
                            {
                                "const": MaskingType.IP_ADDRESS.value,
                                "description": "IP Address",
                            },
                            {
                                "const": MaskingType.HEALTHCARE.value,
                                "description": "Healthcare",
                            },
                            {
                                "const": MaskingType.FINANCIAL.value,
                                "description": "Financial",
                            },
                            {
                                "const": MaskingType.DATE_ISO.value,
                                "description": "Date (ISO)",
                            },
                        ],
                    },
                    "preset_name": {
                        "type": ["string", "null"],
                        "title": "Preset Name",
                        "description": "Name of the specific preset configuration to apply.",
                    },
                    "consistency_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Consistency Fields",
                        "description": "Other fields to mask consistently with the main field.",
                    },
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name used as condition for applying the generalization.",
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition.",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "default": "in",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values of the condition field that trigger the generalization.",
                    },
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field used for k-anonymity risk assessment.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering masking.",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records.",
                    },
                },
                "required": ["field_name"],
            },
            {
                "if": {"properties": {"random_mask": {"const": False}}},
                "then": {"required": ["mask_char"]},
            },
            {
                "if": {
                    "properties": {
                        "mask_strategy": {"const": MaskStrategyEnum.PATTERN.value}
                    }
                },
                "then": {
                    "required": ["pattern_type", "mask_pattern", "preserve_pattern"]
                },
            },
            {
                "if": {"properties": {"random_mask": {"const": True}}},
                "then": {"required": ["mask_char_pool"]},
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_field"],
                },
                "then": {"properties": {"condition_operator": {"type": "string"}}},
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1},
                        "condition_operator": {"type": "string", "minLength": 1},
                    },
                    "required": ["condition_field", "condition_operator"],
                },
                "then": {"properties": {"condition_values": {"type": "array"}}},
            },
        ],
    }
