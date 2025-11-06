"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating partial masking parameters in PAMOLA.CORE.
- Supports fixed, pattern-based, random, and word-based masking strategies
- Handles prefix/suffix/unmasked positions, pattern preservation, and conditional masking
- Integrates with k-anonymity risk assessment and output field configuration
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of partial masking config file
"""

from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.anonymization.commons.masking_presets import MaskingType
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName


class PartialMaskingConfig(OperationConfig):
    """Configuration schema for PartialMaskingOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Partial Masking Operation Configuration",
        "description": "Configuration schema for partial masking operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # ==== Partial Masking Specific Fields ====
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
                        "x-component": "Input",
                        "x-group": GroupName.MASK_APPEARANCE,
                        "x-depend-on": {"random_mask": False},
                        "x-required-on": {"random_mask": False},
                    },
                    "mask_strategy": {
                        "type": "string",
                        "enum": [
                            MaskStrategyEnum.FIXED.value,
                            MaskStrategyEnum.PATTERN.value,
                            MaskStrategyEnum.RANDOM.value,
                            MaskStrategyEnum.WORDS.value,
                        ],
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
                        "x-component": "Select",
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                    },
                    "mask_percentage": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 100,
                        "title": "Mask Percentage",
                        "description": "Percentage of characters to mask randomly (for random strategy).",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {"mask_strategy": MaskStrategyEnum.RANDOM.value},
                    },
                    "unmasked_prefix": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "title": "Unmasked Prefix",
                        "description": "Number of characters at the start to remain visible.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": [
                                MaskStrategyEnum.FIXED.value,
                                MaskStrategyEnum.WORDS.value,
                            ]
                        },
                    },
                    "unmasked_suffix": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "title": "Unmasked Suffix",
                        "description": "Number of characters at the end to remain visible.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": [
                                MaskStrategyEnum.FIXED.value,
                                MaskStrategyEnum.WORDS.value,
                            ]
                        },
                    },
                    "unmasked_positions": {
                        "type": ["array", "null"],
                        "items": {"type": "integer", "minimum": 0},
                        "title": "Unmasked Positions",
                        "description": "Specific index positions to remain unmasked.",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": 0, "description": "0"},
                            {"const": 2, "description": "2"},
                            {"const": 4, "description": "4"},
                        ],
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {"mask_strategy": MaskStrategyEnum.FIXED.value},
                    },
                    "pattern_type": {
                        "type": ["string", "null"],
                        "title": "Pattern Type",
                        "description": "Predefined pattern type (e.g., email, phone) for pattern-based masking.",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "email", "description": "Email address"},
                            {"const": "email_domain", "description": "Email domain"},
                            {"const": "phone", "description": "Phone number"},
                            {"const": "phone_international", "description": "International phone number"},
                            {"const": "phone_us_formatted", "description": "US phone number (formatted)"},
                            {"const": "phone_us_compact", "description": "US phone number (compact)"},
                            {"const": "ssn", "description": "Social Security Number (SSN)"},
                            {"const": "ssn_middle", "description": "Middle part of SSN"},
                            {"const": "credit_card", "description": "Credit card number"},
                            {"const": "credit_card_strict", "description": "Strictly formatted credit card number"},
                            {"const": "ip_address", "description": "IP address"},
                            {"const": "ip_address_last_only", "description": "Last segment of IP address"},
                            {"const": "date_mdy", "description": "Date (MM/DD/YYYY)"},
                            {"const": "date_dmy", "description": "Date (DD/MM/YYYY)"},
                            {"const": "date_ymd", "description": "Date (YYYY/MM/DD)"},
                            {"const": "date_iso", "description": "ISO date (YYYY-MM-DD)"},
                            {"const": "date_year_only", "description": "Year only"},
                            {"const": "birthdate", "description": "Birthdate"},
                            {"const": "birthdate_dmy", "description": "Birthdate (DD/MM/YYYY)"},
                            {"const": "date_month_year", "description": "Date (Month/Year)"},
                            {"const": "date_dotted", "description": "Date with dots (DD.MM.YYYY)"},
                            {"const": "account_number", "description": "Account number"},
                            {"const": "account_number_last_only", "description": "Last digits of account number"},
                            {"const": "license_plate", "description": "License plate number"},
                            {"const": "driver_license", "description": "Driver's license number"},
                            {"const": "passport", "description": "Passport number"},
                            {"const": "iban", "description": "IBAN (International Bank Account Number)"},
                            {"const": "url", "description": "Website URL"},
                            {"const": "username", "description": "Username"},
                            {"const": "medical_record", "description": "Medical record number"},
                            {"const": "health_insurance_number", "description": "Health insurance number"},
                            {"const": "icd10_code", "description": "ICD-10 medical code"},
                            {"const": "patient_id", "description": "Patient ID"}
                        ],
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                        "x-depend-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                        "x-required-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                    },
                    "mask_pattern": {
                        "type": ["string", "null"],
                        "title": "Mask Pattern",
                        "description": "Custom regex pattern for masking (pattern strategy).",
                        "x-component": "Input",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                        "x-required-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                    },
                    "preserve_pattern": {
                        "type": ["string", "null"],
                        "title": "Preserve Pattern",
                        "description": "Regex pattern to preserve (mask all except matches).",
                        "x-component": "Input",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                        "x-required-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                    },
                    "preserve_separators": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Separators",
                        "description": "Whether to keep separators (e.g., '-', '_', '.') unchanged.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "preserve_word_boundaries": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Word Boundaries",
                        "description": "Whether to avoid masking across word boundaries.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                        "x-depend-on": {"mask_strategy": MaskStrategyEnum.WORDS.value},
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": True,
                        "title": "Case-Sensitive Matching",
                        "description": "Whether pattern matching is case-sensitive.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "random_mask": {
                        "type": "boolean",
                        "default": False,
                        "title": "Random Mask",
                        "description": "Use random characters from a pool instead of a fixed mask_char.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.MASK_APPEARANCE,
                    },
                    "mask_char_pool": {
                        "type": ["string", "null"],
                        "title": "Mask Character Pool",
                        "description": "Pool of characters to randomly sample from if random_mask is True.",
                        "x-component": "Input",
                        "x-group": GroupName.MASK_APPEARANCE,
                        "x-depend-on": {"random_mask": True},
                        "x-required-on": {"random_mask": True},
                    },
                    "preset_type": {
                        "type": ["string", "null"],
                        "title": "Preset Type",
                        "description": "Preset category for reusable masking templates.",
                        "oneOf": [
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
                        "x-component": "Select",
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                    },
                    "preset_name": {
                        "type": ["string", "null"],
                        "title": "Preset Name",
                        "description": "Name of the specific preset configuration to apply.",
                        "x-component": "Depend-Select",
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                        "x-depend-map": {
                            "depend_on": "preset_type",
                            "options_map": {
                                MaskingType.EMAIL.value: [
                                    {"label": "FULL_DOMAIN", "value": "FULL_DOMAIN"},
                                    {"label": "DOMAIN_ONLY", "value": "DOMAIN_ONLY"},
                                    {
                                        "label": "PARTIAL_DOMAIN",
                                        "value": "PARTIAL_DOMAIN",
                                    },
                                    {
                                        "label": "PRIVACY_FOCUSED",
                                        "value": "PRIVACY_FOCUSED",
                                    },
                                    {
                                        "label": "GDPR_COMPLIANT",
                                        "value": "GDPR_COMPLIANT",
                                    },
                                    {
                                        "label": "UTILITY_BALANCED",
                                        "value": "UTILITY_BALANCED",
                                    },
                                    {
                                        "label": "MINIMAL_EXPOSURE",
                                        "value": "MINIMAL_EXPOSURE",
                                    },
                                ],
                                MaskingType.PHONE.value: [
                                    {"label": "US_STANDARD", "value": "US_STANDARD"},
                                    {"label": "US_FORMATTED", "value": "US_FORMATTED"},
                                    {
                                        "label": "INTERNATIONAL",
                                        "value": "INTERNATIONAL",
                                    },
                                    {
                                        "label": "LAST_FOUR_ONLY",
                                        "value": "LAST_FOUR_ONLY",
                                    },
                                    {
                                        "label": "AREA_CODE_ONLY",
                                        "value": "AREA_CODE_ONLY",
                                    },
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                ],
                                MaskingType.CREDIT_CARD.value: [
                                    {
                                        "label": "PCI_COMPLIANT",
                                        "value": "PCI_COMPLIANT",
                                    },
                                    {"label": "STRICT", "value": "STRICT"},
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                    {"label": "NUMERIC_ONLY", "value": "NUMERIC_ONLY"},
                                    {
                                        "label": "FIRST_LAST_FOUR",
                                        "value": "FIRST_LAST_FOUR",
                                    },
                                ],
                                MaskingType.SSN.value: [
                                    {"label": "LAST_FOUR", "value": "LAST_FOUR"},
                                    {"label": "FIRST_THREE", "value": "FIRST_THREE"},
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                    {"label": "NUMERIC_MASK", "value": "NUMERIC_MASK"},
                                    {
                                        "label": "AREA_NUMBER_ONLY",
                                        "value": "AREA_NUMBER_ONLY",
                                    },
                                ],
                                MaskingType.IP_ADDRESS.value: [
                                    {"label": "SUBNET_MASK", "value": "SUBNET_MASK"},
                                    {"label": "NETWORK_ONLY", "value": "NETWORK_ONLY"},
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                    {"label": "ZERO_MASK", "value": "ZERO_MASK"},
                                    {
                                        "label": "PRIVATE_NETWORK",
                                        "value": "PRIVATE_NETWORK",
                                    },
                                ],
                                MaskingType.HEALTHCARE.value: [
                                    {
                                        "label": "MEDICAL_RECORD",
                                        "value": "MEDICAL_RECORD",
                                    },
                                    {"label": "PATIENT_ID", "value": "PATIENT_ID"},
                                    {"label": "NPI_NUMBER", "value": "NPI_NUMBER"},
                                    {"label": "DEA_NUMBER", "value": "DEA_NUMBER"},
                                ],
                                MaskingType.FINANCIAL.value: [
                                    {
                                        "label": "ACCOUNT_NUMBER",
                                        "value": "ACCOUNT_NUMBER",
                                    },
                                    {
                                        "label": "ROUTING_NUMBER",
                                        "value": "ROUTING_NUMBER",
                                    },
                                    {
                                        "label": "BANK_STANDARD",
                                        "value": "BANK_STANDARD",
                                    },
                                    {"label": "SWIFT_CODE", "value": "SWIFT_CODE"},
                                    {"label": "IBAN", "value": "IBAN"},
                                    {"label": "CREDIT_LIMIT", "value": "CREDIT_LIMIT"},
                                    {"label": "LOAN_NUMBER", "value": "LOAN_NUMBER"},
                                ],
                                MaskingType.DATE_ISO.value: [
                                    {"label": "MASK_DAY", "value": "MASK_DAY"},
                                    {"label": "MASK_MONTH", "value": "MASK_MONTH"},
                                    {"label": "MASK_YEAR", "value": "MASK_YEAR"},
                                    {"label": "MASK_FULL", "value": "MASK_FULL"},
                                ],
                            },
                        },
                    },
                    "consistency_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Consistency Fields",
                        "description": "Other fields to mask consistently with the main field.",
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "x-component": "Select",
                        "description": "Field name used as condition for applying the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition.",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "default": "in",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"condition_field": "not_null"},
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "x-component": "Input",  # ArrayItems
                        "description": "Values of the condition field that trigger the generalization.",
                        "items": {"type": "string"},
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                    },
                    # K-anonymity integration
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
        ],
    }
