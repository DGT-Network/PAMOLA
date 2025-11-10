"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Full Masking Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating full masking parameters in PAMOLA.CORE.
- Supports character masking, length preservation, randomization, format-aware masking, and conditional masking
- Integrates with k-anonymity risk assessment and output field configuration
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of full masking config file
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName


class FullMaskingConfig(OperationConfig):
    """Configuration for FullMaskingOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Full Masking Operation Configuration",
        "description": "Configuration schema for full masking operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to apply full masking.",
                    },
                    "random_mask": {
                        "type": "boolean",
                        "default": False,
                        "title": "Random Mask",
                        "description": "Whether to use random characters from a pool instead of a fixed mask_char.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.CORE_MASKING_RULES,
                    },
                    "mask_char": {
                        "type": "string",
                        "default": "*",
                        "title": "Mask Character",
                        "description": "Character used for masking the field values.",
                        "x-component": "Input",
                        "x-group": GroupName.CORE_MASKING_RULES,
                        "x-depend-on": {"random_mask": False},
                        "x-required-on": {"random_mask": False},
                    },
                    "preserve_length": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Length",
                        "description": "Whether to preserve the original string length of masked values.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.CORE_MASKING_RULES,
                    },
                    "fixed_length": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "title": "Fixed Length",
                        "description": "Fixed output length for all masked values. If None, uses input length.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_MASKING_RULES,
                        "x-depend-on": {"preserve_length": False},
                        "x-required-on": {"preserve_length": False},
                    },
                    "mask_char_pool": {
                        "type": ["string", "null"],
                        "title": "Mask Character Pool",
                        "description": "Pool of characters to randomly sample from if random_mask is True.",
                        "x-component": "Input",
                        "x-group": GroupName.CORE_MASKING_RULES,
                        "x-depend-on": {"preserve_length": True},
                        "x-required-on": {"preserve_length": True},
                    },
                    "preserve_format": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Format",
                        "description": "Whether to preserve data format or structure (e.g., keep dashes or parentheses).",
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "format_patterns": {
                        "type": ["array", "null"],
                        "title": "Format Patterns",
                        "description": "Custom regex patterns for identifying and preserving data formats.",
                        "items": {
                            "properties": {
                                "select_type": {
                                    "title": "Select Type",
                                    "description": "Select Type",
                                    "type": "string",
                                    "x-component": "Select",
                                    "enum": [
                                        "phone",
                                        "ssn",
                                        "credit_card",
                                        "email",
                                        "date",
                                    ],
                                    "oneOf": [
                                        {
                                            "const": "phone",
                                            "description": "Phone Number",
                                        },
                                        {
                                            "const": "ssn",
                                            "description": "SSN",
                                        },
                                        {
                                            "const": "credit_card",
                                            "description": "Credit Card",
                                        },
                                        {
                                            "const": "email",
                                            "description": "Email",
                                        },
                                        {
                                            "const": "date",
                                            "description": "Date",
                                        },
                                    ],
                                },
                                "pattern": {
                                    "title": "Pattern",
                                    "description": "Value (e.g., r'(\d{3})-(\d{3})-(\d{4})')",
                                    "type": "string",
                                    "x-component": "Input",
                                },
                            },
                            "type": "object",
                        },
                        "x-component": "ArrayItems",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                        "x-depend-on": {"preserve_format": True},
                        "x-required-on": {"preserve_format": True},
                    },
                    "numeric_output": {
                        "type": "string",
                        "enum": ["string", "numeric", "preserve"],
                        "default": "string",
                        "title": "Numeric Output",
                        "description": "Defines the output type for numeric fields: string, numeric, or preserve.",
                        "oneOf": [
                            {"const": "string", "description": "string"},
                            {"const": "numeric", "description": "numeric"},
                            {"const": "preserve", "description": "preserve"},
                        ],
                        "x-component": "Select",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "date_format": {
                        "type": ["string", "null"],
                        "title": "Date Format",
                        "description": "Date format string to use when masking datetime fields.",
                        "x-component": "Input",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "x-component": "Select",
                        "description": "Field name used as condition for applying the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_FIELD],
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
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_OPERATOR],
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "x-component": "Input",  # ArrayItems
                        "description": "Values of the condition field that trigger the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_VALUES],
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
                "required": ["field_name", "mask_char"],
            },
        ],
    }
