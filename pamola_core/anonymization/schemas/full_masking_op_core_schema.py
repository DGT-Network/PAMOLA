"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Full Masking Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of full masking configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for character masking with format preservation
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Character masking with configurable mask characters
- Random masking from character pools
- Length and format preservation options
- Conditional masking based on field values
- K-anonymity risk assessment integration

Changelog:
1.0.0 - 2025-11-18 - Initial creation of full masking core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class FullMaskingConfig(OperationConfig):
    """
    Core configuration schema for FullMasking backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Full Masking Operation Core Configuration",
        "description": "Core schema for full masking operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
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
                        "title": "Random Mask",
                        "default": False,
                        "description": "Whether to use random characters from a pool instead of a fixed mask_char.",
                    },
                    "mask_char": {
                        "type": "string",
                        "title": "Mask Character",
                        "default": "*",
                        "description": "Character used for masking the field values.",
                    },
                    "preserve_length": {
                        "type": "boolean",
                        "title": "Preserve Length",
                        "default": True,
                        "description": "Whether to preserve the original string length of masked values.",
                    },
                    "fixed_length": {
                        "type": ["integer", "null"],
                        "title": "Fixed Length",
                        "minimum": 0,
                        "description": "Fixed output length for all masked values. If None, uses input length.",
                    },
                    "mask_char_pool": {
                        "type": ["string", "null"],
                        "title": "Mask Character Pool",
                        "description": "Pool of characters to randomly sample from if random_mask is True.",
                    },
                    "preserve_format": {
                        "type": "boolean",
                        "title": "Preserve Format",
                        "default": False,
                        "description": "Whether to preserve data format or structure (e.g., keep dashes or parentheses).",
                    },
                    "format_patterns": {
                        "type": ["array", "null"],
                        "title": "Format Patterns",
                        "description": "Custom regex patterns for identifying and preserving data formats.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "select_type": {
                                    "title": "Select Type",
                                    "description": "Select Type",
                                    "type": "string",
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
                                },
                            },
                        },
                    },
                    "numeric_output": {
                        "type": "string",
                        "default": "string",
                        "title": "Numeric Output",
                        "description": "Defines the output type for numeric fields: string, numeric, or preserve.",
                        "oneOf": [
                            {"const": "string", "description": "string"},
                            {"const": "numeric", "description": "numeric"},
                            {"const": "preserve", "description": "preserve"},
                        ],
                    },
                    "date_format": {
                        "type": ["string", "null"],
                        "title": "Date Format",
                        "description": "Date format string to use when masking datetime fields.",
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
                "required": ["field_name", "mask_char"],
            },
            {
                "if": {"properties": {"random_mask": {"const": False}}},
                "then": {"required": ["mask_char"]},
            },
            {
                "if": {"properties": {"preserve_length": {"const": False}}},
                "then": {"required": ["fixed_length"]},
            },
            {
                "if": {"properties": {"preserve_length": {"const": True}}},
                "then": {"required": ["mask_char_pool"]},
            },
            {
                "if": {"properties": {"preserve_format": {"const": True}}},
                "then": {"required": ["format_patterns"]},
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {
                            "type": "string",
                            "minLength": 1
                        }
                    },
                    "required": ["condition_field"]
                },
                "then": {
                    "properties": {
                        "condition_operator": {
                            "type": "string"
                        }
                    }
                }
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {
                            "type": "string",
                            "minLength": 1
                        },
                        "condition_operator": {
                            "type": "string",
                            "minLength": 1
                        }
                    },
                    "required": ["condition_field", "condition_operator"]
                },
                "then": {
                    "properties": {
                        "condition_values": {
                            "type": "array"
                        }
                    }
                }
            },
        ],
    }