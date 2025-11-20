"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of uniform numeric noise configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for additive and multiplicative noise application
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Additive and multiplicative noise strategies
- Output bounds and constraints (min/max values)
- Statistical scaling by standard deviation
- Secure random generation support
- Conditional noise application based on field values
- Multi-condition logic support
- K-anonymity risk assessment integration

Changelog:
1.0.0 - 2025-11-18 - Initial creation of uniform numeric noise core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class UniformNumericNoiseConfig(OperationConfig):
    """
    Core configuration schema for UniformNumericNoise backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Uniform Numeric Noise Operation Core Configuration",
        "description": "Core schema for uniform numeric noise operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field to apply noise to.",
                    },
                    "noise_type": {
                        "type": "string",
                        "oneOf": [
                            {"const": "additive", "description": "Additive noise"},
                            {
                                "const": "multiplicative",
                                "description": "Multiplicative noise",
                            },
                        ],
                        "default": "additive",
                        "title": "Noise Type",
                        "description": "Type of noise: 'additive' (add noise) or 'multiplicative' (scale by noise).",
                    },
                    "noise_range": {
                        "type": ["number", "array"],
                        "title": "Noise Range",
                        "default": 0.1,
                    },
                    "output_min": {
                        "type": ["number", "null"],
                        "title": "Output Minimum",
                        "description": "Minimum allowed value after noise is applied.",
                    },
                    "output_max": {
                        "type": ["number", "null"],
                        "title": "Output Maximum",
                        "description": "Maximum allowed value after noise is applied.",
                    },
                    "preserve_zero": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Zero",
                        "description": "If True, zero values will not be changed by noise.",
                    },
                    "round_to_integer": {
                        "type": ["boolean", "null"],
                        "default": False,
                        "title": "Round to Integer",
                        "description": "If True, round the result to the nearest integer.",
                    },
                    "scale_by_std": {
                        "type": "boolean",
                        "default": False,
                        "title": "Scale by Std",
                        "description": "If True, scale noise by the standard deviation of the field.",
                    },
                    "scale_factor": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1.0,
                        "title": "Scale Factor",
                        "description": "Multiplier for the noise magnitude.",
                    },
                    "random_seed": {
                        "type": ["integer", "null"],
                        "title": "Random Seed",
                        "description": "Seed for reproducible random noise (ignored if use_secure_random is True).",
                    },
                    "use_secure_random": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use Secure Random",
                        "description": "If True, use a cryptographically secure random generator.",
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
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "title": "Condition Field",
                                    "description": "Field name for the condition.",
                                },
                                "operator": {
                                    "type": "string",
                                    "title": "Condition Operator",
                                    "oneOf": [
                                        {"const": "in", "description": "In"},
                                        {"const": "not_in", "description": "Not in"},
                                        {"const": "gt", "description": "Greater than"},
                                        {"const": "lt", "description": "Less than"},
                                        {"const": "eq", "description": "Equal to"},
                                        {"const": "ne", "description": "Not equal"},
                                        {
                                            "const": "ge",
                                            "description": "Greater than or equal",
                                        },
                                        {
                                            "const": "le",
                                            "description": "Less than or equal",
                                        },
                                        {"const": "range", "description": "Range"},
                                        {"const": "all", "description": "All"},
                                    ],
                                    "description": "Operator for the condition (e.g., '=', '>', '<', 'in').",
                                },
                                "values": {
                                    "type": "array",
                                    "title": "Condition Value",
                                    "description": "Value(s) for the condition.",
                                },
                            },
                        },
                        "title": "Multi-Conditions",
                        "description": "List of multi-field conditions for custom noise application logic.",
                    },
                    "condition_logic": {
                        "type": "string",
                        "title": "Condition Logic",
                        "description": "Logical operator for combining multiple conditions (e.g., 'AND', 'OR').",
                        "default": "AND",
                        "oneOf": [
                            {"const": "AND", "description": "AND"},
                            {"const": "OR", "description": "OR"},
                        ],
                    },
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field containing k-anonymity risk scores for suppression based on risk.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering noise application.",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records.",
                    },
                },
                "required": ["field_name", "noise_range", "noise_type"],
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
