"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Suppression Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of attribute suppression configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for attribute-level suppression operations
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Primary and additional field suppression
- Suppression mode configuration (REMOVE)
- Column metadata preservation options
- Conditional suppression based on field values
- Multi-condition logic support
- K-anonymity risk-based filtering

Changelog:
1.0.0 - 2025-11-18 - Initial creation of attribute suppression core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class AttributeSuppressionConfig(OperationConfig):
    """
    Core configuration schema for AttributeSuppression backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Attribute Suppression Operation Core Configuration",
        "description": "Core schema for attribute-level suppression operations configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Primary field to apply suppression.",
                    },
                    "additional_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Additional Fields",
                        "description": "Other fields to include in suppression operation.",
                    },
                    "suppression_mode": {
                        "type": "string",
                        "default": "REMOVE",
                        "oneOf": [
                            {"const": "REMOVE", "description": "REMOVE"},
                        ],
                        "title": "Suppression Mode",
                        "description": "Suppression strategy to apply (e.g., REMOVE).",
                    },
                    "save_suppressed_schema": {
                        "type": "boolean",
                        "default": True,
                        "title": "Save Column Metadata",
                        "description": "Whether to save the schema after suppression.",
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
                        "title": "Risk Score Field",
                        "description": "Field used for k-anonymity risk assessment.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "default": 5,
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering suppression.",
                    },
                },
                "required": ["field_name", "suppression_mode"],
            },
            {
                "if": {
                    "properties": {"condition_operator": {"type": "string"}},
                    "required": ["condition_operator"],
                },
                "then": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_field"],
                },
            },
            {
                "if": {
                    "properties": {"condition_values": {"type": "array"}},
                    "required": ["condition_values"],
                },
                "then": {
                    "properties": {
                        "condition_operator": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_operator"],
                },
            },
            {
                "if": {
                    "properties": {"condition_logic": {"type": "string"}},
                    "required": ["condition_logic"],
                },
                "then": {
                    "properties": {
                        "multi_conditions": {"type": "array", "minItems": 1}
                    },
                    "required": ["multi_conditions"],
                },
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
