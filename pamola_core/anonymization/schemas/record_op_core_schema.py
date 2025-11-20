"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Record Suppression Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
Updated:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of record suppression configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for record-level suppression operations
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multiple suppression conditions (null, value, range, risk, custom)
- Value-based and range-based suppression
- Risk-based filtering with k-anonymity scores
- Multi-condition custom logic support
- Suppressed record preservation options
- Suppression reason tracking

Changelog:
1.1.0 - 2025-11-18 - Refactored into separate core schema
1.0.0 - 2025-01-15 - Initial creation of record suppression config file
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class RecordSuppressionConfig(OperationConfig):
    """
    Core configuration schema for RecordSuppression backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Record Suppression Operation Core Configuration",
        "description": "Core schema for record suppression operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to evaluate for suppression.",
                    },
                    "suppression_mode": {
                        "type": "string",
                        "title": "Suppression Mode",
                        "oneOf": [{"const": "REMOVE", "description": "Remove"}],
                        "default": "REMOVE",
                        "description": "Suppression mode. Only 'REMOVE' (remove entire record) is supported.",
                    },
                    "suppression_condition": {
                        "type": "string",
                        "title": "Suppression Condition",
                        "oneOf": [
                            {"const": "null", "description": "Null"},
                            {"const": "value", "description": "Value"},
                            {"const": "range", "description": "Range"},
                            {"const": "risk", "description": "Risk"},
                            {"const": "custom", "description": "Custom"},
                        ],
                        "default": "null",
                        "description": "Condition for suppressing records: 'null', 'value', 'range', 'risk', or 'custom'.",
                    },
                    "suppression_values": {
                        "type": ["array", "null"],
                        "title": "Suppression Values",
                        "description": "List of values to match for suppression (used with 'value' condition).",
                    },
                    "suppression_range": {
                        "type": ["array", "null"],
                        "title": "Suppression Range",
                        "description": "Range [min, max] for suppression (used with 'range' condition).",
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
                        "description": "List of multi-field conditions for custom suppression logic.",
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
                        "default": 5.0,
                        "description": "Threshold for k-anonymity risk triggering suppression.",
                    },
                    "save_suppressed_records": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Suppressed Records",
                        "description": "Whether to save removed records to a separate artifact.",
                    },
                    "suppression_reason_field": {
                        "type": "string",
                        "default": "_suppression_reason",
                        "title": "Suppression Reason Field",
                        "description": "Field name for storing the reason for suppression in the output.",
                    },
                },
                "required": ["field_name", "suppression_condition"],
            },
            {
                "if": {"properties": {"suppression_condition": {"const": "value"}}},
                "then": {"required": ["suppression_values"]},
            },
            {
                "if": {"properties": {"suppression_condition": {"const": "range"}}},
                "then": {"required": ["suppression_range"]},
            },
            {
                "if": {"properties": {"suppression_condition": {"const": "risk"}}},
                "then": {"required": ["ka_risk_field"]},
            },
            {
                "if": {
                    "properties": {"ka_risk_field": {"type": "string", "minLength": 1}},
                    "required": ["ka_risk_field"],
                },
                "then": {"required": ["risk_threshold"]},
            },
        ],
    }
