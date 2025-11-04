"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Record Suppression Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
Updated:       2025-01-16
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating record suppression operations in PAMOLA.CORE.
Supports parameters for field names, suppression modes, and control options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.1.0 - 2025-01-16 - Updated with x-group organization matching pattern
1.0.0 - 2025-01-15 - Initial creation of record suppression config file
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class RecordSuppressionConfig(OperationConfig):
    """Configuration for RecordSuppressionOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Record Suppression Operation Configuration",
        "description": "Configuration schema for record suppression operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # === Core parameters ===
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to evaluate for suppression.",
                        "x-component": "Select",
                    },
                    "suppression_mode": {
                        "type": "string",
                        "title": "Suppression Mode",
                        "oneOf": [{"const": "REMOVE", "description": "Remove"}],
                        "default": "REMOVE",
                        "description": "Suppression mode. Only 'REMOVE' (remove entire record) is supported.",
                        "x-component": "Select",
                    },
                    # === Core Suppression Rule ===
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
                        "x-component": "Select",
                        "description": "Condition for suppressing records: 'null', 'value', 'range', 'risk', or 'custom'.",
                        "x-group": GroupName.CORE_SUPPRESSION_RULE,
                    },
                    "suppression_values": {
                        "type": ["array", "null"],
                        "title": "Suppression Values",
                        "description": "List of values to match for suppression (used with 'value' condition).",
                        "items": {
                            "type": "string",
                            "x-component": "Input",
                        },
                        "x-component": "ArrayItems",
                        "x-group": GroupName.CORE_SUPPRESSION_RULE,
                        "x-depend-on": {"suppression_condition": "value"},
                        "x-required-on": {"suppression_condition": "value"},
                    },
                    "suppression_range": {
                        "type": ["array", "null"],
                        "title": "Suppression Range",
                        "description": "Range [min, max] for suppression (used with 'range' condition).",
                        "items": {
                            "type": "number",
                            "x-component": "NumberPicker",
                        },
                        "minItems": 2,
                        "maxItems": 2,
                        "x-component": "ArrayItems",
                        "x-group": GroupName.CORE_SUPPRESSION_RULE,
                        "x-depend-on": {"suppression_condition": "range"},
                        "x-required-on": {"suppression_condition": "range"},
                    },
                    # === Advanced Conditional Rules ===
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "title": "Condition Field",
                                    "x-component": "Select",
                                    "description": "Field name for the condition.",
                                },
                                "operator": {
                                    "type": "string",
                                    "title": "Condition Operator",
                                    "x-component": "Select",
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
                                    "x-depend-on": {"field": "not_null"},
                                    "description": "Operator for the condition (e.g., '=', '>', '<', 'in').",
                                },
                                "values": {
                                    "type": "array",
                                    "title": "Condition Value",
                                    "x-component": "Input",
                                    "description": "Value(s) for the condition.",
                                    "x-depend-on": {
                                        "field": "not_null",
                                        "operator": "not_null",
                                    },
                                },
                            },
                        },
                        "title": "Multi-Conditions",
                        "description": "List of multi-field conditions for custom suppression logic.",
                        "x-component": "ArrayItems",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                    },
                    "condition_logic": {
                        "type": "string",
                        "title": "Condition Logic",
                        "description": "Logical expression for combining multi-field conditions (e.g., 'AND', 'OR').",
                        "oneOf": [
                            {"const": "AND", "description": "AND"},
                            {"const": "OR", "description": "OR"},
                        ],
                        "x-component": "Select",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                        "x-depend-on": {"multi_conditions": "not_null"},
                    },
                    # === Risk-Based Filtering ===
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field containing k-anonymity risk scores for suppression based on risk.",
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering suppression.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                        "x-depend-on": {"ka_risk_field": "not_null"},
                        "x-required-on": {"ka_risk_field": "not_null"},
                    },
                    # === Operation Behavior/Output ===
                    "save_suppressed_records": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Suppressed Records",
                        "description": "Whether to save removed records to a separate artifact.",
                        "x-component": "Switch",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "suppression_reason_field": {
                        "type": "string",
                        "default": "_suppression_reason",
                        "title": "Suppression Reason Field",
                        "description": "Field name for storing the reason for suppression in the output.",
                        "x-component": "Input",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                },
                "required": ["field_name", "suppression_condition"],
            },
            # === Conditional logic for suppression-specific requirements ===
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
                "then": {"required": ["ka_risk_field", "risk_threshold"]},
            },
            {
                "if": {
                    "properties": {"risk_threshold": {"type": "number"}},
                    "required": ["risk_threshold"],
                },
                "then": {
                    "properties": {"ka_risk_field": {"type": "string", "minLength": 1}},
                    "required": ["ka_risk_field"],
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
        ],
    }
