"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Suppression Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating attribute suppression operations in PAMOLA.CORE.
Supports parameters for field names, additional fields, suppression modes, and multi-field conditions.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of attribute suppression config file
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class AttributeSuppressionConfig(OperationConfig):
    """Configuration schema for AttributeSuppressionOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Attribute Suppression Operation Configuration",
        "description": "Configuration schema for attribute-level suppression operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common parameters
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
                        "enum": ["REMOVE"],
                        "title": "Suppression Mode",
                        "description": "Suppression strategy to apply (e.g., REMOVE).",
                    },
                    "save_suppressed_schema": {
                        "type": "boolean",
                        "default": True,
                        "title": "Save Column Metadata",
                        "description": "Whether to save the schema after suppression.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "x-component": "Select",
                        "description": "Field name used as condition for applying the generalization.",
                        "x-group": GroupName.SIMPLE_CONDITIONAL_RULE,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
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
                        "x-group": GroupName.SIMPLE_CONDITIONAL_RULE,
                        "x-depend-on": {"condition_field": "not_null"},
                        "x-custom-function": [
                            CustomFunctions.UPDATE_CONDITION_OPERATOR
                        ],
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "x-component": "Input",  # ArrayItems
                        "description": "Values of the condition field that trigger the generalization.",
                        "x-group": GroupName.SIMPLE_CONDITIONAL_RULE,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_VALUES],
                    },
                    # Multi-field conditions
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "x-component": "ArrayItems",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "title": "Condition Field",
                                    "x-component": "Select",
                                    "description": "Field name for the condition.",
                                    "x-decorator-props": {
                                        "layout": "vertical",
                                        "style": {"width": "250px", "marginBottom": 8},
                                    },
                                    "x-custom-function": [
                                        CustomFunctions.UPDATE_FIELD_OPTIONS
                                    ],
                                },
                                "operator": {
                                    "type": "string",
                                    "title": "Condition Operator",
                                    "x-component": "Select",
                                    "oneOf": [
                                        {
                                            "const": "in",
                                            "description": "In",
                                        },
                                        {
                                            "const": "not_in",
                                            "description": "Not in",
                                        },
                                        {
                                            "const": "gt",
                                            "description": "Greater than",
                                        },
                                        {
                                            "const": "lt",
                                            "description": "Less than",
                                        },
                                        {
                                            "const": "eq",
                                            "description": "Equal to",
                                        },
                                        {
                                            "const": "ne",
                                            "description": "Not equal",
                                        },
                                        {
                                            "const": "ge",
                                            "description": "Greater than or equal",
                                        },
                                        {
                                            "const": "le",
                                            "description": "Less than or equal",
                                        },
                                        {
                                            "const": "range",
                                            "description": "Range",
                                        },
                                        {
                                            "const": "all",
                                            "description": "All",
                                        },
                                    ],
                                    "x-depend-on": {"field": "not_null"},
                                    "x-decorator-props": {
                                        "layout": "vertical",
                                        "style": {"width": "250px", "marginBottom": 8},
                                    },
                                    "description": "Operator for the condition (e.g., '=', '>', '<', 'in').",
                                    "x-custom-function": [
                                        CustomFunctions.UPDATE_CONDITION_OPERATOR
                                    ],
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
                                    "x-decorator-props": {
                                        "layout": "vertical",
                                        "style": {"width": "250px", "marginBottom": 8},
                                    },
                                    "x-custom-function": [
                                        CustomFunctions.UPDATE_CONDITION_VALUES
                                    ],
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
                            {
                                "const": "AND",
                                "description": "AND",
                            },
                            {
                                "const": "OR",
                                "description": "OR",
                            },
                        ],
                        "x-component": "Select",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                        "x-depend-on": {"multi_conditions": "not_null"},
                    },
                    # K-anonymity integration
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "Risk Score Field",
                        "description": "Field used for k-anonymity risk assessment.",
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_INT64_FIELD_OPTIONS
                        ],
                    },
                    "risk_threshold": {
                        "type": "number",
                        "default": 5,
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering suppression.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                        "x-depend-on": {"ka_risk_field": "not_null"},
                    },
                },
                "required": ["field_name", "suppression_mode"],
            },
            # === Conditional logic for strategy-specific requirements ===
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
        ],
    }
