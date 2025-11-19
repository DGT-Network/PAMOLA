"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating uniform numeric noise operations in PAMOLA.CORE.
Supports parameters for field names, noise ranges, and noise types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of uniform numeric noise config file
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName


class UniformNumericNoiseConfig(OperationConfig):
    """Configuration for UniformNumericNoiseOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Uniform Numeric Noise Operation Configuration",
        "description": "Configuration schema for uniform numeric noise operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # ==== Noise Parameters ====
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field to apply noise to.",
                    },
                    "noise_type": {
                        "type": "string",
                        "enum": ["additive", "multiplicative"],
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
                        "x-component": "Select",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range": {
                        "type": ["number", "array"],
                        "title": "Noise Range",
                        "x-component": CustomComponents.NUMERIC_RANGE_MODE,
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                        "name": "noise_range",
                        "default": 0.1,
                    },
                    # ==== Bounds and Constraints ====
                    "output_min": {
                        "type": ["number", "null"],
                        "title": "Output Minimum",
                        "description": "Minimum allowed value after noise is applied.",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "output_max": {
                        "type": ["number", "null"],
                        "title": "Output Maximum",
                        "description": "Maximum allowed value after noise is applied.",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "preserve_zero": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Zero",
                        "description": "If True, zero values will not be changed by noise.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    # ==== Integer Handling ====
                    "round_to_integer": {
                        "type": ["boolean", "null"],
                        "default": False,
                        "title": "Round to Integer",
                        "description": "If True, round the result to the nearest integer.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    # ==== Statistical Parameters ====
                    "scale_by_std": {
                        "type": "boolean",
                        "default": False,
                        "title": "Scale by Std",
                        "description": "If True, scale noise by the standard deviation of the field.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "scale_factor": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1.0,
                        "title": "Scale Factor",
                        "description": "Multiplier for the noise magnitude.",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    # ==== Randomization ====
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
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "x-component": "Select",
                        "description": "Field name used as condition for applying the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
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
                        "x-group": GroupName.CONDITIONAL_LOGIC,
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
                        "x-group": GroupName.CONDITIONAL_LOGIC,
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
                        "x-group": GroupName.CONDITIONAL_LOGIC,
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
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"multi_conditions": "not_null"},
                    },
                    # K-anonymity integration
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
        ],
    }
