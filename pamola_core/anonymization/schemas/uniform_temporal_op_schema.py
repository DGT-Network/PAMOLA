"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating uniform temporal noise operations in PAMOLA.CORE.
Supports parameters for field names, temporal noise ranges, and noise types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of uniform temporal noise config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName


class UniformTemporalNoiseConfig(OperationConfig):
    """Configuration for UniformTemporalNoiseOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Uniform Temporal Noise Operation Configuration",
        "description": "Configuration schema for uniform temporal noise operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields from base config
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the datetime field to which uniform temporal noise will be applied.",
                    },
                    # Temporal noise parameters
                    "noise_range_days": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Days)",
                        "description": "Maximum absolute value of random time shift in days. Must be positive if specified.",
                        "minimum": 0,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range_hours": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Hours)",
                        "description": "Maximum absolute value of random time shift in hours. Must be positive if specified.",
                        "minimum": 0,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range_minutes": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Minutes)",
                        "description": "Maximum absolute value of random time shift in minutes. Must be positive if specified.",
                        "minimum": 0,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range_seconds": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Seconds)",
                        "description": "Maximum absolute value of random time shift in seconds. Must be positive if specified.",
                        "minimum": 0,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range": {
                        "type": ["object", "null"],
                        "title": "Noise Range (Composite)",
                        "description": "Dictionary specifying one or more time shift ranges (days, hours, minutes, seconds). Overrides individual parameters if provided.",
                        "properties": {
                            "noise_range_days": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Days)",
                                "description": "Maximum absolute value of random time shift in days (composite).",
                            },
                            "noise_range_hours": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Hours)",
                                "description": "Maximum absolute value of random time shift in hours (composite).",
                            },
                            "noise_range_minutes": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Minutes)",
                                "description": "Maximum absolute value of random time shift in minutes (composite).",
                            },
                            "noise_range_seconds": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Seconds)",
                                "description": "Maximum absolute value of random time shift in seconds (composite).",
                            },
                        },
                    },
                    # Direction control
                    "direction": {
                        "type": "string",
                        "enum": ["both", "forward", "backward"],
                        "default": "both",
                        "title": "Direction",
                        "description": "Direction of time shift: 'both' (forward and backward), 'forward' (future only), or 'backward' (past only).",
                        "oneOf": [
                            {"const": "both", "description": "Both"},
                            {"const": "forward", "description": "Forward"},
                            {"const": "backward", "description": "Backward"},
                        ],
                        "x-component": "Select",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    # Boundary constraints
                    "min_datetime": {
                        "type": ["string", "null"],
                        "title": "Minimum Datetime",
                        "description": "Minimum allowed datetime after noise is applied. Values below this will be clipped.",
                        "x-component": "DatePicker",
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "max_datetime": {
                        "type": ["string", "null"],
                        "title": "Maximum Datetime",
                        "description": "Maximum allowed datetime after noise is applied. Values above this will be clipped.",
                        "x-component": "DatePicker",
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    # Special date handling
                    "preserve_special_dates": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Special Dates",
                        "description": "If true, specified special dates will not be shifted.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.PRESERVATION_RULES,
                    },
                    "special_dates": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Special Dates",
                        "description": "List of dates (as strings) to preserve unchanged during noise application.",
                        "x-component": "DatePicker",
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                        "x-group": GroupName.PRESERVATION_RULES,
                        "x-depend-on": {"preserve_special_dates": True},
                        "x-required-on": {"preserve_special_dates": True},
                    },
                    "preserve_weekends": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Weekends",
                        "description": "If true, ensures that weekend/weekday status is preserved after noise is applied.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.PRESERVATION_RULES,
                    },
                    "preserve_time_of_day": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Time of Day",
                        "description": "If true, only the date part is shifted; the original time-of-day is preserved.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.PRESERVATION_RULES,
                    },
                    # Granularity
                    "output_granularity": {
                        "type": ["string", "null"],
                        "enum": ["day", "hour", "minute", "second", None],
                        "title": "Output Granularity",
                        "description": "Granularity to which output datetimes are rounded: 'day', 'hour', 'minute', or 'second'.",
                        "oneOf": [
                            {"const": "", "description": ""},
                            {"const": "day", "description": "Day"},
                            {"const": "hour", "description": "Hour"},
                            {"const": "minute", "description": "Minute"},
                            {"const": "second", "description": "Second"},
                        ],
                        "x-component": "Select",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    # Reproducibility
                    "random_seed": {
                        "type": ["integer", "null"],
                        "title": "Random Seed",
                        "description": "Seed for random number generator (ignored if use_secure_random is true).",
                    },
                    "use_secure_random": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use Secure Random",
                        "description": "If true, uses cryptographically secure random number generation.",
                    },
                    # Multi-field conditions
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "x-component": "ArrayItems",
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
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "x-component": "Select",
                        "description": "Field name used as condition for applying the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": ["update_condition_field"],
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
                        "x-custom-function": ["update_condition_operator"],
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
                        "x-custom-function": ["update_condition_values"],
                    },
                    # K-anonymity integration
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-Anonymity Risk Field",
                        "description": "Field used for k-anonymity risk assessment.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk.",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling records identified as vulnerable (e.g., 'suppress', 'flag').",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
