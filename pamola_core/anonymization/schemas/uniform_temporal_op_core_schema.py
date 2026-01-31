"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of uniform temporal noise configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for temporal noise application on datetime fields
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multiple temporal noise range specifications (days, hours, minutes, seconds)
- Directional control (both, forward, backward)
- Boundary constraints with min/max datetime
- Special date preservation and weekend handling
- Output granularity control
- Conditional noise application based on field values
- Multi-condition logic support
- K-anonymity risk assessment integration

Changelog:
1.0.0 - 2025-11-18 - Initial creation of uniform temporal noise core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class UniformTemporalNoiseConfig(OperationConfig):
    """
    Core configuration schema for UniformTemporalNoise backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Uniform Temporal Noise Operation Core Configuration",
        "description": "Core schema for uniform temporal noise operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the datetime field to which uniform temporal noise will be applied.",
                    },
                    "noise_range_days": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Days)",
                        "description": "Maximum absolute value of random time shift in days. Must be positive if specified.",
                        "minimum": 0,
                    },
                    "noise_range_hours": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Hours)",
                        "description": "Maximum absolute value of random time shift in hours. Must be positive if specified.",
                        "minimum": 0,
                    },
                    "noise_range_minutes": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Minutes)",
                        "description": "Maximum absolute value of random time shift in minutes. Must be positive if specified.",
                        "minimum": 0,
                    },
                    "noise_range_seconds": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Seconds)",
                        "description": "Maximum absolute value of random time shift in seconds. Must be positive if specified.",
                        "minimum": 0,
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
                    "direction": {
                        "type": "string",
                        "default": "both",
                        "title": "Direction",
                        "description": "Direction of time shift: 'both' (forward and backward), 'forward' (future only), or 'backward' (past only).",
                        "oneOf": [
                            {"const": "both", "description": "Both"},
                            {"const": "forward", "description": "Forward"},
                            {"const": "backward", "description": "Backward"},
                        ],
                    },
                    "min_datetime": {
                        "type": ["string", "null"],
                        "title": "Minimum Datetime",
                        "description": "Minimum allowed datetime after noise is applied. Values below this will be clipped.",
                    },
                    "max_datetime": {
                        "type": ["string", "null"],
                        "title": "Maximum Datetime",
                        "description": "Maximum allowed datetime after noise is applied. Values above this will be clipped.",
                    },
                    "preserve_special_dates": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Special Dates",
                        "description": "If true, specified special dates will not be shifted.",
                    },
                    "special_dates": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Special Dates",
                        "description": "List of dates (as strings) to preserve unchanged during noise application.",
                    },
                    "preserve_weekends": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Weekends",
                        "description": "If true, ensures that weekend/weekday status is preserved after noise is applied.",
                    },
                    "preserve_time_of_day": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Time of Day",
                        "description": "If true, only the date part is shifted; the original time-of-day is preserved.",
                    },
                    "output_granularity": {
                        "type": ["string", "null"],
                        "title": "Output Granularity",
                        "default": None,
                        "description": "Granularity to which output datetimes are rounded: 'day', 'hour', 'minute', or 'second'.",
                        "oneOf": [
                            {"type": "null"},
                            {"const": "day", "description": "Day"},
                            {"const": "hour", "description": "Hour"},
                            {"const": "minute", "description": "Minute"},
                            {"const": "second", "description": "Second"},
                        ],
                    },
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
            {
                "if": {"properties": {"preserve_special_dates": {"const": True}}},
                "then": {"required": ["special_dates"]},
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
