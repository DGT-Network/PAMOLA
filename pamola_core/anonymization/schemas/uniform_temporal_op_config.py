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
                        "description": "Name of the datetime field to which uniform temporal noise will be applied."
                    },
                    # Temporal noise parameters
                    "noise_range_days": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Days)",
                        "description": "Maximum absolute value of random time shift in days. Must be positive if specified."
                    },
                    "noise_range_hours": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Hours)",
                        "description": "Maximum absolute value of random time shift in hours. Must be positive if specified."
                    },
                    "noise_range_minutes": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Minutes)",
                        "description": "Maximum absolute value of random time shift in minutes. Must be positive if specified."
                    },
                    "noise_range_seconds": {
                        "type": ["number", "null"],
                        "title": "Noise Range (Seconds)",
                        "description": "Maximum absolute value of random time shift in seconds. Must be positive if specified."
                    },
                    "noise_range": {
                        "type": ["object", "null"],
                        "title": "Noise Range (Composite)",
                        "description": "Dictionary specifying one or more time shift ranges (days, hours, minutes, seconds). Overrides individual parameters if provided.",
                        "properties": {
                            "noise_range_days": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Days)",
                                "description": "Maximum absolute value of random time shift in days (composite)."
                            },
                            "noise_range_hours": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Hours)",
                                "description": "Maximum absolute value of random time shift in hours (composite)."
                            },
                            "noise_range_minutes": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Minutes)",
                                "description": "Maximum absolute value of random time shift in minutes (composite)."
                            },
                            "noise_range_seconds": {
                                "type": ["number", "null"],
                                "title": "Noise Range (Seconds)",
                                "description": "Maximum absolute value of random time shift in seconds (composite)."
                            },
                        },
                    },
                    # Direction control
                    "direction": {
                        "type": "string",
                        "enum": ["both", "forward", "backward"],
                        "default": "both",
                        "title": "Direction",
                        "description": "Direction of time shift: 'both' (forward and backward), 'forward' (future only), or 'backward' (past only)."
                    },
                    # Boundary constraints
                    "min_datetime": {
                        "type": ["string", "null"],
                        "title": "Minimum Datetime",
                        "description": "Minimum allowed datetime after noise is applied. Values below this will be clipped."
                    },
                    "max_datetime": {
                        "type": ["string", "null"],
                        "title": "Maximum Datetime",
                        "description": "Maximum allowed datetime after noise is applied. Values above this will be clipped."
                    },
                    # Special date handling
                    "preserve_special_dates": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Special Dates",
                        "description": "If true, specified special dates will not be shifted."
                    },
                    "special_dates": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Special Dates",
                        "description": "List of dates (as strings) to preserve unchanged during noise application."
                    },
                    "preserve_weekends": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Weekends",
                        "description": "If true, ensures that weekend/weekday status is preserved after noise is applied."
                    },
                    "preserve_time_of_day": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Time of Day",
                        "description": "If true, only the date part is shifted; the original time-of-day is preserved."
                    },
                    # Granularity
                    "output_granularity": {
                        "type": ["string", "null"],
                        "enum": ["day", "hour", "minute", "second", None],
                        "title": "Output Granularity",
                        "description": "Granularity to which output datetimes are rounded: 'day', 'hour', 'minute', or 'second'."
                    },
                    # Reproducibility
                    "random_seed": {
                        "type": ["integer", "null"],
                        "title": "Random Seed",
                        "description": "Seed for random number generator (ignored if use_secure_random is true)."
                    },
                    "use_secure_random": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use Secure Random",
                        "description": "If true, uses cryptographically secure random number generation."
                    },
                    # Multi-field conditions
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "items": {"type": "object"},
                        "title": "Multi-Conditions",
                        "description": "List of condition objects for advanced conditional processing."
                    },
                    "condition_logic": {
                        "type": "string",
                        "title": "Condition Logic",
                        "description": "Logical operator for combining multiple conditions (e.g., 'AND', 'OR')."
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name to use for conditional noise application."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Condition Values",
                        "description": "List of values for conditional noise application."
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Operator for conditional processing (e.g., '=', '!=', 'in')."
                    },
                    # K-anonymity integration
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-Anonymity Risk Field",
                        "description": "Field used for k-anonymity risk assessment."
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk."
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling records identified as vulnerable (e.g., 'suppress', 'flag')."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }