"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Datetime Generalization Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of datetime generalization configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for multiple generalization strategies (rounding, binning, component, relative)
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multiple datetime generalization strategies with strategy-specific parameter validation
- Timezone handling and custom format support
- Privacy threshold controls and quasi-identifier management
- Custom binning and component-based generalization
- Relative date conversion capabilities

Changelog:
1.0.0 - 2025-11-18 - Initial creation of datetime generalization core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class DateTimeGeneralizationConfig(OperationConfig):
    """
    Core configuration schema for DateTimeGeneralization backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Datetime Generalization Operation Core Configuration",
        "description": "Core schema for datetime generalization operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the datetime field to generalize.",
                    },
                    "strategy": {
                        "type": "string",
                        "title": "Strategy",
                        "oneOf": [
                            {"const": "binning", "description": "Binning"},
                            {"const": "rounding", "description": "Rounding"},
                            {"const": "component", "description": "Component"},
                            {"const": "relative", "description": "Relative"},
                        ],
                        "default": "binning",
                        "description": (
                            "Generalization strategy to apply: rounding, binning, component, or relative.\n"
                            "- 'binning': Groups timestamps into defined intervals.\n"
                            "- 'rounding': Reduces precision to a larger time unit.\n"
                            "- 'component': Keeps only specific parts of the date/time.\n"
                            "- 'relative': Converts timestamps into relative descriptions."
                        ),
                    },
                    "rounding_unit": {
                        "type": "string",
                        "default": "day",
                        "title": "Rounding Unit",
                        "description": "Unit for rounding datetime values (e.g., year, month, day, hour).",
                        "oneOf": [
                            {"const": "year", "description": "Year"},
                            {"const": "quarter", "description": "Quarter"},
                            {"const": "month", "description": "Month"},
                            {"const": "week", "description": "Week"},
                            {"const": "day", "description": "Day"},
                            {"const": "hour", "description": "Hour"},
                        ],
                    },
                    "bin_type": {
                        "type": "string",
                        "default": "day_range",
                        "title": "Binning Type",
                        "description": "Type of binning to apply (e.g., day_range, hour_range, business_period, seasonal, custom).",
                        "oneOf": [
                            {"const": "hour_range", "description": "Hour Range"},
                            {"const": "day_range", "description": "Day Range"},
                            {
                                "const": "business_period",
                                "description": "Business Period",
                            },
                            {"const": "seasonal", "description": "Seasonal"},
                            {"const": "custom", "description": "Custom"},
                        ],
                    },
                    "interval_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 7,
                        "title": "Interval Size",
                        "description": "Size of each binning interval.",
                    },
                    "interval_unit": {
                        "type": "string",
                        "default": "days",
                        "title": "Interval Unit",
                        "description": "Unit for binning interval (e.g., days, weeks, months).",
                        "oneOf": [
                            {"const": "hours", "description": "Hours"},
                            {"const": "days", "description": "Days"},
                            {"const": "weeks", "description": "Weeks"},
                            {"const": "months", "description": "Months"},
                        ],
                    },
                    "reference_date": {
                        "type": ["string", "null"],
                        "title": "Reference Date",
                        "description": "Reference date for binning alignment. Accepts string or null.",
                    },
                    "custom_bins": {
                        "type": ["array", "null"],
                        "title": "Custom Bins",
                        "description": "User-defined bin boundaries for custom binning.",
                    },
                    "keep_components": {
                        "type": ["array", "null"],
                        "title": "Components to Keep",
                        "description": "List of datetime components to keep (e.g., year, month, day, hour, minute, weekday).",
                        "oneOf": [
                            {"type": "null"},
                            {"const": "year", "description": "Years"},
                            {"const": "month", "description": "Month"},
                            {"const": "day", "description": "Day"},
                            {"const": "hour", "description": "Hour"},
                            {"const": "minute", "description": "Minute"},
                            {"const": "weekday", "description": "Weekday"},
                        ],
                    },
                    "strftime_output_format": {
                        "type": ["string", "null"],
                        "title": "Custom Output Format",
                        "description": "Custom output format for datetime string conversion.",
                    },
                    "timezone_handling": {
                        "type": "string",
                        "default": "preserve",
                        "title": "Timezone Handling",
                        "description": (
                            "How to handle timezones:\n"
                            "preserve: Keeps the original timezone (default).\n"
                            "utc: Converts all timestamps to UTC for consistency.\n"
                            "remove: Strips all timezone information."
                        ),
                        "oneOf": [
                            {"const": "preserve", "description": "Preserve"},
                            {"const": "utc", "description": "UTC"},
                            {"const": "remove", "description": "Remove"},
                        ],
                    },
                    "default_timezone": {
                        "type": "string",
                        "default": "UTC",
                        "title": "Default Timezone",
                        "description": "Default timezone to use if missing in input.",
                    },
                    "input_formats": {
                        "type": ["array", "null"],
                        "title": "Custom Input Formats",
                        "description": "Accepted input datetime formats.",
                    },
                    "min_privacy_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "title": "Minimum Privacy Threshold",
                        "description": "Minimum privacy preservation threshold (ratio of unique value reduction).",
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Quasi-identifiers",
                        "description": "List of quasi-identifier fields to consider for privacy checks.",
                    },
                },
                "required": ["field_name", "strategy"],
            },
            {
                "if": {"properties": {"strategy": {"const": "rounding"}}},
                "then": {"required": ["rounding_unit"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "binning"}}},
                "then": {"required": ["bin_type"]},
            },
            {
                "if": {
                    "properties": {
                        "bin_type": {
                            "oneOf": [{"const": "hour_range"}, {"const": "day_range"}]
                        }
                    }
                },
                "then": {"required": ["interval_size", "interval_unit"]},
            },
            {
                "if": {"properties": {"bin_type": {"const": "custom"}}},
                "then": {"required": ["custom_bins"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "component"}}},
                "then": {"required": ["keep_components"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "relative"}}},
                "then": {
                    "properties": {"reference_date": {"type": "string", "minLength": 1}}
                },
            },
            {
                "if": {"properties": {"timezone_handling": {"const": "utc"}}},
                "then": {"required": ["default_timezone"]},
            },
        ],
    }
