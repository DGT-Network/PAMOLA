"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Datetime Generalization Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating datetime generalization parameters in PAMOLA.CORE.
- Supports multiple generalization strategies (rounding, binning, component, relative)
- Handles timezone, custom bins, privacy thresholds, and output formatting
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of datetime generalization config file
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class DateTimeGeneralizationConfig(OperationConfig):
    """Configuration for DateTimeGeneralizationOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Datetime Generalization Operation Configuration",
        "description": "Configuration schema for datetime generalization operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common fields
            {
                "type": "object",
                "properties": {
                    # Required fields
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "x-component": "Select",
                        "description": "Name of the datetime field to generalize.",
                    },
                    "strategy": {
                        "type": "string",
                        "title": "Strategy",
                        "x-component": "Select",
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
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    # --- Rounding parameters ---
                    "rounding_unit": {
                        "type": "string",
                        "default": "day",
                        "title": "Rounding Unit",
                        "description": "Unit for rounding datetime values (e.g., year, month, day, hour).",
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "rounding"},
                        "x-required-on": {"strategy": "rounding"},
                        "oneOf": [
                            {"const": "year", "description": "Year"},
                            {"const": "quarter", "description": "Quarter"},
                            {"const": "month", "description": "Month"},
                            {"const": "week", "description": "Week"},
                            {"const": "day", "description": "Day"},
                            {"const": "hour", "description": "Hour"},
                        ],
                    },
                    # --- Binning parameters ---
                    "bin_type": {
                        "type": "string",
                        "default": "day_range",
                        "title": "Binning Type",
                        "description": "Type of binning to apply (e.g., day_range, hour_range, business_period, seasonal, custom).",
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "binning"},
                        "x-required-on": {"strategy": "binning"},
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
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"bin_type": ["hour_range", "day_range"]},
                        "x-required-on": {"bin_type": ["hour_range", "day_range"]},
                    },
                    "interval_unit": {
                        "type": "string",
                        "default": "days",
                        "title": "Interval Unit",
                        "description": "Unit for binning interval (e.g., days, weeks, months).",
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"bin_type": ["hour_range", "day_range"]},
                        "x-required-on": {"bin_type": ["hour_range", "day_range"]},
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
                        "x-component": "DatePicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "relative"},
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                    },
                    "custom_bins": {
                        "type": ["array", "null"],
                        "title": "Custom Bins",
                        "description": "User-defined bin boundaries for custom binning.",
                        "x-component": "DatePickerArray",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"bin_type": "custom"},
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "getPopupContainer": "{{(node) => node?.parentElement || document.body}}",
                            "placeholder": "YYYY-MM-DD",
                        },
                    },
                    # --- Component-based generalization ---
                    "keep_components": {
                        "type": ["array", "null"],
                        "title": "Components to Keep",
                        "description": "List of datetime components to keep (e.g., year, month, day, hour, minute, weekday).",
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "component"},
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "year",
                                        "month",
                                        "day",
                                        "hour",
                                        "minute",
                                        "weekday",
                                    ],
                                },
                            },
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
                        "x-component": "Input",
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                    },
                    # --- Timezone and format handling ---
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
                        "x-component": "Select",
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
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
                        "x-component": "Input",
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                        "x-depend-on": {"timezone_handling": "utc"},
                    },
                    "input_formats": {
                        "type": ["array", "null"],
                        "title": "Custom Input Formats",
                        "description": "Accepted input datetime formats.",
                        "x-component": "DateFormatArray",
                        "x-component-props": {
                            "formatActions": "{{ supportedFormatActions }}",
                            "placeholder": "Custom datetime pattern",
                        },
                        "x-group": GroupName.FORMATTING_AND_TIMEZONE,
                    },
                    # --- Privacy & QI ---
                    "min_privacy_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.0,
                        "title": "Minimum Privacy Threshold",
                        "description": "Minimum privacy preservation threshold (ratio of unique value reduction).",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Quasi-identifiers",
                        "description": "List of quasi-identifier fields to consider for privacy checks.",
                        "visible": False,
                    },
                },
                "required": ["field_name", "strategy"],
            },
            # === Conditional logic for strategy-specific requirements ===
            {
                "if": {"properties": {"strategy": {"const": "rounding"}}},
                "then": {"required": ["rounding_unit"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "binning"}}},
                "then": {"required": ["bin_type"]},
            },
            {
                "if": {"properties": {"bin_type": {"const": "hour_range"}}},
                "then": {"required": ["interval_size", "interval_unit"]},
            },
            {
                "if": {"properties": {"bin_type": {"const": "day_range"}}},
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
