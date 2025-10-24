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
                        "description": "Name of the datetime field to generalize."
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["rounding", "binning", "component", "relative"],
                        "title": "Generalization Strategy",
                        "description": "Generalization strategy to apply: rounding, binning, component, or relative."
                    },
                    # --- Rounding parameters ---
                    "rounding_unit": {
                        "type": "string",
                        "enum": ["year", "quarter", "month", "week", "day", "hour"],
                        "title": "Rounding Unit",
                        "description": "Unit for rounding datetime values (e.g., year, month, day, hour)."
                    },
                    # --- Binning parameters ---
                    "bin_type": {
                        "type": "string",
                        "enum": [
                            "hour_range",
                            "day_range",
                            "business_period",
                            "seasonal",
                            "custom",
                        ],
                        "title": "Binning Type",
                        "description": "Type of binning to apply (e.g., day_range, hour_range, business_period, seasonal, custom)."
                    },
                    "interval_size": {
                        "type": "integer",
                        "minimum": 1,
                        "title": "Interval Size",
                        "description": "Size of each binning interval."
                    },
                    "interval_unit": {
                        "type": "string",
                        "enum": ["hours", "days", "weeks", "months"],
                        "title": "Interval Unit",
                        "description": "Unit for binning interval (e.g., days, weeks, months)."
                    },
                    "reference_date": {
                        "type": ["string", "null"],
                        "title": "Reference Date",
                        "description": "Reference date for binning alignment. Accepts string or null."
                    },
                    "custom_bins": {
                        "type": ["array", "null"],
                        "items": {"type": ["string", "number"]},
                        "title": "Custom Bins",
                        "description": "User-defined bin boundaries for custom binning."
                    },
                    # --- Component-based generalization ---
                    "keep_components": {
                        "type": ["array", "null"],
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
                        "title": "Keep Components",
                        "description": "List of datetime components to keep (e.g., year, month, day, hour, minute, weekday)."
                    },
                    "strftime_output_format": {
                        "type": ["string", "null"],
                        "title": "Strftime Output Format",
                        "description": "Custom output format for datetime string conversion."
                    },
                    # --- Timezone and format handling ---
                    "timezone_handling": {
                        "type": "string",
                        "enum": ["preserve", "utc", "remove"],
                        "default": "preserve",
                        "title": "Timezone Handling",
                        "description": "How to handle timezones: preserve, convert to UTC, or remove timezone info."
                    },
                    "default_timezone": {
                        "type": "string",
                        "default": "UTC",
                        "title": "Default Timezone",
                        "description": "Default timezone to use if missing in input."
                    },
                    "input_formats": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Input Formats",
                        "description": "Accepted input datetime formats."
                    },
                    # --- Privacy & QI ---
                    "min_privacy_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "title": "Minimum Privacy Threshold",
                        "description": "Minimum privacy preservation threshold (ratio of unique value reduction)."
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Quasi-identifiers",
                        "description": "List of quasi-identifier fields to consider for privacy checks."
                    },
                    # Output field name configuration
                    "output_field_name": {
                        "type": ["string", "null"],
                        "title": "Output Field Name",
                        "description": "Custom output field name (for ENRICH mode)."
                    },
                },
                "required": ["field_name", "strategy"],
            },
        ],
    }