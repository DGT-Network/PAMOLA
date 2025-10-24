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
        "allOf": [
            BaseOperationConfig.schema,  # merge all common fields
            {
                "type": "object",
                "properties": {
                    # Required fields
                    "field_name": {"type": "string"},
                    "strategy": {
                        "type": "string",
                        "enum": ["rounding", "binning", "component", "relative"],
                    },
                    # --- Rounding parameters ---
                    "rounding_unit": {
                        "type": "string",
                        "enum": ["year", "quarter", "month", "week", "day", "hour"],
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
                    },
                    "interval_size": {"type": "integer", "minimum": 1},
                    "interval_unit": {
                        "type": "string",
                        "enum": ["hours", "days", "weeks", "months"],
                    },
                    "reference_date": {"type": ["string", "null"]},
                    "custom_bins": {
                        "type": ["array", "null"],
                        "items": {"type": ["string", "number"]},
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
                    },
                    "strftime_output_format": {"type": ["string", "null"]},
                    # --- Timezone and format handling ---
                    "timezone_handling": {
                        "type": "string",
                        "enum": ["preserve", "utc", "remove"],
                        "default": "preserve",
                    },
                    "default_timezone": {"type": "string", "default": "UTC"},
                    "input_formats": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    # --- Privacy & QI ---
                    "min_privacy_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    # Output field name configuration
                    "output_field_name": {"type": ["string", "null"]},
                },
                "required": ["field_name", "strategy"],
            },
        ],
    }