"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Phone Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating phone profiling operations in PAMOLA.CORE.
Supports parameters for field names, frequency, patterns, and country codes.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of phone config file
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class PhoneOperationConfig(OperationConfig):
    """Configuration for PhoneOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Phone Operation Configuration",
        "description": "Configuration schema for phone profiling operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- Phone-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "x-component": "Select",
                        "description": "Name of the phone number field (column) to analyze. This should be a column in the DataFrame containing phone numbers.",
                    },
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 1,
                        "title": "Minimum Frequency",
                        "x-component": "NumberPicker",
                        "description": "Minimum number of occurrences for a phone number or component to be included in the results. Values appearing fewer times will be excluded from the output.",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                    "country_codes": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Country Codes",
                        "x-component": "Select",
                        "description": "List of country codes to restrict the analysis to specific countries. If null, all detected country codes will be included.",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                    "patterns_csv": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Patterns CSV",
                        "x-component": "Upload",
                        "description": "Path to a CSV file containing phone number patterns for validation and parsing. If null, default patterns will be used.",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                },
                "required": ["field_name"],
            },
        ],
    }