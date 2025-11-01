"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Date Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating date profiling operations in PAMOLA.CORE.
Supports parameters for field names, year ranges, id columns, and profile types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of date config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class DateOperationConfig(OperationConfig):
    """Configuration for DateOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Date Operation Configuration",
        "description": "Configuration schema for date profiling operations. Defines parameters for analyzing a date field, including year range, group/UID columns, and birth date detection.",
        "allOf": [
            BaseOperationConfig.schema,  # merge base common fields
            {
                "type": "object",
                "properties": {
                    # --- field & profiling options ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the date field (column) to analyze. Must exist in the input DataFrame."
                    },
                    "min_year": {
                        "type": "integer",
                        "title": "Minimum Year",
                        "description": "Minimum valid year for anomaly detection in the date field. Dates before this year are considered anomalies.",
                        "minimum": 0,
                        "default": 1940
                    },
                    "max_year": {
                        "type": "integer",
                        "title": "Maximum Year",
                        "description": "Maximum valid year for anomaly detection in the date field. Dates after this year are considered anomalies.",
                        "minimum": 0,
                        "default": 2005
                    },
                    "id_column": {
                        "type": ["string", "null"],
                        "title": "Group ID Column",
                        "description": "Optional column name to use for group-based analysis (e.g., grouping by user or entity).",
                        "default": None
                    },
                    "uid_column": {
                        "type": ["string", "null"],
                        "title": "UID Column",
                        "description": "Optional column name to use for unique identifier (UID) analysis.",
                        "default": None
                    },
                    "profile_type": {
                        "type": "string",
                        "title": "Profile Type",
                        "description": "Type of profiling for organizing artifacts. Default is 'date'.",
                        "enum": ["date"],
                        "default": "date"
                    },
                    "is_birth_date": {
                        "type": ["boolean", "null"],
                        "title": "Is Birth Date Field",
                        "description": "Whether the field is a birth date field. If None, auto-detected based on field name.",
                        "default": None
                    },
                },
                "required": ["field_name"],
            },
        ],
    }