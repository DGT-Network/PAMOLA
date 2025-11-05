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

from pamola_core.common.enum.form_groups import GroupName
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
                        "type": "number",
                        "title": "Min year",
                        "description": "Minimum valid year for anomaly detection in the date field. Dates before this year are considered anomalies.",
                        "minimum": 1,
                        "maximum": 9999,
                        "default": 1940,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.VALIDATION_RANGE,
                    },
                    "max_year": {
                        "type": "number",
                        "title": "Max year",
                        "description": "Maximum valid year for anomaly detection in the date field. Dates after this year are considered anomalies.",
                        "minimum": 1,
                        "maximum": 9999,
                        "default": 2005,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.VALIDATION_RANGE,
                    },
                    "id_column": {
                        "type": ["string", "null"],
                        "title": "ID column",
                        "description": "Optional column name to use for group-based analysis (e.g., grouping by user or entity).",
                        "default": None,
                        "x-component": "Select",
                        "x-group": GroupName.DATA_QUALITY_ANALYSIS,
                    },
                    "uid_column": {
                        "type": ["string", "null"],
                        "title": "UID column",
                        "description": "Optional column name to use for unique identifier (UID) analysis.",
                        "default": None,
                        "x-component": "Select",
                        "x-group": GroupName.DATA_QUALITY_ANALYSIS,
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
                        "title": "Is birth date",
                        "description": "Whether the field is a birth date field. If None, auto-detected based on field name.",
                        "default": False,
                        "x-component": "Checkbox",
                        "x-group": GroupName.DATA_QUALITY_ANALYSIS,
                    },
                },
                "required": ["field_name"],
            },
        ],
    }