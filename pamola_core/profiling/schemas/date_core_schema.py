"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Date Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of date profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines date analysis parameters, year range validation, and birth date detection
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for date analysis
- Year range validation controls (min/max year)
- ID and UID column configuration for grouped analysis
- Birth date detection and profile type settings

Changelog:
1.0.0 - 2025-01-15 - Initial creation of date profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class DateOperationConfig(OperationConfig):
    """
    Core configuration schema for DateOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Date Operation Core Configuration",
        "description": "Core schema for date profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the date field (column) to analyze. Must exist in the input DataFrame.",
                    },
                    "min_year": {
                        "type": "number",
                        "title": "Min year",
                        "description": "Minimum valid year for anomaly detection in the date field. Dates before this year are considered anomalies.",
                        "minimum": 1,
                        "maximum": 9999,
                        "default": 1940,
                    },
                    "max_year": {
                        "type": "number",
                        "title": "Max year",
                        "description": "Maximum valid year for anomaly detection in the date field. Dates after this year are considered anomalies.",
                        "minimum": 1,
                        "maximum": 9999,
                        "default": 2005,
                    },
                    "id_column": {
                        "type": ["string", "null"],
                        "title": "ID column",
                        "description": "Optional column name to use for group-based analysis (e.g., grouping by user or entity).",
                        "default": None,
                    },
                    "uid_column": {
                        "type": ["string", "null"],
                        "title": "UID column",
                        "description": "Optional column name to use for unique identifier (UID) analysis.",
                        "default": None,
                    },
                    "profile_type": {
                        "type": "string",
                        "title": "Profile Type",
                        "description": "Type of profiling for organizing artifacts. Default is 'date'.",
                        "default": "date",
                    },
                    "is_birth_date": {
                        "type": ["boolean", "null"],
                        "title": "Is birth date",
                        "description": "Whether the field is a birth date field. If None, auto-detected based on field name.",
                        "default": False,
                    },
                },
                "required": ["field_name", "min_year", "max_year"],
            },
        ],
    }
