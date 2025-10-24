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
        "allOf": [
            BaseOperationConfig.schema,  # merge base common fields
            {
                "type": "object",
                "properties": {
                    # --- field & profiling options ---
                    "field_name": {"type": "string"},
                    "min_year": {"type": "integer", "minimum": 0, "default": 1940},
                    "max_year": {"type": "integer", "minimum": 0, "default": 2005},
                    "id_column": {"type": ["string", "null"], "default": None},
                    "uid_column": {"type": ["string", "null"], "default": None},
                    "profile_type": {
                        "type": "string",
                        "enum": ["date"],
                        "default": "date",
                    },
                    "is_birth_date": {"type": ["boolean", "null"], "default": None},
                },
                "required": ["field_name"],
            },
        ],
    }