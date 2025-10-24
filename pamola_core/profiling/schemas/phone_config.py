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

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class PhoneOperationConfig(OperationConfig):
    """Configuration for PhoneOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- Phone-specific fields ---
                    "field_name": {"type": "string"},
                    "min_frequency": {"type": "integer", "minimum": 1, "default": 1},
                    "patterns_csv": {"type": ["string", "null"], "default": None},
                    "country_codes": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                    },
                },
                "required": ["field_name"],
            },
        ],
    }

