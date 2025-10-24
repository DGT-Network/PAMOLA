"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating email profiling operations in PAMOLA.CORE.
Supports parameters for field names, top N values, frequency, and profile types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of email config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class EmailOperationConfig(OperationConfig):
    """Configuration for EmailOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common operation fields
            {
                "type": "object",
                "properties": {
                    # --- Email-specific parameters ---
                    "field_name": {"type": "string"},
                    "top_n": {"type": "integer", "minimum": 1, "default": 20},
                    "min_frequency": {"type": "integer", "minimum": 1, "default": 1},
                    "profile_type": {
                        "type": "string",
                        "enum": ["email"],
                        "default": "email",
                    },
                    "analyze_privacy_risk": {"type": "boolean", "default": True},
                },
                "required": ["field_name"],
            },
        ],
    }