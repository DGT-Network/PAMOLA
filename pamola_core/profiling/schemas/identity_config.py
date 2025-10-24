"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Identity Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating identity analysis operations in PAMOLA.CORE.
Supports parameters for uid fields, reference fields, and id fields.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of identity config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class IdentityAnalysisOperationConfig(OperationConfig):
    """Configuration for IdentityAnalysisOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields from BaseOperationConfig
            {
                "type": "object",
                "properties": {
                    # --- Operation-specific fields ---
                    "uid_field": {"type": "string"},
                    "reference_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "id_field": {"type": ["string", "null"], "default": None},
                    "top_n": {"type": "integer", "minimum": 1, "default": 15},
                    "check_cross_matches": {"type": "boolean", "default": True},
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                    },
                    "fuzzy_matching": {"type": "boolean", "default": False},
                },
                "required": ["uid_field", "reference_fields"],
            },
        ],
    }

