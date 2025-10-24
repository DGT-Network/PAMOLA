"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        MVF Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating MVF analysis operations in PAMOLA.CORE.
Supports parameters for field names, top N values, frequency, format types, and parsing options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of MVF config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class MVFAnalysisOperationConfig(OperationConfig):
    """Configuration for MVFOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge shared base fields
            {
                "type": "object",
                "properties": {
                    # --- Operation-specific fields ---
                    "field_name": {"type": "string"},
                    "top_n": {"type": "integer", "minimum": 1, "default": 20},
                    "min_frequency": {"type": "integer", "minimum": 1, "default": 1},
                    "format_type": {"type": ["string", "null"], "default": None},
                    "parse_kwargs": {"type": "object", "default": {}},
                },
                "required": ["field_name"],
            },
        ],
    }

