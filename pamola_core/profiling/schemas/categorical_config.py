"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating categorical profiling operations in PAMOLA.CORE.
Supports parameters for field names, top N values, frequency, and profile types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of categorical config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class CategoricalOperationConfig(OperationConfig):
    """Configuration for CategoricalOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge BaseOperation common fields
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "top_n": {"type": "integer", "minimum": 1, "default": 15},
                    "min_frequency": {"type": "integer", "minimum": 1, "default": 1},
                    "profile_type": {
                        "type": "string",
                        "enum": ["categorical", "string", "text"],
                        "default": "categorical",
                    },
                    "analyze_anomalies": {"type": "boolean", "default": True},
                },
                "required": ["field_name"],
            },
        ],
    }
