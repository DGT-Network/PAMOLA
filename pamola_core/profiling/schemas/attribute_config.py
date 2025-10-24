"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating attribute profiling operations in PAMOLA.CORE.
Supports parameters for attribute dictionaries, language, and profiling options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of attribute config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class DataAttributeProfilerOperationConfig(OperationConfig):
    """Configuration for DataAttributeProfilerOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema, # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "dictionary_path": {"type": ["string", "null"], "default": None},
                    "language": {"type": "string", "default": "en"},
                    "sample_size": {"type": "integer", "minimum": 1, "default": 10},
                    "max_columns": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "default": None,
                    },
                    "id_column": {"type": ["string", "null"], "default": None},
                },
                "required": [],
            },
        ],
    }

