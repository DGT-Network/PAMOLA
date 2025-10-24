"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Remove Fields Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating remove fields operations in PAMOLA.CORE.
- Supports explicit field removal and pattern-based removal for transformation pipelines
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of remove fields config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class RemoveFieldsOperationConfig(OperationConfig):
    """Configuration for RemoveFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "fields_to_remove": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "pattern": {"type": ["string", "null"]},
                },
            },
        ],
    }
