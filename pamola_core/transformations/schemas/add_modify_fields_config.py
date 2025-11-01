"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add/Modify Fields Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating add/modify fields operations in PAMOLA.CORE.
- Supports field operations and lookup tables for transformation pipelines
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of add/modify fields config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class AddOrModifyFieldsOperationConfig(OperationConfig):
    """Configuration for AddOrModifyFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Add/Modify Fields Operation Configuration",
        "description": "Configuration schema for add/modify fields operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "field_operations": {
                        "type": ["object", "null"],
                        "title": "Field Operations",
                        "description": "Dictionary defining operations to add or modify fields.",
                    },
                    "lookup_tables": {
                        "type": ["object", "null"],
                        "title": "Lookup Tables",
                        "description": "Lookup tables for field transformations.",
                    },
                },
            },
        ],
    }
