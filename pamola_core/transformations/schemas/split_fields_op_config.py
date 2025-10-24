"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split Fields Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating split fields operations in PAMOLA.CORE.
- Supports splitting by ID field and field groups, with optional ID inclusion
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split fields config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class SplitFieldsOperationConfig(OperationConfig):
    """Configuration for SplitFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "id_field": {"type": ["string", "null"]},
                    "field_groups": {
                        "type": ["object", "null"],
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "include_id_field": {"type": "boolean", "default": True},
                },
            },
        ],
    }

