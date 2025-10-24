"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Merge Datasets Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating merge datasets operations in PAMOLA.CORE.
- Supports left/right dataset selection, join keys, join types, suffixes, and relationship types
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of merge datasets config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class MergeDatasetsOperationConfig(OperationConfig):
    """Configuration for MergeDatasetsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "left_dataset_name": {"type": "string"},
                    "right_dataset_name": {"type": ["string", "null"]},
                    "right_dataset_path": {"type": ["string", "null"]},
                    "left_key": {"type": "string"},
                    "right_key": {"type": ["string", "null"]},
                    "join_type": {
                        "type": "string",
                        "enum": ["inner", "left", "right", "outer"],
                    },
                    "suffixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": ["auto", "one-to-one", "one-to-many"],
                    },
                },
                "required": [
                    "left_dataset_name",
                    "left_key",
                    "right_dataset_name",
                    "right_key",
                    "join_type",
                    "relationship_type",
                ],
            },
        ],
    }
