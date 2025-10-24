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
        "title": "MergeDatasetsOperationConfig",
        "description": "Schema for merging two datasets based on key fields, join type, and relationship type. Supports left/right dataset selection, join keys, suffixes, and relationship validation.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "MergeDatasetsOperationConfig Properties",
                "description": "Properties for configuring dataset merging, including dataset names, join keys, join type, suffixes, and relationship type.",
                "properties": {
                    "left_dataset_name": {
                        "type": "string",
                        "title": "Left Dataset Name",
                        "description": "Name of the main (left) dataset to merge."
                    },
                    "right_dataset_name": {
                        "type": ["string", "null"],
                        "title": "Right Dataset Name",
                        "description": "Name of the right (lookup) dataset to merge."
                    },
                    "right_dataset_path": {
                        "type": ["string", "null"],
                        "title": "Right Dataset Path",
                        "description": "File path to the right dataset if not using a named dataset."
                    },
                    "left_key": {
                        "type": "string",
                        "title": "Left Key",
                        "description": "Key field in the left dataset for joining."
                    },
                    "right_key": {
                        "type": ["string", "null"],
                        "title": "Right Key",
                        "description": "Key field in the right dataset for joining. Defaults to left_key if not set."
                    },
                    "join_type": {
                        "type": "string",
                        "enum": ["inner", "left", "right", "outer"],
                        "title": "Join Type",
                        "description": "Type of join to perform: inner, left, right, or outer."
                    },
                    "suffixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                        "title": "Suffixes",
                        "description": "Suffixes to apply to overlapping columns in the merged dataset."
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": ["auto", "one-to-one", "one-to-many"],
                        "title": "Relationship Type",
                        "description": "Type of relationship between datasets: auto-detect, one-to-one, or one-to-many."
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
