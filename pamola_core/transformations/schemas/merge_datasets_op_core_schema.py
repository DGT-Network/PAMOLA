"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Merge Datasets Core Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of merge datasets configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines dataset merging parameters including join keys, types, and relationships
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Left and right dataset configuration
- Join key specification for both datasets
- Join type selection (inner, left, right, outer)
- Relationship type validation (one-to-one, one-to-many)
- Suffix configuration for overlapping columns

Changelog:
1.0.0 - 2025-01-15 - Initial creation of merge datasets core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class MergeDatasetsOperationConfig(OperationConfig):
    """
    Core configuration schema for MergeDatasetsOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Merge Datasets Operation Core Configuration",
        "description": "Core schema for Merge Datasets operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "left_dataset_name": {
                        "type": "string",
                        "title": "Left Dataset Name",
                        "description": "Name of the main (left) dataset to merge.",
                    },
                    "right_dataset_name": {
                        "type": ["string", "null"],
                        "title": "Right Dataset Name",
                        "description": "Name of the right (lookup) dataset to merge.",
                    },
                    "right_dataset_path": {
                        "type": ["string", "null"],
                        "title": "Right Dataset Path",
                        "description": "File path to the right dataset if not using a named dataset.",
                    },
                    "left_key": {
                        "type": "string",
                        "title": "Left Key",
                        "description": "Key field in the left dataset for joining.",
                    },
                    "right_key": {
                        "type": ["string", "null"],
                        "title": "Right Key",
                        "description": "Key field in the right dataset for joining. Defaults to left_key if not set.",
                    },
                    "join_type": {
                        "type": "string",
                        "oneOf": [
                            {"const": "inner", "description": "Inner"},
                            {"const": "left", "description": "Left"},
                            {"const": "right", "description": "Right"},
                            {"const": "outer", "description": "Outer"},
                        ],
                        "title": "Join Type",
                        "description": "Type of join to perform: inner, left, right, or outer.",
                        "default": "left",
                    },
                    "relationship_type": {
                        "type": "string",
                        "oneOf": [
                            {"const": "auto", "description": "Auto-detect"},
                            {"const": "one-to-one", "description": "One-to-One"},
                            {"const": "one-to-many", "description": "One-to-Many"},
                        ],
                        "title": "Relationship Type",
                        "default": "auto",
                        "description": "Type of relationship between datasets: auto-detect, one-to-one, or one-to-many.",
                    },
                    "suffixes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "minItems": 2,
                        "maxItems": 2,
                        "default": ["_x", "_y"],
                        "title": "Suffixes",
                        "description": "Suffixes to apply to overlapping columns in the merged dataset.",
                    },
                },
                "required": [
                    "left_dataset_name",
                    "left_key",
                    "right_dataset_name",
                    "join_type",
                    "relationship_type",
                ],
            },
        ],
    }
