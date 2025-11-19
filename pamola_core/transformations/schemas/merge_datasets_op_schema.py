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

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
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
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.JOIN_KEYS,
                    },
                    "right_key": {
                        "type": ["string", "null"],
                        "title": "Right Key",
                        "description": "Key field in the right dataset for joining. Defaults to left_key if not set.",
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.JOIN_KEYS,
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
                        "x-component": "Select",
                        "default": "left",
                        "x-group": GroupName.INPUT_DATASETS,
                    },
                    "relationship_type": {
                        "type": "string",
                        "oneOf": [
                            {"const": "auto", "description": "Auto-detect"},
                            {"const": "one-to-one", "description": "One-to-One"},
                            {"const": "one-to-many", "description": "One-to-Many"},
                        ],
                        "title": "Relationship Type",
                        "x-component": "Select",
                        "default": "auto",
                        "x-group": GroupName.INPUT_DATASETS,
                        "description": "Type of relationship between datasets: auto-detect, one-to-one, or one-to-many.",
                    },
                    "suffixes": {
                        "type": "array",
                        "x-component": "ArrayItems",
                        "items": {
                            "type": "string",
                            "x-component": "Input",
                            "x-items-title": [
                                "Left Column Suffix",
                                "Right Column Suffix",
                            ],
                            "x-item-params": ["left", "right"],
                        },
                        "minItems": 2,
                        "maxItems": 2,
                        "default": ["_x", "_y"],
                        "title": "Suffixes",
                        "x-group": GroupName.SUFFIXES,
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
