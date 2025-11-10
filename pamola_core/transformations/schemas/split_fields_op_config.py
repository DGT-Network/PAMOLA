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
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class SplitFieldsOperationConfig(OperationConfig):
    """Configuration for SplitFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "title": "SplitFieldsOperationConfig",
        "description": "Schema for splitting a dataset into multiple subsets based on field groups, with optional inclusion of an ID field. Supports flexible field grouping for privacy and modular data processing.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "SplitFieldsOperationConfig Properties",
                "description": "Properties for configuring field-based splitting, including ID field, field groups, and ID inclusion flag.",
                "properties": {
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "x-component": "Select",
                        "description": "Name of the field used as a unique identifier to be included in all splits. If null, no ID field is included.",
                        "x-group": GroupName.INPUT_SETTINGS,
                    },
                    "include_id_field": {
                        "type": "boolean",
                        "default": True,
                        "title": "Include ID Field",
                        "x-component": "Checkbox",
                        "description": "Whether to include the ID field in all output splits. Default is true.",
                        "x-group": GroupName.INPUT_SETTINGS,
                        "x-depend-on": {"id_field": "not_null"},
                    },
                     "field_groups": {
                        "type": ["object", "null"],
                        "title": "Field Groups",
                        "x-component": "ArrayItems",
                        "description": "Dictionary mapping group names to lists of field names for each split. If null, no field grouping is applied.",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "x-group": GroupName.FIELD_GROUPS_CONFIGURATION,
                    },
                },
                "required": ["field_groups"],
            },
        ],
    }