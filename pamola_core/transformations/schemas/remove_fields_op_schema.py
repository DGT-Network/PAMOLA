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

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class RemoveFieldsOperationConfig(OperationConfig):
    """Configuration for RemoveFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "title": "RemoveFieldsOperationConfig",
        "description": "Schema for removing one or more fields from a dataset, supporting both explicit field lists and regex pattern-based removal. Used for privacy and data minimization in transformation pipelines.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "RemoveFieldsOperationConfig Properties",
                "description": "Properties for configuring field removal, including explicit field names and regex pattern.",
                "properties": {
                    "pattern": {
                        "type": ["string", "null"],
                        "title": "Pattern",
                        "x-component": "Input",
                        "x-group": GroupName.FIELD_REMOVAL,
                        "description": "Regex pattern to match field names for removal. If null, no pattern-based removal is performed.",
                    },
                    "fields_to_remove": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Fields to Remove",
                        "x-component": "Select",
                        "x-group": GroupName.FIELD_REMOVAL,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "description": "List of field names to remove from the dataset. If null, no explicit fields are removed.",
                    },
                },
            },
        ],
    }
