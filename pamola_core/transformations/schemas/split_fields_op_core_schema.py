"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split Fields Core Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of split fields configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines field-based splitting parameters including ID field and field groups
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- ID field specification for unique identifier inclusion
- Field groups configuration for organizing splits
- Optional ID field inclusion control
- Flexible field grouping for privacy preservation

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split fields core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class SplitFieldsOperationConfig(OperationConfig):
    """
    Core configuration schema for SplitFieldsOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Split Fields Operation Core Configuration",
        "description": "Core schema for Split Fields operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "description": "Name of the field used as a unique identifier to be included in all splits. If null, no ID field is included.",
                    },
                    "include_id_field": {
                        "type": "boolean",
                        "default": True,
                        "title": "Include ID Field",
                        "description": "Whether to include the ID field in all output splits. Default is true.",
                    },
                    "field_groups": {
                        "type": ["object", "null"],
                        "title": "Field Groups",
                        "description": "Dictionary mapping group names to lists of field names for each split. If null, no field grouping is applied.",
                    },
                },
                "required": ["id_field", "field_groups"],
            },
        ],
    }
