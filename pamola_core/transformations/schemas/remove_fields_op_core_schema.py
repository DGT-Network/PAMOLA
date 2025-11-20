"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Remove Fields Core Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of remove fields configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines field removal parameters for explicit and pattern-based removal
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Explicit field list removal configuration
- Pattern-based removal using regex
- Support for privacy and data minimization workflows
- Flexible field removal strategies

Changelog:
1.0.0 - 2025-01-15 - Initial creation of remove fields core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class RemoveFieldsOperationConfig(OperationConfig):
    """
    Core configuration schema for RemoveFieldsOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Remove Fields Operation Core Configuration",
        "description": "Core schema for Remove Fields operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": ["string", "null"],
                        "title": "Pattern",
                        "description": "Regex pattern to match field names for removal. If null, no pattern-based removal is performed.",
                    },
                    "fields_to_remove": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Fields to Remove",
                        "description": "List of field names to remove from the dataset. If null, no explicit fields are removed.",
                    },
                },
            },
        ],
    }
