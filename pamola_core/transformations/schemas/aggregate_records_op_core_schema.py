"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Aggregate Records Core Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of aggregate records configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines record aggregation parameters including group by and aggregation functions
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Group by field configuration with minimum field requirement
- Standard aggregation function mapping
- Custom aggregation expression support
- SQL-like group by and aggregate functionality

Changelog:
1.0.0 - 2025-01-15 - Initial creation of aggregate records core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class AggregateRecordsOperationConfig(OperationConfig):
    """
    Core configuration schema for AggregateRecordsOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Aggregate Records Operation Core Configuration",
        "description": "Core schema for Aggregate Records operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "group_by_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "title": "Group By Fields",
                        "description": "List of fields to group records by (like SQL GROUP BY). Must have at least one field.",
                    },
                    "aggregations": {
                        "type": "object",
                        "title": "Aggregations",
                        "description": "Dictionary mapping field names to a list of aggregation functions (e.g., sum, mean, count).",
                    },
                    "custom_aggregations": {
                        "type": "object",
                        "title": "Custom Aggregations",
                        "description": "Dictionary mapping field names to custom aggregation function names or expressions.",
                    },
                },
                "required": ["group_by_fields", "aggregations"],
            },
        ],
    }
