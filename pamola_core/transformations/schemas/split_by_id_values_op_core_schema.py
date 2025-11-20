"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split By ID Values Core Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of split by ID values configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines dataset partitioning parameters including ID field, partition methods, and value groups
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- ID field specification for record identification
- Partition method selection (equal_size, random, modulo)
- Number of partitions configuration
- Value groups for explicit partition assignment
- Invalid value handling

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split by ID values core schema
"""

from pamola_core.transformations.splitting.split_by_id_values_op import PartitionMethod
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class SplitByIDValuesOperationConfig(OperationConfig):
    """
    Core configuration schema for SplitByIDValuesOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Split By ID Values Operation Core Configuration",
        "description": "Core schema for Split By ID Values operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "id_field": {
                        "type": "string",
                        "title": "ID Field",
                        "description": "Name of the field used to identify records for splitting. Required for all partitioning strategies.",
                    },
                    "partition_method": {
                        "type": "string",
                        "oneOf": [
                            {
                                "const": PartitionMethod.EQUAL_SIZE.value,
                                "description": "Equal Size",
                            },
                            {
                                "const": PartitionMethod.RANDOM.value,
                                "description": "Random",
                            },
                            {
                                "const": PartitionMethod.MODULO.value,
                                "description": "Modulo",
                            },
                        ],
                        "title": "Partition Method",
                        "description": "Partitioning strategy to use when value_groups is not provided. Options: 'equal_size', 'random', or 'modulo'.",
                    },
                    "number_of_partitions": {
                        "type": "integer",
                        "minimum": 1,
                        "title": "Number of Partitions",
                        "description": "Number of partitions to create when using automatic partitioning (equal size, random, or modulo). Ignored if value_groups is provided.",
                    },
                    "value_groups": {
                        "type": ["object", "null"],
                        "title": "Value Groups",
                        "description": "Dictionary mapping group names to lists of ID values for explicit group-based splitting. If null, automatic partitioning is used.",
                    },
                    "invalid_values": {
                        "type": ["object", "null"],
                        "title": "Invalid Values",
                        "description": "Dictionary of invalid or excluded ID values to ignore during splitting. Optional.",
                    },
                },
                "required": ["id_field"],
            },
        ],
    }
