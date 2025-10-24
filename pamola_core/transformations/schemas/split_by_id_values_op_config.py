"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split By ID Values Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating split by ID values operations in PAMOLA.CORE.
- Supports partitioning by ID, value groups, partition methods, and invalid value handling
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split by ID values config file
"""
from pamola_core.transformations.splitting.split_by_id_values_op import PartitionMethod
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class SplitByIDValuesOperationConfig(OperationConfig):
    """Configuration for SplitByIDValuesOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "id_field": {"type": "string"},
                    "value_groups": {"type": ["object", "null"]},
                    "number_of_partitions": {"type": "integer", "minimum": 0},
                    "partition_method": {
                        "type": "string",
                        "enum": [
                            PartitionMethod.EQUAL_SIZE.value,
                            PartitionMethod.RANDOM.value,
                            PartitionMethod.MODULO.value,
                        ],
                    },
                    "invalid_values": {"type": ["object", "null"]},
                },
            },
        ],
    }
