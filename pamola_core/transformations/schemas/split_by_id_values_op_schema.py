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

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.transformations.splitting.split_by_id_values_op import PartitionMethod
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class SplitByIDValuesOperationConfig(OperationConfig):
    """Configuration for SplitByIDValuesOperation with BaseOperationConfig merged."""

    schema = {
        "title": "SplitByIDValuesOperationConfig",
        "description": "Schema for splitting a dataset into multiple subsets based on ID field values or partitioning strategies. Supports explicit value groups, equal-size/random/modulo partitioning, and invalid value handling.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "SplitByIDValuesOperationConfig Properties",
                "description": "Properties for configuring dataset splitting by ID, value groups, partition method, and invalid value handling.",
                "properties": {
                    "id_field": {
                        "type": "string",
                        "title": "ID Field",
                        "x-component": "Select",
                        "description": "Name of the field used to identify records for splitting. Required for all partitioning strategies.",
                        "x-group": GroupName.ID_FIELD,
                    },
                    "partition_method": {
                        "type": "string",
                        "x-component": "Select",
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
                        "x-group": GroupName.PARTITION_SETTINGS,
                    },
                    "number_of_partitions": {
                        "type": "integer",
                        "minimum": 1,
                        "title": "Number of Partitions",
                        "x-component": "NumberPicker",
                        "description": "Number of partitions to create when using automatic partitioning (equal size, random, or modulo). Ignored if value_groups is provided.",
                        "x-group": GroupName.PARTITION_SETTINGS,
                        "x-depend-on": {"partition_method": "not_null"},
                    },
                    "value_groups": {
                        "type": ["object", "null"],
                        "title": "Value Groups",
                        "x-component": "Input",
                        "description": "Dictionary mapping group names to lists of ID values for explicit group-based splitting. If null, automatic partitioning is used.",
                        "x-group": GroupName.VALUE_GROUPS,
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
