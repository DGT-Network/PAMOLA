"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Aggregate Records Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating aggregate records operations in PAMOLA.CORE.
- Supports group by fields, standard and custom aggregations for transformation pipelines
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of aggregate records config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class AggregateRecordsOperationConfig(OperationConfig):
    """Configuration for AggregateRecordsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "group_by_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "aggregations": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "custom_aggregations": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["group_by_fields", "aggregations"],
            },
        ],
    }