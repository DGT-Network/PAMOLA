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
        "title": "AggregateRecordsOperationConfig",
        "description": "Schema for configuring record aggregation operations in PAMOLA.CORE. Supports group by fields, standard and custom aggregations.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "AggregateRecordsOperationConfig Properties",
                "description": "Properties for grouping and aggregating records in transformation pipelines.",
                "properties": {
                    "group_by_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "title": "Group By Fields",
                        "description": "List of fields to group records by (like SQL GROUP BY). Must have at least one field."
                    },
                    "aggregations": {
                        "type": "object",
                        "title": "Aggregations",
                        "description": "Dictionary mapping field names to a list of aggregation functions (e.g., sum, mean, count).",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"},
                            "title": "Aggregation Functions",
                            "description": "List of aggregation functions to apply to the field."
                        },
                    },
                    "custom_aggregations": {
                        "type": "object",
                        "title": "Custom Aggregations",
                        "description": "Dictionary mapping field names to custom aggregation function names or expressions.",
                        "additionalProperties": {"type": "string", "title": "Custom Aggregation Function", "description": "Custom aggregation function or expression for the field."},
                    },
                },
                "required": ["group_by_fields", "aggregations"],
            },
        ],
    }