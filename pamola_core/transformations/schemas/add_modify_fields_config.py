"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add/Modify Fields Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating add/modify fields operations in PAMOLA.CORE.
- Supports field operations and lookup tables for transformation pipelines
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of add/modify fields config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class AddOrModifyFieldsOperationConfig(OperationConfig):
    """Configuration for AddOrModifyFieldsOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "field_operations": {
                        "type": "object",
                        "patternProperties": {
                            "^.+$": {
                                "type": "object",
                                "properties": {
                                    "operation_type": {
                                        "title": "Operation Type",
                                        "type": "string",
                                        "enum": [
                                            "add_constant",
                                            "add_from_lookup",
                                            "add_conditional",
                                            "modify_constant",
                                            "modify_from_lookup",
                                            "modify_conditional",
                                            "modify_expression"
                                        ]
                                    },
                                    "constant_value": {},
                                    "base_on_column": { "type": "string", "title": "Base On Column" },
                                    "lookup_table_name": { "type": "string", "title": "Lookup Table Name" },
                                    "condition": { "type": "string", "title": "Condition" },
                                    "value_if_true": { "type": "string", "title": "Value If True" },
                                    "value_if_false": { "type": "string", "title": "Value If False" },
                                    "expression": { "type": "string", "title": "Expression" },
                                    "expression_character": { "type": "string", "title": "Expression Character" }
                                },
                                "required": ["operation_type"],
                                "additionalProperties": True
                            }
                        },
                        "additionalProperties": False
                    },
                    "lookup_tables": {
                        "type": "object",
                        "patternProperties": {
                            "^.*$": {
                                "type": ["string", "object"],
                                "additionalProperties": True,
                                "title": "Lookup Table Path"
                            }
                        },
                        "additionalProperties": False
                    },
                },
            },
        ],
    }