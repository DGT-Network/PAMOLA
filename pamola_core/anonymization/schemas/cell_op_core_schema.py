"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Cell Suppression Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of cell suppression configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for cell-level suppression operations
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multiple suppression strategies (null, mean, median, mode, constant, group-based)
- Automatic suppression triggers (outlier, rare, null detection)
- Group-based suppression with minimum group size
- Outlier detection methods (IQR, Z-score)
- Conditional suppression based on field values

Changelog:
1.0.0 - 2025-11-18 - Initial creation of cell suppression core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CellSuppressionConfig(OperationConfig):
    """
    Core configuration schema for CellSuppression backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Cell Suppression Operation Core Configuration",
        "description": "Core schema for cell-level suppression operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "The column containing cells to suppress.",
                    },
                    "suppression_strategy": {
                        "type": "string",
                        "default": "null",
                        "title": "Suppression Strategy",
                        "oneOf": [
                            {"const": "null", "description": "Null"},
                            {"const": "mean", "description": "Mean"},
                            {"const": "median", "description": "Median"},
                            {"const": "mode", "description": "Mode"},
                            {"const": "constant", "description": "Constant"},
                            {"const": "group_mean", "description": "Group Mean"},
                            {"const": "group_mode", "description": "Group Mode"},
                        ],
                        "description": "Suppression method to apply. Supported: null, mean, median, mode, constant, group_mean, group_mode.",
                    },
                    "suppression_value": {
                        "type": ["string", "number", "null"],
                        "title": "Suppression Value",
                        "description": "Replacement value when using the 'constant' strategy.",
                    },
                    "group_by_field": {
                        "type": ["string", "null"],
                        "title": "Group By Field",
                        "description": "Column for group-based suppression (required for group_mean or group_mode).",
                    },
                    "min_group_size": {
                        "type": "number",
                        "minimum": 1,
                        "default": 5,
                        "title": "Minimum Group Size",
                        "description": "Minimum group size for valid group-level suppression.",
                    },
                    "suppress_if": {
                        "type": ["string", "null"],
                        "title": "Suppress If",
                        "oneOf": [
                            {"type": "null"},
                            {"const": "outlier", "description": "Outlier"},
                            {"const": "rare", "description": "Rare"},
                            {"const": "null", "description": "Null"},
                        ],
                        "description": "Automatic suppression trigger. One of: outlier, rare, null. If set, this disables the custom Condition Field options.",
                    },
                    "outlier_method": {
                        "type": "string",
                        "default": "iqr",
                        "title": "Outlier Method",
                        "oneOf": [
                            {"const": "iqr", "description": "IQR"},
                            {"const": "zscore", "description": "Z-Score"},
                        ],
                        "description": "Outlier detection method if suppress_if is 'outlier'.",
                    },
                    "outlier_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1.5,
                        "title": "Outlier Threshold",
                        "description": "Threshold for outlier detection.",
                    },
                    "rare_threshold": {
                        "type": "number",
                        "minimum": 1,
                        "default": 10,
                        "title": "Rare Threshold",
                        "description": "Frequency threshold for rare value detection.",
                    },
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name for conditional processing.",
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition.",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "default": "in",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values for conditional processing.",
                    },
                },
                "required": ["field_name", "suppression_strategy"],
            },
            {
                "if": {"properties": {"suppression_strategy": {"const": "constant"}}},
                "then": {"required": ["suppression_value"]},
            },
            {
                "if": {
                    "properties": {
                        "suppression_strategy": {
                            "oneOf": [{"const": "group_mean"}, {"const": "group_mode"}]
                        }
                    }
                },
                "then": {"required": ["group_by_field"]},
            },
            {
                "if": {"properties": {"suppress_if": {"const": "outlier"}}},
                "then": {"required": ["outlier_method", "outlier_threshold"]},
            },
            {
                "if": {"properties": {"suppress_if": {"const": "rare"}}},
                "then": {"required": ["rare_threshold"]},
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_field"],
                },
                "then": {"properties": {"condition_operator": {"type": "string"}}},
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1},
                        "condition_operator": {"type": "string", "minLength": 1},
                    },
                    "required": ["condition_field", "condition_operator"],
                },
                "then": {"properties": {"condition_values": {"type": "array"}}},
            },
        ],
    }
