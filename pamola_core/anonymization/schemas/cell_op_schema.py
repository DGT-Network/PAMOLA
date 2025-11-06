"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Cell Suppression Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating cell suppression operations in PAMOLA.CORE.
Supports parameters for field names, suppression strategies, and control options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of cell suppression config file
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class CellSuppressionConfig(OperationConfig):
    """Configuration schema for CellSuppressionOperation with with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Cell Suppression Operation Configuration",
        "description": "Configuration schema for cell-level suppression operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common base fields
            {
                "type": "object",
                "properties": {
                    # Suppression-specific fields
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "x-component": "Select",
                        "description": "The column containing cells to suppress.",
                    },
                    "suppression_strategy": {
                        "type": "string",
                        "title": "Suppression Strategy",
                        "x-component": "Select",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
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
                        "x-component": "Input",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                        "description": "Replacement value when using the 'constant' strategy.",
                        "x-depend-on": {"suppression_strategy": "constant"},
                        "x-required-on": {"suppression_strategy": "constant"},
                    },
                    "group_by_field": {
                        "type": ["string", "null"],
                        "title": "Group By Field",
                        "x-component": "Input",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                        "description": "Column for group-based suppression (required for group_mean or group_mode).",
                        "x-depend-on": {
                            "suppression_strategy": ["group_mean", "group_mode"]
                        },
                        "x-required-on": {
                            "suppression_strategy": ["group_mean", "group_mode"]
                        },
                    },
                    "min_group_size": {
                        "type": "number",
                        "minimum": 1,
                        "default": 5,
                        "title": "Minimum Group Size",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                        "description": "Minimum group size for valid group-level suppression.",
                        "x-depend-on": {"group_by_field": "not_null"},
                    },
                    "suppress_if": {
                        "type": ["string", "null"],
                        "title": "Suppress If",
                        "x-component": "Select",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "oneOf": [
                            {"const": "outlier", "description": "Outlier"},
                            {"const": "rare", "description": "Rare"},
                            {"const": "null", "description": "Null"},
                        ],
                        "description": "Automatic suppression trigger. One of: outlier, rare, null. If set, this disables the custom Condition Field options.",
                    },
                    "outlier_method": {
                        "type": "string",
                        "title": "Outlier Method",
                        "x-component": "Select",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "oneOf": [
                            {"const": "iqr", "description": "IQR"},
                            {"const": "zscore", "description": "Z-Score"},
                        ],
                        "description": "Outlier detection method if suppress_if is 'outlier'.",
                        "x-depend-on": {"suppress_if": "outlier"},
                    },
                    "outlier_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1.5,
                        "title": "Outlier Threshold",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "description": "Threshold for outlier detection.",
                        "x-depend-on": {"suppress_if": "outlier"},
                    },
                    "rare_threshold": {
                        "type": "number",
                        "minimum": 1,
                        "default": 10,
                        "title": "Rare Threshold",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "description": "Frequency threshold for rare value detection.",
                        "x-depend-on": {"suppress_if": "rare"},
                    },
                    # Conditional processing
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "default": "in",
                        "x-component": "Input",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "description": "Field to check for conditional suppression. Active only if Suppression Trigger is not set.",
                        "x-depend-on": {"suppress_if": "null"},
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "description": "Operator for condition evaluation. Active only if Suppression Trigger is not set.",
                        "x-depend-on": {
                            "suppress_if": "null",
                            "condition_field": "not_null",
                        },
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "x-component": "Input",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "items": {"type": "string"},
                        "description": "Values that trigger conditional suppression. Active only if Suppression Trigger is not set.",
                        "x-depend-on": {
                            "suppress_if": "null",
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                    },
                },
                "required": ["field_name", "suppression_strategy"],
            },
            # === Conditional logic for strategy-specific requirements ===
            {
                "if": {"properties": {"suppression_strategy": {"const": "constant"}}},
                "then": {"required": ["suppression_value"]},
            },
            {
                "if": {
                    "properties": {
                        "suppression_strategy": {"enum": ["group_mean", "group_mode"]}
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
                    "properties": {"condition_operator": {"type": "string"}},
                    "required": ["condition_operator"],
                },
                "then": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_field"],
                },
            },
            {
                "if": {
                    "properties": {"condition_values": {"type": "array"}},
                    "required": ["condition_values"],
                },
                "then": {
                    "properties": {
                        "condition_operator": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_operator"],
                },
            },
            {
                "if": {
                    "anyOf": [
                        {"required": ["output_field_name"]},
                        {"required": ["column_prefix"]},
                    ]
                },
                "then": {
                    "properties": {"mode": {"const": "ENRICH"}},
                    "required": ["mode"],
                },
            },
        ],
    }
