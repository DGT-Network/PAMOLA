"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Generalization Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of numeric generalization configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines binning, rounding, and range-based numeric anonymization strategies
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Strategy-based numeric generalization (binning, rounding, range)
- Conditional validation logic for strategy-specific parameters
- K-anonymity risk threshold integration
- Quasi-identifier support for contextual anonymization

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric generalization core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class NumericGeneralizationConfig(OperationConfig):
    """
    Core configuration schema for NumericGeneralization backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Numeric Generalization Operation Core Configuration",
        "description": "Core schema for numeric generalization operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field to generalize.",
                    },
                    "strategy": {
                        "type": "string",
                        "title": "Strategy",
                        "default": "binning",
                        "oneOf": [
                            {"const": "binning", "description": "Binning"},
                            {"const": "rounding", "description": "Rounding"},
                            {"const": "range", "description": "Range"},
                        ],
                        "description": (
                            "Defines how numerical values are generalized:\n"
                            "- 'binning': group numbers into discrete bins\n"
                            "- 'rounding': reduce precision to a fixed number of digits\n"
                            "- 'range': replace values by defined numeric ranges"
                        ),
                    },
                    "binning_method": {
                        "type": "string",
                        "title": "Binning Method",
                        "default": "equal_width",
                        "oneOf": [
                            {"const": "equal_width", "description": "Equal width"},
                            {
                                "const": "equal_frequency",
                                "description": "Equal frequency",
                            },
                            {"const": "quantile", "description": "Quantile-based"},
                        ],
                    },
                    "bin_count": {
                        "type": "integer",
                        "title": "Bin Count",
                        "default": 10,
                        "minimum": 2,
                        "description": "Number of bins to divide numeric values into (for 'binning' strategy).",
                    },
                    "precision": {
                        "type": "integer",
                        "title": "Precision",
                        "description": "Number of decimal places to retain when rounding numeric values.",
                    },
                    "range_limits": {
                        "type": ["array", "null"],
                        "title": "Range Limits",
                        "items": {
                            "type": "number",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "description": (
                            "Custom range intervals for numeric generalization.\n"
                            "Each range is defined as a two-element array [min, max]."
                        ),
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "title": "Quasi-Identifiers",
                        "description": (
                            "List of related fields used to determine quasi-identifiers "
                            "for risk-based anonymization."
                        ),
                        "items": {"type": "string"},
                    },
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name used as condition for applying the generalization.",
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "default": "in",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "description": "Comparison operator used in the condition.",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values of the condition field that trigger the generalization.",
                    },
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-Anonymity Risk Field",
                        "description": "Field name containing precomputed risk scores for k-anonymity.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Maximum acceptable risk value for anonymization.",
                        "default": 5.0,
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "default": "suppress",
                        "oneOf": [
                            {"const": "suppress", "description": "Suppress"},
                            {"const": "remove", "description": "Remove"},
                            {"const": "mean", "description": "Mean"},
                            {"const": "mode", "description": "Mode"},
                            {"const": "custom", "description": "Custom"},
                        ],
                        "description": "Action to apply to records exceeding the risk threshold.",
                    },
                },
                "required": ["field_name", "strategy"],
            },
            # Strategy-specific required fields
            {
                "if": {"properties": {"strategy": {"const": "binning"}}},
                "then": {"required": ["bin_count", "binning_method"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "rounding"}}},
                "then": {"required": ["precision"]},
            },
            {
                "if": {"properties": {"strategy": {"const": "range"}}},
                "then": {"required": ["range_limits"]},
            },
            # Conditional logic dependencies
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
            # K-anonymity risk threshold dependency
            {
                "if": {
                    "properties": {"ka_risk_field": {"type": "string", "minLength": 1}},
                    "required": ["ka_risk_field"],
                },
                "then": {"properties": {"risk_threshold": {"type": "number"}}},
            },
        ],
    }
