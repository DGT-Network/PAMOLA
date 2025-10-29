"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Generalization Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating numeric generalization parameters in PAMOLA.CORE.
- Supports binning, rounding, and range-based strategies for numeric anonymization
- Handles precision, custom ranges, conditional logic, and k-anonymity integration
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric generalization config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class NumericGeneralizationConfig(OperationConfig):
    """
    Configuration schema for NumericGeneralizationOperation.

    Extends BaseOperationConfig with numeric-specific anonymization and generalization parameters,
    including binning, rounding, and range-based strategies.
    """

    schema = {
        "type": "object",
        "title": "Numeric Generalization Configuration",
        "description": (
            "Configuration options for generalizing or anonymizing numerical fields. "
            "Supports binning, rounding, and range-based strategies."
        ),
        "allOf": [
            BaseOperationConfig.schema,  # merge base schema
            {
                "type": "object",
                "properties": {
                    # === Core numeric fields ===
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field to generalize."
                    },
                    "strategy": {
                        "type": "string",
                        "title": "Strategy",
                        "oneOf": [
                            {"const": "binning", "description": "Binning"},
                            {"const": "rounding", "description": "Rounding"},
                            {"const": "range", "description": "Range"}
                        ],
                        "description": (
                            "Defines how numerical values are generalized:\n"
                            "- 'binning': group numbers into discrete bins\n"
                            "- 'rounding': reduce precision to a fixed number of digits\n"
                            "- 'range': replace values by defined numeric ranges"
                        )
                    },

                    # === Binning ===
                    "bin_count": {
                        "type": "integer",
                        "minimum": 2,
                        "title": "Bin Count",
                        "description": "Number of bins to divide numeric values into (for 'binning' strategy)."
                    },
                    "binning_method": {
                        "title": "Binning Method",
                        "oneOf": [
                            {"const": "equal_width", "description": "Equal width"},
                            {"const": "equal_frequency", "description": "Equal frequency"},
                            {"const": "quantile", "description": "Quantile-based"}
                        ]
                    },

                    # === Rounding ===
                    "precision": {
                        "type": "integer",
                        "title": "Precision",
                        "description": "Number of decimal places to retain when rounding numeric values."
                    },

                    # === Range-based generalization ===
                    "range_limits": {
                        "type": ["array", "null"],
                        "title": "Range Limits",
                        "description": (
                            "Custom range intervals for numeric generalization.\n"
                            "Each range is defined as a two-element array [min, max]."
                        ),
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },

                    # === Contextual anonymization ===
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "title": "Quasi-Identifiers",
                        "description": (
                            "List of related fields used to determine quasi-identifiers "
                            "for risk-based anonymization."
                        ),
                        "items": {"type": "string"}
                    },

                    # === Conditional generalization ===
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name used as condition for applying the generalization."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values of the condition field that trigger the generalization.",
                        "items": {"type": "string"}
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition (e.g., 'in', 'not_in', 'gt', 'lt', 'eq', 'range').",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"}
                        ]
                    },

                    # === K-Anonymity integration ===
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-Anonymity Risk Field",
                        "description": "Field name containing precomputed risk scores for k-anonymity."
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Maximum acceptable risk value for anonymization."
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Action to apply to records exceeding the risk threshold."
                    },

                    # === Output ===
                    "output_field_name": {
                        "type": ["string", "null"],
                        "title": "Output Field Name",
                        "description": "Optional custom name for the generated or modified output field."
                    },
                },
                "required": ["field_name", "strategy"],

                # Add UI dependency logic for condition_field
                "dependencies": {
                    "condition_field": {
                        "oneOf": [
                            {
                                "properties": {
                                    "condition_field": { "type": "null" }
                                }
                            },
                            {
                                "properties": {
                                    "condition_field": {
                                        "type": "string",
                                        "minLength": 1
                                    },
                                    "condition_values": {
                                        "type": ["array", "null"],
                                        "title": "Condition Values",
                                        "items": {"type": "string"},
                                        "description": "Displayed when condition_field has a value."
                                    }
                                }
                            }
                        ]
                    },
                    "condition_values": {
                        "oneOf": [
                            {
                                "properties": {
                                    "condition_values": {"type": "null"}
                                }
                            },
                            {
                                "properties": {
                                    "condition_values": {
                                        "type": "array",
                                        "minItems": 1
                                    },
                                    "condition_operator": {
                                        "type": "string",
                                        "title": "Condition Operator",
                                        "description": "Comparison operator used in the condition (e.g., 'in', 'not_in', 'gt', 'lt', 'eq', 'range').",
                                        "oneOf": [
                                            {"const": "in", "description": "In"},
                                            {"const": "not_in", "description": "Not in"},
                                            {"const": "gt", "description": "Greater than"},
                                            {"const": "lt", "description": "Less than"},
                                            {"const": "eq", "description": "Equal to"},
                                            {"const": "range", "description": "Range"}
                                        ]
                                    }
                                }
                            }
                        ]
                    },
                },
            },
            # === Conditional logic for strategy-specific requirements ===
            {
                "if": {"properties": {"strategy": {"const": "binning"}}},
                "then": {"required": ["bin_count", "binning_method"]}
            },
            {
                "if": {"properties": {"strategy": {"const": "rounding"}}},
                "then": {"required": ["precision"]}
            },
            {
                "if": {"properties": {"strategy": {"const": "range"}}},
                "then": {"required": ["range_limits"]}
            }
        ],
    }
