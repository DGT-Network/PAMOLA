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

from pamola_core.common.enum.form_groups import GroupName
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
                        "x-component": "Select",
                        "description": "Name of the numeric field to generalize.",
                    },
                    "strategy": {
                        "type": "string",
                        "title": "Strategy",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "binning", "description": "Binning"},
                            {"const": "rounding", "description": "Rounding"},
                            {"const": "range", "description": "Range"},
                        ],
                        "default": "binning",
                        "description": (
                            "Defines how numerical values are generalized:\n"
                            "- 'binning': group numbers into discrete bins\n"
                            "- 'rounding': reduce precision to a fixed number of digits\n"
                            "- 'range': replace values by defined numeric ranges"
                        ),
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    # === Binning ===
                    "binning_method": {
                        "title": "Binning Method",
                        "type": "string",
                        "x-component": "Select",
                        "default": "equal_width",
                        "oneOf": [
                            {"const": "equal_width", "description": "Equal width"},
                            {
                                "const": "equal_frequency",
                                "description": "Equal frequency",
                            },
                            {"const": "quantile", "description": "Quantile-based"},
                        ],
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "binning"},
                        "x-required-on": {"strategy": "binning"},
                    },
                    "bin_count": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 10,
                        "title": "Bin Count",
                        "x-component": "NumberPicker",
                        "description": "Number of bins to divide numeric values into (for 'binning' strategy).",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "binning"},
                        "x-required-on": {"strategy": "binning"},
                    },
                    # === Rounding ===
                    "precision": {
                        "type": "integer",
                        "title": "Precision",
                        "x-component": "NumberPicker",
                        "description": "Number of decimal places to retain when rounding numeric values.",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "rounding"},
                        "x-required-on": {"strategy": "rounding"},
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
                            "type": "number",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "x-items-title": ["Min", "Max"],
                            "x-component": "NumberPicker",
                            "maxItems": 2,
                        },
                        "x-component": "ArrayItems",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "range"},
                        "x-required-on": {"strategy": "range"},
                    },
                    # === Contextual anonymization ===
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "title": "Quasi-Identifiers",
                        "description": (
                            "List of related fields used to determine quasi-identifiers "
                            "for risk-based anonymization."
                        ),
                        "items": {"type": "string"},
                        "x-component": "Select",
                    },
                    # === Conditional generalization ===
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "x-component": "Select",
                        "description": "Field name used as condition for applying the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": ["update_condition_field"],
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition.",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "default": "in",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"condition_field": "not_null"},
                        "x-custom-function": ["update_condition_operator"],
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "x-component": "Input",  # ArrayItems
                        "description": "Values of the condition field that trigger the generalization.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                        "x-custom-function": ["update_condition_values"],
                    },
                    # === K-Anonymity integration ===
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-Anonymity Risk Field",
                        "x-component": "Select",
                        "description": "Field name containing precomputed risk scores for k-anonymity.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "x-component": "NumberPicker",
                        "description": "Maximum acceptable risk value for anonymization.",
                        "default": 5.0,
                        "x-depend-on": {"ka_risk_field": "not_null"},
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "x-component": "Select",
                        "description": "Action to apply to records exceeding the risk threshold.",
                        "oneOf": [
                            {"const": "suppress", "description": "Suppress"},
                            {"const": "remove", "description": "Remove"},
                            {"const": "mean", "description": "Mean"},
                            {"const": "mode", "description": "Mode"},
                            {"const": "custom", "description": "Custom"},
                        ],
                        "default": "suppress",
                    },
                },
                "required": ["field_name", "strategy"],
            },
            # === Conditional logic for strategy-specific requirements ===
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
        ],
    }
