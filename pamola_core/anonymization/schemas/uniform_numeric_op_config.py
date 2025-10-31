"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating uniform numeric noise operations in PAMOLA.CORE.
Supports parameters for field names, noise ranges, and noise types.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of uniform numeric noise config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class UniformNumericNoiseConfig(OperationConfig):
    """Configuration for UniformNumericNoiseOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Uniform Numeric Noise Operation Configuration",
        "description": "Configuration schema for uniform numeric noise operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # ==== Noise Parameters ====
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the numeric field to apply noise to."
                    },
                    "noise_range": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        ],
                        "title": "Noise Range",
                        "description": "Range of uniform noise to add. Use a single number for symmetric range (±value), or a two-element array [min, max] for asymmetric range."
                    },
                    "noise_type": {
                        "type": "string",
                        "enum": ["additive", "multiplicative"],
                        "default": "additive",
                        "title": "Noise Type",
                        "description": "Type of noise: 'additive' (add noise) or 'multiplicative' (scale by noise)."
                    },
                    # ==== Bounds and Constraints ====
                    "output_min": {
                        "type": ["number", "null"],
                        "title": "Output Minimum",
                        "description": "Minimum allowed value after noise is applied."
                    },
                    "output_max": {
                        "type": ["number", "null"],
                        "title": "Output Maximum",
                        "description": "Maximum allowed value after noise is applied."
                    },
                    "preserve_zero": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Zero",
                        "description": "If True, zero values will not be changed by noise."
                    },
                    # ==== Integer Handling ====
                    "round_to_integer": {
                        "type": ["boolean", "null"],
                        "title": "Round to Integer",
                        "description": "If True, round the result to the nearest integer."
                    },
                    # ==== Statistical Parameters ====
                    "scale_by_std": {
                        "type": "boolean",
                        "default": False,
                        "title": "Scale by Std",
                        "description": "If True, scale noise by the standard deviation of the field."
                    },
                    "scale_factor": {
                        "type": "number",
                        "minimum": 0,
                        "default": 1.0,
                        "title": "Scale Factor",
                        "description": "Multiplier for the noise magnitude."
                    },
                    # ==== Randomization ====
                    "random_seed": {
                        "type": ["integer", "null"],
                        "title": "Random Seed",
                        "description": "Seed for reproducible random noise (ignored if use_secure_random is True)."
                    },
                    "use_secure_random": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use Secure Random",
                        "description": "If True, use a cryptographically secure random generator."
                    },
                    # Multi-field conditions
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "items": {"type": "object"},
                        "title": "Multi-Conditions",
                        "description": "List of multi-field conditions for custom noise application logic."
                    },
                    "condition_logic": {
                        "type": "string",
                        "title": "Condition Logic",
                        "description": "Logical expression for combining multi-field conditions (e.g., 'AND', 'OR')."
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field to check for conditional noise application."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Condition Values",
                        "description": "Values that trigger conditional noise application."
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Operator for condition evaluation (e.g., '=', '>', '<', 'in')."
                    },
                    # K-anonymity integration
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field containing k-anonymity risk scores for suppression based on risk."
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering noise application."
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records."
                    },
                },
                "required": ["field_name", "noise_range"],
            },
        ],
    }
