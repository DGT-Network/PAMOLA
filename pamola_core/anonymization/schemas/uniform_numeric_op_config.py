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
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # ==== Noise Parameters ====
                    "field_name": {"type": "string"},
                    "noise_range": {
                        "oneOf": [
                            {"type": "number"},
                            {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        ]
                    },
                    "noise_type": {
                        "type": "string",
                        "enum": ["additive", "multiplicative"],
                        "default": "additive",
                    },
                    # ==== Bounds and Constraints ====
                    "output_min": {"type": ["number", "null"]},
                    "output_max": {"type": ["number", "null"]},
                    "preserve_zero": {"type": "boolean", "default": False},
                    # ==== Integer Handling ====
                    "round_to_integer": {"type": ["boolean", "null"]},
                    # ==== Statistical Parameters ====
                    "scale_by_std": {"type": "boolean", "default": False},
                    "scale_factor": {"type": "number", "minimum": 0, "default": 1.0},
                    # ==== Randomization ====
                    "random_seed": {"type": ["integer", "null"]},
                    "use_secure_random": {"type": "boolean", "default": True},
                    # Multi-field conditions
                    "multi_conditions": {"type": ["array", "null"], "items": {"type": "object"}},
                    "condition_logic": {"type": "string"},
                    # Conditional processing parameters
                    "condition_field": {"type": ["string", "null"]},
                    "condition_values": {"type": ["array", "null"], "items": {"type": "string"}},
                    "condition_operator": {"type": "string"},
                    # K-anonymity integration
                    "ka_risk_field": {"type": ["string", "null"]},
                    "risk_threshold": {"type": "number"},
                    "vulnerable_record_strategy": {"type": "string"},
                    # Output field name configuration
                    "output_field_name": {"type": ["string", "null"]},
                },
                "required": ["field_name", "noise_range"],
            },
        ],
    }
