"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Full Masking Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating full masking parameters in PAMOLA.CORE.
- Supports character masking, length preservation, randomization, format-aware masking, and conditional masking
- Integrates with k-anonymity risk assessment and output field configuration
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of full masking config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FullMaskingConfig(OperationConfig):
    """Configuration for FullMaskingOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "mask_char": {"type": "string", "default": "*"},
                    "preserve_length": {"type": "boolean", "default": True},
                    "fixed_length": {"type": ["integer", "null"], "minimum": 0},
                    "random_mask": {"type": "boolean", "default": False},
                    "mask_char_pool": {"type": ["string", "null"]},
                    "preserve_format": {"type": "boolean", "default": False},
                    "format_patterns": {"type": ["object", "null"]},
                    "numeric_output": {
                        "type": "string",
                        "enum": ["string", "numeric", "preserve"],
                        "default": "string",
                    },
                    "date_format": {"type": ["string", "null"]},
                    # Conditional processing parameters
                    "condition_field": {"type": ["string", "null"]},
                    "condition_values": {"type": ["array", "null"]},
                    "condition_operator": {"type": "string"},
                    # K-anonymity integration
                    "ka_risk_field": {"type": ["string", "null"]},
                    "risk_threshold": {"type": "number"},
                    "vulnerable_record_strategy": {"type": "string"},
                    # Output field name configuration
                    "output_field_name": {"type": ["string", "null"]},
                },
                "required": ["field_name", "mask_char"],
            },
        ],
    }
