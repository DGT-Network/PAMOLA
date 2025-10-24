"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating partial masking parameters in PAMOLA.CORE.
- Supports fixed, pattern-based, random, and word-based masking strategies
- Handles prefix/suffix/unmasked positions, pattern preservation, and conditional masking
- Integrates with k-anonymity risk assessment and output field configuration
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of partial masking config file
"""

from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class PartialMaskingConfig(OperationConfig):
    """Configuration schema for PartialMaskingOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # ==== Partial Masking Specific Fields ====
                    "field_name": {"type": "string"},
                    "mask_char": {"type": "string", "default": "*"},
                    "mask_strategy": {
                        "type": "string",
                        "enum": [
                            MaskStrategyEnum.FIXED.value,
                            MaskStrategyEnum.PATTERN.value,
                            MaskStrategyEnum.RANDOM.value,
                            MaskStrategyEnum.WORDS.value,
                        ],
                        "default": MaskStrategyEnum.FIXED.value,
                    },
                    "mask_percentage": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "unmasked_prefix": {"type": "integer", "minimum": 0, "default": 0},
                    "unmasked_suffix": {"type": "integer", "minimum": 0, "default": 0},
                    "unmasked_positions": {
                        "type": ["array", "null"],
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "pattern_type": {"type": ["string", "null"]},
                    "mask_pattern": {"type": ["string", "null"]},
                    "preserve_pattern": {"type": ["string", "null"]},
                    "preserve_separators": {"type": "boolean", "default": True},
                    "preserve_word_boundaries": {"type": "boolean", "default": False},
                    "case_sensitive": {"type": "boolean", "default": True},
                    "random_mask": {"type": "boolean", "default": False},
                    "mask_char_pool": {"type": ["string", "null"]},
                    "preset_type": {"type": ["string", "null"]},
                    "preset_name": {"type": ["string", "null"]},
                    "consistency_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
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
                "required": ["field_name"],
            },
        ],
    }