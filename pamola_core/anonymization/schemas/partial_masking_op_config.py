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
        "title": "Partial Masking Operation Configuration",
        "description": "Configuration schema for partial masking operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # ==== Partial Masking Specific Fields ====
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to apply partial masking."
                    },
                    "mask_char": {
                        "type": "string",
                        "default": "*",
                        "title": "Mask Character",
                        "description": "Character used to mask sensitive content."
                    },
                    "mask_strategy": {
                        "type": "string",
                        "enum": [
                            MaskStrategyEnum.FIXED.value,
                            MaskStrategyEnum.PATTERN.value,
                            MaskStrategyEnum.RANDOM.value,
                            MaskStrategyEnum.WORDS.value,
                        ],
                        "default": MaskStrategyEnum.FIXED.value,
                        "title": "Masking Strategy",
                        "description": "Strategy for masking: fixed, pattern, random, or words."
                    },
                    "mask_percentage": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 100,
                        "title": "Mask Percentage",
                        "description": "Percentage of characters to mask randomly (for random strategy)."
                    },
                    "unmasked_prefix": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "title": "Unmasked Prefix",
                        "description": "Number of characters at the start to remain visible."
                    },
                    "unmasked_suffix": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "title": "Unmasked Suffix",
                        "description": "Number of characters at the end to remain visible."
                    },
                    "unmasked_positions": {
                        "type": ["array", "null"],
                        "items": {"type": "integer", "minimum": 0},
                        "title": "Unmasked Positions",
                        "description": "Specific index positions to remain unmasked."
                    },
                    "pattern_type": {
                        "type": ["string", "null"],
                        "title": "Pattern Type",
                        "description": "Predefined pattern type (e.g., email, phone) for pattern-based masking."
                    },
                    "mask_pattern": {
                        "type": ["string", "null"],
                        "title": "Mask Pattern",
                        "description": "Custom regex pattern for masking (pattern strategy)."
                    },
                    "preserve_pattern": {
                        "type": ["string", "null"],
                        "title": "Preserve Pattern",
                        "description": "Regex pattern to preserve (mask all except matches)."
                    },
                    "preserve_separators": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Separators",
                        "description": "Whether to keep separators (e.g., '-', '_', '.') unchanged."
                    },
                    "preserve_word_boundaries": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Word Boundaries",
                        "description": "Whether to avoid masking across word boundaries."
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": True,
                        "title": "Case Sensitive",
                        "description": "Whether pattern matching is case-sensitive."
                    },
                    "random_mask": {
                        "type": "boolean",
                        "default": False,
                        "title": "Random Mask",
                        "description": "Use random characters from a pool instead of a fixed mask_char."
                    },
                    "mask_char_pool": {
                        "type": ["string", "null"],
                        "title": "Mask Character Pool",
                        "description": "Pool of characters to randomly sample from if random_mask is True."
                    },
                    "preset_type": {
                        "type": ["string", "null"],
                        "title": "Preset Type",
                        "description": "Preset category for reusable masking templates."
                    },
                    "preset_name": {
                        "type": ["string", "null"],
                        "title": "Preset Name",
                        "description": "Name of the specific preset configuration to apply."
                    },
                    "consistency_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Consistency Fields",
                        "description": "Other fields to mask consistently with the main field."
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field to check for conditional masking."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Condition Values",
                        "description": "Values that trigger conditional masking."
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Operator for condition evaluation (e.g., EQUALS, IN)."
                    },
                    # K-anonymity integration
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field used for k-anonymity risk assessment."
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering masking."
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }