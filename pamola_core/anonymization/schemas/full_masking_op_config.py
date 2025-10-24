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
        "title": "Full Masking Operation Configuration",
        "description": "Configuration schema for full masking operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to apply full masking."
                    },
                    "mask_char": {
                        "type": "string",
                        "default": "*",
                        "title": "Mask Character",
                        "description": "Character used for masking the field values."
                    },
                    "preserve_length": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Length",
                        "description": "Whether to preserve the original string length of masked values."
                    },
                    "fixed_length": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "title": "Fixed Length",
                        "description": "Fixed output length for all masked values. If None, uses input length."
                    },
                    "random_mask": {
                        "type": "boolean",
                        "default": False,
                        "title": "Random Mask",
                        "description": "Whether to use random characters from a pool instead of a fixed mask_char."
                    },
                    "mask_char_pool": {
                        "type": ["string", "null"],
                        "title": "Mask Character Pool",
                        "description": "Pool of characters to randomly sample from if random_mask is True."
                    },
                    "preserve_format": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Format",
                        "description": "Whether to preserve data format or structure (e.g., keep dashes or parentheses)."
                    },
                    "format_patterns": {
                        "type": ["object", "null"],
                        "title": "Format Patterns",
                        "description": "Custom regex patterns for identifying and preserving data formats."
                    },
                    "numeric_output": {
                        "type": "string",
                        "enum": ["string", "numeric", "preserve"],
                        "default": "string",
                        "title": "Numeric Output",
                        "description": "Defines the output type for numeric fields: string, numeric, or preserve."
                    },
                    "date_format": {
                        "type": ["string", "null"],
                        "title": "Date Format",
                        "description": "Date format string to use when masking datetime fields."
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field to check for conditional masking."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
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
                    # Output field name configuration
                    "output_field_name": {
                        "type": ["string", "null"],
                        "title": "Output Field Name",
                        "description": "Custom output field name (for ENRICH mode)."
                    },
                },
                "required": ["field_name", "mask_char"],
            },
        ],
    }
