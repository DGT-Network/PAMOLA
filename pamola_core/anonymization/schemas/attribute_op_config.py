"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Suppression Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating attribute suppression operations in PAMOLA.CORE.
Supports parameters for field names, additional fields, suppression modes, and multi-field conditions.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of attribute suppression config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class AttributeSuppressionConfig(OperationConfig):
    """Configuration schema for AttributeSuppressionOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Attribute Suppression Operation Configuration",
        "description": "Configuration schema for attribute-level suppression operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common parameters
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string", "title": "Field Name", "description": "Primary field to apply suppression."},
                    "additional_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Additional Fields",
                        "description": "Other fields to include in suppression operation.",
                    },
                    "suppression_mode": {
                        "type": "string",
                        "enum": ["REMOVE"],
                        "title": "Suppression Mode",
                        "description": "Suppression strategy to apply (e.g., REMOVE)."
                    },
                    "save_suppressed_schema": {
                        "type": "boolean",
                        "title": "Save Suppressed Schema",
                        "description": "Whether to save the schema after suppression."
                    },
                    # Multi-field conditions
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "items": {"type": "object"},
                        "title": "Multi-field Conditions",
                        "description": "List of conditions for multi-field suppression."
                    },
                    "condition_logic": {
                        "type": "string",
                        "title": "Condition Logic",
                        "description": "Logic to combine multiple conditions (e.g., AND, OR)."
                    },
                    # Conditional processing parameters
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field to check for conditional suppression."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Condition Values",
                        "description": "Values that trigger conditional suppression."
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
                        "description": "Threshold for k-anonymity risk triggering suppression."
                    },
                },
                "required": ["field_name", "suppression_mode"],
            },
        ],
    }
