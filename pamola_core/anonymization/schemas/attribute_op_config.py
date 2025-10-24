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
        "allOf": [
            BaseOperationConfig.schema,  # merge common parameters
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "additional_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "suppression_mode": {"type": "string", "enum": ["REMOVE"]},
                    "save_suppressed_schema": {"type": "boolean"},
                    # Multi-field conditions
                    "multi_conditions": {"type": ["array", "null"]},
                    "condition_logic": {"type": "string"},
                    # Conditional processing parameters
                    "condition_field": {"type": ["string", "null"]},
                    "condition_values": {"type": ["array", "null"]},
                    "condition_operator": {"type": "string"},
                    # K-anonymity integration
                    "ka_risk_field": {"type": ["string", "null"]},
                    "risk_threshold": {"type": "number"},
                },
                "required": ["field_name", "suppression_mode"],
            },
        ],
    }
