"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Record Suppression Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating record suppression operations in PAMOLA.CORE.
Supports parameters for field names, suppression modes, and control options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of record suppression config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class RecordSuppressionConfig(OperationConfig):
    """Configuration for RecordSuppressionOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # Core parameters
                    "field_name": {"type": "string"},
                    "suppression_mode": {
                        "type": "string",
                        "enum": ["REMOVE"],
                        "default": "REMOVE",
                    },
                    # Suppression control parameters
                    "suppression_condition": {
                        "type": "string",
                        "enum": ["null", "value", "range", "risk", "custom"],
                        "default": "null",
                    },
                    "suppression_values": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                    "suppression_range": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    # Output control
                    "save_suppressed_records": {
                        "type": "boolean",
                        "default": False,
                    },
                    "suppression_reason_field": {
                        "type": "string",
                        "default": "_suppression_reason",
                    },
                    # Multi-field conditions
                    "multi_conditions": {"type": ["array", "null"]},
                    "condition_logic": {"type": "string"},
                    # K-anonymity integration
                    "ka_risk_field": {"type": ["string", "null"]},
                    "risk_threshold": {"type": "number"},
                },
                "required": ["field_name", "suppression_condition"],
            },
        ],
    }

