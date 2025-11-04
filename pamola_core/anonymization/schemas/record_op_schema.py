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
        "title": "Record Suppression Operation Configuration",
        "description": "Configuration schema for record suppression operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common fields
            {
                "type": "object",
                "properties": {
                    # Core parameters
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to evaluate for suppression.",
                    },
                    "suppression_mode": {
                        "type": "string",
                        "enum": ["REMOVE"],
                        "default": "REMOVE",
                        "title": "Suppression Mode",
                        "description": "Suppression mode. Only 'REMOVE' (remove entire record) is supported.",
                    },
                    # Suppression control parameters
                    "suppression_condition": {
                        "type": "string",
                        "enum": ["null", "value", "range", "risk", "custom"],
                        "default": "null",
                        "title": "Suppression Condition",
                        "description": "Condition for suppressing records: 'null', 'value', 'range', 'risk', or 'custom'.",
                    },
                    "suppression_values": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Suppression Values",
                        "description": "List of values to match for suppression (used with 'value' condition).",
                    },
                    "suppression_range": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "title": "Suppression Range",
                        "description": "Range [min, max] for suppression (used with 'range' condition).",
                    },
                    # Output control
                    "save_suppressed_records": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Suppressed Records",
                        "description": "Whether to save removed records to a separate artifact.",
                    },
                    "suppression_reason_field": {
                        "type": "string",
                        "default": "_suppression_reason",
                        "title": "Suppression Reason Field",
                        "description": "Field name for storing the reason for suppression in the output.",
                    },
                    # Multi-field conditions
                    "multi_conditions": {
                        "type": ["array", "null"],
                        "items": {"type": "object"},
                        "title": "Multi-Conditions",
                        "description": "List of multi-field conditions for custom suppression logic.",
                    },
                    "condition_logic": {
                        "type": "string",
                        "title": "Condition Logic",
                        "description": "Logical expression for combining multi-field conditions (e.g., 'AND', 'OR').",
                    },
                    # K-anonymity integration
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field containing k-anonymity risk scores for suppression based on risk.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Threshold for k-anonymity risk triggering suppression.",
                    },
                },
                "required": ["field_name", "suppression_condition"],
            },
        ],
    }
