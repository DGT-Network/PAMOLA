"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Consistent Mapping Pseudonymization Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of consistent mapping
pseudonymization configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines mapping storage, pseudonym generation, and encryption parameters
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure

Changelog:
1.0.0 - 2025-06-15 - Initial creation from inline schema in mapping_op.py
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class ConsistentMappingPseudonymizationConfig(OperationConfig):
    """
    Core configuration schema for ConsistentMappingPseudonymizationOperation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    # AES-256-GCM key must never be persisted to disk in plaintext. The
    # base save_config() pipeline replaces this key with a placeholder.
    SENSITIVE_KEYS = frozenset({"mapping_encryption_key"})

    schema = {
        "type": "object",
        "title": "Consistent Mapping Pseudonymization Operation Core Configuration",
        "description": "Core schema for consistent mapping pseudonymization operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to pseudonymize.",
                    },
                    "additional_fields": {
                        "type": ["array", "null"],
                        "title": "Additional Fields",
                        "items": {"type": "string"},
                        "description": "Additional fields for compound pseudonymization.",
                    },
                    "mapping_file": {
                        "type": ["string", "null"],
                        "title": "Mapping File",
                        "description": "Path to mapping file (auto-generated if None).",
                    },
                    "mapping_format": {
                        "type": "string",
                        "title": "Mapping Format",
                        "default": "csv",
                        "oneOf": [
                            {"const": "csv", "description": "CSV"},
                            {"const": "json", "description": "JSON"},
                        ],
                        "description": "Storage format for the encrypted mapping file.",
                    },
                    "pseudonym_type": {
                        "type": "string",
                        "title": "Pseudonym Type",
                        "default": "uuid",
                        "oneOf": [
                            {"const": "uuid", "description": "UUID"},
                            {"const": "sequential", "description": "Sequential"},
                            {"const": "random_string", "description": "Random string"},
                        ],
                        "description": (
                            "Method for generating pseudonyms:\n"
                            "- 'uuid': UUID v4 identifiers\n"
                            "- 'sequential': Sequential numbered identifiers\n"
                            "- 'random_string': Random alphanumeric strings"
                        ),
                    },
                    "pseudonym_prefix": {
                        "type": ["string", "null"],
                        "title": "Pseudonym Prefix",
                        "description": "Optional prefix for generated pseudonyms.",
                    },
                    "pseudonym_suffix": {
                        "type": ["string", "null"],
                        "title": "Pseudonym Suffix",
                        "description": "Optional suffix for generated pseudonyms.",
                    },
                    "pseudonym_length": {
                        "type": "integer",
                        "title": "Pseudonym Length",
                        "default": 36,
                        "minimum": 4,
                        "maximum": 64,
                        "description": "Length of generated pseudonyms (for random_string type).",
                    },
                    "mapping_encryption_key": {
                        "type": "string",
                        "title": "Mapping Encryption Key",
                        "description": "256-bit encryption key as hex string for mapping storage.",
                    },
                    "create_if_not_exists": {
                        "type": "boolean",
                        "title": "Create If Not Exists",
                        "default": True,
                        "description": "Create new mapping file if one doesn't exist.",
                    },
                    "backup_on_update": {
                        "type": "boolean",
                        "title": "Backup On Update",
                        "default": True,
                        "description": "Create backup before updating mapping file.",
                    },
                    "persist_frequency": {
                        "type": "integer",
                        "title": "Persist Frequency",
                        "default": 1000,
                        "minimum": 1,
                        "description": "Save mapping to disk after this many new entries.",
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "title": "Quasi-Identifiers",
                        "items": {"type": "string"},
                        "description": "Fields used for privacy metrics calculation.",
                    },
                    "compound_mode": {
                        "type": "boolean",
                        "title": "Compound Mode",
                        "default": False,
                        "description": "Create compound identifiers from multiple fields.",
                    },
                    "compound_separator": {
                        "type": "string",
                        "title": "Compound Separator",
                        "default": "|",
                        "description": "Separator for compound identifier values.",
                    },
                    "compound_null_handling": {
                        "type": "string",
                        "title": "Compound Null Handling",
                        "default": "skip",
                        "oneOf": [
                            {"const": "skip", "description": "Skip null fields"},
                            {"const": "empty", "description": "Use empty string"},
                            {"const": "null", "description": "Use 'null' literal"},
                        ],
                        "description": "How to handle null values in compound identifiers.",
                    },
                    # Conditional anonymization fields
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field used for conditional pseudonymization.",
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "default": "in",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "description": "Comparison operator for the condition.",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values that trigger the pseudonymization.",
                    },
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-Anonymity Risk Field",
                        "description": "Field containing precomputed risk scores.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "default": 5.0,
                        "description": "Maximum acceptable risk value.",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "default": "pseudonymize",
                        "oneOf": [
                            {"const": "pseudonymize", "description": "Pseudonymize"},
                            {"const": "suppress", "description": "Suppress"},
                            {"const": "remove", "description": "Remove"},
                        ],
                        "description": "Action for records exceeding the risk threshold.",
                    },
                },
                "required": ["field_name", "mapping_encryption_key"],
            },
            # Pseudonym type-specific validation
            {
                "if": {"properties": {"pseudonym_type": {"const": "random_string"}}},
                "then": {
                    "properties": {
                        "pseudonym_length": {"type": "integer", "minimum": 4}
                    },
                    "required": ["pseudonym_length"],
                },
            },
            # Compound mode requires additional_fields
            {
                "if": {"properties": {"compound_mode": {"const": True}}},
                "then": {
                    "properties": {
                        "additional_fields": {
                            "type": "array",
                            "minItems": 1,
                        }
                    },
                    "required": ["additional_fields"],
                },
            },
            # Conditional logic dependencies
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_field"],
                },
                "then": {"properties": {"condition_operator": {"type": "string"}}},
            },
            # K-anonymity risk threshold dependency
            {
                "if": {
                    "properties": {
                        "ka_risk_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["ka_risk_field"],
                },
                "then": {"properties": {"risk_threshold": {"type": "number"}}},
            },
        ],
    }
