"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Hash-Based Pseudonymization Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of hash-based pseudonymization
configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines salt/pepper, algorithm, output format, and collision handling
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure

Changelog:
1.0.0 - 2025-06-15 - Initial creation from inline schema in hash_based_op.py
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class HashBasedPseudonymizationConfig(OperationConfig):
    """
    Core configuration schema for HashBasedPseudonymizationOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Hash-Based Pseudonymization Operation Core Configuration",
        "description": "Core schema for hash-based pseudonymization operation configuration.",
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
                    "algorithm": {
                        "type": "string",
                        "title": "Hash Algorithm",
                        "default": "sha3_256",
                        "oneOf": [
                            {"const": "sha3_256", "description": "SHA3-256"},
                            {"const": "sha3_512", "description": "SHA3-512"},
                        ],
                        "description": (
                            "Cryptographic hash algorithm to use:\n"
                            "- 'sha3_256': SHA3-256 (default, recommended)\n"
                            "- 'sha3_512': SHA3-512 (longer output, higher security)"
                        ),
                    },
                    "salt_config": {
                        "type": "object",
                        "title": "Salt Configuration",
                        "properties": {
                            "source": {
                                "type": "string",
                                "enum": ["parameter", "file"],
                                "description": "Salt source: 'parameter' for inline, 'file' for external.",
                            },
                            "value": {
                                "type": ["string", "null"],
                                "description": "Hex-encoded salt value (for 'parameter' source).",
                            },
                            "field_name": {
                                "type": ["string", "null"],
                                "description": "Field-specific key in salt file (for 'file' source).",
                            },
                        },
                        "required": ["source"],
                        "description": "Configuration for salt used in hashing.",
                    },
                    "salt_file": {"type": ["string", "null"]},
                    "use_pepper": {
                        "type": "boolean",
                        "title": "Use Pepper",
                        "default": True,
                        "description": "Whether to use a session-specific pepper for extra security.",
                    },
                    "pepper_length": {
                        "type": "integer",
                        "title": "Pepper Length",
                        "default": 32,
                        "minimum": 16,
                        "description": "Length of pepper in bytes.",
                    },
                    "hash_output_format": {
                        "type": "string",
                        "title": "Hash Output Format",
                        "default": "hex",
                        "oneOf": [
                            {"const": "hex", "description": "Hexadecimal"},
                            {"const": "base64", "description": "Base64"},
                            {"const": "base32", "description": "Base32"},
                            {"const": "base58", "description": "Base58"},
                            {"const": "uuid", "description": "UUID-style"},
                        ],
                        "description": "Encoding format for the pseudonym output. Distinct from output_format (file format) in base schema.",
                    },
                    "output_length": {
                        "type": ["integer", "null"],
                        "title": "Output Length",
                        "minimum": 8,
                        "description": "Truncate pseudonym output to this length (null for full length).",
                    },
                    "output_prefix": {
                        "type": ["string", "null"],
                        "title": "Output Prefix",
                        "description": "Optional prefix to prepend to pseudonyms.",
                    },
                    "output_suffix": {
                        "type": ["string", "null"],
                        "title": "Output Suffix",
                        "description": "Optional suffix to append to pseudonyms.",
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
                "required": ["field_name", "algorithm", "salt_config"],
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
                    "properties": {"ka_risk_field": {"type": "string", "minLength": 1}},
                    "required": ["ka_risk_field"],
                },
                "then": {"properties": {"risk_threshold": {"type": "number"}}},
            },
        ],
    }
