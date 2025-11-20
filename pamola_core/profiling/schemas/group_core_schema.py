"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Group Analyzer Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of group analyzer configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines group analysis parameters, variance thresholds, and hash algorithm settings
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Variance threshold configuration for group significance
- Large group detection and analysis settings
- Hash algorithm selection (md5, minhash)
- Conditional minhash similarity threshold validation
- Field configuration mapping for weighted analysis

Changelog:
1.0.0 - 2025-01-15 - Initial creation of group analyzer core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class GroupAnalyzerOperationConfig(OperationConfig):
    """
    Core configuration schema for GroupAnalyzerOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Group Operation Core Configuration",
        "description": "Core schema for group profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Group Field Name",
                        "description": "Name of the field (column) used to define groups for analysis. Must exist in the input DataFrame.",
                    },
                    "variance_threshold": {
                        "type": "number",
                        "title": "Variance Threshold",
                        "description": "Minimum variance required for a group to be considered significant in the analysis.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.2,
                    },
                    "large_group_threshold": {
                        "type": "integer",
                        "title": "Large Group Size Threshold",
                        "description": "Minimum number of records for a group to be considered 'large'.",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 100,
                    },
                    "large_group_variance_threshold": {
                        "type": "number",
                        "title": "Large Group Variance Threshold",
                        "description": "Variance threshold for large groups. Used to identify significant variation within large groups.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.05,
                    },
                    "text_length_threshold": {
                        "type": "integer",
                        "title": "Text Length Threshold",
                        "description": "Minimum text length required for a group to be included in the analysis.",
                        "minimum": 0,
                        "maximum": 10000,
                        "default": 100,
                    },
                    "hash_algorithm": {
                        "type": "string",
                        "title": "Hash Algorithm",
                        "description": "Hashing algorithm to use for group analysis. Options are 'md5' for standard hashing or 'minhash' for similarity-based analysis.",
                        "default": "md5",
                        "oneOf": [
                            {"const": "md5", "description": "Md5"},
                            {"const": "minhash", "description": "Minhash"},
                        ],
                    },
                    "minhash_similarity_threshold": {
                        "type": "number",
                        "title": "Minhash Similarity Threshold",
                        "description": "Similarity threshold (between 0.0 and 1.0) for minhash-based group analysis. Groups with similarity above this threshold are considered similar.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                    },
                    "fields_config": {
                        "type": "object",
                        "title": "Fields Configuration",
                        "description": "Dictionary mapping field names to integer configuration values (e.g., weights or thresholds) for group analysis.",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                        },
                    },
                },
                "required": ["field_name", "fields_config"],
            },
            {
                "if": {"properties": {"hash_algorithm": {"const": "minhash"}}},
                "then": {"required": ["minhash_similarity_threshold"]},
            },
        ],
    }
