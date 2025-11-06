"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Group Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating group profiling operations in PAMOLA.CORE.
Supports parameters for field names, field configs, thresholds, and profiling options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of group config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class GroupAnalyzerOperationConfig(OperationConfig):
    """Configuration schema for GroupAnalyzerOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Group Analyzer Operation Configuration",
        "description": "Configuration schema for group profiling operations. Defines parameters for analyzing groups in a dataset, including field selection, thresholds, and similarity settings.",
        "allOf": [
            BaseOperationConfig.schema,  # merge base common fields
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Group Field Name",
                        "description": "Name of the field (column) used to define groups for analysis. Must exist in the input DataFrame.",
                    },
                    "fields_config": {
                        "type": "object",
                        "title": "Fields Configuration",
                        "description": "Dictionary mapping field names to integer configuration values (e.g., weights or thresholds) for group analysis. Must have at least one property.",
                        "minProperties": 1,
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                    "text_length_threshold": {
                        "type": "integer",
                        "title": "Text Length Threshold",
                        "description": "Minimum text length required for a group to be included in the analysis.",
                        "minimum": 0,
                        "default": 100,
                    },
                    "variance_threshold": {
                        "type": "number",
                        "title": "Variance Threshold",
                        "description": "Minimum variance required for a group to be considered significant in the analysis.",
                        "minimum": 0.0,
                        "default": 0.2,
                    },
                    "large_group_threshold": {
                        "type": "integer",
                        "title": "Large Group Size Threshold",
                        "description": "Minimum number of records for a group to be considered 'large'.",
                        "minimum": 1,
                        "default": 100,
                    },
                    "large_group_variance_threshold": {
                        "type": "number",
                        "title": "Large Group Variance Threshold",
                        "description": "Variance threshold for large groups. Used to identify significant variation within large groups.",
                        "minimum": 0.0,
                        "default": 0.05,
                    },
                    "hash_algorithm": {
                        "type": "string",
                        "title": "Hash Algorithm",
                        "description": "Hashing algorithm to use for group analysis. Options are 'md5' for standard hashing or 'minhash' for similarity-based analysis.",
                        "enum": ["md5", "minhash"],
                        "default": "md5",
                    },
                    "minhash_similarity_threshold": {
                        "type": "number",
                        "title": "Minhash Similarity Threshold",
                        "description": "Similarity threshold (between 0.0 and 1.0) for minhash-based group analysis. Groups with similarity above this threshold are considered similar.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                    },
                },
                "required": ["field_name", "fields_config"],
            },
        ],
    }
