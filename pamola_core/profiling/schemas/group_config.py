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
        "allOf": [
            BaseOperationConfig.schema,  # merge base common fields
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "fields_config": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                    # Thresholds and Variance
                    "text_length_threshold": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 100,
                    },
                    "variance_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "default": 0.2,
                    },
                    "large_group_threshold": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 100,
                    },
                    "large_group_variance_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "default": 0.05,
                    },
                    # Hashing & Minhash
                    "hash_algorithm": {
                        "type": "string",
                        "enum": ["md5", "minhash"],
                        "default": "md5",
                    },
                    "minhash_similarity_threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                    },
                },
                "required": ["field_name", "fields_config"],
            },
        ],
    }