"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating text semantic categorizer operations in PAMOLA.CORE.
Supports parameters for field names, entity types, dictionaries, and clustering options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of text config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class TextSemanticCategorizerOperationConfig(OperationConfig):
    """Configuration for TextSemanticCategorizerOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- Text Semantic Categorizer specific fields ---
                    "field_name": {"type": "string"},
                    "id_field": {"type": ["string", "null"]},
                    "entity_type": {"type": "string", "default": "generic"},
                    "dictionary_path": {"type": ["string", "null"], "default": None},
                    "min_word_length": {"type": "integer", "minimum": 1, "default": 3},
                    "clustering_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                    },
                    "use_ner": {"type": "boolean", "default": True},
                    "perform_categorization": {"type": "boolean", "default": True},
                    "perform_clustering": {"type": "boolean", "default": True},
                    "match_strategy": {
                        "type": "string",
                        "enum": [
                            "specific_first",
                            "domain_prefer",
                            "alias_only",
                            "user_override",
                        ],
                        "default": "specific_first",
                    },
                },
                "required": ["field_name", "entity_type"],
            },
        ],
    }

