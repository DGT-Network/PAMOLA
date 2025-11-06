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
        "title": "Text Semantic Categorizer Operation Configuration",
        "description": "Configuration schema for text semantic categorization operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- Text Semantic Categorizer specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the text field (column) to analyze. This should be a column in the DataFrame containing text data for semantic categorization.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "description": "Optional field name containing unique identifiers for each record. Used to track and reference individual text entries.",
                    },
                    "entity_type": {
                        "type": "string",
                        "default": "generic",
                        "title": "Entity Type",
                        "description": "Type of entity to extract or categorize (e.g., 'person', 'organization', 'location', or 'generic'). Determines the semantic focus of the analysis.",
                    },
                    "dictionary_path": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Dictionary Path",
                        "description": "Path to a dictionary file for category or alias matching. If null, default or no dictionary-based categorization is used.",
                    },
                    "min_word_length": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 3,
                        "title": "Minimum Word Length",
                        "description": "Minimum length of words to consider for entity extraction and categorization. Shorter words will be ignored.",
                    },
                    "clustering_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "title": "Clustering Threshold",
                        "description": "Similarity threshold for clustering unresolved or unmatched text values. Values above this threshold are grouped together.",
                    },
                    "use_ner": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use NER",
                        "description": "Whether to use Named Entity Recognition (NER) for entity extraction and categorization. If true, NER-based methods are applied.",
                    },
                    "perform_categorization": {
                        "type": "boolean",
                        "default": True,
                        "title": "Perform Categorization",
                        "description": "Whether to perform semantic categorization of text values. If false, only basic statistics or clustering may be performed.",
                    },
                    "perform_clustering": {
                        "type": "boolean",
                        "default": True,
                        "title": "Perform Clustering",
                        "description": "Whether to cluster unresolved or unmatched text values based on similarity. Helps group similar but uncategorized entries.",
                    },
                    "match_strategy": {
                        "type": "string",
                        "enum": [
                            "specific_first",
                            "domain_prefer",
                            "alias_only",
                            "user_override",
                        ],
                        "default": "specific_first",
                        "title": "Match Strategy",
                        "description": "Strategy for matching text values to categories or aliases. Options control the order and preference for specific, domain, alias, or user-defined matches.",
                    },
                },
                "required": ["field_name", "entity_type"],
            },
        ],
    }
