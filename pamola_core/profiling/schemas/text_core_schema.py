"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Semantic Categorizer Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of text semantic categorizer configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines text categorization parameters, entity types, matching strategies, and clustering options
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name and entity type specification
- Dictionary-based matching configuration
- Match strategy and clustering threshold controls
- NER and clustering enablement flags

Changelog:
1.0.0 - 2025-01-15 - Initial creation of text semantic categorizer core schema
1.1.0 - 2025-11-11 - Updated with enhanced entity type options
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class TextSemanticCategorizerOperationConfig(OperationConfig):
    """
    Core configuration schema for TextSemanticCategorizerOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Text Semantic Categorizer Operation Core Configuration",
        "description": "Core schema for text profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the text field (column) to analyze for semantic categorization.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "description": "Optional field name containing unique identifiers for each record.",
                    },
                    "entity_type": {
                        "type": "string",
                        "title": "Entity Type",
                        "oneOf": [
                            {"const": "job", "description": "Job"},
                            {"const": "organization", "description": "Organization"},
                            {"const": "skill", "description": "Skill"},
                            {"const": "generic", "description": "Generic"},
                        ],
                        "default": "generic",
                        "description": (
                            "Type of entity to extract or categorize:\n"
                            "- 'job': optimizes for job titles and positions\n"
                            "- 'organization': for company names\n"
                            "- 'skill': for technical skills\n"
                            "- 'generic': for general text"
                        ),
                    },
                    "dictionary_path": {
                        "type": ["string", "null"],
                        "title": "Dictionary Path",
                        "default": None,
                        "description": (
                            "Path to a file containing semantic category definitions and patterns.\n"
                            "Leave blank to use default categories."
                        ),
                    },
                    "min_word_length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "title": "Minimum Word Length",
                        "description": "Minimum number of characters a word must have for token-based analysis and keyword matching.",
                    },
                    "match_strategy": {
                        "type": "string",
                        "title": "Match Strategy",
                        "oneOf": [
                            {
                                "const": "specific_first",
                                "description": "Specific First",
                            },
                            {"const": "domain_prefer", "description": "Domain Prefer"},
                            {"const": "alias_only", "description": "Alias Only"},
                            {"const": "user_override", "description": "User Override"},
                        ],
                        "default": "specific_first",
                        "description": (
                            "Strategy for resolving conflicts when text matches multiple categories:\n"
                            "- 'specific_first': prioritizes deeper hierarchy levels\n"
                            "- 'domain_prefer': favors categories from primary domain\n"
                            "- 'alias_only': uses only alias fields\n"
                            "- 'user_override': applies manual prioritization rules"
                        ),
                    },
                    "clustering_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "title": "Clustering Threshold",
                        "description": "Minimum similarity score (0-1) required for clustering unmatched text entries into groups.",
                    },
                    "use_ner": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use NER",
                        "description": "Activates Named Entity Recognition (NER) for text items not matched by the dictionary.",
                    },
                    "perform_categorization": {
                        "type": "boolean",
                        "default": True,
                        "title": "Perform Categorization",
                        "description": "Activates similarity-based clustering for text entries not matched by dictionary or NER.",
                    },
                    "perform_clustering": {
                        "type": "boolean",
                        "default": True,
                        "title": "Perform Clustering",
                        "description": "Activates similarity-based clustering for text entries not matched by dictionary or NER.",
                    },
                },
                "required": [
                    "field_name",
                    "entity_type",
                    "min_word_length",
                    "match_strategy",
                    "clustering_threshold",
                ],
            },
        ],
    }
