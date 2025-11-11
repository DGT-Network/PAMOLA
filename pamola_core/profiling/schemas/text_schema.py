"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Text Semantic Categorizer Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating text semantic categorizer operations in PAMOLA.CORE.
- Supports parameters for field names, entity types, dictionaries, and clustering options
- Handles NER, similarity-based clustering, and match strategy configurations
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of text config file
1.1.0 - 2025-11-11 - Updated with x-component, x-group, and x-custom-function attributes
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class TextSemanticCategorizerOperationConfig(OperationConfig):
    """
    Configuration schema for TextSemanticCategorizerOperation.

    Extends BaseOperationConfig with text semantic categorization parameters,
    including entity recognition, NER, clustering, and dictionary-based matching.
    """

    schema = {
        "type": "object",
        "title": "Text Semantic Categorizer Operation Configuration",
        "description": (
            "Configuration options for semantic categorization of text fields. "
            "Supports entity extraction, NER, clustering, and dictionary-based matching."
        ),
        "allOf": [
            BaseOperationConfig.schema,  # merge base schema
            {
                "type": "object",
                "properties": {
                    # === Core text fields ===
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "x-component": "Select",
                        "description": "Name of the text field (column) to analyze for semantic categorization.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "x-component": "Select",
                        "description": "Optional field name containing unique identifiers for each record.",
                    },
                    # === Semantic configuration ===
                    "entity_type": {
                        "type": "string",
                        "title": "Entity Type",
                        "x-component": "Select",
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
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "dictionary_path": {
                        "type": ["string", "null"],
                        "title": "Dictionary Path",
                        "x-component": "Upload",
                        "default": None,
                        "description": (
                            "Path to a file containing semantic category definitions and patterns.\n"
                            "Leave blank to use default categories."
                        ),
                        "x-group": GroupName.DICTIONARY_CONFIGURATION,
                    },
                    "min_word_length": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 3,
                        "title": "Minimum Word Length",
                        "x-component": "NumberPicker",
                        "description": "Minimum number of characters a word must have for token-based analysis and keyword matching.",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    # === Matching and clustering ===
                    "match_strategy": {
                        "type": "string",
                        "title": "Match Strategy",
                        "x-component": "Select",
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
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "clustering_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "title": "Clustering Threshold",
                        "x-component": "NumberPicker",
                        "description": "Minimum similarity score (0-1) required for clustering unmatched text entries into groups.",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    # === Advanced options ===
                    "use_ner": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use NER",
                        "x-component": "Switch",
                        "description": "Activates Named Entity Recognition (NER) for text items not matched by the dictionary.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                    "perform_categorization": {
                        "type": "boolean",
                        "default": True,
                        "title": "Perform Categorization",
                        "x-component": "Switch",
                        "description": "Activates similarity-based clustering for text entries not matched by dictionary or NER.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                    "perform_clustering": {
                        "type": "boolean",
                        "default": True,
                        "title": "Perform Clustering",
                        "x-component": "Switch",
                        "description": "Activates similarity-based clustering for text entries not matched by dictionary or NER.",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                    },
                },
                "required": ["field_name", "entity_type"],
            },
        ],
    }
