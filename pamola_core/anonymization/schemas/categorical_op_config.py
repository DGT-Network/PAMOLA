"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Generalization Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating categorical generalization parameters in PAMOLA.CORE.
- Supports multiple generalization strategies (hierarchy, frequency, dictionary, etc.)
- Controls handling of rare values, unknown values, text normalization, and privacy checks (k-anonymity, disclosure risk)
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of categorical generalization config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.anonymization.commons.categorical_config import (
    GROUP_RARE_VALUES,
    MAX_HIERARCHY_LEVELS,
    STRATEGY_VALUES,
    SUPPORTED_DICT_FORMATS,
    TEXT_NORM_VALUES,
)


class CategoricalGeneralizationConfig(OperationConfig):
    """
    Configuration for CategoricalGeneralizationOperation with BaseOperationConfig merged.

    This class extends OperationConfig with categorical-specific parameters
    and validation rules. It provides JSON Schema-based validation and
    strategy-specific parameter management.
    """

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    # Required fields
                    "field_name": {
                        "type": "string",
                        "description": "Target field name for generalization",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": STRATEGY_VALUES,
                        "description": "Generalization strategy to apply",
                    },
                    # Dictionary parameters
                    "external_dictionary_path": {
                        "type": ["string", "null"],
                        "description": "Path to external hierarchy dictionary file",
                    },
                    "dictionary_format": {
                        "type": "string",
                        "enum": SUPPORTED_DICT_FORMATS,
                        "description": "Dictionary file format (auto-detected by default)",
                    },
                    "hierarchy_level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_HIERARCHY_LEVELS,
                        "description": f"Hierarchy level (1-{MAX_HIERARCHY_LEVELS})",
                    },
                    # Frequency-based parameters
                    "merge_low_freq": {
                        "type": "boolean",
                        "description": "Merge low-frequency categories",
                    },
                    "min_group_size": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Minimum group size for privacy",
                    },
                    "freq_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Frequency threshold (0-1) for category preservation",
                    },
                    "max_categories": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Maximum number of categories to preserve",
                    },
                    # Unknown value handling
                    "allow_unknown": {
                        "type": "boolean",
                        "description": "Allow unknown values in output",
                    },
                    "unknown_value": {
                        "type": "string",
                        "description": "Placeholder string for unknown values",
                    },
                    "group_rare_as": {
                        "type": "string",
                        "enum": GROUP_RARE_VALUES,
                        "description": "Strategy for grouping rare categories",
                    },
                    "rare_value_template": {
                        "type": "string",
                        "pattern": ".*\\{n\\}.*",
                        "description": "Template for numbered rare values (must contain {n})",
                    },
                    # Text processing
                    "text_normalization": {
                        "type": "string",
                        "enum": TEXT_NORM_VALUES,
                        "description": "Text normalization level",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Use case-sensitive category matching",
                    },
                    "fuzzy_matching": {
                        "type": "boolean",
                        "description": "Enable fuzzy string matching",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Similarity threshold for fuzzy matching (0-1)",
                    },
                    # Privacy controls
                    "privacy_check_enabled": {
                        "type": "boolean",
                        "description": "Enable privacy validation checks",
                    },
                    "min_acceptable_k": {
                        "type": "integer",
                        "minimum": 2,
                        "description": "Minimum k-anonymity (must be ≥2)",
                    },
                    "max_acceptable_disclosure_risk": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Maximum acceptable disclosure risk (0-1)",
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "List of quasi-identifier field names",
                    },
                    # Conditional processing
                    "condition_field": {
                        "type": ["string", "null"],
                        "description": "Field name for conditional processing",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "description": "Values for conditional processing",
                        "items": {
                            "type": "string"
                        },
                    },
                    "condition_operator": {
                        "type": "string",
                        "description": "Conditional operator (in|not_in|eq|ne)",
                    },
                    # Risk assessment
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "description": "Field for k-anonymity risk assessment",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "description": "Risk threshold for vulnerability detection",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "description": "Strategy for handling vulnerable records",
                    },
                    # Output field name configuration
                    "output_field_name": {
                        "type": ["string", "null"],
                        "description": "Custom output field name (for ENRICH mode)",
                    },
                },
                "required": ["field_name", "strategy"],
            },
        ],
    }
