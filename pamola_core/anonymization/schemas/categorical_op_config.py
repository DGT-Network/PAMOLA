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
        "title": "Categorical Generalization Operation Configuration",
        "description": "Configuration schema for categorical generalization operations.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    # Required fields
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Target field name for categorical operation.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": STRATEGY_VALUES,
                        "title": "Generalization Strategy",
                        "description": "Generalization strategy to apply (e.g., hierarchy, frequency, dictionary).",
                    },
                    # Dictionary parameters
                    "external_dictionary_path": {
                        "type": ["string", "null"],
                        "title": "External Dictionary Path",
                        "description": "Path to external hierarchy or mapping dictionary file.",
                    },
                    "dictionary_format": {
                        "type": "string",
                        "enum": SUPPORTED_DICT_FORMATS,
                        "title": "Dictionary Format",
                        "description": "Dictionary file format (auto-detected by default).",
                    },
                    "hierarchy_level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_HIERARCHY_LEVELS,
                        "title": "Hierarchy Level",
                        "description": f"Hierarchy level to generalize to (1-{MAX_HIERARCHY_LEVELS}).",
                    },
                    # Frequency-based parameters
                    "merge_low_freq": {
                        "type": "boolean",
                        "title": "Merge Low Frequency",
                        "description": "Merge low-frequency categories into a single group.",
                    },
                    "min_group_size": {
                        "type": "integer",
                        "minimum": 1,
                        "title": "Minimum Group Size",
                        "description": "Minimum group size for privacy protection.",
                    },
                    "freq_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "title": "Frequency Threshold",
                        "description": "Frequency threshold (0-1) for category preservation.",
                    },
                    "max_categories": {
                        "type": "integer",
                        "minimum": 0,
                        "title": "Max Categories",
                        "description": "Maximum number of categories to preserve.",
                    },
                    # Unknown value handling
                    "allow_unknown": {
                        "type": "boolean",
                        "title": "Allow Unknown",
                        "description": "Allow unknown values in output.",
                    },
                    "unknown_value": {
                        "type": "string",
                        "title": "Unknown Value Placeholder",
                        "description": "Placeholder string for unknown values.",
                    },
                    "group_rare_as": {
                        "type": "string",
                        "enum": GROUP_RARE_VALUES,
                        "title": "Group Rare As",
                        "description": "Strategy for grouping rare categories.",
                    },
                    "rare_value_template": {
                        "type": "string",
                        "pattern": ".*\\{n\\}.*",
                        "title": "Rare Value Template",
                        "description": "Template for numbered rare values (must contain {n}).",
                    },
                    # Text processing
                    "text_normalization": {
                        "type": "string",
                        "enum": TEXT_NORM_VALUES,
                        "title": "Text Normalization",
                        "description": "Text normalization level to apply.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "title": "Case Sensitive",
                        "description": "Use case-sensitive category matching.",
                    },
                    "fuzzy_matching": {
                        "type": "boolean",
                        "title": "Fuzzy Matching",
                        "description": "Enable fuzzy string matching for categories.",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "title": "Similarity Threshold",
                        "description": "Similarity threshold for fuzzy matching (0-1).",
                    },
                    # Privacy controls
                    "privacy_check_enabled": {
                        "type": "boolean",
                        "title": "Privacy Check Enabled",
                        "description": "Enable privacy validation checks (e.g., k-anonymity).",
                    },
                    "min_acceptable_k": {
                        "type": "integer",
                        "minimum": 2,
                        "title": "Minimum Acceptable k",
                        "description": "Minimum k-anonymity (must be ≥2).",
                    },
                    "max_acceptable_disclosure_risk": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "title": "Max Acceptable Disclosure Risk",
                        "description": "Maximum acceptable disclosure risk (0-1).",
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "title": "Quasi-identifiers",
                        "description": "List of quasi-identifier field names.",
                    },
                    # Conditional processing
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name for conditional processing.",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values for conditional processing.",
                        "items": {
                            "type": "string"
                        },
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Conditional operator (in|not_in|eq|ne).",
                    },
                    # Risk assessment
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field for k-anonymity risk assessment.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "title": "Risk Threshold",
                        "description": "Risk threshold for vulnerability detection.",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records.",
                    },
                },
                "required": ["field_name", "strategy"],
            },
        ],
    }
