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
from pamola_core.common.enum.form_groups import GroupName
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
                        "title": "Strategy",
                        "description": "Generalization strategy to apply (e.g., hierarchy, frequency, dictionary).",
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "oneOf": [
                            {"const": "hierarchy", "description": "Hierarchy"},
                            {"const": "merge_low_freq", "description": "Merge Low Frequency"},
                            {"const": "frequency_based", "description": "Frequency Based"},
                        ],
                    },
                    # Dictionary parameters
                    "external_dictionary_path": {
                        "type": ["string", "null"],
                        "title": "External Dictionary Path",
                        "description": "Path to external hierarchy or mapping dictionary file.",
                        "x-component": "Upload",
                        "x-group": GroupName.HIERARCHY_SETTINGS,
                        "x-depend-on": { "strategy": "hierarchy" },
                        "x-required-on": { "strategy": "hierarchy" },
                    },
                    "dictionary_format": {
                        "type": "string",
                        "enum": SUPPORTED_DICT_FORMATS,
                        "title": "Dictionary Format",
                        "description": "Dictionary file format (auto-detected by default).",
                        "default": ".csv",
                        "x-component": "Input",
                        "x-group": GroupName.HIERARCHY_SETTINGS,
                        "x-depend-on": { "strategy": "hierarchy" },
                        "x-required-on": { "strategy": "hierarchy" },
                    },
                    "hierarchy_level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_HIERARCHY_LEVELS,
                        "title": "Hierarchy Level",
                        "description": f"Hierarchy level to generalize to (1-{MAX_HIERARCHY_LEVELS}).",
                        "default": 1,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.HIERARCHY_SETTINGS,
                        "x-depend-on": { "strategy": "hierarchy" },
                        "x-required-on": { "strategy": "hierarchy" },
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
                        "default": 10,
                        "title": "Minimum Group Size",
                        "description": "Minimum group size for privacy protection.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": { "strategy": ["merge_low_freq", "frequency_based"] },
                        "x-required-on": { "strategy": ["merge_low_freq", "frequency_based"] },
                    },
                    "freq_threshold": {
                        "type": "number",
                        "default": 0.01,
                        "minimum": 0,
                        "maximum": 1,
                        "title": "Frequency Threshold",
                        "description": "Frequency threshold (0-1) for category preservation.",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": { "strategy": "merge_low_freq" },
                        "x-required-on": { "strategy": "merge_low_freq" },
                    },
                    "max_categories": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 1000000,
                        "title": "Max Categories",
                        "description": "Maximum number of categories to preserve.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": { "strategy": "frequency_based" },
                        "x-required-on": { "strategy": "frequency_based" },
                    },
                    # Unknown value handling
                    "allow_unknown": {
                        "type": "boolean",
                        "default": True,
                        "title": "Allow Unknown",
                        "description": "Allow unknown values in output.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                    },
                    "unknown_value": {
                        "type": "string",
                        "default": "OTHER",
                        "title": "Unknown Value Placeholder",
                        "description": "Placeholder string for unknown values.",
                        "x-component": "Input",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "x-depend-on": { "allow_unknown": True },
                        "x-required-on": { "allow_unknown": True },
                    },
                    "group_rare_as": {
                        "type": "string",
                        "enum": GROUP_RARE_VALUES,
                        "title": "Group Rare As",
                        "description": "Strategy for grouping rare categories.",
                        "x-component": "Select",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": { "strategy": ["merge_low_freq", "frequency_based"] },
                        "x-required-on": { "strategy": ["merge_low_freq", "frequency_based"] },
                        "oneOf": [
                            {"const": "OTHER", "description": "OTHER"},
                            {"const": "CATEGORY_N", "description": "CATEGORY_N"},
                            {"const": "RARE_N", "description": "RARE_N"},
                        ],
                    },
                    "rare_value_template": {
                        "type": "string",
                        "pattern": ".*\\{n\\}.*",
                        "default": "OTHER_1",
                        "title": "Rare Value Template",
                        "description": "Template for numbered rare values (must contain {n}).",
                        "x-component": "Input",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": { "group_rare_as": ["CATEGORY_N", "RARE_N"] },
                        "x-required-on": { "group_rare_as": ["CATEGORY_N", "RARE_N"] },
                    },
                    # Text processing
                    "text_normalization": {
                        "type": "string",
                        "enum": TEXT_NORM_VALUES,
                        "title": "Text Normalization",
                        "description": "Text normalization level to apply.",
                        "x-component": "Select",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "oneOf": [
                            {"const": "NONE", "description": "none"},
                            {"const": "BASIC", "description": "basic"},
                            {"const": "ADVANCED", "description": "advanced"},
                            {"const": "AGGRESSIVE", "description": "aggressive"},
                        ],
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "title": "Case Sensitive",
                        "description": "Use case-sensitive category matching.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                    },
                    "fuzzy_matching": {
                        "type": "boolean",
                        "default": False,
                        "title": "Fuzzy Matching",
                        "description": "Enable fuzzy string matching for categories.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "x-depend-on": { "strategy": "hierarchy" },
                        "x-required-on": { "strategy": "hierarchy" },
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.85,
                        "title": "Similarity Threshold",
                        "description": "Similarity threshold for fuzzy matching (0-1).",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "x-depend-on": { "fuzzy_matching": True },
                        "x-required-on": { "fuzzy_matching": True },
                    },
                    # Privacy controls
                    "privacy_check_enabled": {
                        "type": "boolean",
                        "default": True,
                        "title": "Privacy Check Enabled",
                        "description": "Enable privacy validation checks (e.g., k-anonymity).",
                        "x-component": "Checkbox",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                    "min_acceptable_k": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 5,
                        "title": "Minimum Acceptable k",
                        "description": "Minimum k-anonymity (must be ≥2).",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-depend-on": { "privacy_check_enabled": True },
                        "x-required-on": { "privacy_check_enabled": True },
                    },
                    "max_acceptable_disclosure_risk": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                        "title": "Max Acceptable Disclosure Risk",
                        "description": "Maximum acceptable disclosure risk (0-1).",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-depend-on": { "privacy_check_enabled": True },
                        "x-required-on": { "privacy_check_enabled": True },
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "x-component": "Input"
                        },
                        "title": "Quasi-identifiers",
                        "description": "List of quasi-identifier field names.",
                        "x-component": "ArrayItems",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-depend-on": { "privacy_check_enabled": True },
                        "x-required-on": { "privacy_check_enabled": True },
                    },
                    # Conditional processing
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name for conditional processing.",
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": ["update_condition_field"]
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition.",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"}
                        ],
                        "default": "in",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": { "condition_field": "not_null" },
                        "x-custom-function": ["update_condition_operator"]
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values for conditional processing.",
                        "items": {
                            "type": "string"
                        },
                        "x-component": "Input",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": { "condition_field": "not_null", "condition_operator": "not_null"},
                        "x-custom-function": ["update_condition_values"]
                    },
                    # Risk assessment
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field for k-anonymity risk assessment.",
                        "x-component": "Input",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                    "risk_threshold": {
                        "type": "number",
                        "default": 5,
                        "title": "Risk Threshold",
                        "description": "Risk threshold for vulnerability detection.",
                        "x-component": "FloatPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "default": "suppress",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records.",
                        "x-component": "Input",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                },
                "required": ["field_name", "strategy"],
            },
            # === Conditional logic for strategy-specific requirements ===
            {
                "if": {"properties": {"strategy": {"const": "hierarchy"}}},
                "then": {"required": ["external_dictionary_path", "dictionary_format", "hierarchy_level"]}
            },
            {
                "if": {"properties": {"strategy": {"const": "merge_low_freq"}}},
                "then": {"required": ["min_group_size", "freq_threshold", "group_rare_as"]}
            },
            {
                "if": {"properties": {"strategy": {"const": "frequency_based"}}},
                "then": {"required": ["min_group_size", "max_categories", "group_rare_as"]}
            },
            {
                "if": {"properties": {"allow_unknown": {"const": True}}},
                "then": {"required": ["unknown_value"]}
            },
            {
                "if": {
                    "properties": {"group_rare_as": {"enum": ["CATEGORY_N", "RARE_N"]}},
                    "required": ["group_rare_as"]
                },
                "then": {"required": ["rare_value_template"]}
            },
            {
                "if": {"properties": {"fuzzy_matching": {"const": True}}},
                "then": {"required": ["similarity_threshold"]}
            },
            {
                "if": {"properties": {"privacy_check_enabled": {"const": True}}},
                "then": {"required": ["min_acceptable_k", "max_acceptable_disclosure_risk", "quasi_identifiers"]}
            }
        ],
    }
