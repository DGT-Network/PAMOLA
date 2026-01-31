"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Generalization Core Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of categorical generalization configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines validation rules for multiple generalization strategies (hierarchy, frequency-based, merge low frequency)
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multiple generalization strategies with strategy-specific parameter validation
- Privacy protection controls (k-anonymity, disclosure risk)
- Text normalization and fuzzy matching capabilities
- Conditional processing and risk-based assessment
- Unknown value and rare category handling

Changelog:
1.0.0 - 2025-11-18 - Initial creation of categorical generalization core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig
from pamola_core.anonymization.commons.categorical_config import MAX_HIERARCHY_LEVELS


class CategoricalGeneralizationConfig(OperationConfig):
    """
    Core configuration schema for CategoricalGeneralization backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Categorical Generalization Operation Core Configuration",
        "description": "Core schema for categorical generalization operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Target field name for categorical operation.",
                    },
                    "strategy": {
                        "type": "string",
                        "default": "hierarchy",
                        "title": "Strategy",
                        "description": "Generalization strategy to apply (e.g., hierarchy, frequency, dictionary).",
                        "oneOf": [
                            {"const": "hierarchy", "description": "Hierarchy"},
                            {
                                "const": "merge_low_freq",
                                "description": "Merge Low Frequency",
                            },
                            {
                                "const": "frequency_based",
                                "description": "Frequency Based",
                            },
                        ],
                    },
                    "external_dictionary_path": {
                        "type": ["string", "null"],
                        "title": "External Dictionary Path",
                        "description": "Path to external hierarchy or mapping dictionary file.",
                    },
                    "dictionary_format": {
                        "type": "string",
                        "title": "Dictionary Format",
                        "description": "Dictionary file format (auto-detected by default).",
                        "default": "auto",
                        "oneOf": [
                            {"const": "auto", "description": "Auto"},
                            {"const": "json", "description": "Json"},
                            {"const": "csv", "description": "Csv"},
                        ],
                    },
                    "hierarchy_level": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_HIERARCHY_LEVELS,
                        "title": "Hierarchy Level",
                        "description": f"Hierarchy level to generalize to (1-{MAX_HIERARCHY_LEVELS}).",
                        "default": 1,
                    },
                    "min_group_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                        "title": "Minimum Group Size",
                        "description": "Minimum group size for privacy protection.",
                    },
                    "freq_threshold": {
                        "type": "number",
                        "default": 0.01,
                        "minimum": 0,
                        "maximum": 1,
                        "title": "Frequency Threshold",
                        "description": "Frequency threshold (0-1) for category preservation.",
                    },
                    "max_categories": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 1000000,
                        "title": "Max Categories",
                        "description": "Maximum number of categories to preserve.",
                    },
                    "allow_unknown": {
                        "type": "boolean",
                        "default": True,
                        "title": "Allow Unknown",
                        "description": "Allow unknown values in output.",
                    },
                    "unknown_value": {
                        "type": "string",
                        "default": "OTHER",
                        "title": "Unknown Value Placeholder",
                        "description": "Placeholder string for unknown values.",
                    },
                    "group_rare_as": {
                        "type": "string",
                        "default": "OTHER",
                        "title": "Group Rare As",
                        "description": "Strategy for grouping rare categories.",
                        "oneOf": [
                            {"const": "OTHER", "description": "OTHER"},
                            {"const": "CATEGORY_N", "description": "CATEGORY_N"},
                            {"const": "RARE_N", "description": "RARE_N"},
                        ],
                    },
                    "rare_value_template": {
                        "type": "string",
                        "default": "OTHER_1",
                        "title": "Rare Value Template",
                        "description": "Template for numbered rare values (must contain {n}).",
                    },
                    "text_normalization": {
                        "type": "string",
                        "default": "basic",
                        "title": "Text Normalization",
                        "description": "Text normalization level to apply.",
                        "oneOf": [
                            {"const": "none", "description": "None"},
                            {"const": "basic", "description": "Basic"},
                            {"const": "advanced", "description": "Advanced"},
                            {"const": "aggressive", "description": "Aggressive"},
                        ],
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "title": "Case Sensitive",
                        "description": "Use case-sensitive category matching.",
                    },
                    "fuzzy_matching": {
                        "type": "boolean",
                        "default": False,
                        "title": "Fuzzy Matching",
                        "description": "Enable fuzzy string matching for categories.",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.85,
                        "title": "Similarity Threshold",
                        "description": "Similarity threshold for fuzzy matching (0-1).",
                    },
                    "privacy_check_enabled": {
                        "type": "boolean",
                        "default": True,
                        "title": "Privacy Check Enabled",
                        "description": "Enable privacy validation checks (e.g., k-anonymity).",
                    },
                    "min_acceptable_k": {
                        "type": "integer",
                        "minimum": 2,
                        "default": 5,
                        "title": "Minimum Acceptable k",
                        "description": "Minimum k-anonymity (must be ≥2).",
                    },
                    "max_acceptable_disclosure_risk": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                        "title": "Max Acceptable Disclosure Risk",
                        "description": "Maximum acceptable disclosure risk (0-1).",
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Quasi-identifiers",
                        "description": "List of quasi-identifier field names.",
                    },
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field name for conditional processing.",
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Comparison operator used in the condition.",
                        "oneOf": [
                            {"const": "in", "description": "In"},
                            {"const": "not_in", "description": "Not in"},
                            {"const": "gt", "description": "Greater than"},
                            {"const": "lt", "description": "Less than"},
                            {"const": "eq", "description": "Equal to"},
                            {"const": "range", "description": "Range"},
                        ],
                        "default": "in",
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values for conditional processing.",
                    },
                    "ka_risk_field": {
                        "type": ["string", "null"],
                        "title": "K-anonymity Risk Field",
                        "description": "Field for k-anonymity risk assessment.",
                    },
                    "risk_threshold": {
                        "type": "number",
                        "default": 5,
                        "title": "Risk Threshold",
                        "description": "Risk threshold for vulnerability detection.",
                    },
                    "vulnerable_record_strategy": {
                        "type": "string",
                        "default": "suppress",
                        "title": "Vulnerable Record Strategy",
                        "description": "Strategy for handling vulnerable records.",
                    },
                },
                "required": ["field_name", "strategy"],
            },
            {
                "if": {"properties": {"strategy": {"const": "hierarchy"}}},
                "then": {
                    "required": [
                        "external_dictionary_path",
                        "dictionary_format",
                        "hierarchy_level",
                        "fuzzy_matching",
                    ]
                },
            },
            {
                "if": {"properties": {"strategy": {"const": "merge_low_freq"}}},
                "then": {
                    "required": ["min_group_size", "freq_threshold", "group_rare_as"]
                },
            },
            {
                "if": {"properties": {"strategy": {"const": "frequency_based"}}},
                "then": {
                    "required": ["min_group_size", "max_categories", "group_rare_as"]
                },
            },
            {
                "if": {"properties": {"allow_unknown": {"const": True}}},
                "then": {"required": ["unknown_value"]},
            },
            {
                "if": {
                    "properties": {
                        "group_rare_as": {
                            "oneOf": [{"const": "CATEGORY_N"}, {"const": "RARE_N"}]
                        }
                    },
                    "required": ["group_rare_as"],
                },
                "then": {"required": ["rare_value_template"]},
            },
            {
                "if": {"properties": {"fuzzy_matching": {"const": True}}},
                "then": {"required": ["similarity_threshold"]},
            },
            {
                "if": {"properties": {"privacy_check_enabled": {"const": True}}},
                "then": {
                    "required": [
                        "min_acceptable_k",
                        "max_acceptable_disclosure_risk",
                        "quasi_identifiers",
                    ]
                },
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1}
                    },
                    "required": ["condition_field"],
                },
                "then": {"properties": {"condition_operator": {"type": "string"}}},
            },
            {
                "if": {
                    "properties": {
                        "condition_field": {"type": "string", "minLength": 1},
                        "condition_operator": {"type": "string", "minLength": 1},
                    },
                    "required": ["condition_field", "condition_operator"],
                },
                "then": {"properties": {"condition_values": {"type": "array"}}},
            },
        ],
    }
