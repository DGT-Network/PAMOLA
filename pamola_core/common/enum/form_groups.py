"""
Form group definitions for UI configuration.

This module provides metadata for organizing operation configuration
fields into logical groups in the UI.
"""

from enum import Enum
from typing import List, Dict


class GroupName(str, Enum):
    """Form field groups for organizing configuration UI."""

    CORE_GENERALIZATION_STRATEGY = "core_generalization_strategy"
    CONDITIONAL_LOGIC = "conditional_logic"
    OPERATION_BEHAVIOR_OUTPUT = "operation_behavior_output"
    HIERARCHY_SETTINGS = "hierarchy_settings"
    FREQUENCY_GROUPING_SETTINGS = "frequency_grouping_settings"
    TEXT_VALUE_HANDLING = "text_value_handling"
    RISK_BASED_PROCESSING_AND_PRIVACY = "risk_based_processing_and_privacy"
    FORMATTING_AND_TIMEZONE = "formatting_and_timezone"
    CORE_NOISE_STRATEGY = "core_noise_strategy"
    PRESERVATION_RULES = "preservation_rules"
    OUTPUT_FORMATTING_CONSTRAINTS = "output_formatting_constraints"
    CORE_MASKING_RULES = "core_masking_rules"
    MASK_APPEARANCE = "mask_appearance"
    MASKING_RULES = "masking_rules"
    FORMATTING_AND_STRUCTURE = "formatting_and_structure"
    CORE_MASKING_STRATEGY = "core_masking_strategy"
    CORE_SUPPRESSION_RULE = "core_suppression_rule"
    ADVANCED_CONDITIONAL_RULES = "advanced_conditional_rules"
    RISK_BASED_FILTERING = "risk_based_filtering"
    CORE_SUPPRESSION_STRATEGY = "core_suppression_strategy"
    SIMPLE_CONDITIONAL_RULE = "simple_conditional_rule"
    SUPPRESSION_RULES = "suppression_rules"
    IDENTIFIER_CONFIGURATION = "identifier_configuration"
    ANALYSIS_PARAMETERS = "analysis_parameters"
    NAME_GENERATION_STYLE = "name_generation_style"
    GENDER_CONFIGURATION = "gender_configuration"
    CONSISTENCY_STRATEGY = "consistency_strategy"
    REGIONAL_CONFIGURATION = "regional_configuration"
    FORMATTING_RULES = "formatting_rules"
    GENERATION_LOGIC = "generation_logic"
    EMAIL_GENERATION_STYLE = "email_generation_style"
    DATA_SOURCES_FOR_GENERATION = "data_sources_for_generation"
    ORGANIZATION_GENERATION_STYLE = "organization_generation_style"
    CONTEXT_AND_DATA_SOURCES = "context_and_data_sources"
    VALIDATION_RANGE = "validation_range"
    DATA_QUALITY_ANALYSIS = "data_quality_analysis"
    CURRENCY_PARSING_SETTINGS = "currency_parsing_settings"
    DISTRIBUTION_AND_ANALYSIS_SETTINGS = "distribution_and_analysis_settings"


GROUP_TITLES: Dict[GroupName, str] = {
    GroupName.CORE_GENERALIZATION_STRATEGY: "Core Generalization Strategy",
    GroupName.CONDITIONAL_LOGIC: "Conditional Logic",
    GroupName.OPERATION_BEHAVIOR_OUTPUT: "Operation Behavior & Output",
    GroupName.HIERARCHY_SETTINGS: "Hierarchy Settings",
    GroupName.FREQUENCY_GROUPING_SETTINGS: "Frequency & Grouping Settings",
    GroupName.TEXT_VALUE_HANDLING: "Text & Value Handling",
    GroupName.RISK_BASED_PROCESSING_AND_PRIVACY: "Risk-Based Processing & Privacy",
    GroupName.FORMATTING_AND_TIMEZONE: "Formatting & Timezone",
    GroupName.CORE_NOISE_STRATEGY: "Core Noise Strategy",
    GroupName.PRESERVATION_RULES: "Preservation Rules",
    GroupName.OUTPUT_FORMATTING_CONSTRAINTS: "Output Formatting Constraints",
    GroupName.CORE_MASKING_RULES: "Core Masking Rules",
    GroupName.MASK_APPEARANCE: "Mask Appearance",
    GroupName.MASKING_RULES: "Masking Rules",
    GroupName.FORMATTING_AND_STRUCTURE: "Formatting & Structure",
    GroupName.CORE_MASKING_STRATEGY: "Core Masking Strategy",
    GroupName.CORE_SUPPRESSION_RULE: "Core Suppression Rule",
    GroupName.ADVANCED_CONDITIONAL_RULES: "Advanced Conditional Rules",
    GroupName.RISK_BASED_FILTERING: "Risk-Based Filtering",
    GroupName.CORE_SUPPRESSION_STRATEGY: "Core Suppression Strategy",
    GroupName.SUPPRESSION_RULES: "Suppression Rules",
    GroupName.SIMPLE_CONDITIONAL_RULE: "Simple Conditional Rule",
    GroupName.IDENTIFIER_CONFIGURATION: "Identifier Configuration",
    GroupName.ANALYSIS_PARAMETERS: "Analysis Parameters",
    GroupName.NAME_GENERATION_STYLE: "Name Generation Style",
    GroupName.GENDER_CONFIGURATION: "Gender Configuration",
    GroupName.CONSISTENCY_STRATEGY: "Consistency Strategy",
    GroupName.REGIONAL_CONFIGURATION: "Regional Configuration",
    GroupName.FORMATTING_RULES: "Formatting Rules",
    GroupName.GENERATION_LOGIC: "Generation Logic",
    GroupName.EMAIL_GENERATION_STYLE: "Email Generation Style",
    GroupName.DATA_SOURCES_FOR_GENERATION: "Data Sources for Generation",
    GroupName.ORGANIZATION_GENERATION_STYLE: "Organization Generation Style",
    GroupName.CONTEXT_AND_DATA_SOURCES: "Context & Data Sources",
    GroupName.VALIDATION_RANGE: "Validation Range",
    GroupName.DATA_QUALITY_ANALYSIS: "Data Quality Analysis",
    GroupName.CURRENCY_PARSING_SETTINGS: "Currency Parsing Settings",
    GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS: "Distribution & Analysis Settings",
}


OPERATION_CONFIG_GROUPS: Dict[str, List[GroupName]] = {
    "NumericGeneralizationConfig": [
        GroupName.CORE_GENERALIZATION_STRATEGY,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "CategoricalGeneralizationConfig": [
        GroupName.CORE_GENERALIZATION_STRATEGY,
        GroupName.HIERARCHY_SETTINGS,
        GroupName.FREQUENCY_GROUPING_SETTINGS,
        GroupName.TEXT_VALUE_HANDLING,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "DateTimeGeneralizationConfig": [
        GroupName.CORE_GENERALIZATION_STRATEGY,
        GroupName.FORMATTING_AND_TIMEZONE,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "FullMaskingConfig": [
        GroupName.CORE_MASKING_RULES,
        GroupName.FORMATTING_AND_STRUCTURE,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "PartialMaskingConfig": [
        GroupName.CORE_MASKING_STRATEGY,
        GroupName.MASK_APPEARANCE,
        GroupName.MASKING_RULES,
        GroupName.FORMATTING_AND_STRUCTURE,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "UniformNumericNoiseConfig": [
        GroupName.CORE_NOISE_STRATEGY,
        GroupName.MASK_APPEARANCE,
        GroupName.MASKING_RULES,
        GroupName.FORMATTING_AND_STRUCTURE,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "UniformTemporalNoiseConfig": [
        GroupName.CORE_NOISE_STRATEGY,
        GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "RecordSuppressionConfig": [
        GroupName.CORE_SUPPRESSION_RULE,
        GroupName.ADVANCED_CONDITIONAL_RULES,
        GroupName.RISK_BASED_FILTERING,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "CellSuppressionConfig": [
        GroupName.CORE_SUPPRESSION_STRATEGY,
        GroupName.SUPPRESSION_RULES,
        GroupName.CONDITIONAL_LOGIC,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "AttributeSuppressionConfig": [
        GroupName.SIMPLE_CONDITIONAL_RULE,
        GroupName.ADVANCED_CONDITIONAL_RULES,
        GroupName.RISK_BASED_FILTERING,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "EmailOperationConfig": [
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "IdentityAnalysisOperationConfig": [
        GroupName.IDENTIFIER_CONFIGURATION,
        GroupName.ANALYSIS_PARAMETERS,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "MVFAnalysisOperationConfig": [
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "NumericOperationConfig": [
        GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
    ],
    "FakeNameOperationConfig": [
        GroupName.NAME_GENERATION_STYLE,
        GroupName.GENDER_CONFIGURATION,
        GroupName.CONSISTENCY_STRATEGY,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "FakePhoneOperationConfig": [
        GroupName.REGIONAL_CONFIGURATION,
        GroupName.FORMATTING_RULES,
        GroupName.GENERATION_LOGIC,
        GroupName.CONSISTENCY_STRATEGY,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    'FakeEmailOperationConfig': [
        GroupName.EMAIL_GENERATION_STYLE,
        GroupName.DATA_SOURCES_FOR_GENERATION,
        GroupName.CONSISTENCY_STRATEGY,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    'FakeOrganizationOperationConfig': [
        GroupName.ORGANIZATION_GENERATION_STYLE,
        GroupName.CONTEXT_AND_DATA_SOURCES,
        GroupName.CONSISTENCY_STRATEGY,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "DateOperationConfig": [
        GroupName.VALIDATION_RANGE,
        GroupName.DATA_QUALITY_ANALYSIS,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "CurrencyOperationConfig": [
        GroupName.CURRENCY_PARSING_SETTINGS,
        GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ],
    "CorrelationOperationConfig": [
        GroupName.OPERATION_BEHAVIOR_OUTPUT,
    ]
}


def get_groups_for_operation(operation_config_type: str) -> List[GroupName]:
    """
    Get ordered groups for a specific operation config type.

    Args:
        operation_config_type: The operation config type (e.g., 'NumericGeneralizationConfig')

    Returns:
        List of GroupName enums in display order

    Raises:
        ValueError: If operation_config_type is not configured
    """
    if operation_config_type not in OPERATION_CONFIG_GROUPS:
        raise ValueError(
            f"Unknown operation config type: '{operation_config_type}'. "
            f"Available: {list(OPERATION_CONFIG_GROUPS.keys())}"
        )
    return OPERATION_CONFIG_GROUPS[operation_config_type]


def get_groups_with_titles(operation_config_type: str) -> List[Dict[str, str]]:
    """
    Get group metadata for an operation config type.

    Args:
        operation_config_type: The operation config type

    Returns:
        List of dicts with 'name' and 'title' keys

    Example:
        >>> get_groups_with_titles("NumericGeneralizationConfig")
        [
            {"name": "core_generalization_strategy", "title": "Core Generalization Strategy"},
            {"name": "conditional_logic", "title": "Conditional Logic"},
            {"name": "operation_behavior_output", "title": "Operation Behavior & Output"}
        ]
    """
    groups = get_groups_for_operation(operation_config_type)
    return [{"name": group.value, "title": GROUP_TITLES[group]} for group in groups]
