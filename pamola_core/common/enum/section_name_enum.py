from enum import Enum

class SectionName(str, Enum):
    CORE_GENERALIZATION_STRATEGY = "core_generalization_strategy"
    HIERARCHY_SETTINGS = "hierarchy_settings"
    FREQUENCY_GROUPING_SETTINGS = "frequency_grouping_settings"
    TEXT_VALUE_HANDLING = "text_value_handling"
    CONDITION_LOGIC = "conditional_logic"
    RISK_BASED_PROCESSING_AND_PRIVACY = "risk_based_processing_and_privacy"
    FORMATTING_AND_TIMEZONE = "formatting_and_timezone"
    OPERATION_BEHAVIOR_OUTPUT = "operation_behavior_output"


SECTION_NAME_TITLE = {
    SectionName.CORE_GENERALIZATION_STRATEGY: "Core Generalization Strategy",
    SectionName.HIERARCHY_SETTINGS: "Hierarchy Settings",
    SectionName.FREQUENCY_GROUPING_SETTINGS: "Frequency & Grouping Settings",
    SectionName.TEXT_VALUE_HANDLING: "Text & Value Handling",
    SectionName.CONDITION_LOGIC: "Conditional Logic",
    SectionName.RISK_BASED_PROCESSING_AND_PRIVACY: "Risk-Based Processing & Privacy",
    SectionName.FORMATTING_AND_TIMEZONE: "Formatting & Timezone",
    SectionName.OPERATION_BEHAVIOR_OUTPUT: "Operation Behavior & Output",
}