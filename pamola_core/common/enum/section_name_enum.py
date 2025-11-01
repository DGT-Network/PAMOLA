from enum import Enum

class SectionName(str, Enum):
    CORE_GENERALIZATION_STRATEGY = "core_generalization_strategy"
    CONDITION_LOGIC = "conditional_logic"
    OPERATION_BEHAVIOR_OUTPUT = "operation_behavior_output"


SECTION_NAME_TITLE = {
    SectionName.CORE_GENERALIZATION_STRATEGY: "Core Generalization Strategy",
    SectionName.CONDITION_LOGIC: "Conditional Logic",
    SectionName.OPERATION_BEHAVIOR_OUTPUT: "Operation Behavior & Output",
}