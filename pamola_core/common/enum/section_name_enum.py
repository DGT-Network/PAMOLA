from enum import Enum

class SectionName(str, Enum):
    CORE_GENERALIZATION_STRATEGY = "Core Generalization Strategy"
    CONDITION_LOGIC = "Conditional Logic"
    OPERATION_BEHAVIOR_OUTPUT = "Operation Behavior & Output"


SECTION_NAME_TITLE = {
    SectionName.CORE_GENERALIZATION_STRATEGY: "Core Generalization Strategy",
    SectionName.CONDITION_LOGIC: "Conditional Logic",
    SectionName.OPERATION_BEHAVIOR_OUTPUT: "Operation Behavior & Output",
}