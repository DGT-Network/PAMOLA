"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Suppression UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of attribute suppression configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Conditional field visibility using x-depend-on directives
- No business logic validation - purely presentational metadata

UI Component Types:
- Checkbox: Boolean toggles
- Select: Dropdown menus for enums/oneOf
- Input: Text input fields
- NumberPicker: Numeric inputs with validation
- ArrayItems: Array field component for multi-conditions

Changelog:
1.0.0 - 2025-11-18 - Initial creation of attribute suppression UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class AttributeSuppressionUIConfig(OperationConfig):
    """
    UI configuration schema for AttributeSuppression form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Attribute Suppression Operation UI Configuration",
        "description": "UI schema for configuring attribute-level suppression operations.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "additional_fields": {
                        "x-component": "Select",
                    },
                    "suppression_mode": {
                        "x-component": "Select",
                    },
                    "save_suppressed_schema": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "condition_field": {
                        "x-component": "Select",
                        "x-group": GroupName.SIMPLE_CONDITIONAL_RULE,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "condition_operator": {
                        "x-component": "Select",
                        "x-group": GroupName.SIMPLE_CONDITIONAL_RULE,
                        "x-depend-on": {"condition_field": "not_null"},
                        "x-custom-function": [
                            CustomFunctions.UPDATE_CONDITION_OPERATOR
                        ],
                    },
                    "condition_values": {
                        "x-component": "Input",
                        "x-group": GroupName.SIMPLE_CONDITIONAL_RULE,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_VALUES],
                    },
                    "multi_conditions": {
                        "x-component": "ArrayItems",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                        "x-items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "x-component": "Select",
                                    "x-decorator-props": {
                                        "layout": "vertical",
                                        "style": {"width": "250px", "marginBottom": 8},
                                    },
                                    "x-custom-function": [
                                        CustomFunctions.UPDATE_FIELD_OPTIONS
                                    ],
                                },
                                "operator": {
                                    "x-component": "Select",
                                    "x-depend-on": {"field": "not_null"},
                                    "x-decorator-props": {
                                        "layout": "vertical",
                                        "style": {"width": "250px", "marginBottom": 8},
                                    },
                                    "x-custom-function": [
                                        CustomFunctions.UPDATE_CONDITION_OPERATOR
                                    ],
                                },
                                "values": {
                                    "x-component": "Input",
                                    "x-depend-on": {
                                        "field": "not_null",
                                        "operator": "not_null",
                                    },
                                    "x-decorator-props": {
                                        "layout": "vertical",
                                        "style": {"width": "250px", "marginBottom": 8},
                                    },
                                    "x-custom-function": [
                                        CustomFunctions.UPDATE_CONDITION_VALUES
                                    ],
                                },
                            },
                        },
                    },
                    "condition_logic": {
                        "x-component": "Select",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                        "x-depend-on": {"multi_conditions": "not_null"},
                    },
                    "ka_risk_field": {
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_INT64_FIELD_OPTIONS
                        ],
                    },
                    "risk_threshold": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                        "x-depend-on": {"ka_risk_field": "not_null"},
                    },
                }
            },
        ],
    }
