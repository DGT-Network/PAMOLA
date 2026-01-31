"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Record Suppression UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
Updated:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of record suppression configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Conditional field visibility using x-depend-on directives
- Dynamic group toggling based on suppression condition
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menus for modes and conditions
- StringArray: Custom array input for suppression values
- RangeNumber: Custom range selector component
- ArrayItems: Array field component for multi-conditions
- Checkbox: Boolean toggles
- Input: Text input fields
- NumberPicker: Numeric inputs with validation

Changelog:
1.1.0 - 2025-11-18 - Refactored into separate UI schema
1.0.0 - 2025-01-15 - Initial creation of record suppression config file
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class RecordSuppressionUIConfig(OperationConfig):
    """
    UI configuration schema for RecordSuppression form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Record Suppression Operation UI Configuration",
        "description": "UI schema for configuring record suppression operations.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "suppression_mode": {
                        "x-component": "Select",
                    },
                    "suppression_condition": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_SUPPRESSION_RULE,
                        "x-toggle-groups": {
                            "custom": [GroupName.ADVANCED_CONDITIONAL_RULES],
                            "risk": [GroupName.RISK_BASED_FILTERING],
                        },
                    },
                    "suppression_values": {
                        "x-component": CustomComponents.STRING_ARRAY,
                        "x-group": GroupName.CORE_SUPPRESSION_RULE,
                        "x-depend-on": {"suppression_condition": "value"},
                        "x-required-on": {"suppression_condition": "value"},
                    },
                    "suppression_range": {
                        "x-component": CustomComponents.RANGE_NUMBER,
                        "x-group": GroupName.CORE_SUPPRESSION_RULE,
                        "x-depend-on": {"suppression_condition": "range"},
                        "x-required-on": {"suppression_condition": "range"},
                    },
                    "multi_conditions": {
                        "x-component": "ArrayItems",
                        "x-group": GroupName.ADVANCED_CONDITIONAL_RULES,
                        "x-depend-on": {"suppression_condition": "custom"},
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
                        "x-depend-on": {"suppression_condition": "risk"},
                        "x-required-on": {"suppression_condition": "risk"},
                    },
                    "risk_threshold": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.RISK_BASED_FILTERING,
                        "x-depend-on": {"ka_risk_field": "not_null"},
                        "x-required-on": {"ka_risk_field": "not_null"},
                    },
                    "save_suppressed_records": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "suppression_reason_field": {
                        "x-component": "Input",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                }
            },
        ],
    }
