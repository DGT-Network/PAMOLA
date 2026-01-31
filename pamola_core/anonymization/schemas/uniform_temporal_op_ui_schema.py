"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of uniform temporal noise configurations in PAMOLA.CORE.
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
- NumberPicker: Numeric inputs for noise ranges
- Select: Dropdown menus for direction and granularity
- DatePicker: Date selection for boundaries
- DatePickerArray: Array of date pickers for special dates
- Checkbox: Boolean toggles for preservation rules
- Input: Text input fields
- ArrayItems: Array field component for multi-conditions

Changelog:
1.0.0 - 2025-11-18 - Initial creation of uniform temporal noise UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class UniformTemporalNoiseUIConfig(OperationConfig):
    """
    UI configuration schema for UniformTemporalNoise form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Uniform Temporal Noise UI Configuration",
        "description": "UI schema for configuring uniform temporal noise anonymization operation.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "noise_range_days": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range_hours": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range_minutes": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range_seconds": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range": {
                        "x-component": "Object",
                    },
                    "direction": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "min_datetime": {
                        "x-component": "DatePicker",
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "max_datetime": {
                        "x-component": "DatePicker",
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "preserve_special_dates": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.PRESERVATION_RULES,
                    },
                    "special_dates": {
                        "x-component": "DatePickerArray",
                        "x-component-props": {
                            "format": "YYYY-MM-DD",
                            "placeholder": "Select date",
                        },
                        "x-group": GroupName.PRESERVATION_RULES,
                        "x-depend-on": {"preserve_special_dates": True},
                        "x-required-on": {"preserve_special_dates": True},
                    },
                    "preserve_weekends": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.PRESERVATION_RULES,
                    },
                    "preserve_time_of_day": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.PRESERVATION_RULES,
                    },
                    "output_granularity": {
                        "x-component": "Select",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "random_seed": {
                        "x-component": "NumberPicker",
                    },
                    "use_secure_random": {
                        "x-component": "Checkbox",
                    },
                    "condition_field": {
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "condition_operator": {
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"condition_field": "not_null"},
                        "x-custom-function": [
                            CustomFunctions.UPDATE_CONDITION_OPERATOR
                        ],
                    },
                    "condition_values": {
                        "x-component": "Input",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_VALUES],
                    },
                    "multi_conditions": {
                        "x-component": "ArrayItems",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
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
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"multi_conditions": "not_null"},
                    },
                    "ka_risk_field": {
                        "x-component": "Select",
                    },
                    "risk_threshold": {
                        "x-component": "FloatPicker",
                    },
                    "vulnerable_record_strategy": {
                        "x-component": "Input",
                    },
                }
            },
        ],
    }
