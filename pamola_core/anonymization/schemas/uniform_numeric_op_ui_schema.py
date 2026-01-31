"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of uniform numeric noise configurations in PAMOLA.CORE.
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
- Select: Dropdown menus for enums/oneOf
- Checkbox: Boolean toggles
- FloatPicker: Floating-point numeric inputs
- Input: Text input fields
- ArrayItems: Array field component for multi-conditions
- NumericRangeMode: Custom component for noise range

Changelog:
1.0.0 - 2025-11-18 - Initial creation of uniform numeric noise UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class UniformNumericNoiseUIConfig(OperationConfig):
    """
    UI configuration schema for UniformNumericNoise form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Uniform Numeric Noise UI Configuration",
        "description": "UI schema for configuring uniform numeric noise anonymization operation.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "noise_type": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "noise_range": {
                        "x-component": CustomComponents.NUMERIC_RANGE_MODE,
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "output_min": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "output_max": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "preserve_zero": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "round_to_integer": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "scale_by_std": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
                    },
                    "scale_factor": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.CORE_NOISE_STRATEGY,
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
