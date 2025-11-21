"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Full Masking UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of full masking configurations in PAMOLA.CORE.
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
- ArrayItems: Array field component for format patterns

Changelog:
1.0.0 - 2025-11-18 - Initial creation of full masking UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class FullMaskingUIConfig(OperationConfig):
    """
    UI configuration schema for FullMasking form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Full Masking Configuration UI Schema",
        "description": "UI schema for full masking operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "random_mask": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.CORE_MASKING_RULES,
                    },
                    "mask_char": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_MASKING_RULES,
                        "x-depend-on": {"random_mask": False},
                        "x-required-on": {"random_mask": False},
                    },
                    "preserve_length": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.CORE_MASKING_RULES,
                    },
                    "fixed_length": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_MASKING_RULES,
                        "x-depend-on": {"preserve_length": False},
                        "x-required-on": {"preserve_length": False},
                    },
                    "mask_char_pool": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_MASKING_RULES,
                        "x-depend-on": {"random_mask": True},
                        "x-required-on": {"random_mask": True},
                    },
                    "preserve_format": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "format_patterns": {
                        "x-component": CustomComponents.FORMAT_PATTERNS,
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                        "x-depend-on": {"preserve_format": True},
                        "x-required-on": {"preserve_format": True},
                    },
                    "numeric_output": {
                        "x-component": "Select",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "date_format": {
                        "x-component": CustomComponents.DATE_FORMAT,
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
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
                    "ka_risk_field": {
                        "x-component": "Select",
                    },
                    "risk_threshold": {
                        "x-component": "FloatPicker",
                    },
                    "vulnerable_record_strategy": {
                        "x-component": "Input",
                    },
                },
            },
        ],
    }
