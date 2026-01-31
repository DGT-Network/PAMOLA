"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Generalization UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of numeric generalization configurations in PAMOLA.CORE.
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
- ArrayItems: Array field editors

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric generalization UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class NumericGeneralizationUIConfig(OperationConfig):
    """
    UI configuration schema for NumericGeneralization form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Numeric Generalization Configuration UI Schema",
        "description": "UI schema for numeric generalization operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "binning_method": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "binning"},
                        "x-required-on": {"strategy": "binning"},
                    },
                    "bin_count": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "binning"},
                        "x-required-on": {"strategy": "binning"},
                    },
                    "precision": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "rounding"},
                        "x-required-on": {"strategy": "rounding"},
                    },
                    "range_limits": {
                        "x-component": "ArrayItems",
                        "x-items": {
                            "x-items-title": ["Min", "Max"],
                            "x-component": "NumberPicker",
                        },
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                        "x-depend-on": {"strategy": "range"},
                        "x-required-on": {"strategy": "range"},
                    },
                    "quasi_identifiers": {
                        "x-component": "Select",
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
                        "x-component": "NumberPicker",
                        "x-depend-on": {"ka_risk_field": "not_null"},
                    },
                    "vulnerable_record_strategy": {
                        "x-component": "Select",
                    },
                },
            },
        ],
    }
