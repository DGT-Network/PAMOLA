"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Cell Suppression UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of cell suppression configurations in PAMOLA.CORE.
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
- Select: Dropdown menus for strategies and methods
- Input: Text input for suppression values
- NumberPicker: Numeric inputs for thresholds
- FloatPicker: Floating-point numeric inputs

Changelog:
1.0.0 - 2025-11-18 - Initial creation of cell suppression UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class CellSuppressionUIConfig(OperationConfig):
    """
    UI configuration schema for CellSuppression form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Cell Suppression Operation UI Configuration",
        "description": "UI schema for configuring cell-level suppression operations.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "suppression_strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                    },
                    "suppression_value": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                        "x-depend-on": {"suppression_strategy": "constant"},
                        "x-required-on": {"suppression_strategy": "constant"},
                    },
                    "group_by_field": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                        "x-depend-on": {
                            "suppression_strategy": ["group_mean", "group_mode"]
                        },
                        "x-required-on": {
                            "suppression_strategy": ["group_mean", "group_mode"]
                        },
                    },
                    "min_group_size": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_SUPPRESSION_STRATEGY,
                        "x-depend-on": {"group_by_field": "not_null"},
                    },
                    "suppress_if": {
                        "x-component": "Select",
                        "x-group": GroupName.SUPPRESSION_RULES,
                    },
                    "outlier_method": {
                        "x-component": "Select",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "x-depend-on": {"suppress_if": "outlier"},
                    },
                    "outlier_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "x-depend-on": {"suppress_if": "outlier"},
                    },
                    "rare_threshold": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.SUPPRESSION_RULES,
                        "x-depend-on": {"suppress_if": "rare"},
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
                }
            },
        ],
    }
