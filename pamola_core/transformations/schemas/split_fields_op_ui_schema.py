"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split Fields UI Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of split fields configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on, x-custom-function)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Conditional field visibility using x-depend-on directives
- Custom functions for dynamic field options
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menu for ID field selection
- Checkbox: Boolean toggle for ID field inclusion
- FieldGroupArray: Custom component for field group configuration

Changelog:
1.0.0 - 2025-01-15 - Initial creation of split fields UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class SplitFieldsOperationUIConfig(OperationConfig):
    """
    UI configuration schema for SplitFieldsOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Split Fields Operation UI Configuration",
        "description": "UI schema for Split Fields operation configuration form.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "id_field": {
                        "x-component": "Select",
                        "x-group": GroupName.INPUT_SETTINGS,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "include_id_field": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.INPUT_SETTINGS,
                        "x-depend-on": {"id_field": "not_null"},
                    },
                    "field_groups": {
                        "x-component": "FieldGroupArray",
                        "x-group": GroupName.FIELD_GROUPS_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                },
            },
        ],
    }
