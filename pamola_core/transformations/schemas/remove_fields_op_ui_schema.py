"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Remove Fields UI Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of remove fields configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-custom-function)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- Custom functions for dynamic field options
- No business logic validation - purely presentational metadata

UI Component Types:
- Input: Text input for regex pattern
- Select: Multi-select dropdown for field selection

Changelog:
1.0.0 - 2025-01-15 - Initial creation of remove fields UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class RemoveFieldsOperationUIConfig(OperationConfig):
    """
    UI configuration schema for RemoveFieldsOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Remove Fields Operation UI Configuration",
        "description": "UI schema for remove fields operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "x-component": "Input",
                        "x-group": GroupName.FIELD_REMOVAL,
                    },
                    "fields_to_remove": {
                        "x-component": "Select",
                        "x-group": GroupName.FIELD_REMOVAL,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                }
            },
        ],
    }
