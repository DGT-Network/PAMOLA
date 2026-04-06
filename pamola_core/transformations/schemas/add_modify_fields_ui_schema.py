"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Add or Modify Fields UI Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of add/modify fields configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- Input: Text input for field operations and lookup table configuration

Changelog:
1.0.0 - 2025-01-15 - Initial creation of add/modify fields UI schema
1.1.0 - 2025-11-11 - Updated with enhanced UI controls
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class AddOrModifyFieldsOperationUIConfig(OperationConfig):
    """
    UI configuration schema for AddOrModifyFieldsOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Add or Modify Fields Operation UI Configuration",
        "description": "UI schema for add or modify fields operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_operations": {
                        "x-component": CustomComponents.FIELD_DOUBLE_SELECT_INPUT_ADD_OR_MODIFY,
                        "x-group": GroupName.FIELD_OPERATIONS_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.INIT_FIELD_DOUBLE_SELECT],
                    },
                    "lookup_tables": {
                        "x-component": CustomComponents.FIELD_SELECT_UPLOAD_FILE_INPUT_ADD_OR_MODIFY,
                        "x-group": GroupName.LOOKUP_TABLE_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.INIT_UPLOAD],
                    },
                },
            },
        ],
    }
