"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Clean Invalid Values UI Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of clean invalid values configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- Input: Text input for constraint dictionaries, file paths, and replacement strategies

Changelog:
1.0.0 - 2025-01-15 - Initial creation of clean invalid values UI schema
1.1.0 - 2025-11-11 - Updated with enhanced UI controls
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class CleanInvalidValuesOperationUIConfig(OperationConfig):
    """
    UI configuration schema for CleanInvalidValuesOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Clean Invalid Values Operation UI Configuration",
        "description": "UI schema for clean invalid values operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_constraints": {
                        "x-component": CustomComponents.FIELD_DOUBLE_SELECT_INPUT_CLEAN_INVALID,
                        "x-group": GroupName.FIELD_CONSTRAINTS_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.INIT_FIELD_DOUBLE_SELECT],
                    },
                    "whitelist_path": {
                        "x-component": CustomComponents.FIELD_SELECT_UPLOAD_FILE_INPUT_CLEAN_INVALID,
                        "x-group": GroupName.WHITELIST_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS, CustomFunctions.INIT_UPLOAD],
                    },
                    "blacklist_path": {
                        "x-component": CustomComponents.FIELD_SELECT_UPLOAD_FILE_INPUT_CLEAN_INVALID,
                        "x-group": GroupName.BLACKLIST_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS, CustomFunctions.INIT_UPLOAD],
                    },
                    "null_replacement": {
                        "x-component": CustomComponents.FIELD_NULL_REPLACEMENT_INPUT,
                        "x-group": GroupName.NULL_REPLACEMENT_CONFIGURATION,
                         "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS, CustomFunctions.INIT_FIELD_STRATEGY_OPTIONS],
                    },
                },
            },
        ],
    }
