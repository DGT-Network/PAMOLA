"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Impute Missing Values UI Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of impute missing values configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- Input: Text input for field strategies and invalid values configuration

Changelog:
1.0.0 - 2025-01-15 - Initial creation of impute missing values UI schema
1.1.0 - 2025-11-11 - Updated with enhanced UI controls
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class ImputeMissingValuesOperationUIConfig(OperationConfig):
    """
    UI configuration schema for ImputeMissingValuesOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Impute Missing Values Operation UI Configuration",
        "description": "UI schema for impute missing values operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_strategies": {
                        "x-component": CustomComponents.FIELD_IMPUTE_STRATEGY,
                        "x-group": GroupName.FIELD_STRATEGIES_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS, CustomFunctions.INIT_DATA_TYPE_OPTIONS],
                    },
                    "invalid_values": {
                        "x-component": CustomComponents.VALUE_GROUP_ARRAY,
                        "x-group": GroupName.INVALID_VALUES_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                },
            },
        ],
    }
