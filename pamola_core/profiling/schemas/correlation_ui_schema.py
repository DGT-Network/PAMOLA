"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Profiler UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of correlation profiling configurations in PAMOLA.CORE.
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
- Select: Dropdown menus for enums and field selection

Changelog:
1.0.0 - 2025-01-15 - Initial creation of correlation profiler UI schema
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class CorrelationOperationUIConfig(OperationConfig):
    """
    UI configuration schema for CorrelationOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Correlation Operation UI Configuration",
        "description": "UI schema for correlation profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field1": {
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.FIELD_SETTINGS,
                    },
                    "field2": {
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.FIELD_SETTINGS,
                    },
                    "method": {
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "null_handling": {
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "mvf_parser": {
                        "x-component": "Input",
                    },
                }
            },
        ],
    }
