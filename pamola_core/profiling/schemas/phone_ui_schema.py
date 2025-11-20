"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Phone Profiler UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of phone profiling configurations in PAMOLA.CORE.
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
- Select: Dropdown menus for field and country code selection
- NumberPicker: Integer numeric inputs with validation
- Upload: File upload for pattern CSV

Changelog:
1.0.0 - 2025-01-15 - Initial creation of phone profiler UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class PhoneOperationUIConfig(OperationConfig):
    """
    UI configuration schema for PhoneOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Phone Operation UI Configuration",
        "description": "UI schema for phone profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "min_frequency": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                    "country_codes": {
                        "x-component": "Select",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_DEFAULT_COUNTRY_OPTIONS
                        ],
                    },
                    "patterns_csv": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                }
            },
        ],
    }
