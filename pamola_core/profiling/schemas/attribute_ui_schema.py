"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Data Attribute Profiler UI Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of attribute profiling configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group)
- Defines component types, grouping, and display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

Key Features:
- Component mappings for each field type
- Logical grouping via x-group for organized form layout
- No business logic validation - purely presentational metadata

UI Component Types:
- Select: Dropdown menus for enums
- NumberPicker: Numeric inputs with validation
- Upload: File upload components

Changelog:
1.0.0 - 2025-01-15 - Initial creation of data attribute profiler UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class DataAttributeProfilerOperationUIConfig(OperationConfig):
    """
    UI configuration schema for DataAttributeProfilerOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Attribute Profiler Operation UI Configuration",
        "description": "UI schema for attribute profiling operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "language": {
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "sample_size": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "max_columns": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "dictionary_path": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.DICTIONARY_CONFIGURATION,
                    },
                }
            },
        ],
    }
