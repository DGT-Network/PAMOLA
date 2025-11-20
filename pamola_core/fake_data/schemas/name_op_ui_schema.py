"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Name UI Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of fake name generation configurations in PAMOLA.CORE.
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
- Upload: File upload components

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake name UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class FakeNameOperationUIConfig(OperationConfig):
    """
    UI configuration schema for FakeNameOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Fake Name Operation UI Configuration",
        "description": "UI schema for configuring fake name generation operations.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "language": {
                        "x-component": "Select",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "format": {
                        "x-component": "Select",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "case": {
                        "x-component": "Select",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "use_faker": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "dictionaries": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "gender_from_name": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "gender_field": {
                        "x-component": "Select",
                        "x-depend-on": {"gender_from_name": False},
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "f_m_ratio": {
                        "x-component": "NumberPicker",
                        "x-depend-on": {
                            "gender_field": "null",
                            "gender_from_name": False,
                        },
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "consistency_mechanism": {
                        "x-component": "Select",
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "id_field": {
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "key": {
                        "x-component": "Input",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "context_salt": {
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "mapping_store_path": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "save_mapping": {
                        "x-component": "Checkbox",
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                }
            },
        ],
    }
