"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Organization UI Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of fake organization generation configurations in PAMOLA.CORE.
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
1.0.0 - 2025-01-15 - Initial creation of fake organization UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class FakeOrganizationOperationUIConfig(OperationConfig):
    """
    UI configuration schema for FakeOrganizationOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Fake Organization Operation UI Configuration",
        "description": "UI schema for fake organization operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "organization_type": {
                        "x-component": "Select",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                    },
                    "region": {
                        "x-component": "Select",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                    },
                    "industry": {
                        "x-component": "Select",
                        "x-depend-on": {"organization_type": "industry"},
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                    },
                    "preserve_type": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                    },
                    "add_prefix_probability": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                    },
                    "add_suffix_probability": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                    },
                    "type_field": {
                        "x-component": "Input",
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "region_field": {
                        "x-component": "Select",
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "dictionaries": {
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "x-component": CustomComponents.UPLOAD,
                    },
                    "prefixes": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                    },
                    "suffixes": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                    },
                    "consistency_mechanism": {
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "id_field": {
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "key": {
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-component": "Input",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                    },
                    "context_salt": {
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-component": "Input",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                    },
                    "mapping_store_path": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                    },
                    "save_mapping": {
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-component": "Checkbox",
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                    },
                    "max_retries": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "detailed_metrics": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "collect_type_distribution": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                }
            },
        ],
    }
