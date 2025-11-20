"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Email UI Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of fake email generation configurations in PAMOLA.CORE.
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
1.0.0 - 2025-01-15 - Initial creation of fake email UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class FakeEmailOperationUIConfig(OperationConfig):
    """
    UI configuration schema for FakeEmailOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Fake Email Operation UI Configuration",
        "description": "UI schema for configuring fake email generation operations.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "format": {
                        "x-component": "Select",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                    },
                    "format_ratio": {
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "x-component": "Input",
                        "x-depend-on": {"format": "null"},
                    },
                    "separator_options": {
                        "x-component": "Select",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                    },
                    "number_suffix_probability": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                    },
                    "preserve_domain_ratio": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                    },
                    "business_domain_ratio": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                    },
                    "max_length": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                    },
                    "first_name_field": {
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "last_name_field": {
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "full_name_field": {
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "name_format": {
                        "x-component": "Select",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-depend-on": {"full_name_field": "not_null"},
                    },
                    "nicknames_dict": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                    },
                    "domains": {
                        "x-component": "Select",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                    },
                    "generator": {
                        "x-component": "Object",
                    },
                    "generator_params": {
                        "x-component": "Object",
                    },
                    "consistency_mechanism": {
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "id_field": {
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "key": {
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "x-component": "Input",
                    },
                    "context_salt": {
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "x-component": "Input",
                    },
                    "mapping_store_path": {
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                    },
                    "mapping_store": {
                        "x-component": "Object",
                    },
                    "save_mapping": {
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                    },
                    "validate_source": {
                        "x-component": "Select",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "handle_invalid_email": {
                        "x-component": "Select",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                    },
                    "detailed_metrics": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "max_retries": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                }
            },
        ],
    }
