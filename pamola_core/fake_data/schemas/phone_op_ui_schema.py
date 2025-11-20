"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Phone UI Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of fake phone generation configurations in PAMOLA.CORE.
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
- Upload: File upload components

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake phone UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class FakePhoneOperationUIConfig(OperationConfig):
    """
    UI configuration schema for FakePhoneOperation form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Fake Phone Operation UI Configuration",
        "description": "UI schema for configuring fake phone generation operations.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {"x-component": "Select"},
                    "region": {
                        "x-component": "Input",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                    },
                    "default_country": {
                        "x-component": "Select",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_DEFAULT_COUNTRY_OPTIONS
                        ],
                    },
                    "country_codes": {
                        "x-component": "Select",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_DEFAULT_COUNTRY_OPTIONS
                        ],
                    },
                    "country_code_field": {
                        "x-component": "Select",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "operator_codes_dict": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                    },
                    "format": {
                        "x-component": "Select",
                        "x-group": GroupName.FORMATTING_RULES,
                        "x-custom-function": [CustomFunctions.UPDATE_FAKE_PHONE_FORMAT],
                    },
                    "preserve_country_code": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.GENERATION_LOGIC,
                    },
                    "preserve_operator_code": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.GENERATION_LOGIC,
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
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                    },
                    "context_salt": {
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                    },
                    "mapping_store_path": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                    },
                    "save_mapping": {
                        "x-component": "Checkbox",
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "validate_source": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "handle_invalid_phone": {
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "detailed_metrics": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "max_retries": {
                        "x-component": "Input",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                }
            },
        ],
    }
