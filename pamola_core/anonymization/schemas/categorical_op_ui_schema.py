"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Generalization UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of categorical generalization configurations in PAMOLA.CORE.
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
- FloatPicker: Floating-point numeric inputs
- Upload: File upload component for dictionaries

Changelog:
1.0.0 - 2025-11-18 - Initial creation of categorical generalization UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class CategoricalGeneralizationUIConfig(OperationConfig):
    """
    UI configuration schema for CategoricalGeneralization form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Categorical Generalization UI Configuration",
        "description": "UI schema for categorical generalization operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "external_dictionary_path": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.HIERARCHY_SETTINGS,
                        "x-depend-on": {"strategy": "hierarchy"},
                        "x-required-on": {"strategy": "hierarchy"},
                    },
                    "dictionary_format": {
                        "x-component": "Select",
                        "x-group": GroupName.HIERARCHY_SETTINGS,
                        "x-depend-on": {"strategy": "hierarchy"},
                        "x-required-on": {"strategy": "hierarchy"},
                    },
                    "hierarchy_level": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.HIERARCHY_SETTINGS,
                        "x-depend-on": {"strategy": "hierarchy"},
                        "x-required-on": {"strategy": "hierarchy"},
                    },
                    "min_group_size": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": {
                            "strategy": ["merge_low_freq", "frequency_based"]
                        },
                        "x-required-on": {
                            "strategy": ["merge_low_freq", "frequency_based"]
                        },
                    },
                    "freq_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": {"strategy": "merge_low_freq"},
                        "x-required-on": {"strategy": "merge_low_freq"},
                    },
                    "max_categories": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": {"strategy": "frequency_based"},
                        "x-required-on": {"strategy": "frequency_based"},
                    },
                    "allow_unknown": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                    },
                    "unknown_value": {
                        "x-component": "Input",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "x-depend-on": {"allow_unknown": True},
                        "x-required-on": {"allow_unknown": True},
                    },
                    "group_rare_as": {
                        "x-component": "Select",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": {
                            "strategy": ["merge_low_freq", "frequency_based"]
                        },
                        "x-required-on": {
                            "strategy": ["merge_low_freq", "frequency_based"]
                        },
                    },
                    "rare_value_template": {
                        "x-component": "Input",
                        "x-group": GroupName.FREQUENCY_GROUPING_SETTINGS,
                        "x-depend-on": {"group_rare_as": ["CATEGORY_N", "RARE_N"]},
                        "x-required-on": {"group_rare_as": ["CATEGORY_N", "RARE_N"]},
                    },
                    "text_normalization": {
                        "x-component": "Select",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                    },
                    "case_sensitive": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                    },
                    "fuzzy_matching": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "x-depend-on": {"strategy": "hierarchy"},
                        "x-required-on": {"strategy": "hierarchy"},
                    },
                    "similarity_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.TEXT_VALUE_HANDLING,
                        "x-depend-on": {"fuzzy_matching": True},
                        "x-required-on": {"fuzzy_matching": True},
                    },
                    "privacy_check_enabled": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                    "min_acceptable_k": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-depend-on": {"privacy_check_enabled": True},
                        "x-required-on": {"privacy_check_enabled": True},
                    },
                    "max_acceptable_disclosure_risk": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-depend-on": {"privacy_check_enabled": True},
                        "x-required-on": {"privacy_check_enabled": True},
                    },
                    "quasi_identifiers": {
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_QUASI_FIELD_OPTIONS
                        ],
                        "x-ignore-depend-fields": True,
                        "x-depend-on": {"privacy_check_enabled": True},
                        "x-required-on": {"privacy_check_enabled": True},
                    },
                    "condition_field": {
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "condition_operator": {
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"condition_field": "not_null"},
                        "x-custom-function": [
                            CustomFunctions.UPDATE_CONDITION_OPERATOR
                        ],
                    },
                    "condition_values": {
                        "x-component": "Input",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {
                            "condition_field": "not_null",
                            "condition_operator": "not_null",
                        },
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_VALUES],
                    },
                    "ka_risk_field": {
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-custom-function": [
                            CustomFunctions.UPDATE_INT64_FIELD_OPTIONS
                        ],
                    },
                    "risk_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                    "vulnerable_record_strategy": {
                        "x-component": "Input",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                }
            },
        ],
    }
