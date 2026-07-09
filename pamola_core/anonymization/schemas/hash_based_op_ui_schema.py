"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Hash-Based Pseudonymization UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of hash-based pseudonymization
configurations in PAMOLA.CORE.
- Contains only UI-specific metadata (x-component, x-group, x-depend-on)
- Defines component types, grouping, and conditional display logic for frontend forms
- Parallel structure to core schema but without validation rules
- Used by frontend to render dynamic configuration forms with proper UX controls

UI Component Types:
- Checkbox: Boolean toggles
- Select: Dropdown menus for enums/oneOf
- Input: Text input fields
- NumberPicker: Integer inputs
- FloatPicker: Float inputs
- Upload: File upload for salt file

Changelog:
1.0.0 - 2025-06-15 - Initial creation
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class HashBasedPseudonymizationUIConfig(OperationConfig):
    """
    UI configuration schema for HashBasedPseudonymization form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Hash-Based Pseudonymization UI Configuration",
        "description": "UI schema for hash-based pseudonymization operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    # --- Primary target ---
                    "field_name": {
                        "x-component": "Select",
                    },
                    # --- Core pseudonymization strategy ---
                    "algorithm": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "hash_output_format": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "output_length": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "output_prefix": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "output_suffix": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    # --- Salt & security settings ---
                    "salt_config": {
                        "x-component": CustomComponents.FIELD_SELECT_UPLOAD_FILE_INPUT,
                        "x-group": GroupName.SALT_AND_SECURITY_SETTINGS,
                    },
                    "salt_file": {
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.SALT_AND_SECURITY_SETTINGS,
                        "x-depend-on": {"salt_config.source": "file"},
                        "x-required-on": {"salt_config.source": "file"},
                    },
                    "use_pepper": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.SALT_AND_SECURITY_SETTINGS,
                    },
                    "pepper_length": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.SALT_AND_SECURITY_SETTINGS,
                        "x-depend-on": {"use_pepper": True},
                    },
                    # --- Compound identifier settings ---
                    "compound_mode": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.COMPOUND_IDENTIFIER_SETTINGS,
                    },
                    "additional_fields": {
                        "x-component": "Select",
                        "x-group": GroupName.COMPOUND_IDENTIFIER_SETTINGS,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-depend-on": {"compound_mode": True},
                        "x-required-on": {"compound_mode": True},
                    },
                    "compound_separator": {
                        "x-component": "Input",
                        "x-group": GroupName.COMPOUND_IDENTIFIER_SETTINGS,
                        "x-depend-on": {"compound_mode": True},
                    },
                    "compound_null_handling": {
                        "x-component": "Select",
                        "x-group": GroupName.COMPOUND_IDENTIFIER_SETTINGS,
                        "x-depend-on": {"compound_mode": True},
                    },
                    # --- Conditional logic ---
                    "condition_field": {
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                    },
                    "condition_operator": {
                        "x-component": "Select",
                        "x-group": GroupName.CONDITIONAL_LOGIC,
                        "x-depend-on": {"condition_field": "not_null"},
                        "x-custom-function": [CustomFunctions.UPDATE_CONDITION_OPERATOR],
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
                    # --- Risk-based processing & privacy ---
                    "quasi_identifiers": {
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-custom-function": [CustomFunctions.UPDATE_QUASI_FIELD_OPTIONS],
                        "x-ignore-depend-fields": True,
                    },
                    "ka_risk_field": {
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-custom-function": [CustomFunctions.UPDATE_INT64_FIELD_OPTIONS],
                    },
                    "risk_threshold": {
                        "x-component": "FloatPicker",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                        "x-depend-on": {"ka_risk_field": "not_null"},
                    },
                    "vulnerable_record_strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.RISK_BASED_PROCESSING_AND_PRIVACY,
                    },
                },
            },
        ],
    }
