"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Consistent Mapping Pseudonymization UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of consistent mapping
pseudonymization configurations in PAMOLA.CORE.
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

Changelog:
1.0.0 - 2025-06-15 - Initial creation
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class ConsistentMappingPseudonymizationUIConfig(OperationConfig):
    """
    UI configuration schema for ConsistentMappingPseudonymization form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Consistent Mapping Pseudonymization UI Configuration",
        "description": "UI schema for consistent mapping pseudonymization operation configuration.",
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
                    "pseudonym_type": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "pseudonym_prefix": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "pseudonym_suffix": {
                        "x-component": "Input",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                    },
                    "pseudonym_length": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_PSEUDONYMIZATION_STRATEGY,
                        "x-depend-on": {"pseudonym_type": "random_string"},
                        "x-required-on": {"pseudonym_type": "random_string"},
                    },
                    # --- Mapping storage settings ---
                    "mapping_encryption_key": {
                        "x-component": "Input",
                        "x-group": GroupName.MAPPING_STORAGE_SETTINGS,
                    },
                    "mapping_file": {
                        "x-component": "Input",
                        "x-group": GroupName.MAPPING_STORAGE_SETTINGS,
                    },
                    "mapping_format": {
                        "x-component": "Select",
                        "x-group": GroupName.MAPPING_STORAGE_SETTINGS,
                    },
                    "create_if_not_exists": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.MAPPING_STORAGE_SETTINGS,
                    },
                    "backup_on_update": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.MAPPING_STORAGE_SETTINGS,
                    },
                    "persist_frequency": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MAPPING_STORAGE_SETTINGS,
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
