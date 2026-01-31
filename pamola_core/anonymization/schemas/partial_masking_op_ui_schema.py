"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking UI Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-18
License:       BSD 3-Clause

Description:
UI metadata schema for Formily-based form rendering of partial masking configurations in PAMOLA.CORE.
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
- DependSelect: Custom dependent dropdown component

Changelog:
1.0.0 - 2025-11-18 - Initial creation of partial masking UI schema
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.common.enum.mask_strategy_enum import MaskStrategyEnum
from pamola_core.anonymization.commons.masking_presets import MaskingType
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_ui_schema import BaseOperationUIConfig


class PartialMaskingUIConfig(OperationConfig):
    """
    UI configuration schema for PartialMasking form rendering.

    Defines Formily component metadata and display logic.
    Used by frontend to build dynamic configuration forms.
    """

    schema = {
        "type": "object",
        "title": "Partial Masking Configuration UI Schema",
        "description": "UI schema for partial masking operation configuration.",
        "allOf": [
            BaseOperationUIConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "x-component": "Select",
                    },
                    "mask_char": {
                        "x-component": "Input",
                        "x-group": GroupName.MASK_APPEARANCE,
                        "x-depend-on": {"random_mask": False},
                        "x-required-on": {"random_mask": False},
                    },
                    "mask_strategy": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                    },
                    "mask_percentage": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {"mask_strategy": MaskStrategyEnum.RANDOM.value},
                    },
                    "unmasked_prefix": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": [
                                MaskStrategyEnum.FIXED.value,
                                MaskStrategyEnum.WORDS.value,
                            ]
                        },
                    },
                    "unmasked_suffix": {
                        "x-component": "NumberPicker",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": [
                                MaskStrategyEnum.FIXED.value,
                                MaskStrategyEnum.WORDS.value,
                            ]
                        },
                    },
                    "unmasked_positions": {
                        "x-component": "Select",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {"mask_strategy": MaskStrategyEnum.FIXED.value},
                    },
                    "pattern_type": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                        "x-depend-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                        "x-required-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                    },
                    "mask_pattern": {
                        "x-component": "Input",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                        "x-required-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                    },
                    "preserve_pattern": {
                        "x-component": "Input",
                        "x-group": GroupName.MASKING_RULES,
                        "x-depend-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                        "x-required-on": {
                            "mask_strategy": MaskStrategyEnum.PATTERN.value
                        },
                    },
                    "preserve_separators": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "preserve_word_boundaries": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                        "x-depend-on": {"mask_strategy": MaskStrategyEnum.WORDS.value},
                    },
                    "case_sensitive": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.FORMATTING_AND_STRUCTURE,
                    },
                    "random_mask": {
                        "x-component": "Checkbox",
                        "x-group": GroupName.MASK_APPEARANCE,
                    },
                    "mask_char_pool": {
                        "x-component": "Input",
                        "x-group": GroupName.MASK_APPEARANCE,
                        "x-depend-on": {"random_mask": True},
                        "x-required-on": {"random_mask": True},
                    },
                    "preset_type": {
                        "x-component": "Select",
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                    },
                    "preset_name": {
                        "x-component": CustomComponents.DEPEND_SELECT,
                        "x-group": GroupName.CORE_MASKING_STRATEGY,
                        "x-depend-map": {
                            "depend_on": "preset_type",
                            "options_map": {
                                MaskingType.EMAIL.value: [
                                    {"label": "FULL_DOMAIN", "value": "FULL_DOMAIN"},
                                    {"label": "DOMAIN_ONLY", "value": "DOMAIN_ONLY"},
                                    {
                                        "label": "PARTIAL_DOMAIN",
                                        "value": "PARTIAL_DOMAIN",
                                    },
                                    {
                                        "label": "PRIVACY_FOCUSED",
                                        "value": "PRIVACY_FOCUSED",
                                    },
                                    {
                                        "label": "GDPR_COMPLIANT",
                                        "value": "GDPR_COMPLIANT",
                                    },
                                    {
                                        "label": "UTILITY_BALANCED",
                                        "value": "UTILITY_BALANCED",
                                    },
                                    {
                                        "label": "MINIMAL_EXPOSURE",
                                        "value": "MINIMAL_EXPOSURE",
                                    },
                                ],
                                MaskingType.PHONE.value: [
                                    {"label": "US_STANDARD", "value": "US_STANDARD"},
                                    {"label": "US_FORMATTED", "value": "US_FORMATTED"},
                                    {
                                        "label": "INTERNATIONAL",
                                        "value": "INTERNATIONAL",
                                    },
                                    {
                                        "label": "LAST_FOUR_ONLY",
                                        "value": "LAST_FOUR_ONLY",
                                    },
                                    {
                                        "label": "AREA_CODE_ONLY",
                                        "value": "AREA_CODE_ONLY",
                                    },
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                ],
                                MaskingType.CREDIT_CARD.value: [
                                    {
                                        "label": "PCI_COMPLIANT",
                                        "value": "PCI_COMPLIANT",
                                    },
                                    {"label": "STRICT", "value": "STRICT"},
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                    {"label": "NUMERIC_ONLY", "value": "NUMERIC_ONLY"},
                                    {
                                        "label": "FIRST_LAST_FOUR",
                                        "value": "FIRST_LAST_FOUR",
                                    },
                                ],
                                MaskingType.SSN.value: [
                                    {"label": "LAST_FOUR", "value": "LAST_FOUR"},
                                    {"label": "FIRST_THREE", "value": "FIRST_THREE"},
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                    {"label": "NUMERIC_MASK", "value": "NUMERIC_MASK"},
                                    {
                                        "label": "AREA_NUMBER_ONLY",
                                        "value": "AREA_NUMBER_ONLY",
                                    },
                                ],
                                MaskingType.IP_ADDRESS.value: [
                                    {"label": "SUBNET_MASK", "value": "SUBNET_MASK"},
                                    {"label": "NETWORK_ONLY", "value": "NETWORK_ONLY"},
                                    {"label": "FULL_MASK", "value": "FULL_MASK"},
                                    {"label": "ZERO_MASK", "value": "ZERO_MASK"},
                                    {
                                        "label": "PRIVATE_NETWORK",
                                        "value": "PRIVATE_NETWORK",
                                    },
                                ],
                                MaskingType.HEALTHCARE.value: [
                                    {
                                        "label": "MEDICAL_RECORD",
                                        "value": "MEDICAL_RECORD",
                                    },
                                    {"label": "PATIENT_ID", "value": "PATIENT_ID"},
                                    {"label": "NPI_NUMBER", "value": "NPI_NUMBER"},
                                    {"label": "DEA_NUMBER", "value": "DEA_NUMBER"},
                                ],
                                MaskingType.FINANCIAL.value: [
                                    {
                                        "label": "ACCOUNT_NUMBER",
                                        "value": "ACCOUNT_NUMBER",
                                    },
                                    {
                                        "label": "ROUTING_NUMBER",
                                        "value": "ROUTING_NUMBER",
                                    },
                                    {
                                        "label": "BANK_STANDARD",
                                        "value": "BANK_STANDARD",
                                    },
                                    {"label": "SWIFT_CODE", "value": "SWIFT_CODE"},
                                    {"label": "IBAN", "value": "IBAN"},
                                    {"label": "CREDIT_LIMIT", "value": "CREDIT_LIMIT"},
                                    {"label": "LOAN_NUMBER", "value": "LOAN_NUMBER"},
                                ],
                                MaskingType.DATE_ISO.value: [
                                    {"label": "MASK_DAY", "value": "MASK_DAY"},
                                    {"label": "MASK_MONTH", "value": "MASK_MONTH"},
                                    {"label": "MASK_YEAR", "value": "MASK_YEAR"},
                                    {"label": "MASK_FULL", "value": "MASK_FULL"},
                                ],
                            },
                        },
                    },
                    "consistency_fields": {
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
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
                    },
                    "risk_threshold": {
                        "x-component": "FloatPicker",
                    },
                    "vulnerable_record_strategy": {
                        "x-component": "Input",
                    },
                }
            },
        ],
    }
