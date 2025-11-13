"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Phone Config Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating fake phone generation operations in PAMOLA.CORE.
- Supports generator parameters, country/region, formats, ratios, and advanced metrics
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake phone config file
"""
from click import Group
from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FakePhoneOperationConfig(OperationConfig):
    """Configuration for FakePhoneOperation with BaseOperationConfig merged."""

    schema = {
        "title": "Fake Phone Operation Config",
        "description": "Configuration schema for generating fake phone numbers. Includes generator parameters, country/region, formats, ratios, and advanced metrics.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "FakePhoneOperationConfig Properties",
                "description": "Properties specific to fake phone number generation and operation control.",
                "properties": {
                    # --- FakePhoneOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to be replaced with a fake phone number."
                    },
                    "region": {
                        "type": ["string", "null"],
                        "title": "Default Region & Format",
                        "x-component": "Input",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "description": "Region for phone number style (e.g., 'global', 'US', 'VN')."
                    },
                    "default_country": {
                        "type": ["string", "null"],
                        "default": "us",
                        "title": "Fallback Country",
                        "x-component": "Select",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "description": "Sets the default region to determine the country code and phone number formatting rules."
                    },
                    "country_codes": {
                        "type": ["array", "null"],
                        "default": None,
                        "title": "Country Codes List",
                        "x-component": "Select",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "description": "Country codes to use for phone number generation (e.g., '+84', 'US')."
                    },
                    "country_code_field": {
                        "type": ["string", "null"],
                        "x-component": "Select",
                        "title": "Country Code Field",
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "description": "Field name containing country codes to guide phone number generation."
                    },
                    "operator_codes_dict": {
                        "type": ["object", "null"],
                        "default": None,
                        "title": "Operator Codes Dictionary",
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.REGIONAL_CONFIGURATION,
                        "description": "Dictionary mapping country codes to lists of operator codes for phone number generation."
                    },
                    "format": {
                        "type": ["string", "null"],
                        "default": None,
                        "x-component": "Select",
                        "title": "Output Format",
                        "x-group": GroupName.FORMATTING_RULES,
                        "description": "Phone number format template to use for generation."
                    },
                    "preserve_country_code": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Country Code",
                        "x-component": "Checkbox",
                        "x-group": GroupName.GENERATION_LOGIC,
                        "description": "Whether to preserve the original country code in the generated phone number."
                    },
                    "preserve_operator_code": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Operator Code",
                        "x-component": "Checkbox",
                        "x-group": GroupName.GENERATION_LOGIC,
                        "description": "Whether to preserve the original operator code in the generated phone number."
                    },

                    # --- GeneratorOperation / BaseOperation common fields ---
                    "consistency_mechanism": {
                        "type": "string",
                        "enum": ["mapping", "prgn"],
                        "default": "prgn",
                        "title": "Consistency Method",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "mapping", "description": "mapping"},
                            {"const": "prgn", "description": "prgn"}
                        ],
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "description": "Mechanism for consistent pseudonymization: 'mapping' for mapping store, 'prgn' for deterministic pseudo-random generation."
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Unique ID Field",
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "description": "Field name used as unique identifier for mapping consistency."
                    },
                    # --- Advanced behavior & metrics ---
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": { "consistency_mechanism": "prgn" },
                        "description": "Optional key for advanced configuration or encryption."
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "PRGN Context Salt",
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": { "consistency_mechanism": "prgn" },
                        "description": "Salt value for context-aware pseudonymization."
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": { "consistency_mechanism": "mapping" },
                        "description": "File path to persistently store mapping for consistent pseudonymization."
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "x-component": "Checkbox",
                        "x-depend-on": { "consistency_mechanism": "mapping" },
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "description": "Whether to save the mapping store to disk after operation."
                    },
                    "validate_source": {
                        "type": "boolean",
                        "default": True,
                        "title": "Validate Source",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Whether to validate source phone numbers before processing."
                    },
                    "handle_invalid_phone": {
                        "type": "string",   
                        "default": "generate",
                        "title": "Invalid Number Handling",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "generate_new", "description": "Generate New"},
                            {"const": "keep_empty", "description": "Keep Empty"},
                            {"const": "generate_with_default_country", "description": "Generate with Default Country"}
                        ],
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Strategy to handle invalid phone numbers in the source data."
                    },
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "title": "Detailed Metrics",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Collect detailed metrics during phone number generation."
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "title": "Max Retries",
                        "x-component": "Input",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Maximum number of retries for generating a valid phone number."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
