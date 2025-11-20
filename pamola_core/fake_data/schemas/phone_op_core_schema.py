"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Phone Core Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of fake phone generation configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines phone generation parameters, country/region codes, format options, and consistency mechanisms
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Regional and country code configuration
- Format and operator code controls
- Preservation options for country and operator codes
- Consistency mechanism validation (mapping vs PRGN)

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake phone core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class FakePhoneOperationConfig(OperationConfig):
    """
    Core configuration schema for FakePhoneOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Fake Phone Operation Core Configuration",
        "description": "Core schema for fake phone operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to be replaced with a fake phone number.",
                    },
                    "region": {
                        "type": ["string", "null"],
                        "title": "Default Region & Format",
                        "description": "Region for phone number style (e.g., 'global', 'US', 'VN').",
                    },
                    "default_country": {
                        "type": ["string", "null"],
                        "default": "us",
                        "title": "Fallback Country",
                        "description": "Sets the default region to determine the country code and phone number formatting rules.",
                    },
                    "country_codes": {
                        "type": ["array", "null"],
                        "default": None,
                        "title": "Country Codes List",
                        "description": "Country codes to use for phone number generation (e.g., '+84', 'US').",
                    },
                    "country_code_field": {
                        "type": ["string", "null"],
                        "title": "Country Code Field",
                        "description": "Field name containing country codes to guide phone number generation.",
                    },
                    "operator_codes_dict": {
                        "type": ["object", "null"],
                        "default": None,
                        "title": "Operator Codes Dictionary",
                        "description": "Dictionary mapping country codes to lists of operator codes for phone number generation.",
                    },
                    "format": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Output Format",
                        "description": "Phone number format template to use for generation.",
                    },
                    "preserve_country_code": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Country Code",
                        "description": "Whether to preserve the original country code in the generated phone number.",
                    },
                    "preserve_operator_code": {
                        "type": "boolean",
                        "default": False,
                        "title": "Preserve Operator Code",
                        "description": "Whether to preserve the original operator code in the generated phone number.",
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "default": "prgn",
                        "title": "Consistency Method",
                        "oneOf": [
                            {"const": "mapping", "description": "mapping"},
                            {"const": "prgn", "description": "prgn"},
                        ],
                        "description": "Mechanism for consistent pseudonymization: 'mapping' for mapping store, 'prgn' for deterministic pseudo-random generation.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Unique ID Field",
                        "description": "Field name used as unique identifier for mapping consistency.",
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "description": "Optional key for advanced configuration or encryption.",
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "PRGN Context Salt",
                        "description": "Salt value for context-aware pseudonymization.",
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "File path to persistently store mapping for consistent pseudonymization.",
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping store to disk after operation.",
                    },
                    "validate_source": {
                        "type": "boolean",
                        "default": True,
                        "title": "Validate Source",
                        "description": "Whether to validate source phone numbers before processing.",
                    },
                    "handle_invalid_phone": {
                        "type": "string",
                        "default": "generate",
                        "title": "Invalid Number Handling",
                        "oneOf": [
                            {"const": "generate_new", "description": "Generate New"},
                            {"const": "keep_empty", "description": "Keep Empty"},
                            {
                                "const": "generate_with_default_country",
                                "description": "Generate with Default Country",
                            },
                        ],
                        "description": "Strategy to handle invalid phone numbers in the source data.",
                    },
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "title": "Detailed Metrics",
                        "description": "Collect detailed metrics during phone number generation.",
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "title": "Max Retries",
                        "description": "Maximum number of retries for generating a valid phone number.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
