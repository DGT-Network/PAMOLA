"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Email Core Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of fake email generation configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines email generation parameters, domain options, name field mapping, and consistency mechanisms
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Email format and generation style controls
- Domain and name field source configuration
- Consistency mechanism validation (mapping vs PRGN)
- Output validation and error handling rules

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake email core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class FakeEmailOperationConfig(OperationConfig):
    """
    Core configuration schema for FakeEmailOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Fake Email Operation Core Configuration",
        "description": "Core schema for fake email operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing email addresses to process.",
                    },
                    "format": {
                        "type": ["string", "null"],
                        "title": "Email Format",
                        "oneOf": [
                            {"const": "name_surname", "description": "Name Surname"},
                            {"const": "surname_name", "description": "Surname Name"},
                            {"const": "nickname", "description": "Nickname"},
                            {
                                "const": "existing_domain",
                                "description": "Existing Domain",
                            },
                        ],
                        "description": "Format for generating email addresses (e.g., 'first_last', 'nickname').",
                    },
                    "format_ratio": {
                        "type": ["object", "null"],
                        "title": "Format Ratio",
                        "description": "Ratio distribution for using different email formats.",
                    },
                    "separator_options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": [".", "_", "-", ""],
                        "oneOf": [
                            {"const": ".", "description": "Dot"},
                            {"const": "_", "description": "Underscore"},
                            {"const": "-", "description": "Dash"},
                            {"const": "", "description": "Blank"},
                        ],
                        "title": "Separator Options",
                        "description": "List of separators to use between name parts in email addresses.",
                    },
                    "number_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                        "title": "Number Suffix Probability",
                        "description": "Probability of adding a numeric suffix to the email local part.",
                    },
                    "preserve_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Preserve Domain Ratio",
                        "description": "Probability of preserving the original domain in the generated email.",
                    },
                    "business_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                        "title": "Business Domain Ratio",
                        "description": "Probability of using a business-related domain in the generated email.",
                    },
                    "max_length": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 254,
                        "title": "Max Email Length",
                        "description": "Maximum allowed length for generated email addresses.",
                    },
                    "first_name_field": {
                        "type": ["string", "null"],
                        "title": "First Name Field",
                        "description": "Column name for first name, used in name-based email generation.",
                    },
                    "last_name_field": {
                        "type": ["string", "null"],
                        "title": "Last Name Field",
                        "description": "Column name for last name, used in name-based email generation.",
                    },
                    "full_name_field": {
                        "type": ["string", "null"],
                        "title": "Full Name Field",
                        "description": "Column name for full name, used in name-based email generation.",
                    },
                    "name_format": {
                        "type": ["string", "null"],
                        "title": "Name Format",
                        "oneOf": [
                            {"const": "FL", "title": "First Last"},
                            {"const": "LF", "title": "Last First"},
                            {"const": "FML", "title": "First Middle Last"},
                            {"const": "LFM", "title": "Last First Middle"},
                            {"const": "F", "title": "First Name"},
                            {"const": "L", "title": "Last Name"},
                        ],
                        "description": "Format of the full name (e.g., 'FL', 'LF').",
                    },
                    "nicknames_dict": {
                        "type": ["string", "null"],
                        "title": "Nicknames Dictionary",
                        "description": "Path to nickname mapping file for generating nickname-based emails.",
                    },
                    "domains": {
                        "type": ["array", "string", "null"],
                        "title": "Domains",
                        "description": "List of domains or path to domain dictionary for email generation.",
                    },
                    "generator": {
                        "type": ["object", "null"],
                        "title": "Generator",
                        "description": "Generator instance or configuration for email generation.",
                    },
                    "generator_params": {
                        "type": ["object", "null"],
                        "title": "Generator Parameters",
                        "description": "Parameters passed to the email generator.",
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "oneOf": [
                            {"const": "mapping", "title": "mapping"},
                            {"const": "prgn", "title": "prgn"},
                        ],
                        "default": "prgn",
                        "title": "Consistency Method",
                        "description": "Controls how consistent synthetic values are generated: 'mapping' for mapping store, 'prgn' for pseudo-random generation.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Unique ID Field",
                        "description": "Field name used as unique identifier for mapping consistency.",
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "description": "Key for encryption or PRGN consistency, if applicable.",
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "PRGN Context Salt",
                        "description": "Additional context salt for PRGN to enhance uniqueness.",
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "Path to store mapping between original and synthetic emails.",
                    },
                    "mapping_store": {
                        "type": ["object", "null"],
                        "title": "Mapping Store",
                        "description": "Object for storing mapping between original and synthetic values.",
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping between original and synthetic emails.",
                    },
                    "validate_source": {
                        "type": "boolean",
                        "default": True,
                        "title": "Validate Source Emails",
                        "description": "Whether to validate input email addresses before generating synthetic ones.",
                    },
                    "handle_invalid_email": {
                        "type": "string",
                        "oneOf": [
                            {"const": "generate_new", "title": "Generate New"},
                            {"const": "keep_empty", "title": "Keep Empty"},
                            {
                                "const": "generate_with_default_domain",
                                "title": "Generate with Default Domain",
                            },
                        ],
                        "default": "generate_new",
                        "title": "Invalid Email Handling",
                        "description": "Strategy for handling invalid emails: generate new, keep empty, or use default domain.",
                    },
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "title": "Detailed Metrics",
                        "description": "Whether to collect detailed metrics during email generation.",
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "title": "Max Retries",
                        "description": "Maximum number of retries for generating a valid synthetic email.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
