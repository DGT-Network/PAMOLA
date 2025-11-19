"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Email Config Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating fake email generation operations in PAMOLA.CORE.
- Supports generator parameters, domain and format options, name fields, validation, and fine-tuning
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake email config file
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FakeEmailOperationConfig(OperationConfig):
    """Configuration for FakeEmailOperation with BaseOperationConfig merged."""

    schema = {
        "title": "Fake Email Operation Config",
        "description": "Configuration schema for FakeEmailOperation. Controls how synthetic email addresses are generated, including domain and format options, name field mapping, validation, consistency mechanisms, and fine-tuning for realistic and statistically consistent email anonymization.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- FakeEmailOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing email addresses to process.",
                    },
                    "format": {
                        "type": ["string", "null"],
                        "title": "Email Format",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "name_surname", "description": "Name Surname"},
                            {"const": "surname_name", "description": "Surname Name"},
                            {"const": "nickname", "description": "Nickname"},
                            {
                                "const": "existing_domain",
                                "description": "Existing Domain",
                            },
                        ],
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "description": "Format for generating email addresses (e.g., 'first_last', 'nickname').",
                    },
                    "format_ratio": {
                        "type": ["object", "null"],
                        "title": "Format Ratio",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "x-component": "Input",
                        "x-depend-on": {"format": "null"},
                        "description": "Ratio distribution for using different email formats.",
                    },
                    # --- Generator fine-tuning fields ---
                    "separator_options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": [".", "_", "-", ""],
                        "x-component": "Select",
                        "oneOf": [
                            {"const": ".", "description": "Dot"},
                            {"const": "_", "description": "Underscore"},
                            {"const": "-", "description": "Dash"},
                            {"const": "", "description": "Blank"},
                        ],
                        "title": "Separator Options",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "description": "List of separators to use between name parts in email addresses.",
                    },
                    "number_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                        "x-component": "NumberPicker",
                        "title": "Number Suffix Probability",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "description": "Probability of adding a numeric suffix to the email local part.",
                    },
                    "preserve_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Preserve Domain Ratio",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "description": "Probability of preserving the original domain in the generated email.",
                    },
                    "business_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                        "title": "Business Domain Ratio",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "description": "Probability of using a business-related domain in the generated email.",
                    },
                    "max_length": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 254,
                        "title": "Max Email Length",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.EMAIL_GENERATION_STYLE,
                        "description": "Maximum allowed length for generated email addresses.",
                    },
                    "first_name_field": {
                        "type": ["string", "null"],
                        "title": "First Name Field",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "description": "Column name for first name, used in name-based email generation.",
                    },
                    "last_name_field": {
                        "type": ["string", "null"],
                        "title": "Last Name Field",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "description": "Column name for last name, used in name-based email generation.",
                    },
                    "full_name_field": {
                        "type": ["string", "null"],
                        "title": "Full Name Field",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
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
                        "x-component": "Select",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "x-depend-on": {"full_name_field": "not_null"},
                        "description": "Format of the full name (e.g., 'FL', 'LF').",
                    },
                    "nicknames_dict": {
                        "type": ["string", "null"],
                        "title": "Nicknames Dictionary",
                        "x-component": CustomComponents.UPLOAD,
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "description": "Path to nickname mapping file for generating nickname-based emails.",
                    },
                    "domains": {
                        "type": ["array", "string", "null"],
                        "title": "Domains",
                        "x-component": "Select",
                        "x-group": GroupName.DATA_SOURCES_FOR_GENERATION,
                        "description": "List of domains or path to domain dictionary for email generation.",
                    },
                    # --- GeneratorOperation-specific fields ---
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
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "title": "Consistency Method",
                        "description": "Controls how consistent synthetic values are generated: 'mapping' for mapping store, 'prgn' for pseudo-random generation.",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Unique ID Field",
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "description": "Field name used as unique identifier for mapping consistency.",
                    },
                    # --- Generator fine-tuning fields ---
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "x-component": "Input",
                        "description": "Key for encryption or PRGN consistency, if applicable.",
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "PRGN Context Salt",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "x-component": "Input",
                        "description": "Additional context salt for PRGN to enhance uniqueness.",
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
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
                        "x-component": "Select",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "description": "Whether to save the mapping between original and synthetic emails.",
                    },
                    "validate_source": {
                        "type": "boolean",
                        "default": True,
                        "x-component": "Select",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
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
                        "x-component": "Select",
                        "x-group": GroupName.OUTPUT_FORMATTING_CONSTRAINTS,
                        "default": "generate_new",
                        "title": "Invalid Email Handling",
                        "description": "Strategy for handling invalid emails: generate new, keep empty, or use default domain.",
                    },
                    # --- Generator fine-tuning fields ---
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "title": "Detailed Metrics",
                        "description": "Whether to collect detailed metrics during email generation.",
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "title": "Max Retries",
                        "description": "Maximum number of retries for generating a valid synthetic email.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
