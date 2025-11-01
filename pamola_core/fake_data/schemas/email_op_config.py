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
                    # --- GeneratorOperation-specific fields ---
                    "generator": {
                        "type": ["object", "null"],
                        "title": "Generator",
                        "description": "Generator instance or configuration for email generation."
                    },
                    "generator_params": {
                        "type": ["object", "null"],
                        "title": "Generator Parameters",
                        "description": "Parameters passed to the email generator."
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "enum": ["mapping", "prgn"],
                        "default": "prgn",
                        "title": "Consistency Mechanism",
                        "description": "Controls how consistent synthetic values are generated: 'mapping' for mapping store, 'prgn' for pseudo-random generation."
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "description": "Field name used as unique identifier for mapping consistency."
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "Path to store mapping between original and synthetic emails."
                    },
                    "mapping_store": {
                        "type": ["object", "null"],
                        "title": "Mapping Store",
                        "description": "Object for storing mapping between original and synthetic values."
                    },
                    "save_mapping": {
                        "type": "boolean", "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping between original and synthetic emails."
                    },
                    # --- FakeEmailOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing email addresses to process."
                    },
                    "domains": {
                        "type": ["array", "string", "null"],
                        "title": "Domains",
                        "description": "List of domains or path to domain dictionary for email generation."
                    },
                    "format": {
                        "type": ["string", "null"],
                        "title": "Email Format",
                        "enum": ["name_surname", "surname_name", "nickname", "existing_domain"],
                        "description": "Format for generating email addresses (e.g., 'first_last', 'nickname')."
                    },
                    "format_ratio": {
                        "type": ["object", "null"],
                        "title": "Format Ratio",
                        "description": "Ratio distribution for using different email formats."
                    },
                    "first_name_field": {
                        "type": ["string", "null"],
                        "title": "First Name Field",
                        "description": "Column name for first name, used in name-based email generation."
                    },
                    "last_name_field": {
                        "type": ["string", "null"],
                        "title": "Last Name Field",
                        "description": "Column name for last name, used in name-based email generation."
                    },
                    "full_name_field": {
                        "type": ["string", "null"],
                        "title": "Full Name Field",
                        "description": "Column name for full name, used in name-based email generation."
                    },
                    "name_format": {
                        "type": ["string", "null"],
                        "title": "Name Format",
                        "enum": ["FL", "LF", "FML", "LFM", "F", "L"],
                        "description": "Format of the full name (e.g., 'FL', 'LF')."
                    },
                    "validate_source": {
                        "type": "boolean", "default": True,
                        "title": "Validate Source",
                        "description": "Whether to validate input email addresses before generating synthetic ones."
                    },
                    "handle_invalid_email": {
                        "type": "string",
                        "enum": [
                            "generate_new",
                            "keep_empty",
                            "generate_with_default_domain",
                        ],
                        "default": "generate_new",
                        "title": "Handle Invalid Email",
                        "description": "Strategy for handling invalid emails: generate new, keep empty, or use default domain."
                    },
                    "nicknames_dict": {
                        "type": ["string", "null"],
                        "title": "Nicknames Dictionary",
                        "description": "Path to nickname mapping file for generating nickname-based emails."
                    },
                    "max_length": {
                        "type": "integer", "minimum": 1, "default": 254,
                        "title": "Max Length",
                        "description": "Maximum allowed length for generated email addresses."
                    },
                    # --- Generator fine-tuning fields ---
                    "separator_options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": [".", "_", "-", ""],
                        "title": "Separator Options",
                        "description": "List of separators to use between name parts in email addresses."
                    },
                    "number_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                        "title": "Number Suffix Probability",
                        "description": "Probability of adding a numeric suffix to the email local part."
                    },
                    "preserve_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Preserve Domain Ratio",
                        "description": "Probability of preserving the original domain in the generated email."
                    },
                    "business_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                        "title": "Business Domain Ratio",
                        "description": "Probability of using a business-related domain in the generated email."
                    },
                    "detailed_metrics": {
                        "type": "boolean", "default": False,
                        "title": "Detailed Metrics",
                        "description": "Whether to collect detailed metrics during email generation."
                    },
                    "max_retries": {
                        "type": "integer", "minimum": 0, "default": 3,
                        "title": "Max Retries",
                        "description": "Maximum number of retries for generating a valid synthetic email."
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "Key",
                        "description": "Key for encryption or PRGN consistency, if applicable."
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "Context Salt",
                        "description": "Additional context salt for PRGN to enhance uniqueness."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }