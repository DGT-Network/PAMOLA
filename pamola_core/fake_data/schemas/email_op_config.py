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
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- GeneratorOperation-specific fields ---
                    "generator": {"type": ["object", "null"]},
                    "generator_params": {"type": ["object", "null"]},
                    "consistency_mechanism": {
                        "type": "string",
                        "enum": ["mapping", "prgn"],
                        "default": "prgn",
                    },
                    "id_field": {"type": ["string", "null"]},
                    "mapping_store_path": {"type": ["string", "null"]},
                    "mapping_store": {"type": ["object", "null"]},
                    "save_mapping": {"type": "boolean", "default": False},
                    "output_field_name": {"type": ["string", "null"]},
                    # --- FakeEmailOperation-specific fields ---
                    "field_name": {"type": "string"},
                    "domains": {"type": ["array", "string", "null"]},
                    "format": {"type": ["string", "null"]},
                    "format_ratio": {"type": ["object", "null"]},
                    "first_name_field": {"type": ["string", "null"]},
                    "last_name_field": {"type": ["string", "null"]},
                    "full_name_field": {"type": ["string", "null"]},
                    "name_format": {"type": ["string", "null"]},
                    "validate_source": {"type": "boolean", "default": True},
                    "handle_invalid_email": {
                        "type": "string",
                        "enum": [
                            "generate_new",
                            "keep_empty",
                            "generate_with_default_domain",
                        ],
                        "default": "generate_new",
                    },
                    "nicknames_dict": {"type": ["string", "null"]},
                    "max_length": {"type": "integer", "minimum": 1, "default": 254},
                    # --- Generator fine-tuning fields ---
                    "separator_options": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": [".", "_", "-", ""],
                    },
                    "number_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.4,
                    },
                    "preserve_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                    },
                    "business_domain_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.2,
                    },
                    "detailed_metrics": {"type": "boolean", "default": False},
                    "max_retries": {"type": "integer", "minimum": 0, "default": 3},
                    "key": {"type": ["string", "null"]},
                    "context_salt": {"type": ["string", "null"]},
                },
                "required": ["field_name"],
            },
        ],
    }