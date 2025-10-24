"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Name Config Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating fake name generation operations in PAMOLA.CORE.
- Supports generator parameters, language, gender, formatting, and dictionary options
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake name config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FakeNameOperationConfig(OperationConfig):
    """Configuration for FakeNameOperation with BaseOperationConfig merged."""

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
                    # --- FakeNameOperation-specific fields ---
                    "field_name": {"type": "string"},
                    "language": {"type": "string", "default": "en"},
                    "gender_field": {"type": ["string", "null"]},
                    "gender_from_name": {"type": "boolean", "default": False},
                    "format": {"type": ["string", "null"]},
                    "f_m_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                    },
                    "use_faker": {"type": "boolean", "default": False},
                    "case": {
                        "type": "string",
                        "enum": ["upper", "lower", "title"],
                        "default": "title",
                    },
                    "dictionaries": {"type": ["object", "null"]},
                    "key": {"type": ["string", "null"]},
                    "context_salt": {"type": ["string", "null"]},
                },
                "required": ["field_name"],
            },
        ],
    }
