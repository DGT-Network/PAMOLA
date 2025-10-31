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
        "title": "Fake Name Operation Configuration",
        "description": "Configuration options for generating fake names to replace real names in datasets.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- GeneratorOperation-specific fields ---
                    "generator": {
                        "type": ["object", "null"],
                        "title": "Generator Instance",
                        "description": "Custom generator instance for name synthesis (advanced usage)."
                    },
                    "generator_params": {
                        "type": ["object", "null"],
                        "title": "Generator Parameters",
                        "description": "Parameters passed to the name generator (language, gender, format, etc.)."
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "enum": ["mapping", "prgn"],
                        "default": "prgn",
                        "title": "Consistency Mechanism",
                        "description": "Mechanism to ensure consistent name generation: 'mapping' (fixed mapping) or 'prgn' (pseudo-random)."
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "description": "Column name used as unique identifier for mapping consistency (optional)."
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "Path to store or load mapping file for consistent name replacement."
                    },
                    "mapping_store": {
                        "type": ["object", "null"],
                        "title": "Mapping Store",
                        "description": "In-memory mapping store object for value-to-synthetic mapping (internal use)."
                    },
                    "save_mapping": {
                        "type": "boolean", "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping between original and synthetic names to disk."
                    },
                    # --- FakeNameOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing real names to be replaced."
                    },
                    "language": {
                        "type": "string", "default": "en",
                        "title": "Language",
                        "description": "Language code for name generation (e.g., 'en', 'vi', 'ru')."
                    },
                    "gender_field": {
                        "type": ["string", "null"],
                        "title": "Gender Field",
                        "description": "Column name providing gender information (if available)."
                    },
                    "gender_from_name": {
                        "type": "boolean", "default": False,
                        "title": "Infer Gender from Name",
                        "description": "Whether to infer gender from the original name value if gender field is not provided."
                    },
                    "format": {
                        "type": ["string", "null"],
                        "title": "Name Format",
                        "enum": ["FML", "FL", "LF", "LFM", "F_L", "L_F"],
                        "description": "Output format for synthetic names (e.g., 'FML', 'FL')."
                    },
                    "f_m_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Female-Male Ratio",
                        "description": "Ratio of female to male names in generated data (0 = all male, 1 = all female, 0.5 = balanced)."
                    },
                    "use_faker": {
                        "type": "boolean", "default": False,
                        "title": "Use Faker",
                        "description": "Whether to use the Faker library for name generation (for more variety)."
                    },
                    "case": {
                        "type": "string",
                        "enum": ["upper", "lower", "title"],
                        "default": "title",
                        "title": "Case Format",
                        "description": "Case formatting for output names: 'upper', 'lower', or 'title'."
                    },
                    "dictionaries": {
                        "type": ["object", "null"],
                        "title": "Custom Dictionaries",
                        "description": "Custom dictionaries for localized name generation (advanced usage)."
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "Key",
                        "description": "Key for encryption or PRGN consistency (if applicable)."
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
