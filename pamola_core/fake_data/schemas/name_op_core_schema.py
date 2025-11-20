"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Name Core Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of fake name generation configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines name generation parameters, language options, gender configuration, and consistency mechanisms
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Name format and case transformation controls
- Language and dictionary configuration
- Gender inference and ratio management
- Consistency mechanism validation (mapping vs PRGN)

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake name core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class FakeNameOperationConfig(OperationConfig):
    """
    Core configuration schema for FakeNameOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Fake Name Operation Core Configuration",
        "description": "Core schema for fake name operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing real names to be replaced.",
                    },
                    "language": {
                        "type": "string",
                        "default": "en",
                        "title": "Language",
                        "description": "Language code for name generation (e.g., 'en', 'vi', 'ru').",
                    },
                    "format": {
                        "type": ["string", "null"],
                        "title": "Name Format",
                        "description": "Output format for synthetic names (e.g., 'FML', 'FL').",
                        "oneOf": [
                            {"const": "FML", "description": "First Middle Last"},
                            {"const": "FL", "description": "First Last"},
                            {"const": "LF", "description": "Last First"},
                            {"const": "LFM", "description": "Last First Middle"},
                            {"const": "F_L", "description": "First_Last"},
                            {"const": "L_F", "description": "Last_First"},
                        ],
                    },
                    "case": {
                        "type": "string",
                        "oneOf": [
                            {"const": "upper", "description": "Upper"},
                            {"const": "lower", "description": "Lower"},
                            {"const": "title", "description": "Title"},
                        ],
                        "default": "title",
                        "title": "Name Case",
                        "description": "Case formatting for output names: 'upper', 'lower', or 'title'.",
                    },
                    "use_faker": {
                        "type": "boolean",
                        "default": False,
                        "title": "Use Faker Library",
                        "description": "Whether to use the Faker library for name generation (for more variety).",
                    },
                    "dictionaries": {
                        "type": ["object", "null"],
                        "title": "Custom Dictionaries",
                        "description": "Custom dictionaries for localized name generation (advanced usage).",
                    },
                    "gender_from_name": {
                        "type": "boolean",
                        "default": False,
                        "title": "Infer Gender from Name",
                        "description": "Whether to infer gender from the original name value if gender field is not provided.",
                    },
                    "gender_field": {
                        "type": ["string", "null"],
                        "title": "Gender Field",
                        "description": "Column name providing gender information (if available).",
                    },
                    "f_m_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Female-to-Male Ratio",
                        "description": "Ratio of female to male names in generated data (0 = all male, 1 = all female, 0.5 = balanced).",
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "oneOf": [
                            {"const": "mapping", "description": "mapping"},
                            {"const": "prgn", "description": "prgn"},
                        ],
                        "default": "prgn",
                        "title": "Consistency Method",
                        "description": "Mechanism to ensure consistent name generation: 'mapping' (fixed mapping) or 'prgn' (pseudo-random).",
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Unique ID Field",
                        "description": "Column name used as unique identifier for mapping consistency (optional).",
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "description": "Key for encryption or PRGN consistency (if applicable).",
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "PRGN Context Salt",
                        "description": "Additional context salt for PRGN to enhance uniqueness.",
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "Path to store or load mapping file for consistent name replacement.",
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping between original and synthetic names to disk.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
