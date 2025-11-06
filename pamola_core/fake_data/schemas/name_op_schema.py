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

from pamola_core.common.enum.form_groups import GroupName
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
                    # --- FakeNameOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "x-component": "Select",
                        "description": "Name of the column containing real names to be replaced.",
                    },
                    "language": {
                        "type": "string",
                        "default": "en",
                        "title": "Language",
                        "x-component": "Select",
                        "description": "Language code for name generation (e.g., 'en', 'vi', 'ru').",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "format": {
                        "type": ["string", "null"],
                        "title": "Name Format",
                        "description": "Output format for synthetic names (e.g., 'FML', 'FL').",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "FML", "description": "First Middle Last"},
                            {"const": "FL", "description": "First Last"},
                            {"const": "LF", "description": "Last First"},
                            {"const": "LFM", "description": "Last First Middle"},
                            {"const": "F_L", "description": "First_Last"},
                            {"const": "L_F", "description": "Last_First"},
                        ],
                        "x-group": GroupName.NAME_GENERATION_STYLE,
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
                        "x-component": "Select",
                        "description": "Case formatting for output names: 'upper', 'lower', or 'title'.",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "use_faker": {
                        "type": "boolean",
                        "default": False,
                        "title": "Use Faker Library",
                        "x-component": "Checkbox",
                        "description": "Whether to use the Faker library for name generation (for more variety).",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "dictionaries": {
                        "type": ["object", "null"],
                        "title": "Custom Dictionaries",
                        "x-component": "Upload",  # TODO: Use custom component when available
                        "description": "Custom dictionaries for localized name generation (advanced usage).",
                        "x-group": GroupName.NAME_GENERATION_STYLE,
                    },
                    "gender_from_name": {
                        "type": "boolean",
                        "default": False,
                        "title": "Infer Gender from Name",
                        "x-component": "Checkbox",
                        "description": "Whether to infer gender from the original name value if gender field is not provided.",
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "gender_field": {
                        "type": ["string", "null"],
                        "title": "Gender Field",
                        "x-component": "Select",
                        "description": "Column name providing gender information (if available).",
                        "x-depend-on": {"gender_from_name": False},
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "f_m_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "x-component": "NumberPicker",
                        "title": "Female-to-Male Ratio",
                        "x-depend-on": {
                            "gender_field": "null",
                            "gender_from_name": False,
                        },
                        "description": "Ratio of female to male names in generated data (0 = all male, 1 = all female, 0.5 = balanced).",
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "mapping", "description": "mapping"},
                            {"const": "prgn", "description": "prgn"},
                        ],
                        "default": "prgn",
                        "title": "Consistency Method",
                        "description": "Mechanism to ensure consistent name generation: 'mapping' (fixed mapping) or 'prgn' (pseudo-random).",
                        "x-group": GroupName.GENDER_CONFIGURATION,
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "Unique ID Field",
                        "x-component": "Select",
                        "description": "Column name used as unique identifier for mapping consistency (optional).",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "x-component": "Input",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "description": "Key for encryption or PRGN consistency (if applicable).",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "PRGN Context Salt",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "description": "Additional context salt for PRGN to enhance uniqueness.",
                        "x-component": "Input",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "x-component": "Upload",
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "description": "Path to store or load mapping file for consistent name replacement.",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "x-component": "Checkbox",
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "description": "Whether to save the mapping between original and synthetic names to disk.",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
