"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Organization Core Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of fake organization generation configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines organization generation parameters, type options, region settings, and consistency mechanisms
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Organization type and region configuration
- Industry-specific generation controls
- Prefix and suffix probability settings
- Consistency mechanism validation (mapping vs PRGN)

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake organization core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class FakeOrganizationOperationConfig(OperationConfig):
    """
    Core configuration schema for FakeOrganizationOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Fake Organization Operation Core Configuration",
        "description": "Core schema for fake organization operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing organization names to process.",
                    },
                    "organization_type": {
                        "type": "string",
                        "default": "general",
                        "title": "Default Organization Type",
                        "oneOf": [
                            {"const": "general", "description": "General"},
                            {"const": "educational", "description": "Educational"},
                            {"const": "manufacturing", "description": "Manufacturing"},
                            {"const": "government", "description": "Government"},
                            {"const": "industry", "description": "Industry"},
                        ],
                        "description": "Type of organization to generate (e.g., 'general', 'industry', 'government').",
                    },
                    "region": {
                        "type": "string",
                        "default": "en",
                        "title": "Default Region",
                        "description": "Region code for localized organization name generation.",
                    },
                    "industry": {
                        "type": ["string", "null"],
                        "title": "Industry Type",
                        "description": "Specific industry name for contextual organization generation.",
                    },
                    "preserve_type": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Original Type",
                        "description": "Whether to preserve the type of organization in the generated name.",
                    },
                    "add_prefix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                        "title": "Add Prefix Probability",
                        "description": "Probability of adding a prefix to the organization name.",
                    },
                    "add_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Add Suffix Probability",
                        "description": "Probability of adding a suffix to the organization name.",
                    },
                    "type_field": {
                        "type": ["string", "null"],
                        "title": "Type Field",
                        "description": "Field in the dataset containing organization type codes.",
                    },
                    "region_field": {
                        "type": ["string", "null"],
                        "title": "Region Field",
                        "description": "Field in the dataset containing region codes.",
                    },
                    "dictionaries": {
                        "type": ["object", "null"],
                        "title": "Dictionaries",
                        "description": "Custom dictionaries for organization name generation.",
                    },
                    "prefixes": {
                        "type": ["object", "null"],
                        "title": "Prefixes",
                        "description": "Prefix dictionary for organization names.",
                    },
                    "suffixes": {
                        "type": ["object", "null"],
                        "title": "Suffixes",
                        "description": "Suffix dictionary for organization names.",
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
                        "description": "Encryption or PRGN key for consistent pseudonymization.",
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "Context Salt",
                        "description": "Contextual salt for PRGN uniqueness.",
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "Path to store mapping between original and synthetic organization names.",
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping between original and synthetic organization names.",
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "title": "Max Retries",
                        "description": "Maximum retries for failed name generations.",
                    },
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "title": "Detailed Metrics",
                        "description": "Whether to enable detailed generation statistics.",
                    },
                    "collect_type_distribution": {
                        "type": "boolean",
                        "default": True,
                        "title": "Collect Type Distribution",
                        "description": "Whether to collect distribution statistics per organization type.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
