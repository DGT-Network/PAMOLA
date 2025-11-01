"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Organization Config Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating fake organization generation operations in PAMOLA.CORE.
- Supports generator parameters, organization type, dictionaries, region, and advanced metrics
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake organization config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FakeOrganizationOperationConfig(OperationConfig):
    """Configuration for FakeOrganizationOperation with BaseOperationConfig merged."""

    schema = {
        "title": "FakeOrganizationOperationConfig",
        "description": "Configuration schema for FakeOrganizationOperation. Controls how synthetic organization names are generated, including type, region, dictionaries, prefix/suffix, and advanced metrics for realistic and consistent organization name anonymization.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "FakeOrganizationOperationConfig Properties",
                "description": "All configuration options for FakeOrganizationOperation, including generator, organization type, dictionaries, region, prefix/suffix, and advanced metrics.",
                "properties": {
                    # --- GeneratorOperation / BaseOperation common fields ---
                    "generator": {
                        "type": ["object", "null"],
                        "title": "Generator",
                        "description": "Generator instance or configuration for organization name generation."
                    },
                    "generator_params": {
                        "type": ["object", "null"],
                        "title": "Generator Parameters",
                        "description": "Parameters passed to the organization generator."
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
                        "description": "Path to store mapping between original and synthetic organization names."
                    },
                    "mapping_store": {
                        "type": ["object", "null"],
                        "title": "Mapping Store",
                        "description": "Object for storing mapping between original and synthetic values."
                    },
                    "save_mapping": {
                        "type": "boolean", "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping between original and synthetic organization names."
                    },
                    # --- FakeOrganizationOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing organization names to process."
                    },
                    "organization_type": {
                        "type": "string", "default": "general",
                        "title": "Organization Type",
                        "description": "Type of organization to generate (e.g., 'general', 'industry', 'government')."
                    },
                    "dictionaries": {
                        "type": ["object", "null"],
                        "title": "Dictionaries",
                        "description": "Custom dictionaries for organization name generation."
                    },
                    "prefixes": {
                        "type": ["object", "null"],
                        "title": "Prefixes",
                        "description": "Prefix dictionary for organization names."
                    },
                    "suffixes": {
                        "type": ["object", "null"],
                        "title": "Suffixes",
                        "description": "Suffix dictionary for organization names."
                    },
                    "add_prefix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                        "title": "Add Prefix Probability",
                        "description": "Probability of adding a prefix to the organization name."
                    },
                    "add_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "title": "Add Suffix Probability",
                        "description": "Probability of adding a suffix to the organization name."
                    },
                    "region": {
                        "type": "string", "default": "en",
                        "title": "Region",
                        "description": "Region code for localized organization name generation."
                    },
                    "preserve_type": {
                        "type": "boolean", "default": True,
                        "title": "Preserve Type",
                        "description": "Whether to preserve the type of organization in the generated name."
                    },
                    "industry": {
                        "type": ["string", "null"],
                        "title": "Industry",
                        "description": "Specific industry name for contextual organization generation."
                    },
                    # --- Advanced behavior & metrics ---
                    "collect_type_distribution": {
                        "type": "boolean", "default": True,
                        "title": "Collect Type Distribution",
                        "description": "Whether to collect distribution statistics per organization type."
                    },
                    "type_field": {
                        "type": ["string", "null"],
                        "title": "Type Field",
                        "description": "Field in the dataset containing organization type codes."
                    },
                    "region_field": {
                        "type": ["string", "null"],
                        "title": "Region Field",
                        "description": "Field in the dataset containing region codes."
                    },
                    "detailed_metrics": {
                        "type": "boolean", "default": False,
                        "title": "Detailed Metrics",
                        "description": "Whether to enable detailed generation statistics."
                    },
                    "max_retries": {
                        "type": "integer", "minimum": 0, "default": 3,
                        "title": "Max Retries",
                        "description": "Maximum retries for failed name generations."
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "Key",
                        "description": "Encryption or PRGN key for consistent pseudonymization."
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "Context Salt",
                        "description": "Contextual salt for PRGN uniqueness."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }