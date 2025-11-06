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

from pamola_core.common.enum.form_groups import GroupName
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
                    # --- FakeOrganizationOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the column containing organization names to process.",
                    },
                    "organization_type": {
                        "type": "string",
                        "default": "general",
                        "title": "Default Organization Type",
                        "x-component": "Select",
                        "oneOf": [
                            {"const": "general", "description": "General"},
                            {"const": "educational", "description": "Educational"},
                            {"const": "manufacturing", "description": "Manufacturing"},
                            {"const": "government", "description": "Government"},
                            {"const": "industry", "description": "Industry"},
                        ],
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                        "description": "Type of organization to generate (e.g., 'general', 'industry', 'government').",
                    },
                    "region": {
                        "type": "string",
                        "default": "en",
                        "title": "Default Region",
                        "x-component": "Select",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                        "description": "Region code for localized organization name generation.",
                    },
                    "industry": {
                        "type": ["string", "null"],
                        "title": "Industry Type",
                        "x-component": "Select",
                        # "oneOf": [
                        #     { "const": "tech", "title": "General" },
                        #     { "const": "finance", "title": "Educational" },
                        #     { "const": "retail", "title": "Manufacturing" },
                        #     { "const": "healthcare", "title": "Government" }
                        # ],
                        "x-depend-on": {"organization_type": "industry"},
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                        "description": "Specific industry name for contextual organization generation.",
                    },
                    "preserve_type": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Original Type",
                        "x-component": "Checkbox",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                        "description": "Whether to preserve the type of organization in the generated name.",
                    },
                    "add_prefix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                        "title": "Add Prefix Probability",
                        "description": "Probability of adding a prefix to the organization name.",
                    },
                    "add_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.ORGANIZATION_GENERATION_STYLE,
                        "title": "Add Suffix Probability",
                        "description": "Probability of adding a suffix to the organization name.",
                    },
                    "type_field": {
                        "type": ["string", "null"],
                        "title": "Type Field",
                        "x-component": "Input",
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "description": "Field in the dataset containing organization type codes.",
                    },
                    "region_field": {
                        "type": ["string", "null"],
                        "title": "Region Field",
                        "x-component": "Select",
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "description": "Field in the dataset containing region codes.",
                    },
                    "dictionaries": {
                        "type": ["object", "null"],
                        "title": "Dictionaries",
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "x-component": "Upload",  # custom input component
                        "description": "Custom dictionaries for organization name generation.",
                    },
                    "prefixes": {
                        "type": ["object", "null"],
                        "title": "Prefixes",
                        "x-component": "Upload",  # custom input component
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "description": "Prefix dictionary for organization names.",
                    },
                    "suffixes": {
                        "type": ["object", "null"],
                        "title": "Suffixes",
                        "x-component": "Upload",  # custom input component
                        "x-group": GroupName.CONTEXT_AND_DATA_SOURCES,
                        "description": "Suffix dictionary for organization names.",
                    },
                    # --- GeneratorOperation / BaseOperation common fields ---
                    "generator": {
                        "type": ["object", "null"],
                        "title": "Generator",
                        "description": "Generator instance or configuration for organization name generation.",
                    },
                    "generator_params": {
                        "type": ["object", "null"],
                        "title": "Generator Parameters",
                        "description": "Parameters passed to the organization generator.",
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
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "description": "Field name used as unique identifier for mapping consistency.",
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "PRGN Key",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-component": "Input",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "description": "Encryption or PRGN key for consistent pseudonymization.",
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "Context Salt",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-component": "Input",
                        "x-depend-on": {"consistency_mechanism": "prgn"},
                        "description": "Contextual salt for PRGN uniqueness.",
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "x-component": "Upload",
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "description": "Path to store mapping between original and synthetic organization names.",
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
                        "x-group": GroupName.CONSISTENCY_STRATEGY,
                        "x-component": "Checkbox",
                        "x-depend-on": {"consistency_mechanism": "mapping"},
                        "description": "Whether to save the mapping between original and synthetic organization names.",
                    },
                    # --- Advanced behavior & metrics ---
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "title": "Max Retries",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Maximum retries for failed name generations.",
                    },
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "title": "Detailed Metrics",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Whether to enable detailed generation statistics.",
                    },
                    "collect_type_distribution": {
                        "type": "boolean",
                        "default": True,
                        "title": "Collect Type Distribution",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "description": "Whether to collect distribution statistics per organization type.",
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
