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
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    # --- GeneratorOperation / BaseOperation common fields ---
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
                    # --- FakeOrganizationOperation-specific fields ---
                    "field_name": {"type": "string"},
                    "organization_type": {"type": "string", "default": "general"},
                    "dictionaries": {"type": ["object", "null"]},
                    "prefixes": {"type": ["object", "null"]},
                    "suffixes": {"type": ["object", "null"]},
                    "add_prefix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                    },
                    "add_suffix_probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.5,
                    },
                    "region": {"type": "string", "default": "en"},
                    "preserve_type": {"type": "boolean", "default": True},
                    "industry": {"type": ["string", "null"]},
                    # --- Advanced behavior & metrics ---
                    "collect_type_distribution": {"type": "boolean", "default": True},
                    "type_field": {"type": ["string", "null"]},
                    "region_field": {"type": ["string", "null"]},
                    "detailed_metrics": {"type": "boolean", "default": False},
                    "max_retries": {"type": "integer", "minimum": 0, "default": 3},
                    "key": {"type": ["string", "null"]},
                    "context_salt": {"type": ["string", "null"]},
                },
                "required": ["field_name"],
            },
        ],
    }