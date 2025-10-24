"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Phone Config Schema
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating fake phone generation operations in PAMOLA.CORE.
- Supports generator parameters, country/region, formats, ratios, and advanced metrics
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of fake phone config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FakePhoneOperationConfig(OperationConfig):
    """Configuration for FakePhoneOperation with BaseOperationConfig merged."""

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
                    # --- FakePhoneOperation-specific fields ---
                    "field_name": {"type": "string"},
                    "country_code": {"type": ["string", "null"], "default": None},
                    "formats": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                    },
                    "region": {"type": ["string", "null"], "default": "global"},
                    "preserve_format": {"type": "boolean", "default": True},
                    "allow_invalid": {"type": "boolean", "default": False},
                    "mobile_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                    },
                    "landline_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                    },
                    "format_variants": {
                        "type": ["object", "null"],
                        "default": None,
                    },
                    "use_international_format": {"type": "boolean", "default": True},
                    # --- Advanced behavior & metrics ---
                    "detailed_metrics": {"type": "boolean", "default": False},
                    "max_retries": {"type": "integer", "minimum": 0, "default": 3},
                    "key": {"type": ["string", "null"]},
                    "context_salt": {"type": ["string", "null"]},
                },
                "required": ["field_name"],
            },
        ],
    }
