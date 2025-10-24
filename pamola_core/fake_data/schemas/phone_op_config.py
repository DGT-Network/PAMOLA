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
        "title": "Fake Phone Operation Config",
        "description": "Configuration schema for generating fake phone numbers. Includes generator parameters, country/region, formats, ratios, and advanced metrics.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "FakePhoneOperationConfig Properties",
                "description": "Properties specific to fake phone number generation and operation control.",
                "properties": {
                    # --- GeneratorOperation / BaseOperation common fields ---
                    "generator": {
                        "type": ["object", "null"],
                        "title": "Generator",
                        "description": "Custom generator instance or parameters for phone number generation."
                    },
                    "generator_params": {
                        "type": ["object", "null"],
                        "title": "Generator Parameters",
                        "description": "Parameters to configure the phone number generator."
                    },
                    "consistency_mechanism": {
                        "type": "string",
                        "enum": ["mapping", "prgn"],
                        "default": "prgn",
                        "title": "Consistency Mechanism",
                        "description": "Mechanism for consistent pseudonymization: 'mapping' for mapping store, 'prgn' for deterministic pseudo-random generation."
                    },
                    "id_field": {
                        "type": ["string", "null"],
                        "title": "ID Field",
                        "description": "Field name used as unique identifier for mapping consistency."
                    },
                    "mapping_store_path": {
                        "type": ["string", "null"],
                        "title": "Mapping Store Path",
                        "description": "File path to persistently store mapping for consistent pseudonymization."
                    },
                    "mapping_store": {
                        "type": ["object", "null"],
                        "title": "Mapping Store",
                        "description": "In-memory mapping store for value consistency."
                    },
                    "save_mapping": {
                        "type": "boolean",
                        "default": False,
                        "title": "Save Mapping",
                        "description": "Whether to save the mapping store to disk after operation."
                    },
                    "output_field_name": {
                        "type": ["string", "null"],
                        "title": "Output Field Name",
                        "description": "Name of the output field for the generated phone number."
                    },
                    # --- FakePhoneOperation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the field to be replaced with a fake phone number."
                    },
                    "country_code": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Country Code",
                        "description": "Country code to use for phone number generation (e.g., '+84', 'US')."
                    },
                    "formats": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Formats",
                        "description": "List of phone number formats to use for generation."
                    },
                    "region": {
                        "type": ["string", "null"],
                        "default": "global",
                        "title": "Region",
                        "description": "Region for phone number style (e.g., 'global', 'US', 'VN')."
                    },
                    "preserve_format": {
                        "type": "boolean",
                        "default": True,
                        "title": "Preserve Format",
                        "description": "Whether to preserve the original phone number format."
                    },
                    "allow_invalid": {
                        "type": "boolean",
                        "default": False,
                        "title": "Allow Invalid",
                        "description": "Allow generation of phone numbers that may not be valid."
                    },
                    "mobile_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.7,
                        "title": "Mobile Ratio",
                        "description": "Ratio of generated numbers that should be mobile numbers."
                    },
                    "landline_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.3,
                        "title": "Landline Ratio",
                        "description": "Ratio of generated numbers that should be landline numbers."
                    },
                    "format_variants": {
                        "type": ["object", "null"],
                        "default": None,
                        "title": "Format Variants",
                        "description": "Dictionary of format variants for phone number generation."
                    },
                    "use_international_format": {
                        "type": "boolean",
                        "default": True,
                        "title": "Use International Format",
                        "description": "Whether to use international phone number format (E.164)."
                    },
                    # --- Advanced behavior & metrics ---
                    "detailed_metrics": {
                        "type": "boolean",
                        "default": False,
                        "title": "Detailed Metrics",
                        "description": "Collect detailed metrics during phone number generation."
                    },
                    "max_retries": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 3,
                        "title": "Max Retries",
                        "description": "Maximum number of retries for generating a valid phone number."
                    },
                    "key": {
                        "type": ["string", "null"],
                        "title": "Key",
                        "description": "Optional key for advanced configuration or encryption."
                    },
                    "context_salt": {
                        "type": ["string", "null"],
                        "title": "Context Salt",
                        "description": "Salt value for context-aware pseudonymization."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }
