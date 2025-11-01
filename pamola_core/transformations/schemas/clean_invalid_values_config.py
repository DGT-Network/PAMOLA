"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Clean Invalid Values Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating clean invalid values operations in PAMOLA.CORE.
- Supports field constraints, whitelist/blacklist paths, and null value replacement
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of clean invalid values config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class CleanInvalidValuesOperationConfig(OperationConfig):
    """Configuration for CleanInvalidValuesOperation with BaseOperationConfig merged."""

    schema = {
        "title": "CleanInvalidValuesOperationConfig",
        "description": "Schema for cleaning or nullifying invalid values based on field constraints, whitelist/blacklist, and null replacement strategies.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "CleanInvalidValuesOperationConfig Properties",
                "description": "Properties for configuring field constraints, whitelist/blacklist, and null replacement in data cleaning operations.",
                "properties": {
                    "field_constraints": {
                        "type": ["object", "null"],
                        "title": "Field Constraints",
                        "description": "Dictionary defining constraints for each field (e.g., min/max, allowed/disallowed values, patterns, etc.)."
                    },
                    "whitelist_path": {
                        "type": ["object", "null"],
                        "title": "Whitelist Path",
                        "description": "Dictionary mapping field names to whitelist file paths for allowed values."
                    },
                    "blacklist_path": {
                        "type": ["object", "null"],
                        "title": "Blacklist Path",
                        "description": "Dictionary mapping field names to blacklist file paths for disallowed values."
                    },
                    "null_replacement": {
                        "type": ["string", "object", "null"],
                        "title": "Null Replacement",
                        "description": "Strategy or dictionary for replacing null values (e.g., mean, median, mode, random_sample, or specific value)."
                    },
                },
            },
        ],
    }