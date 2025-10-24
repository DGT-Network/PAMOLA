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
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "field_constraints": {"type": ["object", "null"]},
                    "whitelist_path": {"type": ["object", "null"]},
                    "blacklist_path": {"type": ["object", "null"]},
                    "null_replacement": {"type": ["string", "object", "null"]},
                },
            },
        ],
    }