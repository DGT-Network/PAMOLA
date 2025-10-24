"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Cell Suppression Config Schema
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating cell suppression operations in PAMOLA.CORE.
Supports parameters for field names, suppression strategies, and control options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of cell suppression config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class CellSuppressionConfig(OperationConfig):
    """Configuration schema for CellSuppressionOperation with with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common base fields
            {
                "type": "object",
                "properties": {
                    # Suppression-specific fields
                    "field_name": {"type": "string"},
                    "suppression_strategy": {
                        "type": "string",
                        "enum": [
                            "null",
                            "mean",
                            "median",
                            "mode",
                            "constant",
                            "group_mean",
                            "group_mode",
                        ],
                    },
                    "suppression_value": {},
                    "group_by_field": {"type": ["string", "null"]},
                    "min_group_size": {"type": "integer", "minimum": 1},
                    "suppress_if": {
                        "type": ["string", "null"],
                        "enum": ["outlier", "rare", "null"],
                    },
                    "outlier_method": {
                        "type": "string",
                        "enum": ["iqr", "zscore"],
                    },
                    "outlier_threshold": {"type": "number", "minimum": 0},
                    "rare_threshold": {"type": "integer", "minimum": 1},
                    # Conditional processing
                    "condition_field": {"type": ["string", "null"]},
                    "condition_values": {"type": ["array", "null"]},
                    "condition_operator": {"type": "string"},
                    # Output field name
                    "output_field_name": {"type": ["string", "null"]},
                },
                "required": ["field_name", "suppression_strategy"],
            },
        ],
    }
