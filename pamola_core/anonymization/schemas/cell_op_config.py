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
        "title": "Cell Suppression Operation Configuration",
        "description": "Configuration schema for cell-level suppression operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common base fields
            {
                "type": "object",
                "properties": {
                    # Suppression-specific fields
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "The column containing cells to suppress."
                    },
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
                        "title": "Suppression Strategy",
                        "description": "Suppression method to apply. Supported: null, mean, median, mode, constant, group_mean, group_mode."
                    },
                    "suppression_value": {
                        "type": ["string", "number", "null"],
                        "title": "Suppression Value",
                        "description": "Replacement value when using the 'constant' strategy."
                    },
                    "group_by_field": {
                        "type": ["string", "null"],
                        "title": "Group By Field",
                        "description": "Column for group-based suppression (required for group_mean or group_mode)."
                    },
                    "min_group_size": {
                        "type": "integer",
                        "minimum": 1,
                        "title": "Minimum Group Size",
                        "description": "Minimum group size for valid group-level suppression."
                    },
                    "suppress_if": {
                        "type": ["string", "null"],
                        "enum": ["outlier", "rare", "null"],
                        "title": "Suppress If",
                        "description": "Automatic suppression trigger. One of: outlier, rare, null."
                    },
                    "outlier_method": {
                        "type": "string",
                        "enum": ["iqr", "zscore"],
                        "title": "Outlier Method",
                        "description": "Outlier detection method if suppress_if is 'outlier'."
                    },
                    "outlier_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "title": "Outlier Threshold",
                        "description": "Threshold for outlier detection."
                    },
                    "rare_threshold": {
                        "type": "integer",
                        "minimum": 1,
                        "title": "Rare Threshold",
                        "description": "Frequency threshold for rare value detection."
                    },
                    # Conditional processing
                    "condition_field": {
                        "type": ["string", "null"],
                        "title": "Condition Field",
                        "description": "Field to check for conditional suppression."
                    },
                    "condition_values": {
                        "type": ["array", "null"],
                        "title": "Condition Values",
                        "description": "Values that trigger conditional suppression."
                    },
                    "condition_operator": {
                        "type": "string",
                        "title": "Condition Operator",
                        "description": "Operator for condition evaluation (e.g., EQUALS, IN)."
                    },
                    # Output field name
                    "output_field_name": {
                        "type": ["string", "null"],
                        "title": "Output Field Name",
                        "description": "Custom output field name (for ENRICH mode)."
                    },
                },
                "required": ["field_name", "suppression_strategy"],
            },
        ],
    }
