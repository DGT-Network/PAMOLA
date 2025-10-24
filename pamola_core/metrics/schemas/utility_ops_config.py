
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Utility Metric Config Schema
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating utility metric operations in PAMOLA.CORE.
Supports generator parameters, metric options, validation, and fine-tuning for utility metrics.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of utility metric config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class UtilityMetricConfig(OperationConfig):
    """
    Configuration for UtilityMetricOperation merged with BaseOperationConfig.
    """

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common/base fields
            {
                "type": "object",
                "properties": {
                    "utility_metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of utility metric identifiers to compute.",
                    },
                    "metric_params": {"type": ["object", "null"]},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "column_mapping": {"type": ["object", "null"]},
                    "normalize": {"type": "boolean", "default": True},
                    "confidence_level": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.95,
                    },
                    "sample_size": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Size of dataset sample used for metric calculation.",
                    },
                },
                "required": ["utility_metrics"],
            },
        ],
    }
