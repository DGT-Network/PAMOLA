"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Privacy Metric Config Schema
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating privacy metric operations in PAMOLA.CORE.
Supports generator parameters, metric options, validation, and fine-tuning for privacy metrics.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of privacy metric config file
"""

from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class PrivacyMetricConfig(OperationConfig):
    """Configuration for PrivacyMetricOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common fields
            {
                "type": "object",
                "properties": {
                    "privacy_metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                PrivacyMetricsType.DCR.value,
                                PrivacyMetricsType.NNDR.value,
                                PrivacyMetricsType.UNIQUENESS.value,
                                PrivacyMetricsType.K_ANONYMITY.value,
                                PrivacyMetricsType.L_DIVERSITY.value,
                            ],
                        },
                        "default": [PrivacyMetricsType.DCR.value],
                    },
                    "metric_params": {"type": ["object", "null"]},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "column_mapping": {"type": ["object", "null"]},
                    "sample_size": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Size of dataset sample used for metric calculation.",
                    },
                },
                "required": ["privacy_metrics"],
            },
        ],
    }
