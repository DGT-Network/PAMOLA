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
                        "title": "Privacy Metrics",
                        "description": "List of privacy metrics to be used in the operation. Supported: DCR (Distance to Closest Record), NNDR (Nearest Neighbor Distance Ratio), UNIQUENESS, K-ANONYMITY, L-DIVERSITY.",
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
                    "metric_params": {
                        "type": ["object", "null"],
                        "title": "Metric Parameters",
                        "description": "Optional dictionary of parameters for each privacy metric (e.g., thresholds, custom settings)."
                    },
                    "columns": {
                        "type": "array",
                        "title": "Columns",
                        "description": "List of column names to evaluate privacy metrics on.",
                        "items": {"type": "string"},
                    },
                    "column_mapping": {
                        "type": ["object", "null"],
                        "title": "Column Mapping",
                        "description": "Optional mapping from original to anonymized column names for metric comparison."
                    },
                    "sample_size": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "title": "Sample Size",
                        "description": "Number of records to sample for metric calculation. If null, use all data."
                    },
                },
                "required": ["privacy_metrics"],
            },
        ],
    }
