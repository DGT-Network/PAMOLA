"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Privacy Metric Core Schema
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of privacy metric operation
configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines selectable privacy metrics, evaluated columns, sampling
- Contains business logic validation rules (type constraints, enums)
- Free of UI metadata - only validation rules and data structure
"""

from pamola_core.common.enum.privacy_metrics_type import PrivacyMetricsType
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class PrivacyMetricConfig(OperationConfig):
    """
    Core configuration schema for PrivacyMetricOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Privacy Metric Operation Core Configuration",
        "description": "Core schema for privacy metric operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "privacy_metrics": {
                        "type": "array",
                        "title": "Privacy Metrics",
                        "description": (
                            "List of privacy metrics to compute. Supported: DCR (Distance to "
                            "Closest Record), NNDR (Nearest Neighbor Distance Ratio), "
                            "UNIQUENESS, K-ANONYMITY, L-DIVERSITY."
                        ),
                        "items": {
                            "type": "string",
                            "oneOf": [
                                {"const": PrivacyMetricsType.DCR.value, "description": "Distance to Closest Record"},
                                {"const": PrivacyMetricsType.NNDR.value, "description": "Nearest Neighbor Distance Ratio"},
                                {"const": PrivacyMetricsType.UNIQUENESS.value, "description": "Uniqueness"},
                                {"const": PrivacyMetricsType.K_ANONYMITY.value, "description": "K-Anonymity"},
                                {"const": PrivacyMetricsType.L_DIVERSITY.value, "description": "L-Diversity"},
                            ],
                        },
                        "default": [PrivacyMetricsType.DCR.value],
                    },
                    "metric_params": {
                        "type": ["object", "null"],
                        "title": "Metric Parameters",
                        "description": "Optional dictionary of parameters for each privacy metric (e.g., thresholds, custom settings).",
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
                        "description": "Optional mapping from original to anonymized column names for metric comparison.",
                    },
                    "sample_size": {
                        "type": ["integer", "null"],
                        "title": "Sample Size",
                        "minimum": 1,
                        "description": "Number of records to sample for metric calculation. If null, use all data.",
                    },
                },
                "required": ["privacy_metrics"],
            },
        ],
    }