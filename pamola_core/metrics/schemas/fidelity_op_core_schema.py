"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Fidelity Metric Core Schema
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of fidelity metric operation
configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines selectable fidelity metrics, evaluated columns, sampling and statistical knobs
- Contains business logic validation rules (type constraints, enums)
- Free of UI metadata - only validation rules and data structure
"""

from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class FidelityConfig(OperationConfig):
    """
    Core configuration schema for FidelityOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Fidelity Metric Operation Core Configuration",
        "description": "Core schema for fidelity metric operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "fidelity_metrics": {
                        "type": "array",
                        "title": "Fidelity Metrics",
                        "description": "List of fidelity metrics to compute between the original and transformed datasets.",
                        "items": {
                            "type": "string",
                            "oneOf": [
                                {"const": FidelityMetricsType.KS.value, "description": "Kolmogorov-Smirnov Test"},
                                {"const": FidelityMetricsType.KL.value, "description": "Kullback-Leibler Divergence"},
                            ],
                        },
                        "default": [
                            FidelityMetricsType.KS.value,
                            FidelityMetricsType.KL.value,
                        ],
                    },
                    "metric_params": {
                        "type": ["object", "null"],
                        "title": "Metric Parameters",
                        "description": "Optional dictionary of parameters for each fidelity metric (e.g., thresholds, custom settings).",
                    },
                    "columns": {
                        "type": "array",
                        "title": "Columns",
                        "description": "List of column names to evaluate fidelity metrics on.",
                        "items": {"type": "string"},
                    },
                    "column_mapping": {
                        "type": ["object", "null"],
                        "title": "Column Mapping",
                        "description": "Optional mapping from original to anonymized column names for metric comparison.",
                    },
                    "normalize": {
                        "type": "boolean",
                        "title": "Normalize",
                        "default": True,
                        "description": "If true, normalize data before computing fidelity metrics.",
                    },
                    "confidence_level": {
                        "type": "number",
                        "title": "Confidence Level",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.95,
                        "description": "Confidence level for statistical tests (e.g., 0.95 for 95% confidence).",
                    },
                    "sample_size": {
                        "type": ["integer", "null"],
                        "title": "Sample Size",
                        "minimum": 1,
                        "description": "Size of dataset sample used for metric calculation (null = use all rows).",
                    },
                },
                "required": ["fidelity_metrics"],
            },
        ],
    }
