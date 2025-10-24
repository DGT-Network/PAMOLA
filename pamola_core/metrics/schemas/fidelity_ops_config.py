"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
-------------------------------------------------------------
Module:        Fidelity Operation Config
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
Updated:       2025-01-24
License:       BSD 3-Clause

Description:
    Configuration schema for fidelity metric operations.
    Defines the structure and validation rules for fidelity-related metrics in the PAMOLA framework.

Purpose:
    - Provide a standardized schema for fidelity metric operations
    - Support validation, export, and integration with frontend tools
    - Ensure consistency and maintainability across modules

Key Features:
    - Schema definition for fidelity metrics
    - Used by operation and schema utilities

"""
from pamola_core.common.enum.fidelity_metrics_type import FidelityMetricsType
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class FidelityConfig(OperationConfig):
    """Configuration for FidelityOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common fields
            {
                "type": "object",
                "properties": {
                    "fidelity_metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                FidelityMetricsType.KS.value,
                                FidelityMetricsType.KL.value,
                            ],
                        },
                        "default": [
                            FidelityMetricsType.KS.value,
                            FidelityMetricsType.KL.value,
                        ],
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
                "required": ["fidelity_metrics"],
            },
        ],
    }
