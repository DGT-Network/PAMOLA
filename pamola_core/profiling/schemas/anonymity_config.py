"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Anonymity Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating anonymity profiling operations in PAMOLA.CORE.
Supports parameters for k-anonymity, analysis modes, and integration with profiling tools.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of anonymity config file
"""

from pamola_core.profiling.analyzers.anonymity import AnalysisMode
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class KAnonymityProfilerOperationConfig(OperationConfig):
    """Configuration for KAnonymityProfilerOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                    },
                    "analysis_mode": {
                        "type": "string",
                        "enum": [
                            AnalysisMode.ANALYZE.value,
                            AnalysisMode.ENRICH.value,
                            AnalysisMode.BOTH.value,
                        ],
                        "default": AnalysisMode.ANALYZE.value,
                    },
                    "threshold_k": {"type": "integer", "minimum": 1, "default": 5},
                    "export_metrics": {"type": "boolean", "default": True},
                    "max_combinations": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 50,
                    },
                    "output_field_suffix": {"type": "string", "default": "k_anon"},
                    "quasi_identifier_sets": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "default": None,
                    },
                    "id_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                    },
                },
                "required": ["quasi_identifiers"],
            },
        ],
    }

