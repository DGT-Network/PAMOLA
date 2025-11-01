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
        "title": "K-Anonymity Profiler Operation Configuration",
        "description": "Configuration schema for k-anonymity profiling operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Quasi-Identifiers",
                        "description": "List of fields used as quasi-identifiers for k-anonymity analysis. These are the columns whose combinations are evaluated for re-identification risk."
                    },
                    "analysis_mode": {
                        "type": "string",
                        "enum": [
                            AnalysisMode.ANALYZE.value,
                            AnalysisMode.ENRICH.value,
                            AnalysisMode.BOTH.value,
                        ],
                        "default": AnalysisMode.ANALYZE.value,
                        "title": "Analysis Mode",
                        "description": "Operation mode: 'ANALYZE' (generate metrics and visualizations), 'ENRICH' (add k-values to the DataFrame), or 'BOTH' (perform both analysis and enrichment)."
                    },
                    "threshold_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "title": "k Threshold",
                        "description": "Threshold for considering records as vulnerable. Records/groups with k < threshold_k are flagged as privacy risks."
                    },
                    "export_metrics": {
                        "type": "boolean",
                        "default": True,
                        "title": "Export Metrics",
                        "description": "If true, export k-anonymity metrics and vulnerability analysis to JSON/CSV files."
                    },
                    "max_combinations": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 50,
                        "title": "Max QI Combinations",
                        "description": "Maximum number of quasi-identifier combinations to analyze. Limits combinatorial explosion for large datasets."
                    },
                    "output_field_suffix": {
                        "type": "string",
                        "default": "k_anon",
                        "title": "Output Field Suffix",
                        "description": "Suffix for the k-anonymity field added in ENRICH mode."
                    },
                    "quasi_identifier_sets": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "default": None,
                        "title": "Quasi-Identifier Sets",
                        "description": "Optional list of pre-defined sets of quasi-identifiers to analyze as combinations. Overrides automatic detection."
                    },
                    "id_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "ID Fields",
                        "description": "List of columns used as record identifiers for grouping or tracking vulnerable records."
                    },
                },
                "required": ["quasi_identifiers"],
            },
        ],
    }

