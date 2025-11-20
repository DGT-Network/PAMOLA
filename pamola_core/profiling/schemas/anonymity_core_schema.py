"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        K-Anonymity Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of k-anonymity profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines k-anonymity analysis parameters, quasi-identifiers, thresholds, and operation modes
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Analysis mode configuration (ANALYZE, ENRICH, BOTH)
- Quasi-identifier and threshold management
- Combination limits for scalability
- Output field and metrics export controls

Changelog:
1.0.0 - 2025-01-15 - Initial creation of k-anonymity profiler core schema
"""

from pamola_core.profiling.analyzers.anonymity import AnalysisMode
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class KAnonymityProfilerOperationConfig(OperationConfig):
    """
    Core configuration schema for KAnonymityProfilerOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "K-Anonymity Profiler Operation Core Configuration",
        "description": "Core schema for k-anonymity profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "analysis_mode": {
                        "type": "string",
                        "default": AnalysisMode.ANALYZE.value,
                        "title": "Analysis Mode",
                        "description": "Operation mode: 'ANALYZE' (generate metrics and visualizations), 'ENRICH' (add k-values to the DataFrame), or 'BOTH' (perform both analysis and enrichment).",
                        "oneOf": [
                            {
                                "const": AnalysisMode.ANALYZE.value,
                                "description": "ANALYZE",
                            },
                            {
                                "const": AnalysisMode.ENRICH.value,
                                "description": "ENRICH",
                            },
                            {
                                "const": AnalysisMode.BOTH.value,
                                "description": "BOTH",
                            },
                        ],
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Individual quasi-identifier fields",
                        "description": "List of fields used as quasi-identifiers for k-anonymity analysis. These are the columns whose combinations are evaluated for re-identification risk.",
                    },
                    "quasi_identifier_sets": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Quasi-Identifier Sets",
                        "description": "Optional list of pre-defined sets of quasi-identifiers to analyze as combinations. Overrides automatic detection.",
                    },
                    "threshold_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "title": "k Threshold",
                        "description": "Threshold for considering records as vulnerable. Records/groups with k < threshold_k are flagged as privacy risks.",
                    },
                    "max_combinations": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 50,
                        "title": "Max QI Combinations",
                        "description": "Maximum number of quasi-identifier combinations to analyze. Limits combinatorial explosion for large datasets.",
                    },
                    "id_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "ID Fields",
                        "description": "List of columns used as record identifiers for grouping or tracking vulnerable records.",
                    },
                    "output_field_suffix": {
                        "type": "string",
                        "default": "k_anon",
                        "title": "Output Field Suffix",
                        "description": "Suffix for the k-anonymity field added in ENRICH mode.",
                    },
                    "export_metrics": {
                        "type": "boolean",
                        "default": True,
                        "title": "Export Metrics",
                        "description": "If true, export k-anonymity metrics and vulnerability analysis to JSON/CSV files.",
                    },
                },
                "required": ["quasi_identifiers"],
            },
        ],
    }
