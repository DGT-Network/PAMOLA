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

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
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
                    "analysis_mode": {
                        "type": "string",
                        "enum": [
                            AnalysisMode.ANALYZE.value,
                            AnalysisMode.ENRICH.value,
                            AnalysisMode.BOTH.value,
                        ],
                        "default": AnalysisMode.ANALYZE.value,
                        "title": "Analysis Mode",
                        "description": "Operation mode: 'ANALYZE' (generate metrics and visualizations), 'ENRICH' (add k-values to the DataFrame), or 'BOTH' (perform both analysis and enrichment).",
                        "x-component": "Select",
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
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "quasi_identifiers": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "Individual quasi-identifier fields",
                        "description": "List of fields used as quasi-identifiers for k-anonymity analysis. These are the columns whose combinations are evaluated for re-identification risk.",
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                        "x-custom-function": ["update_field_options"],
                    },
                    "quasi_identifier_sets": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "default": None,
                        "title": "Quasi-Identifier Sets",
                        "description": "Optional list of pre-defined sets of quasi-identifiers to analyze as combinations. Overrides automatic detection.",
                        "x-component": "Select",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                        "x-custom-function": ["update_field_options"],
                    },
                    "threshold_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "title": "k Threshold",
                        "description": "Threshold for considering records as vulnerable. Records/groups with k < threshold_k are flagged as privacy risks.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "max_combinations": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 50,
                        "title": "Max QI Combinations",
                        "description": "Maximum number of quasi-identifier combinations to analyze. Limits combinatorial explosion for large datasets.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.CORE_GENERALIZATION_STRATEGY,
                    },
                    "id_fields": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "default": None,
                        "title": "ID Fields",
                        "description": "List of columns used as record identifiers for grouping or tracking vulnerable records.",
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "x-custom-function": ["update_field_options"],
                    },
                    "output_field_suffix": {
                        "type": "string",
                        "default": "k_anon",
                        "title": "Output Field Suffix",
                        "description": "Suffix for the k-anonymity field added in ENRICH mode.",
                        "x-component": "Input",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "export_metrics": {
                        "type": "boolean",
                        "default": True,
                        "title": "Export Metrics",
                        "description": "If true, export k-anonymity metrics and vulnerability analysis to JSON/CSV files.",
                        "x-component": "Checkbox",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                },
                "required": ["quasi_identifiers"],
            },
        ],
    }
