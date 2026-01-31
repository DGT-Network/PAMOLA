"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        MVF Analysis Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of MVF (Multi-Valued Field) analysis configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines MVF analysis parameters, frequency thresholds, and format parsing options
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for MVF analysis
- Top N values and minimum frequency controls
- Format type configuration (json, csv, array_string)
- Parse keyword arguments for custom parsing logic

Changelog:
1.0.0 - 2025-01-15 - Initial creation of MVF analysis core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class MVFAnalysisOperationConfig(OperationConfig):
    """
    Core configuration schema for MVFAnalysisOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "MVF Analysis Operation Core Configuration",
        "description": "Core schema for MVF profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the multi-valued field (column) to analyze. This should be a column in the DataFrame where each cell contains multiple values (e.g., a list, set, or delimited string).",
                    },
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "title": "Top N Values",
                        "description": "Number of most frequent values to include in the analysis results. Helps focus on the most common items in the multi-valued field.",
                    },
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 1,
                        "title": "Minimum Frequency",
                        "description": "Minimum number of occurrences for a value to be included in the results. Values appearing fewer times will be excluded from the output.",
                    },
                    "format_type": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Format Type",
                        "description": "Format of the multi-valued field. Can specify how to parse the field (e.g., 'list', 'csv', 'json', or a custom delimiter). If null, auto-detection or default parsing is used.",
                        "oneOf": [
                            {"type": "null"},
                            {"const": "json", "description": "JSON"},
                            {"const": "csv", "description": "CSV"},
                            {"const": "array_string", "description": "Array String"},
                        ],
                    },
                    "parse_kwargs": {
                        "type": "object",
                        "default": {},
                        "title": "Parse Keyword Arguments",
                        "description": "Additional keyword arguments for parsing the multi-valued field. Used to customize parsing logic, such as delimiter, quote character, or other options.",
                    },
                },
                "required": ["field_name", "top_n", "min_frequency"],
            },
        ],
    }
