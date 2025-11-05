"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        MVF Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating MVF analysis operations in PAMOLA.CORE.
Supports parameters for field names, top N values, frequency, format types, and parsing options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of MVF config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig
from pamola_core.common.enum.form_groups import GroupName

class MVFAnalysisOperationConfig(OperationConfig):
    """Configuration for MVFOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "MVF Analysis Operation Configuration",
        "description": "Configuration schema for MVF profiling operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge shared base fields
            {
                "type": "object",
                "properties": {
                    # --- Operation-specific fields ---
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the multi-valued field (column) to analyze. This should be a column in the DataFrame where each cell contains multiple values (e.g., a list, set, or delimited string)."
                    },
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "title": "Top N Values",
                        "description": "Number of most frequent values to include in the analysis results. Helps focus on the most common items in the multi-valued field.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "min_frequency": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 1,
                        "title": "Minimum Frequency",
                        "description": "Minimum number of occurrences for a value to be included in the results. Values appearing fewer times will be excluded from the output.",
                        "x-component": "NumberPicker",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "format_type": {
                        "type": ["string", "null"],
                        "default": "json",
                        "title": "Format Type",
                        "description": "Format of the multi-valued field. Can specify how to parse the field (e.g., 'list', 'csv', 'json', or a custom delimiter). If null, auto-detection or default parsing is used.",
                        "oneOf": [
                            {"const": "json", "description": "JSON"},
                            {"const": "csv", "description": "CSV"},
                            {"const": "array_string", "description": "Array String"},
                        ],
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                    },
                    "parse_kwargs": {
                        "type": "object",
                        "default": {},
                        "title": "Parse Keyword Arguments",
                        "description": "Additional keyword arguments for parsing the multi-valued field. Used to customize parsing logic, such as delimiter, quote character, or other options."
                    },
                },
                "required": ["field_name"],
            },
        ],
    }

