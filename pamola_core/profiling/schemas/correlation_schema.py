"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating correlation profiling operations in PAMOLA.CORE.
Supports parameters for field pairs, correlation methods, and profiling options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of correlation config file
"""

from pamola_core.common.enum.custom_functions import CustomFunctions
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class CorrelationOperationConfig(OperationConfig):
    """Configuration for CorrelationOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Correlation Operation Configuration",
        "description": "Configuration schema for correlation profiling operations. Defines parameters for analyzing the correlation between two fields, including method selection, null handling, and multi-valued field parsing.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "field1": {
                        "type": "string",
                        "title": "First Field Name",
                        "description": "Name of the first field (column) to analyze for correlation. Must exist in the input DataFrame.",
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.FIELD_SETTINGS,
                    },
                    "field2": {
                        "type": "string",
                        "title": "Second Field Name",
                        "description": "Name of the second field (column) to analyze for correlation. Must exist in the input DataFrame.",
                        "x-component": "Select",
                        "x-custom-function": [CustomFunctions.UPDATE_FIELD_OPTIONS],
                        "x-group": GroupName.FIELD_SETTINGS,
                    },
                    "method": {
                        "type": ["string", "null"],
                        "title": "Correlation Method",
                        "description": "Correlation method to use. If None, the method is automatically selected based on the data types of the fields. Supported: 'pearson', 'spearman', 'cramers_v', 'correlation_ratio', 'point_biserial'.",
                        "enum": [
                            "pearson",
                            "spearman",
                            "cramers_v",
                            "correlation_ratio",
                            "point_biserial",
                            None,
                        ],
                        "default": None,
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "oneOf": [
                            {"const": "pearson", "description": "Pearson"},
                            {"const": "spearman", "description": "Spearman"},
                            {"const": "cramers_v", "description": "Cramers V"},
                            {
                                "const": "correlation_ratio",
                                "description": "Correlation Ratio",
                            },
                            {
                                "const": "point_biserial",
                                "description": "Point Biserial",
                            },
                            {"const": None, "description": "None"},
                        ],
                    },
                    "null_handling": {
                        "type": ["string", "null"],
                        "title": "Null Handling Strategy",
                        "description": "Strategy for handling null values in the selected fields. 'drop' removes rows with nulls, 'fill_zero' replaces nulls with zero, 'fill_mean' replaces nulls with the mean value. Default is 'drop'.",
                        "enum": ["drop", "fill_zero", "fill_mean", None],
                        "default": "drop",
                        "x-component": "Select",
                        "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                        "oneOf": [
                            {"const": "drop", "description": "Drop"},
                            {"const": "fill_zero", "description": "Fill Zero"},
                            {"const": "fill_mean", "description": "Fill Mean"},
                        ],
                    },
                    "mvf_parser": {
                        "type": ["string", "null"],
                        "title": "Multi-Valued Field Parser",
                        "description": "Optional string lambda to parse multi-valued fields (MVF), e.g., 'lambda x: x.split(';')'. Used if either field contains delimited values.",
                        "default": None,
                    },
                },
                "required": ["method", "field1", "field2", "null_handling"],
            },
        ],
    }
