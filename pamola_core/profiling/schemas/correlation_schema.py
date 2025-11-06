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
                    },
                    "field2": {
                        "type": "string",
                        "title": "Second Field Name",
                        "description": "Name of the second field (column) to analyze for correlation. Must exist in the input DataFrame.",
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
                "required": ["method", "null_handling"],
            },
        ],
    }


class CorrelationMatrixOperationConfig(OperationConfig):
    """Configuration for CorrelationMatrixOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Correlation Matrix Operation Configuration",
        "description": "Configuration schema for correlation matrix profiling operations. Defines parameters for analyzing correlations between multiple fields, including method selection, minimum threshold, and null handling.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "title": "Fields List",
                        "description": "List of field (column) names to include in the correlation matrix analysis. Must contain at least two fields, all of which must exist in the input DataFrame.",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "default": [],
                    },
                    "methods": {
                        "type": ["object", "null"],
                        "title": "Correlation Methods Mapping",
                        "description": "Optional dictionary mapping field pairs to specific correlation methods. Keys are field pair names (e.g., 'field1_field2'), values are method names ('pearson', 'spearman', etc.). If not provided, methods are auto-selected.",
                        "additionalProperties": {
                            "type": "string",
                            "enum": [
                                "pearson",
                                "spearman",
                                "cramers_v",
                                "correlation_ratio",
                                "point_biserial",
                            ],
                        },
                        "default": None,
                    },
                    "min_threshold": {
                        "type": "number",
                        "title": "Minimum Correlation Threshold",
                        "description": "Minimum absolute correlation value required for a correlation to be considered significant and included in the results. Value must be between 0.0 and 1.0. Default is 0.3.",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.3,
                    },
                    "null_handling": {
                        "type": "string",
                        "title": "Null Handling Strategy",
                        "description": "Strategy for handling null values in the selected fields. 'drop' removes rows with nulls, 'fill_zero' replaces nulls with zero, 'fill_mean' replaces nulls with the mean value. Default is 'drop'.",
                        "enum": ["drop", "fill_zero", "fill_mean"],
                        "default": "drop",
                    },
                },
                "required": ["fields"],
            },
        ],
    }
