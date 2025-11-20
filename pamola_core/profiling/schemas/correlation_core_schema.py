"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of correlation profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines correlation analysis parameters, method selection, and null handling strategies
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field pair specification for correlation analysis
- Correlation method configuration (pearson, spearman, cramers_v, etc.)
- Null handling strategy validation
- Multi-valued field parser support

Changelog:
1.0.0 - 2025-01-15 - Initial creation of correlation profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CorrelationOperationConfig(OperationConfig):
    """
    Core configuration schema for CorrelationOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Correlation Operation Core Configuration",
        "description": "Core schema for correlation profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
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
                        "default": "pearson",
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
                        ],
                    },
                    "null_handling": {
                        "type": ["string", "null"],
                        "title": "Null Handling Strategy",
                        "description": "Strategy for handling null values in the selected fields. 'drop' removes rows with nulls, 'fill_zero' replaces nulls with zero, 'fill_mean' replaces nulls with the mean value. Default is 'drop'.",
                        "default": "drop",
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
