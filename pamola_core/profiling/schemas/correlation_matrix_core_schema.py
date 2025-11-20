"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Matrix Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of correlation matrix configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines multi-field correlation analysis parameters, method mappings, and threshold settings
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Multi-field correlation analysis configuration
- Correlation method mapping per field pair
- Threshold-based filtering for significant correlations
- Null handling strategy validation

Changelog:
1.0.0 - 2025-11-11 - Initial creation of correlation matrix core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CorrelationMatrixOperationConfig(OperationConfig):
    """
    Core configuration schema for CorrelationMatrixOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Correlation Matrix Operation Core Configuration",
        "description": "Core schema for correlation matrix profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "title": "Fields List",
                        "description": (
                            "List of field (column) names to include in the correlation matrix analysis. "
                            "Must contain at least two fields, all of which must exist in the input DataFrame."
                        ),
                        "items": {"type": "string"},
                        "minItems": 2,
                        "default": [],
                    },
                    "methods": {
                        "type": ["object", "null"],
                        "title": "Correlation Methods Mapping",
                        "description": (
                            "Optional dictionary mapping field pairs to specific correlation methods.\n"
                            "Keys are field pair names (e.g., 'field1_field2'), values are method names:\n"
                            "- 'pearson': Linear correlation for numeric data\n"
                            "- 'spearman': Rank-based correlation for monotonic relationships\n"
                            "- 'cramers_v': Association measure for categorical data\n"
                            "- 'correlation_ratio': Numeric-categorical correlation\n"
                            "- 'point_biserial': Numeric-binary correlation\n"
                            "If not provided, methods are auto-selected based on data types."
                        ),
                        "items": {
                            "type": "string",
                            "title": "Correlation Method",
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
                        "default": None,
                    },
                    "min_threshold": {
                        "type": "number",
                        "title": "Minimum Correlation Threshold",
                        "description": (
                            "Minimum absolute correlation value required for a correlation to be "
                            "considered significant and included in the results. "
                            "Value must be between 0.0 and 1.0."
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.3,
                    },
                    "null_handling": {
                        "type": "string",
                        "title": "Null Handling Strategy",
                        "description": (
                            "Strategy for handling null values in the selected fields:\n"
                            "- 'drop': Remove rows containing null values (pairwise deletion)\n"
                            "- 'fill_zero': Replace null values with zero\n"
                            "- 'fill_mean': Replace null values with the mean of each field"
                        ),
                        "oneOf": [
                            {"const": "drop", "description": "Drop"},
                            {"const": "fill_zero", "description": "Fill with Zero"},
                            {"const": "fill_mean", "description": "Fill with Mean"},
                        ],
                        "default": "drop",
                    },
                },
                "required": ["fields"],
            },
        ],
    }
