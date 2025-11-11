"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Matrix Operation Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating correlation matrix parameters in PAMOLA.CORE.
- Analyzes correlations between multiple numeric and categorical fields
- Supports multiple correlation methods (Pearson, Spearman, Cramér's V, etc.)
- Provides threshold-based filtering for significant correlations
- Handles missing data with configurable null handling strategies
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-11-11 - Initial creation of correlation matrix config file
                   - Added x-component, x-group attributes for UI integration
                   - Enhanced descriptions and field organization
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class CorrelationMatrixOperationConfig(OperationConfig):
    """
    Configuration schema for CorrelationMatrixOperation.

    Extends BaseOperationConfig with correlation analysis parameters for multi-field
    relationship discovery, supporting various correlation methods and null handling strategies.
    """

    schema = {
        "type": "object",
        "title": "Correlation Matrix Operation Configuration",
        "description": (
            "Configuration options for correlation matrix profiling operations. "
            "Supports analysis of correlations between multiple fields with customizable methods and thresholds."
        ),
        "allOf": [
            BaseOperationConfig.schema,  # merge base schema
            {
                "type": "object",
                "properties": {
                    # === Core fields ===
                    "fields": {
                        "type": "array",
                        "title": "Fields List",
                        "x-component": "Select",
                        "description": (
                            "List of field (column) names to include in the correlation matrix analysis. "
                            "Must contain at least two fields, all of which must exist in the input DataFrame."
                        ),
                        "items": {"type": "string"},
                        "minItems": 2,
                        "default": [],
                    },
                    # === Correlation configuration ===
                    "methods": {
                        "type": ["object", "null"],
                        "title": "Correlation Methods Mapping",
                        "x-component": "Input",
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
                        "x-group": GroupName.CORRELATION_CONFIGURATION,
                    },
                    "min_threshold": {
                        "type": "number",
                        "title": "Minimum Correlation Threshold",
                        "x-component": "NumberPicker",
                        "description": (
                            "Minimum absolute correlation value required for a correlation to be "
                            "considered significant and included in the results. "
                            "Value must be between 0.0 and 1.0."
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.3,
                        "x-group": GroupName.CORRELATION_CONFIGURATION,
                    },
                    "null_handling": {
                        "type": "string",
                        "title": "Null Handling Strategy",
                        "x-component": "Select",
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
                        "x-group": GroupName.CORRELATION_CONFIGURATION,
                    },
                },
                "required": ["fields"],
            },
        ],
    }
