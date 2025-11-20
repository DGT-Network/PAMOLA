"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Impute Missing Values Core Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of impute missing values configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines imputation strategies for numeric, categorical, date, and text data types
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field-specific imputation strategy configuration
- Invalid value identification and handling
- Multiple imputation methods per data type
- Support for statistical, constant, and sampling strategies

Changelog:
1.0.0 - 2025-01-15 - Initial creation of impute missing values core schema
1.1.0 - 2025-11-11 - Updated with enhanced strategy descriptions
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class ImputeMissingValuesConfig(OperationConfig):
    """
    Core configuration schema for ImputeMissingValuesOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Impute Missing Values Operation Core Configuration",
        "description": "Core schema for Impute Missing Values operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_strategies": {
                        "type": ["object", "null"],
                        "title": "Field Strategies",
                        "description": (
                            "Dictionary mapping field names to imputation strategy configurations.\n\n"
                            "Each field requires:\n"
                            "- data_type: 'numeric', 'categorical', 'date', or 'text'\n"
                            "- imputation_strategy: Method to fill missing values\n"
                            "- constant_value: Value to use (only for 'constant' strategies)\n\n"
                            "Strategies by data type:\n"
                            "- Numeric: 'constant', 'mean', 'median', 'mode', 'min', 'max', 'interpolation'\n"
                            "- Categorical: 'constant', 'mode', 'most_frequent', 'random_sample'\n"
                            "- Date: 'constant_date', 'mean_date', 'median_date', 'mode_date', 'previous_date', 'next_date'\n"
                            "- Text: 'constant', 'most_frequent', 'random_sample'\n\n"
                            "Example: {'age': {'data_type': 'numeric', 'imputation_strategy': 'median'}}"
                        ),
                        "default": None,
                    },
                    "invalid_values": {
                        "type": ["object", "null"],
                        "title": "Invalid Values",
                        "description": (
                            "Dictionary mapping field names to lists of values that should be treated as missing.\n\n"
                            "These values will be replaced with NaN before imputation is applied.\n\n"
                            "Common placeholders:\n"
                            "- Numeric: -1, -999, 0, 99999\n"
                            "- Text: 'N/A', 'Unknown', 'null', '', '--'\n\n"
                            "Example: {'age': [-1, 0], 'name': ['N/A', 'Unknown', '']}"
                        ),
                        "default": None,
                    },
                },
            },
        ],
    }
