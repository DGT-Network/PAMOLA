"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Impute Missing Values Config Schema
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating impute missing values operations in PAMOLA.CORE.
- Supports field-specific imputation strategies for numeric, categorical, date, and text data
- Handles invalid value identification and replacement
- Provides multiple imputation methods (statistical, constant, sampling)
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of impute missing values config file
1.1.0 - 2025-11-11 - Updated with x-component, x-group, and enhanced descriptions
"""

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class ImputeMissingValuesConfig(OperationConfig):
    """
    Configuration schema for ImputeMissingValuesOperation.

    Extends BaseOperationConfig with imputation parameters for handling missing and invalid values
    across different data types (numeric, categorical, date, text).
    """

    schema = {
        "type": "object",
        "title": "Impute Missing Values Configuration",
        "description": (
            "Configuration options for imputing missing or invalid values in datasets. "
            "Supports field-specific strategies for different data types."
        ),
        "allOf": [
            BaseOperationConfig.schema,  # merge base schema
            {
                "type": "object",
                "properties": {
                    # === Imputation strategies ===
                    "field_strategies": {
                        "type": ["object", "null"],
                        "title": "Field Strategies",
                        "x-component": "Input",
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
                        "x-group": GroupName.FIELD_STRATEGIES_CONFIGURATION,
                        "default": None,
                    },
                    # === Invalid values ===
                    "invalid_values": {
                        "type": ["object", "null"],
                        "title": "Invalid Values",
                        "x-component": "Input",
                        "description": (
                            "Dictionary mapping field names to lists of values that should be treated as missing.\n\n"
                            "These values will be replaced with NaN before imputation is applied.\n\n"
                            "Common placeholders:\n"
                            "- Numeric: -1, -999, 0, 99999\n"
                            "- Text: 'N/A', 'Unknown', 'null', '', '--'\n\n"
                            "Example: {'age': [-1, 0], 'name': ['N/A', 'Unknown', '']}"
                        ),
                        "x-group": GroupName.INVALID_VALUES_CONFIGURATION,
                        "default": None,
                    },
                },
            },
        ],
    }
