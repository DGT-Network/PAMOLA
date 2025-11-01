"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Impute Missing Values Config Schema
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating impute missing values operations in PAMOLA.CORE.
- Supports field-specific imputation strategies and invalid value handling
- Compatible with JSON Schema, easy to integrate and extend

Changelog:
1.0.0 - 2025-01-15 - Initial creation of impute missing values config file
"""
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class ImputeMissingValuesConfig(OperationConfig):
    """Configuration for ImputeMissingValuesOperation with BaseOperationConfig merged."""

    schema = {
        "title": "ImputeMissingValuesConfig",
        "description": "Schema for imputing missing or invalid values in datasets. Supports field-specific imputation strategies and invalid value handling.",
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "title": "ImputeMissingValuesConfig Properties",
                "description": "Properties for configuring field-level imputation strategies and invalid value handling.",
                "properties": {
                    "field_strategies": {
                        "type": ["object", "null"],
                        "title": "Field Strategies",
                        "description": "Dictionary mapping field names to imputation strategies (e.g., mean, median, mode, constant, interpolation, etc.)."
                    },
                    "invalid_values": {
                        "type": ["object", "null"],
                        "title": "Invalid Values",
                        "description": "Dictionary mapping field names to lists of values considered invalid and to be imputed."
                    },
                },
            },
        ],
    }
