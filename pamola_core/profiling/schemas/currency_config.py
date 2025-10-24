"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Currency Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating currency profiling operations in PAMOLA.CORE.
Supports parameters for field names, locale, binning, and outlier/normality detection.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of currency config file
"""

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class CurrencyOperationConfig(OperationConfig):
    """Configuration for CurrencyOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "locale": {"type": "string", "default": "en_US"},
                    "bins": {"type": "integer", "minimum": 1, "default": 10},
                    "detect_outliers": {"type": "boolean", "default": True},
                    "test_normality": {"type": "boolean", "default": True},
                },
                "required": ["field_name"],
            },
        ],
    }