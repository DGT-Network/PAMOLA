"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Currency Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of currency profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines currency analysis parameters, locale settings, binning, and statistical tests
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Field name specification for currency analysis
- Locale configuration for currency parsing
- Binning controls for histogram analysis
- Outlier detection and normality testing enablement

Changelog:
1.0.0 - 2025-01-15 - Initial creation of currency profiler core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CurrencyOperationConfig(OperationConfig):
    """
    Core configuration schema for CurrencyOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Currency Operation Core Configuration",
        "description": "Core schema for currency profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the currency field (column) to analyze. Must exist in the input DataFrame.",
                    },
                    "locale": {
                        "type": "string",
                        "title": "Locale",
                        "description": "Locale code for parsing currency values (e.g., 'en_US', 'fr_FR'). Determines currency formatting and symbols.",
                        "default": "en_US",
                        "oneOf": [
                            {"const": "en_US", "description": "En US"},
                            {"const": "de_DE", "description": "De DE"},
                            {"const": "it_IT", "description": "It IT"},
                            {"const": "fr_FR", "description": "Fr FR"},
                            {"const": "en_GB", "description": "En GB"},
                            {"const": "ja_JP", "description": "Ja JP"},
                            {"const": "zh_CN", "description": "Zh CN"},
                            {"const": "ru_RU", "description": "Ru RU"},
                            {"const": "es_ES", "description": "Es ES"},
                        ],
                    },
                    "bins": {
                        "type": "number",
                        "title": "Bins",
                        "description": "Number of bins to use for histogram analysis of the currency field. Must be at least 1.",
                        "minimum": 1,
                        "default": 10,
                        "maximum": 100,
                    },
                    "detect_outliers": {
                        "type": "boolean",
                        "title": "Detect Outliers",
                        "description": "Whether to detect and report outliers in the currency field during analysis.",
                        "default": True,
                    },
                    "test_normality": {
                        "type": "boolean",
                        "title": "Test Normality",
                        "description": "Whether to perform normality testing on the currency field values.",
                        "default": True,
                    },
                },
                "required": ["field_name", "locale", "bins"],
            },
        ],
    }
