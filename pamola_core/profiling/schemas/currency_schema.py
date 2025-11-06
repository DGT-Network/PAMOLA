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

from pamola_core.common.enum.form_groups import GroupName
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class CurrencyOperationConfig(OperationConfig):
    """Configuration for CurrencyOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Currency Operation Configuration",
        "description": "Configuration schema for currency profiling operations. Defines parameters for analyzing a currency field, including locale, binning, and outlier/normality detection.",
        "allOf": [
            BaseOperationConfig.schema,  # merge common base fields
            {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "string",
                        "title": "Field Name",
                        "description": "Name of the currency field (column) to analyze. Must exist in the input DataFrame."
                    },
                    "locale": {
                        "type": "string",
                        "title": "Locale",
                        "description": "Locale code for parsing currency values (e.g., 'en_US', 'fr_FR'). Determines currency formatting and symbols.",
                        "default": "en_US",
                        "x-component": "Select",
                        "x-group": GroupName.CURRENCY_PARSING_SETTINGS,
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
                        ]
                    },
                    "bins": {
                        "type": "number",
                        "title": "Bins",
                        "description": "Number of bins to use for histogram analysis of the currency field. Must be at least 1.",
                        "minimum": 1,
                        "default": 10,
                        "maximum": 100,
                        "x-component": "NumberPicker",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                    "detect_outliers": {
                        "type": "boolean",
                        "title": "Detect Outliers",
                        "description": "Whether to detect and report outliers in the currency field during analysis.",
                        "default": True,
                        "x-component": "Checkbox",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                    "test_normality": {
                        "type": "boolean",
                        "title": "Test Normality",
                        "description": "Whether to perform normality testing on the currency field values.",
                        "default": True,
                        "x-component": "Checkbox",
                        "x-group": GroupName.DISTRIBUTION_AND_ANALYSIS_SETTINGS,
                    },
                },
                "required": ["field_name", "locale", "bins"],
            },
        ],
    }