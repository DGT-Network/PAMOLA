"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Config Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Configuration schema for defining and validating attribute profiling operations in PAMOLA.CORE.
Supports parameters for attribute dictionaries, language, and profiling options.
Compatible with JSON Schema, easy to integrate and extend.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of attribute config file
"""

from pamola_core.common.enum.custom_components import CustomComponents
from pamola_core.common.enum.form_groups import GroupName
from pamola_core.common.enum.language_enum import Language
from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig


class DataAttributeProfilerOperationConfig(OperationConfig):
    """Configuration for DataAttributeProfilerOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Attribute Profiler Operation Configuration",
        "description": "Configuration schema for attribute profiling operations.",
        "allOf": [
            BaseOperationConfig.schema,  # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "default": Language.ENGLISH.value,
                        "title": "Language",
                        "x-component": "Select",
                        "oneOf": [
                            {
                                "const": Language.ENGLISH.value,
                                "description": "English",
                            },
                            {
                                "const": Language.VIETNAMESE.value,
                                "description": "Vietnamese",
                            },
                            {
                                "const": Language.RUSSIAN.value,
                                "description": "Russian",
                            },
                        ],
                        "description": "Language code for keyword matching and attribute role detection (e.g., 'en' for English, 'vi' for Vietnamese).",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "sample_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                        "title": "Sample Size",
                        "x-component": "NumberPicker",
                        "description": "Number of sample values to extract and inspect per column for profiling.",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "max_columns": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "default": None,
                        "title": "Max Columns",
                        "x-component": "NumberPicker",
                        "description": "Maximum number of columns to analyze in the dataset. If null, all columns are analyzed.",
                        "x-group": GroupName.ANALYSIS_CONFIGURATION,
                    },
                    "dictionary_path": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Attribute Dictionary Path",
                        "x-component": CustomComponents.UPLOAD,
                        "description": "Path to a custom attribute dictionary file for role detection. If null, uses the default built-in dictionary.",
                        "x-group": GroupName.DICTIONARY_CONFIGURATION,
                    },
                },
                "required": [],
            },
        ],
    }
