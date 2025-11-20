"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Data Attribute Profiler Core Schema
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of attribute profiling configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines attribute profiling parameters, language settings, and dictionary configuration
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Language configuration for keyword matching
- Sample size and column limits for profiling
- Custom dictionary path support
- Attribute role detection controls

Changelog:
1.0.0 - 2025-01-15 - Initial creation of data attribute profiler core schema
"""

from pamola_core.common.enum.language_enum import Language
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class DataAttributeProfilerOperationConfig(OperationConfig):
    """
    Core configuration schema for DataAttributeProfilerOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Attribute Profiler Operation Configuration",
        "description": "Core schema for attribute profiling operation configuration.",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "default": Language.ENGLISH.value,
                        "title": "Language",
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
                    },
                    "sample_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                        "title": "Sample Size",
                        "description": "Number of sample values to extract and inspect per column for profiling.",
                    },
                    "max_columns": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "default": None,
                        "title": "Max Columns",
                        "description": "Maximum number of columns to analyze in the dataset. If null, all columns are analyzed.",
                    },
                    "dictionary_path": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Attribute Dictionary Path",
                        "description": "Path to a custom attribute dictionary file for role detection. If null, uses the default built-in dictionary.",
                    },
                },
                "required": ["language", "sample_size"],
            },
        ],
    }
