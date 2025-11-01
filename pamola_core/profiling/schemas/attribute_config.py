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

from pamola_core.utils.ops.op_config import BaseOperationConfig, OperationConfig

class DataAttributeProfilerOperationConfig(OperationConfig):
    """Configuration for DataAttributeProfilerOperation with BaseOperationConfig merged."""

    schema = {
        "type": "object",
        "title": "Attribute Profiler Operation Configuration",
        "description": "Configuration schema for attribute profiling operations.",
        "allOf": [
            BaseOperationConfig.schema, # merge all common BaseOperation fields
            {
                "type": "object",
                "properties": {
                    "dictionary_path": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Attribute Dictionary Path",
                        "description": "Path to a custom attribute dictionary file for role detection. If null, uses the default built-in dictionary."
                    },
                    "language": {
                        "type": "string",
                        "default": "en",
                        "title": "Language",
                        "description": "Language code for keyword matching and attribute role detection (e.g., 'en' for English, 'vi' for Vietnamese)."
                    },
                    "sample_size": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 10,
                        "title": "Sample Size",
                        "description": "Number of sample values to extract and inspect per column for profiling."
                    },
                    "max_columns": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "default": None,
                        "title": "Max Columns",
                        "description": "Maximum number of columns to analyze in the dataset. If null, all columns are analyzed."
                    },
                    "id_column": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "ID Column",
                        "description": "Name of the column used as a unique record identifier for record-level analysis. Optional."
                    },
                },
                "required": [],
            },
        ],
    }

