"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Base Operation Core Schema
Package:       pamola_core.utils.ops.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Core JSON Schema definition for backend validation of base operation configurations in PAMOLA.CORE.
- Pure JSON Schema (Draft-07+) for runtime validation and type safety
- Defines performance, output, encryption, and execution control parameters
- Contains business logic validation rules (type constraints, conditionals, enums)
- Free of UI metadata - only validation rules and data structure
- Used by backend services for parameter validation before operation execution

Key Features:
- Memory optimization and adaptive processing controls
- Multiple execution engines (auto, pandas, dask) with parallel processing
- Flexible output formats (CSV, Parquet, JSON) with caching support
- Security features including encryption modes (age, simple, none)
- Visualization configuration with timeout and backend selection
- Conditional validation using JSON Schema if/then patterns

Changelog:
1.0.0 - 2025-01-15 - Initial creation of base operation core schema
"""

from pamola_core.utils.ops.op_config import OperationConfig


class BaseOperationConfig(OperationConfig):
    """
    Core configuration schema for BaseOperation backend validation.

    Defines pure JSON Schema validation rules without UI metadata.
    Used for runtime parameter validation and type checking.
    """

    schema = {
        "type": "object",
        "title": "Base Operation Core Configuration",
        "description": "Core JSON Schema for base operation configuration validation.",
        "properties": {
            "name": {
                "type": "string",
                "title": "Operation Name",
                "description": "Human-readable name of the operation.",
                "default": "",
            },
            "description": {
                "type": "string",
                "title": "Description",
                "description": "Optional detailed description of this operation.",
                "default": "",
            },
            "scope": {
                "type": ["object", "null"],
                "title": "Execution Scope",
                "description": "Optional scope or context within which the operation will execute.",
            },
            "config": {
                "type": ["object", "null"],
                "title": "Custom Configuration",
                "description": "Additional configuration parameters, typically used internally.",
            },
            "optimize_memory": {
                "type": "boolean",
                "title": "Optimize Memory Usage",
                "description": "If true, operations will use memory-efficient data structures.",
                "default": True,
            },
            "adaptive_chunk_size": {
                "type": "boolean",
                "title": "Adaptive Chunk Size",
                "description": "Automatically adjust chunk size based on data volume and system resources.",
                "default": True,
            },
            "mode": {
                "type": "string",
                "title": "Mode",
                "oneOf": [
                    {"const": "REPLACE", "description": "Replace"},
                    {"const": "ENRICH", "description": "Enrich"},
                ],
                "description": "Defines how results will be applied to the dataset: REPLACE overwrites, ENRICH adds new data.",
                "default": "REPLACE",
            },
            "output_field_name": {
                "type": ["string", "null"],
                "title": "Output Field Name",
                "description": "Optional custom name for the generated or modified output field.",
                "default": "",
            },
            "column_prefix": {
                "type": "string",
                "title": "Column Prefix",
                "description": "Prefix to apply to newly generated columns.",
                "default": "_",
            },
            "null_strategy": {
                "type": "string",
                "title": "Handle Null Values",
                "oneOf": [
                    {"const": "PRESERVE", "description": "Preserve"},
                    {"const": "EXCLUDE", "description": "Exclude"},
                    {"const": "ANONYMIZE", "description": "Anonymize"},
                    {"const": "ERROR", "description": "Error"},
                ],
                "description": "Determines how null or missing values are handled during processing.",
                "default": "PRESERVE",
            },
            "engine": {
                "type": "string",
                "oneOf": [
                    {"const": "auto", "description": "Auto"},
                    {"const": "pandas", "description": "Pandas"},
                    {"const": "dask", "description": "Dask"},
                ],
                "title": "Execution Engine",
                "description": "Execution backend used to process data. 'auto' selects the best engine automatically.",
                "default": "auto",
            },
            "use_dask": {
                "type": "boolean",
                "title": "Enable Dask Processing",
                "description": "If true, operations are distributed across multiple Dask workers.",
                "default": False,
            },
            "npartitions": {
                "type": ["integer", "null"],
                "title": "Number of Dask Partitions",
                "description": "Number of partitions to split the dataset into for parallel processing.",
                "minimum": 1,
            },
            "dask_partition_size": {
                "type": ["string", "null"],
                "title": "Dask Partition Size",
                "description": "Approximate size of each Dask partition (e.g. '100MB').",
                "default": "100MB",
            },
            "use_vectorization": {
                "type": "boolean",
                "title": "Enable Vectorization",
                "description": "Use NumPy vectorized operations for faster computation where applicable.",
                "default": False,
            },
            "parallel_processes": {
                "type": ["integer", "null"],
                "title": "Parallel Processes",
                "description": "Number of CPU processes to use for parallel execution.",
                "minimum": 1,
            },
            "chunk_size": {
                "type": "integer",
                "title": "Chunk Size",
                "description": "Number of rows to process per batch when streaming or chunked processing is enabled.",
                "minimum": 1,
                "default": 10000,
            },
            "use_cache": {
                "type": "boolean",
                "title": "Use Result Cache",
                "description": "Cache the operation output to speed up repeated runs with the same inputs.",
                "default": False,
            },
            "output_format": {
                "type": "string",
                "oneOf": [
                    {"const": "csv", "description": "csv"},
                    {"const": "parquet", "description": "parquet"},
                    {"const": "json", "description": "json"},
                ],
                "title": "Output Format",
                "description": "Format used when saving processed output data.",
                "default": "csv",
            },
            "save_output": {
                "type": "boolean",
                "title": "Save Output",
                "description": "If true, persist processed data to disk or database.",
                "default": True,
            },
            "visualization_theme": {
                "type": ["string", "null"],
                "title": "Visualization Theme",
                "description": "Optional color or layout theme for visualizations.",
            },
            "visualization_backend": {
                "type": ["string", "null"],
                "oneOf": [
                    {"const": "plotly", "description": "Plotly"},
                    {"const": "matplotlib", "description": "Matplotlib"},
                ],
                "title": "Visualization Backend",
                "description": "Rendering backend for generated plots and charts.",
                "default": "plotly",
            },
            "visualization_strict": {
                "type": "boolean",
                "title": "Strict Visualization Mode",
                "description": "If true, visualization errors will stop execution instead of being ignored.",
                "default": False,
            },
            "visualization_timeout": {
                "type": "integer",
                "title": "Visualization Timeout (seconds)",
                "description": "Maximum time allowed for generating visualization before timing out.",
                "minimum": 1,
                "default": 120,
            },
            "use_encryption": {
                "type": "boolean",
                "title": "Enable Encryption",
                "description": "Encrypt sensitive data outputs using the selected encryption mode.",
                "default": False,
            },
            "encryption_mode": {
                "type": ["string", "null"],
                "oneOf": [
                    {"const": "age", "description": "Age"},
                    {"const": "simple", "description": "Simple"},
                    {"const": "none", "description": "None"},
                ],
                "title": "Encryption Mode",
                "description": "Algorithm used for encrypting outputs. 'none' disables encryption.",
                "default": "none",
            },
            "encryption_key": {
                "type": ["string", "null"],
                "title": "Encryption Key",
                "description": "Key or passphrase used for encryption when enabled.",
            },
            "force_recalculation": {
                "type": "boolean",
                "title": "Force Recalculation",
                "description": "Re-run operation even if cached results exist.",
                "default": False,
            },
            "generate_visualization": {
                "type": "boolean",
                "title": "Generate Visualization",
                "description": "If true, automatically generate visualization outputs after processing.",
                "default": True,
            },
        },
        "allOf": [
            {
                "if": {"properties": {"mode": {"const": "ENRICH"}}},
                "then": {
                    "anyOf": [
                        {
                            "properties": {
                                "output_field_name": {"type": "string", "minLength": 1}
                            }
                        },
                        {
                            "properties": {
                                "output_field_name": {"type": "null"},
                                "column_prefix": {"type": "string", "minLength": 1},
                            },
                            "required": ["column_prefix"],
                        },
                    ]
                },
            }
        ],
    }
