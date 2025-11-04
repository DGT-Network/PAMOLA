"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Record Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from record anonymization operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for record anonymization.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

RECORD_EXCLUDE_FIELDS = [
    # Fields from BaseOperationConfig
    "name",
    "description",
    "scope",
    "config",
    "optimize_memory",
    "adaptive_chunk_size",
    "mode",
    "column_prefix",
    "output_field_name",
    "null_strategy",
    "engine",
    "use_dask",
    "npartitions",
    "dask_partition_size",
    "use_vectorization",
    "parallel_processes",
    "chunk_size",
    "use_cache",
    "output_format",
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
    "generate_visualization",
    "save_output",
    # Fields from RecordSuppressionConfig
    "field_name",
    "suppression_mode",
]
