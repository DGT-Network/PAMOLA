"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split By ID Values Exclude Fields
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from split by ID values operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for splitting.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS = [
    "name",
    "description",
    "scope",
    "config",
    "optimize_memory",
    "adaptive_chunk_size",
    "mode",
    "output_field_name",
    "column_prefix",
    "null_strategy",
    "use_cache",
    "engine",
    "use_dask",
    "npartitions",
    "dask_partition_size",
    "use_vectorization",
    "parallel_processes",
    "chunk_size",
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
    "invalid_values",
]
