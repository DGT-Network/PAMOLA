"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Identity Exclude Fields
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from identity profiling operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for identity profiling.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

IDENTITY_EXCLUDE_FIELDS = [
    "mode",
    "column_prefix",
    "null_strategy",
    "name",
    "description",
    "scope",
    "config",
    "optimize_memory",
    "adaptive_chunk_size",
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
    "save_output",
    "config",
    "field_name",
    "ka_risk_field",
    "risk_threshold",
    "vulnerable_record_strategy",
]
