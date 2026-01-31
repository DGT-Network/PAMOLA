"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Numeric Noise Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from uniform numeric noise operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for uniform numeric noise.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

RECORD_EXCLUDE_FIELDS = [
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
    "use_secure_random",
    "random_seed",
    "field_name",
    "ka_risk_field",
    "risk_threshold",
    "vulnerable_record_strategy",
]
