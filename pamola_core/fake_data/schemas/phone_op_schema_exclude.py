"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Phone Exclude Fields
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from fake phone operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for fake phone generation.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

PHONE_FAKE_EXCLUDE_FIELDS = [
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
    "field_name",
    "generator",
    "generator_params",
    "mapping_store",
]
