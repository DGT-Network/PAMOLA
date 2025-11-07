"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Remove Fields Exclude Fields
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from remove fields operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for removal.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

REMOVE_FIELDS_EXCLUDE_FIELDS = [
    "name",
    "description",
    "scope",
    "config",
    "optimize_memory",
    "adaptive_chunk_size",
    "output_field_name",
    "column_prefix",
    "null_strategy",
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
    "force_recalculation",
    "generate_visualization",
    "save_output",
    "fields_to_remove",
]
