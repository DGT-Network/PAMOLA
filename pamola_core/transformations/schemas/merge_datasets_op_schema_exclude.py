"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Merge Datasets Exclude Fields
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from merge datasets operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for merging.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

MERGE_DATASETS_EXCLUDE_FIELDS = [
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
    "left_dataset_name",
    "right_dataset_name",
    "right_dataset_path",
]
