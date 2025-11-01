"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Exclude Fields
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from correlation profiling operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for correlation profiling.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

CORRELATION_EXCLUDE_FIELDS = [
    "config",
    "scope",
    "optimize_memory",
    "model",
    "column_prefix",
    "null_strategy",
    "engine",
    "dask_partition_size",
    "output_format",
    "use_dask",
    "npartitions",
    "use_vectorization",
    "parallel_processes",
    "chunk_size",
    "adaptive_chunk_size",
]
CORRELATION_MATRIX_EXCLUDE_FIELDS = [
    "config",
    "scope",
    "optimize_memory",
    "adaptive_chunk_size",
    "mode",
    "column_prefix",
    "null_strategy",
    "engine",
    "dask_partition_size",
    "output_format",
    "save_output",
    "use_dask",
    "npartitions",
    "use_vectorization",
    "parallel_processes",
    "chunk_size",
]
