
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Group Exclude Fields
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from group profiling operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for group profiling.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

GROUP_EXCLUDE_FIELDS = [
  "config",
  "scope",
  "optimize_memory",
  "adaptive_chunk_size",
  "mode",
  "column_prefix",
  "null_strategy",
  "engine",
  "output_format",
  "save_output",
  "use_dask",
  "npartitions",
  "dask_partition_size",
  "use_vectorization",
  "parallel_processes",
  "chunk_size"
]