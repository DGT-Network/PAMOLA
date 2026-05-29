"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Fidelity Metric Exclude Fields
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
Defines field names to exclude from fidelity metric operation configurations
in PAMOLA.CORE. These fields are engine/config-level and should not be exposed
as user-configurable inputs in UI form pickers.
"""

FIDELITY_EXCLUDE_FIELDS = [
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
    "output_format",
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
    "save_output",
    "mode",
    "output_field_name",
    "column_prefix",
    "null_strategy",
]
