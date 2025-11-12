"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Matrix Operation Exclude Fields
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Defines field exclusion rules for CorrelationMatrixOperation UI rendering in PAMOLA.CORE.
- Specifies which configuration fields should be hidden from the user interface
- Controls the visibility of inherited BaseOperationConfig parameters
- Simplifies the UI by exposing only correlation-specific parameters
- Maintains consistency across operation configuration interfaces

Purpose:
This module contains the CORRELATION_MATRIX_EXCLUDE_FIELDS list, which filters out
unnecessary or advanced configuration options from the correlation matrix operation UI.
Only the most relevant parameters are exposed to users, improving usability and reducing
configuration complexity.

Visible Parameters (Not Excluded):
- methods: Correlation method mapping for field pairs
- min_threshold: Minimum correlation threshold for filtering
- null_handling: Strategy for handling missing values
- generate_visualization: Toggle for automatic visualization generation
- force_recalculation: Force re-execution bypassing cache

Hidden Parameters (Excluded):
- All BaseOperationConfig fields except generate_visualization and force_recalculation
- The 'fields' parameter from CorrelationMatrixOperationConfig (handled separately)

Changelog:
1.0.0 - 2025-11-11 - Initial creation of exclude fields list
                   - Defined 29 excluded fields from BaseOperationConfig
                   - Added 'fields' exclusion from CorrelationMatrixOperationConfig
"""

CORRELATION_MATRIX_EXCLUDE_FIELDS = [
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
    "mode",
    "output_field_name",
    "column_prefix",
    "null_strategy",
    "use_cache",
    "output_format",
    "save_output",
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
    "fields",
]
