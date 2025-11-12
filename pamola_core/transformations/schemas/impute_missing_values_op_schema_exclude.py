"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Impute Missing Values Operation Exclude Fields
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Defines field exclusion rules for ImputeMissingValuesOperation UI rendering in PAMOLA.CORE.
- Specifies which configuration fields should be hidden from the user interface
- Controls the visibility of inherited BaseOperationConfig parameters
- Simplifies the UI by exposing only imputation-specific parameters
- Maintains consistency across operation configuration interfaces

Purpose:
This module contains the IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS list, which filters out
unnecessary or advanced configuration options from the impute missing values operation UI.
Only the most relevant parameters are exposed to users, improving usability and reducing
configuration complexity.

Visible Parameters (Not Excluded):
- field_strategies: Imputation strategies per field
- invalid_values: Invalid values per field to be treated as missing
- generate_visualization: Toggle for automatic visualization generation
- force_recalculation: Force re-execution bypassing cache

Hidden Parameters (Excluded):
- All BaseOperationConfig fields except generate_visualization and force_recalculation
- The 'name' parameter from ImputeMissingValuesConfig

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS = [
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
    "output_field_name",
    "null_strategy",
    "use_cache",
    "save_output",
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
]
