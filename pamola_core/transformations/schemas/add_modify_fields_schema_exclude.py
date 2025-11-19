"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add or Modify Fields Operation Exclude Fields
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Defines field exclusion rules for AddOrModifyFieldsOperation UI rendering in PAMOLA.CORE.
- Specifies which configuration fields should be hidden from the user interface
- Controls the visibility of inherited BaseOperationConfig parameters
- Simplifies the UI by exposing only field transformation-specific parameters
- Maintains consistency across operation configuration interfaces

Purpose:
This module contains the ADD_MODIFY_FIELDS_EXCLUDE_FIELDS list, which filters out
unnecessary or advanced configuration options from the add/modify fields operation UI.
Only the most relevant parameters are exposed to users, improving usability and reducing
configuration complexity.

Visible Parameters (Not Excluded):
- field_operations: Operations to add or modify fields
- lookup_tables: Mapping tables for lookup-based operations
- generate_visualization: Toggle for automatic visualization generation
- force_recalculation: Force re-execution bypassing cache

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
1.1.0 - 2025-11-11 - Updated to match UI requirements, excluded 30 fields
"""

ADD_MODIFY_FIELDS_EXCLUDE_FIELDS = [
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
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
]
