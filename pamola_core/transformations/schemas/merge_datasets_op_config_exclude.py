
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
MERGE_DATASETS_EXCLUDE_FIELDS = ["config", "scope", "engine", "mode", "column_prefix", "null_strategy", "output_field_name"]