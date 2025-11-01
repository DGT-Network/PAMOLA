"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Split By ID Values Exclude Fields
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from split by ID values operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for splitting.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS = ["config", "scope", "engine", "mode", "column_prefix", "null_strategy", "output_field_name"]