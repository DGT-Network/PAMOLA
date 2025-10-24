
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Aggregate Records Exclude Fields
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from aggregate records operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for aggregation.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

AGGREGATE_RECORDS_EXCLUDE_FIELDS = ["config", "scope", "engine", "mode", "column_prefix", "null_strategy", "output_field_name"]