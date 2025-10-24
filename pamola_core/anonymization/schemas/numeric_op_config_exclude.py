"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Generalization Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from numeric generalization operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed as numeric data.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""
NUMERIC_GENERALIZATION_EXCLUDE_FIELDS = ["config", "scope", "engine"]