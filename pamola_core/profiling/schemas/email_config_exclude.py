
"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Exclude Fields
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from email profiling operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for email profiling.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

EMAIL_EXCLUDE_FIELDS = ["config", "scope", "engine"]