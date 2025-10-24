"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Add/Modify Fields Exclude Fields
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from add/modify fields operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for transformation.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

ADD_MODIFY_FIELDS_EXCLUDE_FIELDS = ["config", "scope", "engine", "null_strategy"]