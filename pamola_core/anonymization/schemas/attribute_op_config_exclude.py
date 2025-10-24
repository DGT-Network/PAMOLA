"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Attribute Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from attribute anonymization operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for attribute anonymization.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

ATTRIBUTE_EXCLUDE_FIELDS = ["config", "scope", "engine"]