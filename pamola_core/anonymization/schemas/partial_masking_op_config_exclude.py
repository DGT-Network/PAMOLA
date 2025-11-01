"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from partial masking operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for masking.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

PARTIAL_MASKING_EXCLUDE_FIELDS = [
    "config",
    "scope",
    "engine",
    "encryption_mode",
    "encryption_key",
]
