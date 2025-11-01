"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Fake Name Exclude Fields
Package:       pamola_core.fake_data.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from fake name operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for fake name generation.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""
NAME_FAKE_EXCLUDE_FIELDS = ["config", "scope", "engine", "generator", "generator_params"]