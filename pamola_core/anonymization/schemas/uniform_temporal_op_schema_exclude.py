"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Uniform Temporal Noise Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Defines a list of field names to be excluded from uniform temporal noise operations in PAMOLA.CORE.
These fields are typically configuration or engine-related and should not be processed for uniform temporal noise.

Changelog:
1.0.0 - 2025-01-15 - Initial creation of exclude fields list
"""

RECORD_EXCLUDE_FIELDS = ["config", 
                         "scope", 
                         "engine", 
                         "field_name",
                         "noise_range",
                         "random_seed",
                         "use_secure_random",
                         "ka_risk_field", 
                         "risk_threshold", 
                         "vulnerable_record_strategy"]