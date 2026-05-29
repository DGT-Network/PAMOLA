"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Hash-Based Pseudonymization Exclude Fields
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
Defines field names to exclude from hash-based pseudonymization operations
in PAMOLA.CORE. These fields are engine/config-level and should not be selectable
as data columns in UI field pickers.

Changelog:
1.0.0 - 2025-06-15 - Initial creation
"""

HASH_BASED_PSEUDONYMIZATION_EXCLUDE_FIELDS = [
    "name",
    "description",
    "scope",
    "config",
    "optimize_memory",
    "adaptive_chunk_size",
    "engine",
    "use_dask",
    "npartitions",
    "dask_partition_size",
    "use_vectorization",
    "parallel_processes",
    "chunk_size",
    "output_format",
    "visualization_theme",
    "visualization_backend",
    "visualization_strict",
    "visualization_timeout",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
    "save_output",
    "field_name",
    "additional_fields",
    "salt_config",
    "salt_file",
    "quasi_identifiers",
    "ka_risk_field",
    "risk_threshold",
    "vulnerable_record_strategy",
]
