"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL Linkage Operation - excluded fields
Package:       pamola_core.metrics.schemas
License:       BSD 3-Clause

Field names that are engine/config-level and must not be offered as user inputs in
form builders for ``CVPLLinkageOperation``. ``columns``, ``column_mapping`` and
``sample_size`` do not exist for CVPL: the attacker's attributes are ``qids`` and
``auxiliary`` and the source row count is always the denominator.
"""

CVPL_EXCLUDE_FIELDS = [
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
    "generate_visualization",
    "use_encryption",
    "encryption_mode",
    "encryption_key",
    "save_output",
    "mode",
    "output_field_name",
    "overwrite_existing_output",
    "column_prefix",
    "null_strategy",
    "direction",
    "linkage_unit",
    "max_group_pairs",
    "chunk_pairs",
    "tie_tolerance",
    "zero_norm_tolerance",
]
