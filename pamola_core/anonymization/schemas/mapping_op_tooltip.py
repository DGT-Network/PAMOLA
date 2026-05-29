"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Consistent Mapping Pseudonymization Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
Provides user-facing tooltips for consistent mapping pseudonymization configuration
fields in PAMOLA.CORE.
- Explains pseudonym generation, mapping storage, compound, and conditional options
- Designed for integration with Formily and schema-driven UI builders

Changelog:
1.0.0 - 2025-06-15 - Initial creation
"""


class ConsistentMappingPseudonymizationTooltip:
    field_name = (
        "What it does: Selects the column whose values will be replaced with pseudonyms.\n"
        "• The operation maintains a persistent mapping so the same original value always "
        "produces the same pseudonym across runs.\n"
        "• Validation: The field must exist in the dataset."
    )

    additional_fields = (
        "What it does: Extra columns to include when building a compound identifier.\n"
        "• Only applies when Compound Mode is enabled.\n"
        "• Example: Combining 'first_name' and 'last_name' into a single pseudonym."
    )

    pseudonym_type = (
        "What it does: Controls how new pseudonyms are generated.\n"
        "• UUID: Generates a random UUID v4 for each new value. Globally unique and unguessable. (Default)\n"
        "• Sequential: Assigns incrementing numbers (e.g., 000001, 000002). Predictable order.\n"
        "• Random String: Generates a random alphanumeric string of the specified length."
    )

    pseudonym_prefix = (
        "What it does: Prepends a fixed string to every generated pseudonym.\n"
        "• Example: A prefix of 'USR-' on a UUID produces 'USR-550e8400-...'.\n"
        "• Default: None (no prefix)."
    )

    pseudonym_suffix = (
        "What it does: Appends a fixed string to every generated pseudonym.\n"
        "• Example: A suffix of '-ID' on a sequential pseudonym produces '000001-ID'.\n"
        "• Default: None (no suffix)."
    )

    pseudonym_length = (
        "What it does: Sets the total character length of Random String pseudonyms.\n"
        "• The prefix and suffix lengths are subtracted from this value to determine "
        "the random portion length.\n"
        "• Validation: Must be between 4 and 64. Effective random length (after prefix/suffix) must be ≥ 4.\n"
        "• Default: 36"
    )

    mapping_encryption_key = (
        "What it does: The AES-256-GCM encryption key used to secure the mapping file on disk.\n"
        "• Must be a 256-bit key provided as a 64-character hexadecimal string.\n"
        "• Keep this key safe — it is required to re-identify data or load existing mappings.\n"
        "• Security: Never share or log this value."
    )

    mapping_file = (
        "What it does: Specifies the file name for storing the encrypted mapping.\n"
        "• If left empty, a name is auto-generated based on the field name "
        "(e.g., 'email_mapping.csv.enc').\n"
        "• Default: Auto-generated."
    )

    mapping_format = (
        "What it does: Sets the serialization format of the encrypted mapping file.\n"
        "• CSV: Compact, fast for large datasets. (Default)\n"
        "• JSON: Human-readable structure, useful for debugging with decryption."
    )

    create_if_not_exists = (
        "What it does: Automatically creates a new mapping file if one does not already exist.\n"
        "• If disabled and no mapping file is found, the operation will raise an error.\n"
        "• Default: True (enabled)."
    )

    backup_on_update = (
        "What it does: Creates a timestamped backup of the mapping file before saving new entries.\n"
        "• Protects against accidental overwrites or corruption.\n"
        "• Default: True (enabled)."
    )

    persist_frequency = (
        "What it does: Controls how often the in-memory mapping is flushed to disk.\n"
        "• Example: A value of 1000 saves the mapping file every 1000 new entries.\n"
        "• Lower values reduce data loss on failure but increase disk I/O.\n"
        "• Validation: Must be at least 1.\n"
        "• Default: 1000"
    )

    compound_mode = (
        "What it does: Combines the primary field and additional fields into a single compound "
        "identifier before pseudonymization.\n"
        "• Example: 'John' + 'Doe' → compound key 'John|Doe' → single pseudonym.\n"
        "• Requires at least one Additional Field to be specified.\n"
        "• Default: False (disabled)."
    )

    compound_separator = (
        "What it does: The character(s) used to join field values when building a compound identifier.\n"
        "• Example: With separator '|', 'John' and 'Doe' become 'John|Doe'.\n"
        "• Default: '|'"
    )

    compound_null_handling = (
        "What it does: Determines how null values in component fields are handled during compound key creation.\n"
        "• Skip: Null fields are omitted from the compound key.\n"
        "• Empty String: Null fields contribute an empty string to the key.\n"
        "• Null Literal: Null fields contribute the string 'null' to the key.\n"
        "• Default: Skip"
    )

    condition_field = (
        "How it works: Pseudonymization will only be applied to rows where the value in this "
        "column meets the specified condition.\n"
        "• Example: To pseudonymize 'email' only for active users, select 'status' here."
    )

    condition_operator = (
        "What it does: Defines the comparison rule applied to the Condition Field.\n"
        "• Options: in, not_in, gt (greater than), lt (less than), eq (equal), range.\n"
        "• Default: in"
    )

    condition_values = (
        "How it works: The values the Condition Field is compared against.\n"
        "• For 'in' / 'not_in': provide a list (e.g., 'active, pending').\n"
        "• For 'range': provide two values representing min and max."
    )

    quasi_identifiers = (
        "What it does: Columns used to calculate k-anonymity and disclosure risk after pseudonymization.\n"
        "• Example: 'zip_code, birth_year, gender'.\n"
        "• Required to enable post-operation privacy metrics."
    )

    ka_risk_field = (
        "How it works: A column containing pre-computed risk scores (e.g., from K-Anonymity Profiler). "
        "Records exceeding the Risk Threshold receive the Vulnerable Record Strategy.\n"
        "• Example: Select a column named 'k_anonymity_score'."
    )

    risk_threshold = (
        "What it does: Records with a risk score above this value are treated as vulnerable.\n"
        "• Default: 5.0"
    )

    vulnerable_record_strategy = (
        "What it does: Action applied to records that exceed the risk threshold.\n"
        "• Pseudonymize: Apply pseudonymization regardless. (Default)\n"
        "• Suppress: Replace the value with null.\n"
        "• Remove: Drop the record from the output."
    )

    mode = (
        "What it does:\n"
        "• REPLACE: Overwrites the original column with pseudonyms.\n"
        "• ENRICH: Keeps the original column and adds a new column with pseudonyms.\n"
        "• Recommended: Use ENRICH during testing to compare original and pseudonymized values."
    )

    null_strategy = (
        "What it does: Determines how null values in the target field are handled.\n"
        "• PRESERVE: Keep nulls as-is. (Default)\n"
        "• EXCLUDE: Remove rows with null values.\n"
        "• ANONYMIZE: Replace nulls with a generated pseudonym.\n"
        "• ERROR: Stop the operation if any null values are found."
    )

    generate_visualization = (
        "What it does: Enables generation of charts comparing original vs pseudonymized value distributions.\n"
        "• Uncheck for faster execution if visualizations are not needed.\n"
        "• Default: Enabled."
    )


    use_cache = (
        "What it does: Enables caching of operation results on disk.\n"
        "• When enabled, repeated runs with the same inputs reuse cached output instead of recomputing.\n"
        "• Must be enabled for 'Force Recalculation' to take effect.\n"
        "• Default: False (disabled)."
    )

    force_recalculation = (
        "What it does: Bypasses the operation cache and forces full re-processing.\n"
        "• Use when source data has changed but the cache key has not."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "field_name": cls.field_name,
            "additional_fields": cls.additional_fields,
            "pseudonym_type": cls.pseudonym_type,
            "pseudonym_prefix": cls.pseudonym_prefix,
            "pseudonym_suffix": cls.pseudonym_suffix,
            "pseudonym_length": cls.pseudonym_length,
            "mapping_encryption_key": cls.mapping_encryption_key,
            "mapping_file": cls.mapping_file,
            "mapping_format": cls.mapping_format,
            "create_if_not_exists": cls.create_if_not_exists,
            "backup_on_update": cls.backup_on_update,
            "persist_frequency": cls.persist_frequency,
            "compound_mode": cls.compound_mode,
            "compound_separator": cls.compound_separator,
            "compound_null_handling": cls.compound_null_handling,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "quasi_identifiers": cls.quasi_identifiers,
            "ka_risk_field": cls.ka_risk_field,
            "risk_threshold": cls.risk_threshold,
            "vulnerable_record_strategy": cls.vulnerable_record_strategy,
            "mode": cls.mode,
            "null_strategy": cls.null_strategy,
            "generate_visualization": cls.generate_visualization,
            "use_cache": cls.use_cache,
            "force_recalculation": cls.force_recalculation,
        }
