"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Hash-Based Pseudonymization Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-15
License:       BSD 3-Clause

Description:
Provides user-facing tooltips for hash-based pseudonymization configuration
fields in PAMOLA.CORE.
- Explains hashing algorithm, salt/pepper, output format, compound, and conditional options
- Designed for integration with Formily and schema-driven UI builders

Changelog:
1.0.0 - 2025-06-15 - Initial creation
"""


class HashBasedPseudonymizationTooltip:
    field_name = (
        "What it does: Selects the column whose values will be hashed into pseudonyms.\n"
        "• The transformation is one-way (irreversible without the original salt and pepper).\n"
        "• Validation: The field must exist in the dataset."
    )

    additional_fields = (
        "What it does: Extra columns included when building a compound identifier.\n"
        "• Only applies when Compound Mode is enabled.\n"
        "• Example: Hashing 'first_name' + 'last_name' together as one compound value."
    )

    algorithm = (
        "What it does: Selects the cryptographic hash function used to generate pseudonyms.\n"
        "• SHA3-256: 256-bit output, quantum-resistant, recommended for most use cases. (Default)\n"
        "• SHA3-512: 512-bit output, higher security margin at the cost of longer pseudonyms."
    )

    salt_config = (
        "What it does: Configures the salt that is combined with each value before hashing.\n"
        "• Source — Parameter: Provide the salt value directly as a hex string.\n"
        "• Source — File: Load per-field salts from an external encrypted salt file.\n"
        "• Purpose: Prevents rainbow table attacks by making identical values hash differently "
        "across datasets."
    )

    salt_file = (
        "What it does: Path to the external salt file when Salt Source is set to 'File'.\n"
        "• The file contains field-specific salt values keyed by field name.\n"
        "• Validation: Required when Salt Source is 'file'."
    )

    use_pepper = (
        "What it does: Adds a randomly generated session-specific pepper to each hash computation.\n"
        "• Purpose: Provides extra protection — even with the same salt, hashes differ between runs.\n"
        "• Trade-off: Disabling pepper makes hashes deterministic across runs (required for lookups).\n"
        "• Default: True (enabled)."
    )

    pepper_length = (
        "What it does: Length (in bytes) of the randomly generated session pepper.\n"
        "• Longer peppers provide stronger per-session uniqueness.\n"
        "• Validation: Must be at least 16 bytes.\n"
        "• Default: 32 bytes."
    )

    hash_output_format = (
        "What it does: Controls the encoding used to represent the hash bytes as a string.\n"
        "• Hex: Lowercase hexadecimal (e.g., 'a3f2...'). (Default)\n"
        "• Base64: URL-safe base64 without padding — compact representation.\n"
        "• Base32: Uppercase alphanumeric, no padding — case-insensitive and URL-safe.\n"
        "• Base58: Bitcoin-style encoding — no ambiguous characters (0/O/l/I).\n"
        "• UUID: First 16 bytes formatted as a UUID (e.g., 'a3f2c1d0-...')."
    )

    output_length = (
        "What it does: Truncates the pseudonym to the specified number of characters.\n"
        "• Useful for fitting pseudonyms into fixed-length database columns.\n"
        "• Warning: Truncation increases collision probability — use with caution.\n"
        "• Validation: Must be at least 8 characters. Default: None (full-length output)."
    )

    output_prefix = (
        "What it does: Prepends a fixed string to every generated pseudonym.\n"
        "• Example: A prefix of 'H-' produces 'H-a3f2c1d0...'.\n"
        "• Default: None."
    )

    output_suffix = (
        "What it does: Appends a fixed string to every generated pseudonym.\n"
        "• Example: A suffix of '-HASH' produces 'a3f2c1d0...-HASH'.\n"
        "• Default: None."
    )

    compound_mode = (
        "What it does: Combines the primary field and additional fields into a single value "
        "before hashing.\n"
        "• Example: 'John' + 'Doe' → 'John|Doe' → single hash pseudonym.\n"
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
        "• Empty String: Null fields contribute an empty string.\n"
        "• Null Literal: Null fields contribute the string 'null'.\n"
        "• Default: Skip"
    )

    condition_field = (
        "How it works: Hashing will only be applied to rows where the value in this column "
        "meets the specified condition.\n"
        "• Example: To hash 'ssn' only for records flagged as 'sensitive', select 'sensitivity_flag' here."
    )

    condition_operator = (
        "What it does: Defines the comparison rule applied to the Condition Field.\n"
        "• Options: in, not_in, gt (greater than), lt (less than), eq (equal), range.\n"
        "• Default: in"
    )

    condition_values = (
        "How it works: The values the Condition Field is compared against.\n"
        "• For 'in' / 'not_in': provide a list (e.g., 'sensitive, restricted').\n"
        "• For 'range': provide two values representing min and max."
    )

    quasi_identifiers = (
        "What it does: Columns used to calculate k-anonymity and disclosure risk after pseudonymization.\n"
        "• Example: 'zip_code, birth_year, gender'.\n"
        "• Required to enable post-operation privacy metrics."
    )

    ka_risk_field = (
        "How it works: A column containing pre-computed risk scores. Records exceeding the "
        "Risk Threshold receive the Vulnerable Record Strategy.\n"
        "• Example: Select a column named 'k_anonymity_score'."
    )

    risk_threshold = (
        "What it does: Records with a risk score above this value are treated as vulnerable.\n"
        "• Default: 5.0"
    )

    vulnerable_record_strategy = (
        "What it does: Action applied to records that exceed the risk threshold.\n"
        "• Pseudonymize: Apply hashing regardless. (Default)\n"
        "• Suppress: Replace the value with null.\n"
        "• Remove: Drop the record from the output."
    )

    mode = (
        "What it does:\n"
        "• REPLACE: Overwrites the original column with hash pseudonyms.\n"
        "• ENRICH: Keeps the original column and adds a new column with hash pseudonyms.\n"
        "• Recommended: Use ENRICH during testing to compare original and pseudonymized values."
    )

    null_strategy = (
        "What it does: Determines how null values in the target field are handled.\n"
        "• PRESERVE: Keep nulls as-is. (Default)\n"
        "• EXCLUDE: Remove rows with null values.\n"
        "• ANONYMIZE: Hash a null placeholder value.\n"
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
            "algorithm": cls.algorithm,
            "salt_config": cls.salt_config,
            "salt_file": cls.salt_file,
            "use_pepper": cls.use_pepper,
            "pepper_length": cls.pepper_length,
            "hash_output_format": cls.hash_output_format,
            "output_length": cls.output_length,
            "output_prefix": cls.output_prefix,
            "output_suffix": cls.output_suffix,
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
