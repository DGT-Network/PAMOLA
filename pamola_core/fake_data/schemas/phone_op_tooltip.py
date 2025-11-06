"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Phone Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for phone number generation configuration fields in PAMOLA.CORE.
- Explains region, country code, format, consistency, and output options for phone anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of phone generation operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of phone operation tooltip file
"""


class FakePhoneOperationTooltip:
    region = (
        "• What it does: Influences the default country code and formatting style used when no other information is available.\n"
        "• Example: Enter `us` for United States (+1), `ru` for Russia (+7), or `gb` for Great Britain (+44).\n"
        "• Default: `us` (United States)"
    )

    default_country = (
        "• What it does: Acts as a safety net to ensure a number can always be generated.\n"
        "• Example: `us`\n"
        "• Impact: Ensures consistent output even when input data lacks clear regional information.\n"
        "• Default: `us`"
    )

    country_codes = (
        '• How it works: You can provide a simple list for random selection (e.g., ["1", "44", "7"]) or a dictionary with weights for prioritized selection (e.g., {"1": 0.8, "44": 0.2}).\n'
        "• Impact: Allows you to create a dataset with a diverse, international mix of phone numbers.\n"
        "• Default: A pre-defined list of common countries, weighted towards the US/Canada."
    )

    country_code_field = (
        "• What it does: Uses the country code from another column (e.g., a 'Country' column) to generate a phone number appropriate for that record.\n"
        "• Example: If you have a column `country_iso` with values 'US' or 'GB', the operation will use +1 and +44 accordingly.\n"
        "• Note: This parameter is used in the code's logic but is not formally defined in the configuration schema."
    )

    operator_codes_dict = (
        "• What it does: Overrides the built-in lists of operator/area codes.\n"
        "• Example: A `.txt` file containing lines like `+1,201,917,646`.\n"
        "• Impact: Gives you precise control over the prefixes used during number generation."
    )

    format = (
        "• How it works: Use placeholders: CC for Country Code, AAA for Operator/Area Code, and X for random digits. Leave blank to use the standard format for each number's country.\n"
        "• Example: `+CC (AAA) XXX-XX-XX` would produce a number like `+7 (903) 123-45-67`.\n"
        "• Impact: Overrides any default region-specific formatting rules."
    )

    preserve_country_code = (
        "• What it does: Ensures that a US phone number is replaced with another US phone number.\n"
        "• Impact: Highly useful for maintaining the geographic validity of your data.\n"
        "• Default: Enabled"
    )

    preserve_operator_code = (
        "• What it does: Helps maintain the regional or carrier characteristics of the original phone number.\n"
        "• Example: A number starting with `(917)` might be replaced with another number that also starts with `(917)`.\n"
        "• Default: Disabled"
    )

    consistency_mechanism = (
        '• What it does: Guarantees that "+1-555-1234" is always replaced by "+1-555-9876" every time the operation runs.\n'
        "• Options:\n"
        "- `prgn`: A fast, stateless method using a cryptographic key. Best for performance.\n"
        "- `mapping`: A stateful method that stores every original-to-fake pair. Slower but allows you to save and reuse the exact same mappings.\n"
        "• Recommended: `prgn` for most scenarios.\n"
        "• Default: `prgn`"
    )

    id_field = (
        "• What it does: Provides a stable identifier for each record to the generation logic.\n"
        "• Impact: While available, the logic for this specific operation primarily relies on the original phone number value for consistency, not this ID field."
    )

    key = (
        "• What it does: Acts like a password for the generation algorithm. The same key with the same input number will always produce the same output number.\n"
        "• Example: `my-secret-project-key-2025`\n"
        "• Impact: Crucial for consistency. If you lose the key, you cannot reproduce the same fake numbers again."
    )

    context_salt = (
        "• How it works: It's mixed with the key during generation.\n"
        "• Example: Use `dev` for development and `prod` for production. The same phone number and key will generate different results in the two environments.\n"
        "• Impact: Allows you to create multiple, distinct pseudonymized datasets from a single source."
    )

    mapping_store_path = (
        "• What it does: Specifies a file (e.g., C:\\mappings\\phone_numbers.json) that stores the lookup table.\n"
        "• Impact: If the file exists, the operation will reuse its mappings. If not, a new map can be created and saved there."
    )

    save_mapping = (
        "• What it does: Writes the complete list of original numbers and their corresponding fake numbers to a file after the operation finishes.\n"
        "• Impact: Enables perfect consistency across future runs and allows for auditing, but the saved file will contain sensitive original data and must be secured."
    )

    validate_source = (
        "• What it does: Helps identify malformed phone numbers in your data before processing.\n"
        "• Impact: Ensures that preservation logic (like keeping the country code) is only applied to valid numbers.\n"
        "• Default: Enabled"
    )

    handle_invalid_phone = (
        "• Options:\n"
        "- `generate_new`: Replaces the invalid number with a completely new, valid one (default).\n"
        "- `keep_empty`: Replaces the invalid number with a blank value.\n"
        "- `generate_with_default_country`: Creates a new number using the default country setting.\n"
        "• Recommended: `generate_new` to ensure all rows have data."
    )

    mode = (
        "• Options:\n"
        "- `REPLACE`: Overwrites the original phone number column with fake numbers. This is a destructive action.\n"
        "- `ENRICH`: Keeps the original column and adds a new column containing the fake numbers.\n"
        "• Recommended: `ENRICH` for safety and data validation."
    )

    output_field_name = (
        "• How it works: If you leave this blank, a name will be automatically created using the Column Prefix and the original field name.\n"
        "• Example: `fake_phone_number`"
    )

    column_prefix = (
        "• How it works: If your original column is phone and the prefix is fake_, the new column will be named fake_phone.\n"
        "• Example: `fake_`, `anon_`, `synthetic_`"
    )

    null_strategy = (
        "• Options:\n"
        "- `preserve`: Keeps null values as they are (default).\n"
        "- `replace`: Generates a new fake phone number to fill in the blank.\n"
        "• Recommended: `preserve` to maintain the original data's completeness characteristics."
    )

    force_recalculation = (
        "• What it does: Ignores any previously saved results (cache) for this exact operation and re-generates all data from scratch.\n"
        "• Impact: Use this if you have changed a setting or the underlying data and need to ensure the results are fresh."
    )

    detailed_metrics = (
        "• What it does: Provides deeper insights into the operation's output.\n"
        "• Impact: May slightly slow down processing due to the extra data collection.\n"
        "• Default: Disabled"
    )

    max_retries = (
        "• What it does: Increases the operation's resilience against random generation errors.\n"
        "• Impact: A higher value can overcome intermittent issues but might slow down the process if there is a persistent problem.\n"
        "• Default: `3`"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "region": cls.region,
            "default_country": cls.default_country,
            "country_codes": cls.country_codes,
            "country_code_field": cls.country_code_field,
            "operator_codes_dict": cls.operator_codes_dict,
            "format": cls.format,
            "preserve_country_code": cls.preserve_country_code,
            "preserve_operator_code": cls.preserve_operator_code,
            "consistency_mechanism": cls.consistency_mechanism,
            "id_field": cls.id_field,
            "key": cls.key,
            "context_salt": cls.context_salt,
            "mapping_store_path": cls.mapping_store_path,
            "save_mapping": cls.save_mapping,
            "validate_source": cls.validate_source,
            "handle_invalid_phone": cls.handle_invalid_phone,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
            "detailed_metrics": cls.detailed_metrics,
            "max_retries": cls.max_retries,
        }
