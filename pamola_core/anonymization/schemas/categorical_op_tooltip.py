"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numeric Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for numeric generalization configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for numeric anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of numeric operation tooltip file
"""


class CategoricalOpTooltip:
    strategy = (
        "What it does: This is the main control for how categories are grouped or mapped.\n"
        '• Hierarchy: Replaces specific values with their more general parents from a predefined hierarchy (e.g., "San Francisco" becomes "California"). (Default)\n'
        '• Merge Low Frequency: Groups the least common categories into a single "Other" category based on frequency thresholds.\n'
        "• Frequency Based: Keeps a specified number of the most common categories and groups all others."
    )

    external_dictionary_path = (
        "What it does: This file acts as a lookup table to find the more general category for each value.\n"
        "• Example: A file that maps cities to states, or specific job titles to broader job functions.\n"
        "• Validation: Must be a valid path to a `.json` or `.csv` file."
    )

    dictionary_format = (
        "What it does: Specifies how to read the dictionary file.\n"
        "• Auto: Automatically detects the format based on the file extension (.json or .csv). (Default)\n"
        "• JSON / CSV: Explicitly defines the file format."
    )

    hierarchy_level = (
        'What it does: Determines how "high up" the hierarchy to go for the replacement value.\n'
        '• Example: If your hierarchy is `City -> State -> Country`, a level of `1` would turn "Seattle" into "Washington". A level of `2` would turn it into "USA".\n'
        "• Validation: Must be an integer between 1 and 5.\n"
        "• Default: `1`"
    )

    freq_threshold = (
        "What it does: Any category that makes up less than this percentage of the total dataset will be grouped.\n"
        "• Example: A value of `0.01` means any category appearing in less than 1% of rows will be grouped.\n"
        "• Validation: Must be a number between 0 and 1.\n"
        "• Default: `0.01`"
    )

    min_group_size = (
        "What it does: Any category with a count lower than this value will be considered rare and grouped with others.\n"
        "• Example: If set to `10`, any category appearing 9 or fewer times will be grouped.\n"
        "• Impact: This is a key parameter for achieving k-anonymity.\n"
        "• Validation: Must be an integer of 1 or greater.\n"
        "• Default: `10`"
    )

    max_categories = (
        "What it does: This setting keeps the most common values as they are and groups the long tail of less frequent values.\n"
        '• Example: If set to `100`, the top 100 most common categories will be kept, and all other categories will be grouped into an "Other" category.\n'
        "• Default: `1,000,000` (effectively no limit)"
    )

    group_rare_as = (
        "What it does: Controls the naming of the generalized groups.\n"
        '• OTHER: All rare values are combined into a single group named "OTHER". (Default)\n'
        "• CATEGORY_N / RARE_N: Rare values are split into multiple, smaller groups with numbered names (e.g., `RARE_1`, `RARE_2`)."
    )

    rare_value_template = (
        "What it does: Defines the format for the generated group names. The {n} placeholder will be replaced by a number.\n"
        "• Example: A template of `Group_{n}` would create groups named `Group_1`, `Group_2`, etc.\n"
        "• Validation: The string must contain `{n}`.\n"
        "• Default: `OTHER_{n}`"
    )

    text_normalization = (
        "What it does: Standardizes text to ensure that variations of the same category are treated as one.\n"
        "• None: No changes are made.\n"
        "• Basic: Trims whitespace and converts to lowercase. (Default)\n"
        "• Advanced: Also removes special characters.\n"
        "• Aggressive: Keeps only letters and numbers."
    )

    case_sensitive = (
        'How it works: If disabled, "Apple" and "apple" will be treated as the same category during matching and frequency counting.\n'
        "• Impact: Disabling this generally improves matching accuracy but may not be suitable for case-sensitive identifiers.\n"
        "• Default: `False`"
    )

    fuzzy_matching = (
        'How it works: If a value like "Software Dev" isn\'t found in the dictionary, this will try to match it to the closest existing entry, like "Software Developer".\n'
        "• Impact: Increases the number of successful mappings but may introduce inaccuracies if the similarity threshold is too low.\n"
        "• Default: `False`"
    )

    similarity_threshold = (
        "What it does: Controls the sensitivity of fuzzy matching. A higher value requires a closer match.\n"
        "• Example: A value of `0.9` means the strings must be 90% similar to be considered a match.\n"
        "• Validation: Must be a number between 0 and 1.\n"
        "• Default: `0.85`"
    )

    allow_unknown = (
        "How it works: If a value is not found in the hierarchy dictionary and cannot be fuzzy matched, this setting prevents an error. The value will be replaced by the string in the Unknown Value field.\n"
        "• Impact: Ensures the operation completes, but can result in data loss if many values are unmapped.\n"
        "• Default: `True`"
    )

    unknown_value = (
        "What it does: This is the default replacement value for any category that cannot be successfully generalized.\n"
        "• Example: `Other`, `Uncategorized`, `N/A`.\n"
        "• Default: `OTHER`"
    )

    condition_field = (
        "How it works: Generalization will only be applied to rows where the value in this column meets the specified condition.\n"
        "• Example: To generalize the 'job_title' column only for employees in the 'USA' office, you would select 'office_location' here."
    )

    condition_operator = (
        "What it does: Defines how to compare the value in the Condition Field with the Condition Values.\n"
        "• Options: `in`, `not_in`, `equals`, `greater_than`, `less_than`, etc.\n"
        "• Default: `in`"
    )

    condition_values = (
        "How it works: These are the values that the condition must match. For the 'in' operator, you can provide a list of values.\n"
        "• Example: To apply the rule only for the 'USA' and 'Canada' offices, you would enter `USA, Canada`."
    )

    ka_risk_field = (
        "How it works: This is a post-processing step. After generalization, the operation checks this risk score to see if any records are still vulnerable and applies the Vulnerable Record Strategy to them.\n"
        "• Example: Select a column named `k_anonymity_score`."
    )

    risk_threshold = (
        "What it does: Defines the cutoff for identifying vulnerable records. Any record where the Vulnerable Record Field value is less than this threshold will have the Vulnerable Record Strategy applied.\n"
        "• Default: `5.0`"
    )

    vulnerable_record_strategy = (
        "What it does: Provides an extra layer of protection for high-risk records.\n"
        '• Suppress: Hides the generalized value, replacing it with null or "SUPPRESSED".\n'
        "• Mask: Applies a simple mask (e.g., `*`) to the generalized value.\n"
        "• Default: `suppress`"
    )

    privacy_check_enabled = (
        "How it works: After the operation is complete, it will calculate the k-anonymity and disclosure risk of the output data and log a warning if they do not meet the specified thresholds.\n"
        "• Note: This is a check, not an enforcement. It will not automatically change the output.\n"
        "• Default: `True`"
    )

    quasi_identifiers = (
        "What it does: Defines the set of fields to use when calculating k-anonymity and disclosure risk.\n"
        "• Example: `zip_code, birth_date, gender`.\n"
        "• Impact: This is the most critical parameter for performing a meaningful privacy validation."
    )

    min_acceptable_k = (
        "What it does: Sets the target for the post-operation privacy check. A warning will be logged if the final k-anonymity is below this value.\n"
        "• Example: A value of `5` means every combination of quasi-identifiers should appear at least 5 times.\n"
        "• Validation: Must be an integer of 2 or greater.\n"
        "• Default: `5`"
    )

    max_acceptable_disclosure_risk = (
        "What it does: Sets the target for the post-operation privacy check. A warning will be logged if the final disclosure risk is above this value.\n"
        "• Validation: Must be a number between 0 and 1.\n"
        "• Default: `0.2`"
    )

    mode = (
        "What it does:\n"
        "• REPLACE: Overwrites the data in the original column with the generalized values.\n"
        "• ENRICH: Keeps the original column and adds a new column containing the generalized values.\n"
        "• Recommended: Use 'ENRICH' during testing to easily compare the original and generalized data."
    )

    output_field_name = (
        "What it does: Specifies the header for the new column.\n"
        "• Example: If you are generalizing the 'job_title' column, you could name the new column 'job_function'.\n"
        "• Default: If left empty, a name is auto-generated (e.g., `_job_title`)."
    )

    column_prefix = (
        "What it does: Helps in auto-generating a new column name.\n"
        "• Example: If the original column is 'city' and the prefix is `generalized_`, the new column will be named `generalized_city`.\n"
        "• Default: `_`"
    )

    null_strategy = (
        "What it does: Determines the behavior for empty or null values.\n"
        "• PRESERVE: Keep null values as they are. (Default)\n"
        "• EXCLUDE: Removes rows with null values from the output.\n"
        "• ANONYMIZE: Replaces nulls with the Unknown Value.\n"
        "• ERROR: Stop the operation if any null values are found."
    )

    force_recalculation = (
        "What it does: Disables the caching mechanism for this run, forcing the operation to re-process all data from scratch.\n"
        "• Use Case: Enable this if you have changed the underlying data or want to ensure a fresh run for auditing purposes."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "strategy": cls.strategy,
            "external_dictionary_path": cls.external_dictionary_path,
            "dictionary_format": cls.dictionary_format,
            "hierarchy_level": cls.hierarchy_level,
            "freq_threshold": cls.freq_threshold,
            "min_group_size": cls.min_group_size,
            "max_categories": cls.max_categories,
            "group_rare_as": cls.group_rare_as,
            "rare_value_template": cls.rare_value_template,
            "text_normalization": cls.text_normalization,
            "case_sensitive": cls.case_sensitive,
            "fuzzy_matching": cls.fuzzy_matching,
            "similarity_threshold": cls.similarity_threshold,
            "allow_unknown": cls.allow_unknown,
            "unknown_value": cls.unknown_value,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "ka_risk_field": cls.ka_risk_field,
            "risk_threshold": cls.risk_threshold,
            "vulnerable_record_strategy": cls.vulnerable_record_strategy,
            "privacy_check_enabled": cls.privacy_check_enabled,
            "quasi_identifiers": cls.quasi_identifiers,
            "min_acceptable_k": cls.min_acceptable_k,
            "max_acceptable_disclosure_risk": cls.max_acceptable_disclosure_risk,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
        }
