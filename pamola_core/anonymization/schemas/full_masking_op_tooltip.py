"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Full Masking Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Full Masking generalization configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Full Masking anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Full Masking operation tooltip file
"""

class FullMaskingOpTooltip:


    mask_char = (
        "What it does: Defines the character that will hide the original data.\n"
        "• Example: Using `*` will result in masks like `******`. Using `X` will result in `XXXXXXXX`.\n"
        "• Validation: Must be a single character.\n"
        "• Default: `*`."
    )

    preserve_length = (
        "How it works: If a value is 10 characters long, the masked version will also be 10 characters long.\n"
        "• Example: `ABC-123` becomes `***-***`.\n"
        "• Impact: This is useful for maintaining the layout of data in some systems but can leak information about "
        "the length of the original data.\n"
        "• Default: `True`."
    )

    fixed_length = (
        "What it does: Makes every masked value the same length, regardless of its original size.\n"
        "• Example: If set to `8`, both `short` and `a-very-long-value` would be masked as `********`.\n"
        "• Impact: Increases privacy by hiding the original length of the data.\n"
        "• Validation: Must be a non-negative number."
    )

    random_mask = (
        "What it does: Makes the masked output look less uniform by replacing sensitive data with random characters "
        "from a defined pool.\n"
        "• Example: A masked value might look like `aB9@k#P$` instead of `********`.\n"
        "• Impact: Can improve data utility for systems that require varied character inputs, but slightly reduces "
        "privacy by revealing the character set."
    )

    mask_char_pool = (
        "What it does: Defines the characters that will be randomly selected to build the mask.\n"
        "• Example: A pool of `ABC123` means the mask will only contain those characters.\n"
        "• Default: If left empty, a default pool of letters, numbers, and symbols (`!@#$%^&*`) is used."
    )

    preserve_format = (
        "How it works: When enabled, this feature tries to identify the format of the data (like a phone number or "
        "credit card) and only masks the alphanumeric characters, leaving separators intact.\n"
        "• Example: `(555)-123-4567` becomes `(***)-***-****`.\n"
        "• Impact: Improves readability and utility for formatted data."
    )

    format_patterns = (
        "What it does: Allows you to define your own rules for what constitutes a 'format' to be preserved.\n"
        "• Example: You could add a pattern for a product ID like `PROD-\\d{4}` to ensure the `PROD-` prefix and "
        "hyphen are always preserved.\n"
        "• Note: This is an advanced feature. The system includes built-in patterns for common types like phone numbers "
        "and credit cards."
    )

    numeric_output = (
        "What it does: Determines the output for numbers.\n"
        "• String: The masked value will be text (e.g., `****`). (Default)\n"
        "• Numeric: The masked value will be a number (e.g., `9999`).\n"
        "• Preserve: The original number will be kept and not masked."
    )

    date_format = (
        "What it does: Standardizes the text representation of dates before the masking logic is applied.\n"
        "• Example: If your dates are in multiple formats, you can set this to `YYYY-MM-DD` to convert them all to "
        "that format first, ensuring the mask is applied consistently."
    )

    condition_field = (
        "Which column determines if a row is anonymized?\n"
        "The generalization will only apply to rows that match the condition you set on this column.\n\n"
        "Example: Select the 'Country' column to apply this rule only to customers from a specific country."
    )

    condition_operator = (
        "How should the condition be evaluated?\n"
        "This operator compares the 'Condition Field' with the 'Condition Values'.\n\n"
        "• in: The value must be in your list of values.\n"
        "• not in: The value must *not* be in your list.\n"
        "• > (gt), < (lt), etc.: For Full Maskingal or date comparisons."
    )

    condition_values = (
        "Which values should trigger this rule?\n"
        "Provide a list of values to match against the 'Condition Field'.\n\n"
        "Example: If your 'Condition Field' is 'Country' and you enter `USA`, "
        "this rule will only apply to rows where the country is 'USA'. You can enter "
        "multiple values, separated by commas (e.g., `USA, Canada`)."
    )

    mode = (
        "How do you want the output?\n"
        "• REPLACE: Overwrites the original column.\n"
        "• ENRICH: Keeps the original and adds a new, anonymized column."
    )

    output_field_name = (
        "Name the new column. If left blank, a name will be generated automatically (e.g., _age)."
    )

    column_prefix = (
        "Prefix for the new column name. The default is an underscore (_), which turns age into _age."
    )

    null_strategy = (
        "What to do with empty cells?\n"
        "• PRESERVE: Leave them as they are (default).\n"
        "• EXCLUDE: Temporarily remove rows with empty values before processing.\n"
        "• ERROR: Stop the operation if any empty values are found."
    )

    force_recalculation = (
        "Ignore saved results. Check this box to force the operation to run again "
        "instead of using a cached result from a previous run with the same settings."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "mask_char": cls.mask_char,
            "preserve_length": cls.preserve_length,
            "fixed_length": cls.fixed_length,
            "random_mask": cls.random_mask,
            "mask_char_pool": cls.mask_char_pool,
            "preserve_format": cls.preserve_format,
            "format_patterns": cls.format_patterns,
            "numeric_output": cls.numeric_output,
            "date_format": cls.date_format,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
        }
