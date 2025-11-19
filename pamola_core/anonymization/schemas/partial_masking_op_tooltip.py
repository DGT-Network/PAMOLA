"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Partial Masking Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Partial Masking generalization configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Partial Masking anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Partial Masking operation tooltip file
"""


class PartialMaskingOpTooltip:

    mask_strategy = (
        "What it does: This is the main control for how masking is applied. Each strategy uses different parameters "
        "to determine what to hide.\n"
        "• Fixed Position: Masks content based on character positions (e.g., hide everything except the first 4 characters). Uses the Masking Rules section.\n"
        "• Pattern: Masks content based on predefined templates (like email, phone) or a custom regex pattern.\n"
        "• Random: Masks a random percentage of characters in each value.\n"
        "• Words: Masks entire words within the text, useful for free-text fields."
    )

    preset_type = (
        "What it does: Presets are ready-to-use masking configurations for common data types. Selecting a type here "
        "will populate the Preset Name dropdown with relevant options.\n"
        "• Example: Select `Email` to see presets like `Domain Only` or `Privacy Focused`.\n"
        "• Impact: Simplifies configuration by providing expert-recommended settings for specific data types."
    )

    preset_name = (
        "What it does: Applies a pre-configured set of rules for masking.\n"
        "• Example: If you chose `Credit Card` as the type, you could select the `PCI_COMPLIANT` preset to automatically "
        "keep the first 6 and last 4 digits visible.\n"
        "• Impact: Using a preset overrides other manual settings in the Masking Rules and Appearance sections."
    )

    pattern_type = (
        "What it does: Uses a predefined regular expression to identify and mask specific parts of the data.\n"
        "• Example: Selecting `ssn` will automatically mask the first 5 digits of a Social Security Number, leaving only "
        "the last 4 visible (e.g., `***-**-6789`).\n"
        "• Recommended: Use this for standard data formats to ensure correct and consistent masking."
    )

    mask_char = (
        "What it does: Defines the character that will hide the original data.\n"
        "• Example: Using `*` will result in masks like `***-**-1234`. Using `X` will result in `XXX-XX-1234`.\n"
        "• Validation: Must be a single character.\n"
        "• Default: `*`."
    )

    random_mask = (
        "What it does: Makes the masked output look less uniform by replacing sensitive data with random characters from a defined pool.\n"
        "• Example: A masked phone number might look like `555-aB9-4567` instead of `555-***-4567`.\n"
        "• Impact: Can improve data utility for systems that require varied character inputs, but slightly reduces privacy by revealing the character set."
    )

    mask_char_pool = (
        "What it does: Defines the characters that will be randomly selected to build the mask.\n"
        "• Example: A pool of `ABC123` means the mask will only contain those characters.\n"
        "• Default: If left empty, a default pool of letters, numbers, and symbols (`!@#$%^&*`) is used."
    )

    unmasked_prefix = (
        "What it does: Preserves a specified number of characters at the start of each value.\n"
        "• Example: A value of `4` will keep the first four characters and mask the rest (e.g., `ACCT12345` becomes `ACCT****`).\n"
        "• Default: `0`."
    )

    unmasked_suffix = (
        "What it does: Preserves a specified number of characters at the end of each value.\n"
        "• Example: A value of `4` will keep the last four characters and mask the rest (e.g., `123-45-6789` becomes `***-**-6789`).\n"
        "• Default: `0`."
    )

    unmasked_positions = (
        "What it does: Allows you to preserve individual characters by their numerical position, starting from 0.\n"
        "• Example: To keep the first and third characters of `ABCDE`, you would enter `0, 2`. The result would be `A*C**`.\n"
        "• Impact: Provides highly granular control but overrides the simpler Prefix and Suffix settings."
    )

    mask_percentage = (
        "What it does: Randomly masks a portion of each string, which is useful when the data has no fixed structure.\n"
        "• Example: A value of `40` will mask approximately 40% of the characters in each string, at random positions.\n"
        "• Validation: Must be a number between 0 and 100."
    )

    mask_pattern = (
        "What it does: Any part of the text that matches this regular expression will be masked.\n"
        "• Example: To mask a 4-digit year in a date like `MM-DD-YYYY`, you could use the pattern `\\d{4}`.\n"
        "• Impact: Offers powerful, custom control. It is used only if Pattern Type is not selected."
    )

    preserve_pattern = (
        "What it does: Any part of the text that matches this regular expression will be preserved, and everything else will be masked.\n"
        "• Example: To keep only the domain of a URL, you could use the pattern `@[\w.-]+`.\n"
        "• Impact: This is the inverse of Mask Pattern."
    )

    preserve_separators = (
        "How it works: When enabled, characters like hyphens, periods, and at-symbols are not replaced by the masking character. "
        "This helps maintain the readability and format of the original data.\n"
        "• Example: With this on, `123-45-6789` might become `***-**-6789`. With it off, it might become `*********`.\n"
        "• Default: `True`."
    )

    preserve_word_boundaries = (
        "How it works: This option treats each word as a unit. It will apply the Keep Prefix/Suffix rules to each word individually.\n"
        "• Example: Masking `John Fitzgerald Kennedy` with a prefix of 1 might result in `J*** F*** K******`."
    )

    case_sensitive = (
        "How it works: If enabled, a pattern will only match text that has the exact same casing.\n"
        "• Example: If `False`, a pattern searching for `apple` would match `Apple`. If `True`, it would not.\n"
        "• Default: `True`."
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
        "• > (gt), < (lt), etc.: For Partial Maskingal or date comparisons."
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

    output_field_name = "Name the new column. If left blank, a name will be generated automatically (e.g., _age)."

    column_prefix = "Prefix for the new column name. The default is an underscore (_), which turns age into _age."

    null_strategy = (
        "What to do with empty cells?\n"
        "• PRESERVE: Leave them as they are (default).\n"
        "• EXCLUDE: Temporarily remove rows with empty values before processing.\n"
        "• ERROR: Stop the operation if any empty values are found."
    )

    force_recalculation = "What it does: Disables the caching mechanism for this run, forcing the operation to re-process all data from scratch."

    generate_visualization = (
        "What it does: Enables the creation of charts that help you visually understand the impact of the anonymization.\n"
        "• Charts will show changes to either data values (for Generalization, Masking, etc.) or the dataset's structure (for Remove Rows/Columns), depending on the operation.\n"
        "• Note: Enabled by default. Uncheck for faster execution if visuals are not needed."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "mask_strategy": cls.mask_strategy,
            "preset_type": cls.preset_type,
            "preset_name": cls.preset_name,
            "pattern_type": cls.pattern_type,
            "mask_char": cls.mask_char,
            "random_mask": cls.random_mask,
            "mask_char_pool": cls.mask_char_pool,
            "unmasked_prefix": cls.unmasked_prefix,
            "unmasked_suffix": cls.unmasked_suffix,
            "unmasked_positions": cls.unmasked_positions,
            "mask_percentage": cls.mask_percentage,
            "mask_pattern": cls.mask_pattern,
            "preserve_pattern": cls.preserve_pattern,
            "preserve_separators": cls.preserve_separators,
            "preserve_word_boundaries": cls.preserve_word_boundaries,
            "case_sensitive": cls.case_sensitive,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
            "generate_visualization": cls.generate_visualization,
        }
