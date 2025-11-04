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

class NumericOpTooltip:
    strategy = (
        "What it does: Selects the core logic for making your numbers less specific.\n"
        "• Binning: Groups numbers into ranges (e.g., 27 -> '25-30').\n"
        "• Rounding: Reduces decimal precision (e.g., 12.345 -> 12).\n"
        "• Range: Maps values to custom-defined intervals."
    )

    binning_method = (
        "How are ranges created?\n"
        "• Equal Width: Creates bins of the same size (e.g., 0-10, 10-20). Good for evenly spread data.\n"
        "• Equal Frequency: Creates bins with the same number of records in each. Good for skewed data.\n"
        "• Quantile: Creates bins based on specific quantile boundaries (e.g., quartiles, percentiles), "
        "so each bin contains a defined proportion of the data. Useful for highlighting statistical "
        "distribution (e.g., 25%, 50%, 75% splits)."
    )

    bin_count = (
        "Privacy vs. Utility Trade-off:\n"
        "• More bins (e.g., 20): More detail, less privacy.\n"
        "• Fewer bins (e.g., 5): Less detail, more privacy.\n"
        "Recommended: 5-20."
    )

    range_limits = (
        "Define intervals as a list of lists.\n"
        "Example: To group ages into 'Under 18', '18-65', and 'Over 65', enter: "
        "[[0, 18], [18, 65], [65, 120]]\n\n"
        "Values outside these ranges will be labeled automatically."
    )

    precision = (
        "How it works:\n"
        "• Positive (e.g., 2): Rounds to decimals (12.345 -> 12.35).\n"
        "• Zero (0): Rounds to a whole number (12.345 -> 12).\n"
        "• Negative (e.g., -2): Rounds to the left of the decimal (1234 -> 1200)."
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
        "• > (gt), < (lt), etc.: For numerical or date comparisons."
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
            "strategy": cls.strategy,
            "binning_method": cls.binning_method,
            "bin_count": cls.bin_count,
            "range_limits": cls.range_limits,
            "precision": cls.precision,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
        }
