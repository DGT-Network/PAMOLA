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

class CellSuppressionOpTooltip:
    suppression_strategy = (
        "What it does: Defines what the suppressed cells will be replaced with.\n"
        "• Null: Replaces values with an empty (null) value. (Default)\n"
        "• Mean/Median/Mode: Replaces values with the mean, median, or mode of the entire column. Requires a numeric field for Mean/Median.\n"
        "• Constant: Replaces values with a specific text or number you provide.\n"
        "• Group Mean/Mode: Replaces values with the mean or mode of the subgroup the row belongs to, based on the Group By field."
    )

    suppression_value = (
        "What it does: Sets the exact replacement value.\n"
        "• Example: If you enter `REDACTED`, all targeted cells will be replaced with the word \"REDACTED\". If you enter `0`, they will be replaced with the number 0.\n"
        "• Impact: Ensures a uniform, predictable replacement for all suppressed data. The system will attempt to match the data type of this value to the column's original type."
        )

    group_by_field = (
        "How it works: When using a group-based strategy, the replacement value (mean or mode) is calculated separately for each unique value in this column.\n"
        "• Example: To replace salaries with the average salary *per department*, you would select the 'department' column here.\n"
        "• Impact: This provides a more contextually relevant replacement, preserving statistical properties within groups."
        )

    min_group_size = (
        "How it works: If a group is smaller than this size, its members will be replaced with the global (column-wide) mean or mode instead of the group's value. This prevents statistical leakage from very small groups.\n"
        "• Example: If set to `5`, and a department has only 3 employees, their salaries will be replaced by the company-wide average, not their department's average.\n"
        "• Default: `5`"
        )

    suppress_if = (
        "What it does: Provides a quick way to target cells based on common statistical properties.\n"
        "• Outlier: Automatically targets values that are statistically unusual (far from the average).\n"
        "• Rare: Targets values that appear infrequently in the column.\n"
        "• Null: Targets only the cells that are already empty (null).\n"
        "• Note: If you need more specific rules, leave this blank and use the custom Condition Field below."
        )

    outlier_method = (
        "What it does: Selects the algorithm for outlier detection.\n"
        "• IQR (Interquartile Range): A robust method based on the data's median and percentile spread. (Default)\n"
        "• Z-Score: A method based on the data's mean and standard deviation; more sensitive to extreme values."
        )

    outlier_threshold = (
        "How it works: A higher value makes the detection less sensitive, flagging only very extreme values. A lower value makes it more sensitive.\n"
        "• Example: For IQR, the standard value is `1.5`. For Z-Score, a common value is `3`.\n"
        "• Default: `1.5`"
        )

    rare_threshold = (
        "What it does: Any value that appears fewer times than this threshold will be targeted for suppression.\n"
        "• Example: If set to `10`, any value that appears 9 or fewer times in the entire column will be suppressed.\n"
        "• Default: `10`"
        )

    condition_field = (
        "How it works: Suppression will only be applied to rows where the value in this column meets the specified condition.\n"
        "• Example: To suppress the 'salary' column only for employees in the 'Sales' department, you would select 'department' here."
        )

    condition_operator = (
        "What it does: Defines how to compare the value in the Condition Field with the Condition Values.\n"
        "• Options: `in`, `not_in`, `equals`, `greater_than`, `less_than`, etc.\n"
        "• Default: `in`"
        )

    condition_values = (
        "How it works: These are the values that the condition must match. For the 'in' operator, you can provide a list of values.\n"
        "• Example: To suppress salaries for the 'Sales' and 'Marketing' departments, you would enter `Sales, Marketing`."
        )

    mode = (
        "What it does:\n"
        "• REPLACE: Overwrites the data in the original column with the suppressed values.\n"
        "• ENRICH: Keeps the original column and adds a new column containing the suppressed values.\n"
        "• Recommended: Use 'ENRICH' during testing to easily compare the original and suppressed data."
        )

    output_field_name = (
        "What it does: Specifies the header for the new column.\n"
        "• Example: If you are suppressing cells in the 'salary' column, you could name the new column 'salary_suppressed'.\n"
        "• Default: If left empty, a name is auto-generated (e.g., `_salary`)."
        )

    column_prefix = (
        "What it does: Helps in auto-generating a new column name.\n"
        "• Example: If the original column is 'age' and the prefix is `suppressed_`, the new column will be named `suppressed_age`.\n"
        "• Default: `_`"
        )

    null_strategy = (
        "What it does: This is a pre-processing step that runs before the main suppression logic is applied.\n"
        "• PRESERVE: Keep null values as they are. They may still be targeted for suppression if the Suppression Trigger is set to 'Null'.\n"
        "• ERROR: Stop the entire operation if any null values are found.\n"
        "• Recommended: `PRESERVE`"
        )

    force_recalculation = (
        "What it does: Disables the caching mechanism for this run, forcing the operation to re-process all data from scratch.\n"
        "• Use Case: Enable this if you have changed the underlying data or want to ensure a fresh run for auditing purposes."
        )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "suppression_strategy": cls.suppression_strategy,
            "suppression_value": cls.suppression_value,
            "group_by_field": cls.group_by_field,
            "min_group_size": cls.min_group_size,
            "suppress_if": cls.suppress_if,
            "outlier_method": cls.outlier_method,
            "outlier_threshold": cls.outlier_threshold,
            "rare_threshold": cls.rare_threshold,
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
        }
