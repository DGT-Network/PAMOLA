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

class AttributeSuppressionOpTooltip:
    suppression_strategy = (
        "What it does: Defines what the suppressed cells will be replaced with.\n"
        "• Null: Replaces values with an empty (null) value. (Default)\n"
        "• Mean/Median/Mode: Replaces values with the mean, median, or mode of the entire column. Requires a numeric field for Mean/Median.\n"
        "• Constant: Replaces values with a specific text or number you provide.\n"
        "• Group Mean/Mode: Replaces values with the mean or mode of the subgroup the row belongs to, based on the Group By field."
    )
        
    condition_field = (
        "How it works: This operation first filters the dataset to keep only the rows that meet your condition. "
        "Then, it removes the specified column(s) from that filtered subset.\n"
        "• Example: If you set the condition to `department` equals `Sales`, the final output will *only* contain rows for the Sales department, "
        "and the specified columns will be removed from that data."
    )
        
    condition_operator = (
        "What it does: Defines how to compare the value in the Condition Field with the Condition Values.\n"
        "• Options: `in`, `not_in`, `equals`, `greater_than`, `less_than`, etc.\n"
        "• Default: `in`"
    )
        
    condition_values = (
        "How it works: These are the values that the condition must match. For the 'in' operator, you can provide a list of values.\n"
        "• Example: To filter for rows where the 'department' is 'Sales' or 'Marketing', you would enter `Sales, Marketing`."
    )
        
    multi_conditions = (
        "What it does: Allows you to create more advanced filtering logic than the simple condition.\n"
        "• Example: You could filter for rows where `(department == 'Sales' AND status == 'Active')`.\n"
        "• Impact: This provides highly granular control over which rows are included in the final output and overrides the simple Condition Field settings if used."
    )
        
    condition_logic = (
        "What it does: Determines how to combine the rules set in Multi-field Conditions.\n"
        "• AND: All conditions must be true.\n"
        "• OR: Any of the conditions can be true.\n"
        "• Default: `AND`"
    )
        
    ka_risk_field = (
        "How it works: This is used to filter out high-risk rows. The operation will only keep rows where the risk score is less than the specified Risk Threshold.\n"
        "• Example: If you want to process only low-risk records, you can set this to your risk score column and define a threshold."
    )
        
    risk_threshold = (
        "What it does: Defines the cutoff for the risk-based filter.\n"
        "• Example: If the threshold is `5`, any row with a risk score of `5` or higher will be removed from the dataset before the columns are suppressed.\n"
        "• Default: `5.0`"
    )
        
    save_suppressed_schema = (
        "What it does: When enabled, the operation creates a JSON file that documents the properties of the deleted columns, "
        "such as their original data type, number of unique values, null counts, and basic statistics for numeric columns.\n"
        "• Impact: This is highly recommended for auditing and data governance, as it provides a record of what was removed.\n"
        "• Default: `True`"
    )
        
    mode = (
        "What it does: Specifies whether to suppress individual cells or entire rows/columns.\n"
        "• Options: 'cell', 'row', 'column'\n"
        "• Default: 'cell'"
    )
        
    output_field_name = (
        "What it does: Specifies the name for the output field after suppression.\n"
        "• Example: If suppressing 'salary', you might name the output 'salary_suppressed'.\n"
        "• Impact: Helps track which fields have been anonymized."
    )
        
    column_prefix = (
        "What it does: Adds a prefix to all suppressed column names.\n"
        "• Example: Setting 'anon_' would rename 'salary' to 'anon_salary'.\n"
        "• Impact: Useful for batch operations on multiple columns."
    )
        
    null_strategy = (
        "What it does: Defines how to handle null/missing values during suppression.\n"
        "• Options: 'keep' (preserve nulls), 'suppress' (treat as suppressible), 'fill' (replace with strategy value)\n"
        "• Default: 'keep'"
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
            "condition_field": cls.condition_field,
            "condition_operator": cls.condition_operator,
            "condition_values": cls.condition_values,
            "multi_conditions": cls.multi_conditions,
            "condition_logic": cls.condition_logic,
            "ka_risk_field": cls.ka_risk_field,
            "risk_threshold": cls.risk_threshold,
            "save_suppressed_schema": cls.save_suppressed_schema,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
        }
