"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Record Suppression Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for record suppression configuration fields in PAMOLA.CORE.
- Explains suppression conditions, multi-field logic, and risk-based removal options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of data suppression operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of record suppression tooltip file
"""


class RecordSuppressionOpTooltip:
    suppression_condition = (
        "What it does: This is the main trigger for the suppression logic. The operation will remove any row that matches the "
        "selected condition.\n\n"
        "• Null: Removes rows where the primary field is empty. (Default)\n"
        "• Value: Removes rows where the primary field matches a value in your specified list.\n"
        "• Range: Removes rows where the primary field's value falls within a numeric range you define.\n"
        "• Risk: Removes rows based on a pre-calculated risk score.\n"
        "• Custom: Removes rows based on a set of advanced, multi-field rules."
    )

    suppression_values = (
        "What it does: Any row where the primary field's value exactly matches one of the values in this list will be removed.\n\n"
        "• Example: To remove all records for 'Test Account' and 'Inactive User', you would enter `Test Account, Inactive User`.\n"
        "• Validation: This field is required when the condition is 'Value'."
    )

    suppression_range = (
        "What it does: Any row where the primary field's value falls between the minimum and maximum values (inclusive) "
        "will be removed.\n\n"
        "• Example: To remove all records with a 'score' between 0 and 10, you would enter `0, 10`.\n"
        "• Validation: The primary field must be numeric for this condition to work. This field is required when the condition is 'Range'."
    )

    multi_conditions = (
        "What it does: Allows you to create more advanced filtering logic than the simple conditions.\n\n"
        "• Example: You could set up rules to remove records where `(department == 'Sales' AND status == 'Inactive')`.\n"
        "• Impact: This provides highly granular control for targeting specific records for suppression."
    )

    condition_logic = (
        "What it does: Determines how to combine the rules set in Multi-field Conditions.\n\n"
        "• AND: All conditions must be true for the record to be removed.\n"
        "• OR: If any of the conditions are true, the record will be removed.\n"
        "• Default: 'AND'"
    )

    ka_risk_field = (
        "How it works: The operation will look at this column to identify high-risk records that need to be removed.\n\n"
        "• Example: If you have a column named 'k_anonymity_score', you would select it here."
    )

    risk_threshold = (
        "What it does: Defines the cutoff for the risk-based suppression. Any record where the Risk Score Field value is "
        "less than this threshold will be removed.\n\n"
        "• Example: If the threshold is '5', any row with a risk score of 1, 2, 3, or 4 will be removed.\n"
        "• Default: '5.0'"
    )

    save_suppressed_records = (
        "What it does: When enabled, the operation creates a new file (e.g., a CSV) that contains a copy of every row that "
        "was deleted from the main dataset.\n\n"
        "• Impact: This is highly recommended for auditing and data governance, as it provides a clear record of what was "
        "removed and why.\n"
        "• Default: 'False'"
    )

    suppression_reason_field = (
        "What it does: Adds a column to the saved file of removed records, making it easy to see which rule triggered the "
        "suppression for each row.\n\n"
        "• Example: The column might contain reasons like 'Null value in field email' or 'K-anonymity risk below threshold'.\n"
        "• Default: `_suppression_reason`"
    )

    force_recalculation = (
        "What it does: Disables the caching mechanism for this run, forcing the operation to re-process all data from scratch.\n\n"
        "• Use Case: Enable this if you have changed the underlying data or want to ensure a fresh run for auditing purposes."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "suppression_condition": cls.suppression_condition,
            "suppression_values": cls.suppression_values,
            "suppression_range": cls.suppression_range,
            "multi_conditions": cls.multi_conditions,
            "condition_logic": cls.condition_logic,
            "ka_risk_field": cls.ka_risk_field,
            "risk_threshold": cls.risk_threshold,
            "save_suppressed_records": cls.save_suppressed_records,
            "suppression_reason_field": cls.suppression_reason_field,
            "force_recalculation": cls.force_recalculation,
        }
