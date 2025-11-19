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


class RemoveFieldsOperationTooltip:
    pattern = (
        "What it does: Uses regular expression patterns to automatically identify and remove fields whose names match the specified pattern. Must be valid regex syntax. "
        'Example: Pattern match="^temp_" removes all fields starting with "temp_" like "temp_score", "temp_flag", "temp_data"; '
        'Pattern match="_old$" removes fields ending with "_old"'
    )

    fields_to_remove = (
        "What it does: Specifies which columns should be excluded from the split partitions\n"
        "• Select one or more column names to remove before partitioning\n"
        "• Useful for excluding sensitive fields, temporary columns, or irrelevant data from output files\n"
        "• Example: Remove columns like 'password', 'ssn', or 'internal_notes' from all generated partitions\n"
        "• Note: The ID field used for splitting cannot be removed and will always be included"
    )

    output_format = "What it does: Specifies the file format used to save the dataset after field removal processing"

    save_output = (
        "What it does: Controls whether the generated partition files are actually saved to storage or just processed in memory\n"
        "• When checked, creates separate files for each partition that can be downloaded or accessed later.\n"
        "• When unchecked, partitions are created in memory only for immediate downstream processing"
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = (
        "What it does: Forces the operation to recalculate results from scratch, ignoring any existing cached results. Useful when you want "
        "to ensure results reflect any subtle data changes"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "pattern": cls.pattern,
            "fields_to_remove": cls.fields_to_remove,
            "output_format": cls.output_format,
            "save_output": cls.save_output,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
