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

    mode = "What it does: Determines whether the operation replaces the original dataset or creates enriched output with additional fields"

    output_format = "What it does: Specifies the file format used to save the dataset after field removal processing"

    use_cache = "What it does: Enables caching of field removal results to speed up repeated operations with the same input data and field removal settings"

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "pattern": cls.pattern,
            "mode": cls.mode,
            "output_format": cls.output_format,
            "use_cache": cls.use_cache,
        }
