"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Clean Invalid Values Operation Tooltips
Package:       pamola_core.transformation.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Provides detailed tooltips for clean invalid values operation configuration fields in PAMOLA.CORE.
- Explains constraint-based validation, whitelist/blacklist filtering, and null replacement options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of data cleaning operations

Changelog:
1.0.0 - 2025-11-11 - Initial creation of clean invalid values operation tooltip file
"""


class CleanInvalidValuesOperationTooltip:
    field_constraints = (
        "What it does: Defines validation rules for each field to identify invalid values.\n\n"
        "Use this to specify what constitutes a 'valid' value for each column. Values that violate "
        "these constraints will be nullified (replaced with null or a custom replacement value).\n\n"
        "Constraint types:\n"
        "• min/max: Numeric range boundaries (e.g., {'age': {'min': 0, 'max': 120}})\n"
        "• pattern: Regular expression pattern for text validation (e.g., {'email': {'pattern': r'^[\\w.-]+@[\\w.-]+\\.\\w+$'}})\n"
        "• allowed_values: List of valid values (e.g., {'status': {'allowed_values': ['active', 'inactive']}})\n"
        "• data_type: Expected data type (e.g., {'count': {'data_type': 'int'}})\n"
        "• not_null: Whether null values are allowed (e.g., {'id': {'not_null': True}})\n\n"
        "Example: {'age': {'min': 0, 'max': 150}, 'email': {'pattern': r'^[\\w.-]+@'}}\n\n"
        "Leave blank if no constraint-based validation is needed."
    )

    whitelist_path = (
        "What it does: Specifies file paths containing allowed values for each field.\n\n"
        "Values not found in the whitelist will be considered invalid and nullified. "
        "This is useful for validating against predefined lists of acceptable values.\n\n"
        "Format: Dictionary mapping field names to file paths containing valid values (one per line)\n\n"
        "Example: {'country': Path('valid_countries.txt'), 'department': Path('valid_depts.txt')}\n\n"
        "The whitelist file should contain one valid value per line:\n"
        "```\n"
        "USA\n"
        "Canada\n"
        "Mexico\n"
        "```\n\n"
        "Note: Only values matching the whitelist exactly will be kept. Matching is case-sensitive by default."
    )

    blacklist_path = (
        "What it does: Specifies file paths containing prohibited values for each field.\n\n"
        "Values found in the blacklist will be considered invalid and nullified. "
        "This is useful for filtering out known bad values, profanity, or sensitive terms.\n\n"
        "Format: Dictionary mapping field names to file paths containing invalid values (one per line)\n\n"
        "Example: {'username': Path('blocked_names.txt'), 'comment': Path('profanity_list.txt')}\n\n"
        "The blacklist file should contain one invalid value per line:\n"
        "```\n"
        "admin\n"
        "root\n"
        "test\n"
        "```\n\n"
        "Note: Any value matching the blacklist will be nullified. Can be combined with whitelist validation."
    )

    null_replacement = (
        "What it does: Defines what value to use when replacing invalid entries.\n\n"
        "By default, invalid values are replaced with null/None. Use this parameter to specify "
        "a different replacement strategy.\n\n"
        "Options:\n"
        "• String value: Use the same replacement for all fields (e.g., 'INVALID' or 'N/A')\n"
        "• Dictionary: Specify different replacements per field (e.g., {'age': -1, 'status': 'unknown'})\n"
        "• Special values:\n"
        "  - None or null: Replace with null (default)\n"
        "  - 'mean': Replace with field mean (numeric fields only)\n"
        "  - 'median': Replace with field median (numeric fields only)\n"
        "  - 'mode': Replace with most common value\n\n"
        "Examples:\n"
        "• null_replacement='INVALID' → All invalid values become 'INVALID'\n"
        "• null_replacement={'age': 0, 'name': 'Unknown'} → Field-specific replacements\n"
        "• null_replacement='mean' → Replace with statistical mean\n\n"
        "Default: None (replaces with null)"
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = "What it does: Forces the operation to recalculate results from scratch, ignoring any existing cached results. Useful when you want to ensure results reflect any subtle data changes"

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "field_constraints": cls.field_constraints,
            "whitelist_path": cls.whitelist_path,
            "blacklist_path": cls.blacklist_path,
            "null_replacement": cls.null_replacement,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
