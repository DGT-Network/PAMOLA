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

class PhoneOperationTooltip:

    min_frequency = (
        "What it does: Sets the minimum number of times a value (country code, operator code, or messenger) must appear to be included in dictionary CSV files."
    )

    country_codes = (
        "What it does: Limits operator code analysis to phone numbers from specific country codes.\n"
        "• When left empty, all countries are included.\n"
        "• When specified, only phones matching these country codes are analyzed for operator patterns."
    )

    patterns_csv = (
        "What it does: Path or name for the CSV file where extracted phone/operator/messenger patterns will be saved."
    )

    generate_visualization = (
        "What it does: If enabled, generates a visualization (e.g., chart or graph) of the operator/messenger pattern distribution."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "min_frequency": cls.min_frequency,
            "country_codes": cls.country_codes,
            "patterns_csv": cls.patterns_csv,
            "generate_visualization": cls.generate_visualization,
        }
