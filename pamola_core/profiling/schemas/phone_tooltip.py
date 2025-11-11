"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Phone Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for phone profiling configuration fields in PAMOLA.CORE.
- Explains frequency filtering, country code analysis, and pattern detection options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of phone profiling operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of phone operation tooltip file
"""


class PhoneOperationTooltip:
    min_frequency = (
        "What it does: Sets the minimum number of times a value (country code, operator code, or messenger) must appear to be included "
        "in dictionary CSV files."
    )

    country_codes = (
        "What it does: Limits operator code analysis to phone numbers from specific country codes.\n"
        "• When left empty, all countries are included.\n"
        "• When specified, only phones matching these country codes are analyzed for operator patterns."
    )

    patterns_csv = "What it does: Path or name for the CSV file where extracted phone/operator/messenger patterns will be saved."

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = "What it does: Ignore saved results. Check this to force the operation to run again instead of using a cached result."

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "min_frequency": cls.min_frequency,
            "country_codes": cls.country_codes,
            "patterns_csv": cls.patterns_csv,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
