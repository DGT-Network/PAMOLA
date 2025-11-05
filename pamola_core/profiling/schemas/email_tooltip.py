"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Email Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Email Operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Email Operation anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Email Operation tooltip file
"""

class EmailOperationTooltip:

    top_n = (
        "What it does: Limits how many of the most frequent email domains appear in the statistics JSON file "
        "and in the visualization bar chart.\n"
        "• Example: If set to `10`, only the top 10 most common domains (e.g., gmail.com, yahoo.com) will be displayed."
    )

    min_frequency = (
        "What it does: Sets the minimum number of times a domain must appear to be included in the domain dictionary CSV file.\n"
        "• Example: If set to `5`, only domains appearing 5 times or more will be included in the exported list."
    )


    analyze_privacy_risk = (
        "What it does: Performs a privacy risk assessment by analyzing email uniqueness ratio, "
        "personal name patterns, and the potential identifiability of individuals based on email addresses.\n"
        "• Impact: Helps identify high-risk fields that may reveal personal identities or sensitive information."
    )

    generate_visualization = (
        "What it does: When enabled, generates visual reports such as bar charts or distribution graphs for email domain statistics.\n"
        "• Impact: Useful for exploring domain usage patterns visually during data profiling."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "top_n": cls.top_n,
            "min_frequency": cls.min_frequency,
            "generate_visualization": cls.generate_visualization,
            "analyze_privacy_risk": cls.analyze_privacy_risk,
        }
