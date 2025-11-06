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

class CurrencyOpTooltip:
    locale = (
        "What it does: Specifies the regional format (locale) used to parse currency values correctly - "
        "determines decimal separator, thousands separator, and currency symbol interpretation.\n"
        "• Wrong locale causes parsing errors - thousands separators treated as decimals or vice versa, "
        "resulting in values 1000x too large or small.\n"
        "• This is critical for international datasets with mixed currency formats."
    )

    bins = (
        "What it does: Sets the number of intervals (bins) used to group currency values when creating the histogram visualization.\n"
        "• Lower bins show general distribution shape, good for overview.\n"
        "• Higher bins reveal detailed patterns like salary bands or price tiers but can be noisy with small datasets.\n"
        "• Very high bins may create sparse histograms."
    )

    detect_outliers = (
        "What it does: Enables statistical outlier detection using the Interquartile Range (IQR) method "
        "to identify extreme currency values."
    )

    test_normality = (
        "What it does: Performs statistical normality tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov) "
        "to determine if currency data follows a normal distribution."
    )

    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing currency distribution and outliers."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "locale": cls.locale,
            "bins": cls.bins,
            "detect_outliers": cls.detect_outliers,
            "test_normality": cls.test_normality,
            "generate_visualization": cls.generate_visualization,
        }
