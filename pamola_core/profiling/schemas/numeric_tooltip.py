"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Numberic Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Numeric Operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Numeric Operation anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Numeric Operation tooltip file
"""

class NumericOperationTooltip:

    bins = (
        "What it does: Sets the number of intervals (bins) used to group values when creating the histogram visualization."
    )

    near_zero_threshold = (
        "What it does: Defines the threshold below which numeric values are classified as 'near zero' in the special value analysis.\n"
        "• Example: For a field with values like `0.0000005`, `-0.000003`, with a threshold of `1e-10`, these are counted as near-zero.\n"
        "• Example: For financial data with values like `0.001`, `0.005`, a threshold might be set to `0.01` to capture cent-level precision."
    )

    detect_outliers = (
        "What it does: Enables statistical outlier detection using the Interquartile Range (IQR) method to identify extreme values.\n"
        "• Impact: Helps highlight unusually large or small values that may indicate anomalies or data quality issues."
    )

    test_normality = (
        "What it does: Performs statistical normality tests (Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov) "
        "to determine if data follows a normal (Gaussian) distribution.\n"
        "• Impact: Useful for deciding which statistical models or anonymization strategies are appropriate for the data."
    )

    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "bins": cls.bins,
            "near_zero_threshold": cls.near_zero_threshold,
            "detect_outliers": cls.detect_outliers,
            "test_normality": cls.test_normality,
            "generate_visualization": cls.generate_visualization,
        }
