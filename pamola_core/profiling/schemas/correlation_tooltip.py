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

class CorrelationOpTooltip:
    method = (
        "What it does: Specifies which correlation algorithm to use for measuring the relationship between fields, "
        "or allows automatic selection based on field data types\n"
        "• Auto (default): works well for most cases but may not capture non-linear relationships\n"
        "• Pearson: for numeric-numeric with linear relationship and is sensitive to outliers\n"
        "• Spearman: for numeric-numeric with monotonic relationship or outliers\n"
        "• Cramer's V for categorical-categorical\n"
        "• Correlation Ratio for categorical-numeric\n"
        "• Point Biserial for binary-numeric."
    )

    null_handling = (
        "What it does: Determines how the system handles rows with missing (null) values in either Field 1 or "
        "Field 2 before calculating correlation\n"
        "• Drop (default) removes any row where either field has a null value, analyzing only complete pairs.\n"
        "• Fill Zero replaces nulls with 0 for numeric fields or empty string for categorical fields.\n"
        "• Fill Mean replaces numeric nulls with field mean, categorical nulls with mode (most common value)"
    )
    
    generate_visualization = (
        "What it does: Controls whether to generate a visualization (PNG file) showing the relationship between "
        "Field 1 and Field 2."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "method": cls.method,
            "null_handling": cls.null_handling,
            "generate_visualization": cls.generate_visualization,
        }
