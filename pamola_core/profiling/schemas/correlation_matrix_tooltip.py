"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Correlation Matrix Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-11
License:       BSD 3-Clause

Description:
Provides detailed tooltips for correlation matrix operation configuration fields in PAMOLA.CORE.
- Explains field selection, correlation methods, threshold settings, and null handling options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of correlation analysis operations

Changelog:
1.0.0 - 2025-11-11 - Initial creation of correlation matrix operation tooltip file
"""


class CorrelationMatrixOperationTooltip:
    methods = (
        "What it does: Defines the correlation calculation method for specific field pairs.\n\n"
        "Use this to customize which correlation algorithm is used for different field combinations.\n\n"
        "Common methods:\n"
        "• 'pearson': Measures linear relationships (default for most numeric data)\n"
        "• 'spearman': Measures monotonic relationships (better for ranked or non-linear data)\n"
        "• 'kendall': Measures ordinal associations (robust to outliers)\n\n"
        "Example: {('age', 'income'): 'pearson', ('satisfaction', 'rating'): 'spearman'}\n\n"
        "Leave blank to use the default method (pearson) for all field pairs."
    )

    min_threshold = (
        "What it does: Filters out weak correlations below this value.\n\n"
        "Only correlations with absolute values above this threshold will be highlighted or displayed "
        "in the results. This helps focus on statistically significant relationships.\n\n"
        "Correlation strength guide:\n"
        "• 0.0 - 0.3: Weak correlation\n"
        "• 0.3 - 0.7: Moderate correlation\n"
        "• 0.7 - 1.0: Strong correlation\n\n"
        "Default: 0.3 (moderate and strong correlations only)\n\n"
        "Tip: Lower this threshold (e.g., 0.1) to see weaker relationships, or raise it (e.g., 0.5) "
        "to focus only on strong correlations."
    )

    null_handling = (
        "What it does: Determines how missing values are handled during correlation calculation.\n\n"
        "• 'drop': Removes any rows with missing values (pairwise or listwise deletion). "
        "This ensures only complete data is used but may reduce sample size.\n"
        "• 'fill_zero': Replaces missing values with 0 before calculating correlations. "
        "Use with caution as it can skew results.\n"
        "• 'fill_mean': Replaces missing values with the mean of each field. "
        "This preserves sample size but may introduce bias.\n\n"
        "Default: 'drop'\n\n"
        "Recommendation: Use 'drop' for most cases unless you have a specific reason to impute missing data."
    )

    generate_visualization = (
        "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value "
        "count distributions"
    )

    force_recalculation = (
        "What it does: Ignore saved results. Check this to force the operation to run again instead of using a cached result."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "methods": cls.methods,
            "min_threshold": cls.min_threshold,
            "null_handling": cls.null_handling,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }