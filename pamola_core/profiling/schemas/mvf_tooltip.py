"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        MVFAnalysis Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for MVFAnalysis Operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for MVFAnalysis Operation anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of MVFAnalysis Operation tooltip file
"""


class MVFAnalysisOperationTooltip:

    top_n = (
        "What it does: Limits the number of values shown in statistics and charts.\n"
        "• Example: Setting this to `10` will display only the top 10 most frequent categories."
    )

    min_frequency = (
        "What it does: Sets the minimum number of occurrences a value must have to be included in the dictionary export (e.g., CSV).\n"
        "• Example: If set to `5`, only values appearing 5 times or more will be included."
    )

    format_type = (
        "What it does: Provides a format hint to the system when the multi-value field (MVF) uses a non-standard format or when auto-detection might fail.\n"
        "How it works: The system automatically detects common formats (JSON arrays, array strings, CSV) in most cases.\n"
        "• Example: If a field contains both `['A','B']` and `[\"A\",\"B\"]`, specifying `format_type` helps ensure consistent parsing.\n"
        "• Impact: Only needed for unusual or mixed-format fields."
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = (
        "Ignore saved results. Check this box to force the operation to run again "
        "instead of using a cached result from a previous run with the same settings."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "top_n": cls.top_n,
            "min_frequency": cls.min_frequency,
            "format_type": cls.format_type,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
