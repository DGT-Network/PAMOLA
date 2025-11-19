"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Categorical Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Categorical Operation configuration fields in PAMOLA.CORE.
- Explains top N, frequency filtering, visualization, and recalculation options for Categorical Operation profiling
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of categorical profiling operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Categorical Operation tooltip file
"""


class CategoricalTooltip:

    top_n = "What it does: Limit values shown in statistics and charts (e.g., top 10 most frequent)"

    min_frequency = (
        "What it does: Sets minimum number of times a value must appear to be included in dictionary CSV (e.g., min 5 = only values appearing ≥5 times). "
        "Higher values reduce file size but may exclude valid rare categories; Lower values preserve all data but create larger files"
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    force_recalculation = "What it does: Ignore saved results. Check this to force the operation to run again instead of using a cached result."

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "top_n": cls.top_n,
            "min_frequency": cls.min_frequency,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
