"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Fidelity Metric Tooltips
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
Provides user-facing tooltips for fidelity metric operation configuration
fields in PAMOLA.CORE.
- Explains supported metrics, column selection, sampling, and statistical knobs
- Designed for integration with Formily and schema-driven UI builders
"""


class FidelityOperationTooltip:
    fidelity_metrics = (
        "What it does: Selects which fidelity metrics to compute between the original and "
        "transformed datasets.\n"
        "• KS (Kolmogorov-Smirnov): Compares distributions to detect shape differences.\n"
        "• KL (Kullback-Leibler): Measures divergence between two probability distributions.\n"
        "• Default: KS and KL."
    )

    metric_params = (
        "What it does: Optional dictionary of metric-specific parameters keyed by metric name.\n"
        "• Example: {\"ks\": {\"alternative\": \"two-sided\"}, \"kl\": {\"epsilon\": 1e-10}}.\n"
        "• Leave empty to use defaults for every selected metric."
    )

    columns = (
        "What it does: Restricts metric calculation to a subset of columns.\n"
        "• Leave empty to evaluate all comparable columns.\n"
        "• Useful when the dataset contains identifiers or irrelevant fields."
    )

    column_mapping = (
        "What it does: Optional mapping from original column names to transformed column names.\n"
        "• Example: {\"age\": \"age_generalized\"} compares 'age' (original) against 'age_generalized' (transformed).\n"
        "• Use when column names differ between the two datasets."
    )

    normalize = (
        "What it does: Normalizes inputs before computing the selected metrics.\n"
        "• Recommended when columns are on different scales.\n"
        "• Default: True (enabled)."
    )

    confidence_level = (
        "What it does: Confidence level for statistical tests (e.g., KS).\n"
        "• Range: 0 < value <= 1.\n"
        "• Default: 0.95 (95% confidence)."
    )

    sample_size = (
        "What it does: If set, randomly samples this many rows from each dataset before computing metrics.\n"
        "• Use for very large datasets to speed up calculation.\n"
        "• Default: None (use all rows)."
    )


    use_cache = (
        "What it does: Enables caching of operation results on disk.\n"
        "• When enabled, repeated runs with the same inputs reuse cached output instead of recomputing.\n"
        "• Must be enabled for 'Force Recalculation' to take effect.\n"
        "• Default: False (disabled)."
    )

    force_recalculation = (
        "What it does: Bypasses the operation cache and forces full re-processing.\n"
        "• Use when source data has changed but the cache key has not."
    )

    generate_visualization = (
        "What it does: Enables generation of charts visualizing metric results.\n"
        "• Uncheck for faster execution if visualizations are not needed.\n"
        "• Default: Enabled."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "fidelity_metrics": cls.fidelity_metrics,
            "metric_params": cls.metric_params,
            "columns": cls.columns,
            "column_mapping": cls.column_mapping,
            "normalize": cls.normalize,
            "confidence_level": cls.confidence_level,
            "sample_size": cls.sample_size,
            "use_cache": cls.use_cache,
            "force_recalculation": cls.force_recalculation,
            "generate_visualization": cls.generate_visualization,
        }
