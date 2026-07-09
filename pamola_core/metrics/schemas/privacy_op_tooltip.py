"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Privacy Metric Tooltips
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides user-facing tooltips for privacy metric operation configuration
fields in PAMOLA.CORE.
- Explains supported metrics, column selection, and sampling knobs
- Designed for integration with Formily and schema-driven UI builders
"""


class PrivacyMetricOperationTooltip:
    privacy_metrics = (
        "What it does: Selects which privacy metrics to compute against the transformed dataset.\n"
        "• DCR (Distance to Closest Record): Average distance from each record to its nearest neighbor.\n"
        "• NNDR (Nearest Neighbor Distance Ratio): Ratio of distance to closest vs second-closest record.\n"
        "• UNIQUENESS: Identifies how many records are uniquely identifiable.\n"
        "• K-ANONYMITY: Each record is indistinguishable from at least k-1 others on quasi-identifiers.\n"
        "• L-DIVERSITY: Sensitive attributes within each equivalence class are well-represented.\n"
        "• Default: DCR."
    )

    metric_params = (
        "What it does: Optional dictionary of metric-specific parameters keyed by metric name.\n"
        "• Example: {\"dcr\": {\"distance_metric\": \"euclidean\"}, \"k_anonymity\": {\"k\": 5}}.\n"
        "• Leave empty to use defaults for every selected metric."
    )

    columns = (
        "What it does: Restricts metric calculation to a subset of columns.\n"
        "• For DCR/NNDR these should be the columns used for distance comparison.\n"
        "• For K-ANONYMITY / L-DIVERSITY these should be the quasi-identifier columns.\n"
        "• Leave empty to use all comparable columns."
    )

    column_mapping = (
        "What it does: Optional mapping from original column names to transformed column names.\n"
        "• Example: {\"age\": \"age_generalized\"} compares 'age' (original) against 'age_generalized' (transformed).\n"
        "• Use when column names differ between the two datasets."
    )

    sample_size = (
        "What it does: If set, randomly samples this many records before computing metrics.\n"
        "• Use for very large datasets to speed up distance-based calculations.\n"
        "• Default: None (use all records)."
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
        "What it does: Enables generation of charts visualizing privacy metric results.\n"
        "• Uncheck for faster execution if visualizations are not needed.\n"
        "• Default: Enabled."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "privacy_metrics": cls.privacy_metrics,
            "metric_params": cls.metric_params,
            "columns": cls.columns,
            "column_mapping": cls.column_mapping,
            "sample_size": cls.sample_size,
            "use_cache": cls.use_cache,
            "force_recalculation": cls.force_recalculation,
            "generate_visualization": cls.generate_visualization,
        }
