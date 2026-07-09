"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Utility Metric Tooltips
Package:       pamola_core.metrics.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides user-facing tooltips for utility metric operation configuration
fields in PAMOLA.CORE.
- Explains supported metric types (classification/regression), column selection, sampling
- Designed for integration with Formily and schema-driven UI builders
"""


class UtilityMetricOperationTooltip:
    utility_metrics = (
        "What it does: Selects which downstream-task utility metrics to compute.\n"
        "• classification: Trains a classifier on original vs transformed data and reports "
        "AUROC, Accuracy, F1, Precision, and Recall.\n"
        "• regression: Trains a regressor and reports R2, MSE, and MAE.\n"
        "• Choose based on the target column type."
    )

    metric_params = (
        "What it does: Optional dictionary of metric-specific parameters keyed by metric name.\n"
        "• Example: {\"classification\": {\"target\": \"label\", \"models\": [\"rf\", \"lr\"]}}.\n"
        "• Includes target column, model selection, and downstream-task hyperparameters."
    )

    columns = (
        "What it does: Restricts metric calculation to a subset of feature columns.\n"
        "• Leave empty to use all comparable columns.\n"
        "• Exclude target/identifier columns explicitly via metric_params."
    )

    column_mapping = (
        "What it does: Optional mapping from original column names to transformed column names.\n"
        "• Example: {\"age\": \"age_generalized\"} compares 'age' (original) against 'age_generalized' (transformed).\n"
        "• Use when column names differ between the two datasets."
    )

    normalize = (
        "What it does: Normalizes feature columns before training downstream models.\n"
        "• Recommended for distance-sensitive models (e.g., kNN, SVM).\n"
        "• Default: True (enabled)."
    )

    confidence_level = (
        "What it does: Confidence level for statistical reporting of metric results.\n"
        "• Range: 0 < value <= 1.\n"
        "• Default: 0.95 (95% confidence)."
    )

    sample_size = (
        "What it does: If set, randomly samples this many rows from each dataset before training models.\n"
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
        "What it does: Enables generation of charts visualizing utility metric results "
        "(bar plots, precision-recall curves, etc.).\n"
        "• Uncheck for faster execution if visualizations are not needed.\n"
        "• Default: Enabled."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "utility_metrics": cls.utility_metrics,
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
