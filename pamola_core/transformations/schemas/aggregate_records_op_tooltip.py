"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Aggregate Records Operation Tooltips
Package:       pamola_core.transformations.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for aggregate records configuration fields in PAMOLA.CORE.
- Explains grouping, aggregation functions, and output control options
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of aggregation operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of aggregate records tooltip file
"""


class AggregateRecordsOperationTooltip:
    group_by_fields = (
        "What it does: Specifies which field(s) will be used as grouping keys for aggregation. All records with identical values in these fields "
        "will be combined into a single group. This is an SQL-like GROUP BY operation. For example, grouping by 'application_id' will create "
        "one group per unique application_id value, and all records with the same application_id will be aggregated together."
    )

    aggregations = (
        "What it does: Specifies which mathematical or statistical operations to perform on the selected field for each group\n"
        "• Count: Number of non-null values in the group\n"
        "• Sum: Total of all numeric values (numeric fields only)\n"
        "• Mean: Average value (numeric fields only)\n"
        "• Median: Middle value when sorted (numeric fields only)\n"
        "• Min: Smallest value (numeric/date fields)\n"
        "• Max: Largest value (numeric/date fields)\n"
        "• Std: Standard deviation, measure of spread (numeric fields only)\n"
        "• Var: Variance, measure of variability (numeric fields only)\n"
        "• First: First value encountered in the group\n"
        "• Last: Last value encountered in the group\n"
        "• Nunique: Count of distinct/unique values in the group"
    )

    custom_aggregations = (
        "What it does: Specifies which field will have custom aggregation functions applied to it. Similar to standard aggregations, but uses "
        "specialized custom functions instead of built-in functions."
    )

    output_format = (
        "What it does: Specifies the file format for the aggregated output dataset"
    )

    save_output = (
        "What it does: Controls whether the aggregated dataset is saved to disk. When enabled (checked), aggregated dataset is written to "
        "task output directory in the specified format."
    )

    generate_visualization = "What it does: Controls whether visualization charts are generated showing aggregation impact and results"

    force_recalculation = (
        "What it does: Forces the operation to reprocess data even if valid cached results exist. When enabled, cache check is skipped and "
        "fresh aggregation is performed. New results may update the cache."
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "group_by_fields": cls.group_by_fields,
            "aggregations": cls.aggregations,
            "custom_aggregations": cls.custom_aggregations,
            "output_format": cls.output_format,
            "save_output": cls.save_output,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
