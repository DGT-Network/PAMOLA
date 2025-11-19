"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        K-Anonymity Profiler Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for K-Anonymity Profiler Operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for K-Anonymity Profiler Operation anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of K-Anonymity Profiler Operation tooltip file
"""


class KAnonymityProfilerOperationTooltip:

    analysis_mode = (
        "What it does: Determines whether the operation generates privacy reports, adds k-anonymity values to the dataset, or performs both actions.\n"
        "• **Analyze mode**: Creates metrics, visualizations, and identifies vulnerable records **without modifying data**. "
        "Fastest option (read-only analysis).\n"
        "• **Enrich mode**: Adds a new column with k-values for each record based on specified quasi-identifiers. "
        "Slower, but enables downstream filtering and analysis.\n"
        "• **Both mode**: Performs complete analysis and enrichment. "
        "Takes longest but provides a full privacy assessment with enriched data output."
    )

    quasi_identifiers = (
        "What it does: Selects individual fields that, when combined, could potentially identify individuals "
        "(known as quasi-identifiers).\n"
        "• Example: Selecting `[\"age\", \"gender\", \"zipcode\"]` generates combinations such as "
        "`[\"age\", \"gender\"]`, `[\"age\", \"zipcode\"]`, `[\"gender\", \"zipcode\"]`, and `[\"age\", \"gender\", \"zipcode\"]` for analysis."
    )

    quasi_identifier_sets = (
        "What it does: Allows you to specify **exact combinations** of quasi-identifiers to analyze, "
        "instead of testing all possible combinations.\n"
        "• Example: `QI Sets = [[\"age\", \"gender\"], [\"zipcode\", \"education\"]]` analyzes only these two combinations, "
        "saving computation time and focusing on relevant cases."
    )

    threshold_k = (
        "What it does: Sets the minimum group size (k-value) below which records are considered vulnerable to re-identification.\n"
        "• Example: With `Threshold K = 5`, a group with `[age=25, gender=F, zipcode=10001]` containing only 3 records "
        "is flagged as vulnerable because `3 < 5`."
    )

    max_combinations = (
        "What it does: Limits the total number of quasi-identifier combinations to analyze, "
        "preventing excessive computation when many fields are selected.\n"
        "• Example: With 6 quasi-identifiers generating 63 possible combinations, setting `Max Combinations = 50` "
        "analyzes only the first 50 (e.g., all 2-field, all 3-field, and some 4-field combinations)."
    )

    id_fields = (
        "What it does: Specifies which field(s) contain unique record or entity identifiers used to track vulnerable individuals in reports.\n"
        "• Without ID Fields: Vulnerable record reports only show row indices, which may not be stable.\n"
        "• With ID Fields: Reports include meaningful identifiers, allowing data stewards to locate and anonymize specific records."
    )

    output_field_suffix = (
        "What it does: Sets the suffix appended to field combination names when creating k-anonymity columns "
        "in Enrich or Both modes.\n"
        "• Example: With suffix `_kval`, a combination of `[age, zipcode]` creates a column named `age_zipcode_kval`."
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
            "analysis_mode": cls.analysis_mode,
            "quasi_identifiers": cls.quasi_identifiers,
            "quasi_identifier_sets": cls.quasi_identifier_sets,
            "threshold_k": cls.threshold_k,
            "max_combinations": cls.max_combinations,
            "id_fields": cls.id_fields,
            "output_field_suffix": cls.output_field_suffix,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
