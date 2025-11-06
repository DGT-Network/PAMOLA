"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Identity Analysis Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Identity Analysis Operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Identity Analysis Operation anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Identity Analysis Operation tooltip file
"""


class IdentityAnalysisOperationTooltip:

    uid_field = (
        "What it does: Selects the primary identifier field to analyze for consistency with reference fields.\n"
        "• Example: Select `customer_id` to analyze whether each customer_id consistently represents the same person "
        "based on reference fields such as name, date of birth, or email."
    )

    id_field = (
        "What it does: When provided, the system analyzes the distribution of this entity field across UID values to determine "
        "one-to-one vs one-to-many relationships.\n"
        "• Example: Select `resume_id` with UID field=`person_id` to find people with multiple resumes. "
        "Output shows person 123 has 3 resumes, person 456 has 1 resume."
    )

    top_n = (
        "What it does: Limits how many example records are included in mismatch and distribution output files for manual review.\n"
        "• Example: If `top_n=15` and 500 inconsistent UIDs are detected, only the 15 most frequent inconsistencies are included "
        "in the output file for readability. The total count (500) is still reported in the summary metrics."
    )

    min_similarity = (
        "What it does: Sets the minimum similarity score (0.0–1.0) required for two text values to be considered a match when "
        "fuzzy matching is enabled.\n"
        "• Example: `similarity=0.85` treats `john.doe` and `john.do` as matching (typo tolerance).\n"
        "• Lower threshold (e.g., 0.6) might match `J. Smith` with `John Smith`.\n"
        "• Impact: Adjust this to control sensitivity in detecting near-duplicate text values."
    )

    check_cross_matches = (
        "What it does: Enables analysis to find records where reference fields match exactly but UID values differ, "
        "indicating potential duplicate entities or identifier assignment errors.\n"
        "• Example: Finds cases where the same `(first_name, last_name, birth_date)` combination appears with different `customer_id`s.\n"
        "• Example Output: `[Alice, Johnson, 1985-03-20]` appears with `customer_id=789` and `customer_id=890`, "
        "suggesting duplicate records or data entry errors."
    )

    fuzzy_matching = (
        "What it does: Enables fuzzy string matching algorithms so that comparisons between reference field values are based on similarity "
        "scores rather than requiring exact matches.\n"
        "• Impact: Helps detect records that refer to the same entity despite minor spelling or formatting differences."
    )

    generate_visualization = "What it does: Controls whether to generate PNG visualizations showing value distributions, combination frequencies, and value count distributions"

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "uid_field": cls.uid_field,
            "id_field": cls.id_field,
            "top_n": cls.top_n,
            "min_similarity": cls.min_similarity,
            "check_cross_matches": cls.check_cross_matches,
            "fuzzy_matching": cls.fuzzy_matching,
            "generate_visualization": cls.generate_visualization,
        }
