"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Group Analyzer Operation Tooltips
Package:       pamola_core.profiling.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for Group Analyzer Operation configuration fields in PAMOLA.CORE.
- Explains binning, rounding, range, and conditional logic options for Group Analyzer Operation anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of anonymization operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of Group Analyzer Operation tooltip file
"""


class GroupAnalyzerOperationTooltip:

    variance_threshold = (
        "What it does: After calculating weighted variance for each group (combining individual field variances with their weights), "
        "the system compares against this threshold. Groups with variance < threshold are flagged as 'should aggregate'.\n"
        "• Variance of 0.0 = all records identical.\n"
        "• Variance of 1.0 = maximum variation.\n"
        "• Example: With `threshold = 0.2` (20% variation):\n"
        "  – Group `[Software Engineer, Software Developer, Software Engineer]` has low variance → can aggregate.\n"
        "  – Group `[Software Engineer, Teacher, Doctor]` has high variance → cannot aggregate."
    )

    large_group_threshold = (
        "What it does: Sets the minimum number of records required for a group to be classified as 'large' and subject to stricter variance rules "
        "via the Large Group Variance Threshold.\n"
        "• Example: `Threshold = 100` in a 10,000-record dataset:\n"
        "  – Group with 95 records uses normal variance threshold (0.2).\n"
        "  – Group with 120 records uses large group variance threshold (0.05).\n"
        "• Purpose: Ensures large groups have higher similarity (lower variance) before aggregation."
    )

    large_group_variance_threshold = (
        "What it does: Sets a stricter (typically lower) variance threshold applied only to groups exceeding the Large Group Threshold size.\n"
        "• When group size > Large Group Threshold, this value replaces the normal Variance Threshold.\n"
        "• Purpose: Forces large groups to have higher similarity before qualifying for aggregation, "
        "preventing over-generalization of heterogeneous groups."
    )

    text_length_threshold = (
        "What it does: Sets the character length at which text fields switch from direct string comparison "
        "to hash-based comparison for better performance and memory efficiency.\n"
        "• Example: `Threshold = 100` → job titles (~20–50 chars) compared directly, "
        "but long text descriptions (200+ chars) use hashing."
    )

    hash_algorithm = (
        "What it does: Selects the algorithm for comparing long text fields (those exceeding the Text Length Threshold).\n"
        "• **MD5**: Generates exact hash fingerprints — two texts match only if 100% identical (including punctuation/whitespace).\n"
        "• **MinHash**: Generates similarity signatures — two texts match if their similarity score ≥ the MinHash Similarity Threshold.\n"
        "• Example: With MD5, `Software Engineer at Google` ≠ `Software Engineer at Google.` (extra period).\n"
        "  With MinHash at 0.9 similarity, these do match due to near-identical content.\n"
        "• MinHash can also match `Developed web applications` with `Develop web application` (tense/plural variation)."
    )

    minhash_similarity_threshold = (
        "What it does: Defines the minimum similarity score (0.0–1.0) for two texts to be considered a match — "
        "used only when MinHash is selected.\n"
        "• Score 1.0 = identical texts, 0.0 = completely different.\n"
        "• Example: With `threshold = 0.7` (70% similarity):\n"
        "  – Comparing `Senior Software Engineer, developed web apps` with "
        "`Senior Software Developer, developed web applications` gives similarity = 0.75 → match (low variance).\n"
        "  – With threshold = 0.9 → no match (high variance)."
    )

    fields_config = (
        "What it does: Allows assigning integer weight values (0–100) to each field, defining its relative importance when calculating weighted group variance.\n"
        "• Fields with weight = 0 are excluded from analysis.\n"
        "• At least one field must have a weight > 0.\n"
        "• Example: Resume dataset with 50 fields:\n"
        "  – `first_name=0`, `last_name=0` (exclude, PII)\n"
        "  – `job_title=5` (very important)\n"
        "  – `salary=4` (important)\n"
        "  – `education=3` (moderate)\n"
        "  – `city=2` (minor)\n"
        "→ Only fields with non-zero weights are analyzed, prioritizing job/salary over location."
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
            "variance_threshold": cls.variance_threshold,
            "large_group_threshold": cls.large_group_threshold,
            "large_group_variance_threshold": cls.large_group_variance_threshold,
            "text_length_threshold": cls.text_length_threshold,
            "hash_algorithm": cls.hash_algorithm,
            "minhash_similarity_threshold": cls.minhash_similarity_threshold,
            "fields_config": cls.fields_config,
            "generate_visualization": cls.generate_visualization,
            "force_recalculation": cls.force_recalculation,
        }
