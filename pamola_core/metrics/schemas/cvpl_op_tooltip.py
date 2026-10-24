"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL Linkage Operation tooltips
Package:       pamola_core.metrics.schemas
License:       BSD 3-Clause

Plain-language help for the CVPL linkage configuration (Formily / schema builders).
"""


class CVPLLinkageOperationTooltip:
    qids = (
        "What it does: The quasi-identifiers the attacker can use to narrow down candidates in the release.\n"
        "• A released row is a candidate only if its QIDs are compatible with the source record's: equal, "
        "'*' (suppressed), inside a released interval such as [20-29], or an ancestor in a hierarchy you declare.\n"
        "• Each entry may be a name or an object {name, kind, bin_width, hierarchy, generalized}.\n"
        "• A QID the release has generalized without a rule you declared cannot be measured (not applicable)."
    )

    auxiliary = (
        "What it does: Extra attributes the attacker knows and uses to compare records (not to filter).\n"
        "• Only the QIDs and these columns are ever used; nothing is added automatically.\n"
        "• Never list identifier or ground-truth columns here."
    )

    blocking_ladder = (
        "What it does: A list of QID sets, from strict to loose, to measure the attacker with progressively "
        "fewer constraints.\n"
        '• Example: [["age","gender","zip"],["age","gender"]].\n'
        "• When the sets are not nested the headline is the attacker trying every set (union of candidates)."
    )

    tau = (
        "What it does: Minimum cosine similarity (0 < tau <= 1) for a released row to count as a plausible match.\n"
        "• It is a modelling choice of the attacker, not a pass/fail limit.\n"
        "• Default: 0.9."
    )

    fit_strategy = (
        "What it does: Which rows the scaler, encoder and PCA are fitted on.\n"
        "• joint (default): source + release, the original CVPL.\n"
        "• source_only: the source alone; frozen: a previously fitted projection.\n"
        "• To compare several releases of one source use source_only or frozen."
    )

    variance_retained = (
        "What it does: Share of variance the PCA keeps (between min and max components).\n• Default: 0.90."
    )

    min_components = "What it does: Fewest PCA components kept.\n• Default: 3."

    max_components = "What it does: Most PCA components kept.\n• Default: 50."

    max_categories = (
        "What it does: Widest one-hot encoding per categorical attribute; rarer values share one bucket "
        "(reported in the result).\n• Default: 50."
    )

    qid_as_features = (
        "What it does: Whether the QIDs are also part of the vectors compared by similarity.\n• Default: yes."
    )

    null_policy = (
        "What it does: How an unknown (null) QID value is treated.\n"
        "• equal_null (default): matches only null. wildcard: matches anything. incompatible: matches nothing."
    )

    wildcard_tokens = (
        "What it does: Text values that mean 'suppressed, compatible with anything'.\n• Default: ['*']."
    )

    ground_truth = (
        "What it does: Evaluation-only link between source and release: {source_id_column, release_id_column, scope}.\n"
        "• Equal identifiers mean the same subject. The columns are never features and never leave the process.\n"
        "• Without it only the existence measure (cvpl_lr) is computed; the identification measures are "
        "reported as not measured, never as zero."
    )

    seed = "What it does: Recorded in the result; 0 is a valid seed. The measurement itself is deterministic."

    max_comparisons = (
        "What it does: Largest number of (source x candidate) pairs scanned per blocking rung.\n"
        "• A rung above it is reported as not measured (resource limit) instead of running out of memory or time."
    )

    use_cache = (
        "What it does: Enables caching of results on disk; the key covers both datasets and every parameter."
    )

    force_recalculation = "What it does: Bypasses the cache and recomputes."

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "qids": cls.qids,
            "auxiliary": cls.auxiliary,
            "blocking_ladder": cls.blocking_ladder,
            "tau": cls.tau,
            "fit_strategy": cls.fit_strategy,
            "variance_retained": cls.variance_retained,
            "min_components": cls.min_components,
            "max_components": cls.max_components,
            "max_categories": cls.max_categories,
            "qid_as_features": cls.qid_as_features,
            "null_policy": cls.null_policy,
            "wildcard_tokens": cls.wildcard_tokens,
            "ground_truth": cls.ground_truth,
            "seed": cls.seed,
            "max_comparisons": cls.max_comparisons,
            "use_cache": cls.use_cache,
            "force_recalculation": cls.force_recalculation,
        }
