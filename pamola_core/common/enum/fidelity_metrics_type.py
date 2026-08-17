from enum import Enum


class FidelityMetricsType(str, Enum):
    """
    Fidelity metrics supported by :class:`~pamola_core.metrics.operations.fidelity_ops.FidelityOperation`
    for comparing statistical similarity between original and anonymized datasets.

    Members:
    - KS: Kolmogorov-Smirnov test
    - KL: Kullback-Leibler divergence

    Every member listed here is wired in the ``FidelityOperation`` metric
    registry and is selectable. That correspondence is the point of this enum:
    it is a menu, and a menu must not list dishes the kitchen cannot cook.

    Removed in the 1.0 cleanup: ``JS`` (Jensen-Shannon divergence) and
    ``WASSERSTEIN`` (Wasserstein distance). Both were declared here and
    documented as "optional/expandable", but neither was ever registered in
    ``fidelity_ops``, so selecting them could not produce a metric. The only
    Wasserstein implementation in the tree lived in the orphaned
    ``metrics/quality/`` package, which nothing imported and which was removed
    at the same time.

    They can come back the moment there is an implementation plus tests behind
    them - adding a member here is the *last* step of that work, not the first.

    For JS the groundwork is closer than it looks: a working
    ``_jensen_shannon_divergence`` helper already exists in
    ``metrics/fidelity/statistical_fidelity.py``. It is not reachable from
    ``FidelityOperation`` today - that module is imported only by
    ``privacy_models/k_anonymity/calculation.py`` - so wiring JS means giving it
    a metric class in ``metrics/fidelity/distribution/`` alongside ``ks_test``
    and ``kl_divergence``, plus tests. That is a small, well-defined task, not a
    research one.

    A near-identical duplicate of this enum lived at
    ``common/enum/fidelity_metrics.py`` with zero references and was removed at
    the same time.
    """

    KS = "ks"  # Kolmogorov-Smirnov Test
    KL = "kl"  # Kullback-Leibler Divergence
