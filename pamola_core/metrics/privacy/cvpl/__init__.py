"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Package:  pamola_core.metrics.privacy.cvpl
License:  BSD 3-Clause

CVPL (Cluster-Vector-Projection Linkage): an explicit, reproducible linkage
exposure measurement ``B_lambda -> phi -> PCA -> cosine -> tau`` between a source
dataset and its protected release. Independent of the legacy
``pamola_core.attacks.linkage_attack`` (TF-IDF / TruncatedSVD), which measures
something else and is left untouched.

Provenance: the method and its default parameters (variance 0.90, 3-50 components,
tau 0.9, joint fit, wildcard-aware blocking) follow DGT-Network/CVPL @bb25aa9 (MIT,
A. Neuman, V. Khvatov); this implementation was written for CORE and reuses none
of its source files.

Public entry points: :class:`CVPLConfig`, :func:`run_cvpl`, and the metrics
operation ``pamola_core.metrics.CVPLLinkageOperation``.
"""

from .config import (
    ALGORITHM_ID,
    ALGORITHM_VERSION,
    CVPLConfig,
    GroundTruthSpec,
    QIDSpec,
)
from .engine import CVPLResult, run_cvpl
from .projection import CVPLProjection

__all__ = [
    "ALGORITHM_ID",
    "ALGORITHM_VERSION",
    "CVPLConfig",
    "CVPLProjection",
    "CVPLResult",
    "GroundTruthSpec",
    "QIDSpec",
    "run_cvpl",
]
