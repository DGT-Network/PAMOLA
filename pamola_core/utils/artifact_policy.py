"""
Artifact classes.

Everything an operation writes belongs to exactly one :class:`ArtifactClass`:

``DATA``
    Operation outputs (the transformed dataset and its partitions).
``PROTECTED_MAPPING``
    Reversible maps (original <-> substitute). They contain source values, so they are
    written encrypted only (see :func:`require_encryption_key`); without a key the write
    is refused with :class:`ArtifactPolicyError`.
``AGGREGATE_METRIC``
    Numbers and distributions without source values.
``DIAGNOSTIC``
    Files for the operator that may contain source values (profiling value dictionaries,
    examples, ``top_values``, charts labelled with data values). Registered with
    ``contains_source_values=True`` and marked ``not_for_release``.
``EVIDENCE``
    Artifacts intended for publication or audit. Never contain source values.

Rule: nothing that contains source values is classified ``AGGREGATE_METRIC`` or
``EVIDENCE`` (:func:`check_class_and_content`).

The class is derived from the artifact's legacy ``category`` unless given explicitly
(:func:`class_for_category`).
"""

import enum
from typing import Optional

from pamola_core.errors.exceptions import ValidationError


class ArtifactClass(str, enum.Enum):
    """Classification of everything an operation writes."""

    DATA = "data"
    PROTECTED_MAPPING = "protected_mapping"
    AGGREGATE_METRIC = "aggregate_metric"
    DIAGNOSTIC = "diagnostic"
    EVIDENCE = "evidence"


#: Classes allowed to carry source values.
SOURCE_VALUE_CLASSES = frozenset(
    {ArtifactClass.DATA, ArtifactClass.PROTECTED_MAPPING, ArtifactClass.DIAGNOSTIC}
)

# Legacy ``category`` strings -> class. Unknown categories are DIAGNOSTIC (conservative).
_CATEGORY_CLASS = {
    "output": ArtifactClass.DATA,
    "metrics": ArtifactClass.AGGREGATE_METRIC,
    "metric": ArtifactClass.AGGREGATE_METRIC,
    "dictionary": ArtifactClass.DIAGNOSTIC,
    "visualization": ArtifactClass.DIAGNOSTIC,
    "mapping": ArtifactClass.PROTECTED_MAPPING,
    "evidence": ArtifactClass.EVIDENCE,
}


def class_for_category(category: Optional[str]) -> ArtifactClass:
    """Derive the artifact class from the legacy ``category`` string."""
    if category is None:
        return ArtifactClass.DATA
    return _CATEGORY_CLASS.get(str(category).lower(), ArtifactClass.DIAGNOSTIC)


class ArtifactPolicyError(ValidationError):
    """Raised when an artifact would break the artifact rules."""


def check_class_and_content(
    artifact_class: ArtifactClass, contains_source_values: bool
) -> ArtifactClass:
    """Return ``artifact_class`` or raise if source values would sit in a metric/evidence."""
    artifact_class = ArtifactClass(artifact_class)
    if contains_source_values and artifact_class not in SOURCE_VALUE_CLASSES:
        raise ArtifactPolicyError(
            f"an '{artifact_class.value}' artifact may not contain source values; "
            "classify it as 'diagnostic' (not for release)"
        )
    return artifact_class


def require_encryption_key(encryption_key: Optional[str], what: str) -> str:
    """A ``PROTECTED_MAPPING`` is only written encrypted: refuse without a key."""
    if not encryption_key:
        raise ArtifactPolicyError(
            f"{what} holds original values and is only written encrypted. Provide the task "
            "encryption key (encryption_key with use_encryption=True) or disable saving the mapping."
        )
    return encryption_key
