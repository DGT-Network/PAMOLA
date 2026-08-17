"""
Guard: every public operation declares whether it can be batch-processed.

Why this exists
---------------
Five operations - three suppression, two splitting - inherited an abstract
``process_batch`` that raised ``FeatureNotImplementedError``. The limitation
was real, but the only way to discover it was to call the method and catch the
exception. For a library meant to be embedded, "try it and see if it explodes"
is not an interface.

``supports_batch`` turns that into a declared, machine-readable property, and
these tests keep the declaration honest: a class that says ``True`` must
actually implement ``process_batch``, and one that says ``False`` must not
silently start working (which would mean the flag is now a lie in the other
direction).
"""

from __future__ import annotations

import pytest

import pamola_core as P
from pamola_core.utils.ops.op_base import BaseOperation

OPERATION_NAMES = sorted(n for n in P.__all__ if n.endswith("Operation"))

# Operations that cannot be driven as an in-memory DataFrame -> DataFrame
# transform. Kept explicit rather than derived, so that flipping a flag without
# implementing anything fails here loudly.
KNOWN_NON_BATCH = {
    # suppression: statistical criteria / row removal are not batch-local
    "AttributeSuppressionOperation",
    "CellSuppressionOperation",
    "RecordSuppressionOperation",
    # splitting: one dataset in, several out - no single-frame output form
    "SplitByIDValuesOperation",
    "SplitFieldsOperation",
}


def test_operation_inventory_is_not_empty():
    assert len(OPERATION_NAMES) > 30, OPERATION_NAMES


@pytest.mark.parametrize("name", OPERATION_NAMES)
def test_every_operation_declares_supports_batch(name: str):
    """No operation may leave the question unanswered."""
    cls = getattr(P, name)
    assert hasattr(cls, "supports_batch"), (
        f"{name} does not declare supports_batch. Every operation must, so that "
        f"a caller can branch on the flag instead of catching an exception. "
        f"The default lives on BaseOperation."
    )
    assert isinstance(cls.supports_batch, bool), (
        f"{name}.supports_batch must be a bool, got {type(cls.supports_batch)}"
    )


@pytest.mark.parametrize("name", OPERATION_NAMES)
def test_flag_matches_implementation(name: str):
    """A True flag must be backed by a real process_batch."""
    cls = getattr(P, name)
    if cls is BaseOperation or not cls.supports_batch:
        return

    assert hasattr(cls, "process_batch"), (
        f"{name}.supports_batch is True but the class has no process_batch at all"
    )

    # Walk the MRO: the implementation must come from somewhere other than the
    # abstract base that raises.
    owner = next(
        (k.__name__ for k in cls.__mro__ if "process_batch" in vars(k)), None
    )
    assert owner not in (None, "AnonymizationOperation", "TransformationOperation"), (
        f"{name}.supports_batch is True, but process_batch resolves to "
        f"{owner}, which is the abstract version that raises "
        f"FeatureNotImplementedError. Either implement process_batch or set "
        f"supports_batch = False."
    )


@pytest.mark.parametrize("name", sorted(KNOWN_NON_BATCH))
def test_known_non_batch_operations_still_declare_false(name: str):
    """Guards against the flag being flipped without an implementation.

    If an operation here grows a real ``process_batch``, remove it from
    ``KNOWN_NON_BATCH`` in the same change - that is the intended workflow, and
    flipping False -> True is backwards compatible.
    """
    cls = getattr(P, name)
    assert cls.supports_batch is False, (
        f"{name} now claims supports_batch=True. If that is intentional and it "
        f"really implements process_batch, drop it from KNOWN_NON_BATCH here."
    )
