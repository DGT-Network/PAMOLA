"""
Guard: the public API and the declared coverage scope must not drift apart.

Why this exists
---------------
The project maintains its public surface in two hand-written places:

* ``pamola_core/__init__.py`` ``__all__`` — what users may import;
* ``.coveragerc`` ``[run] include`` — the "Public API only" coverage scope
  declared per SRS 4.1.11, and the basis of the project's reported coverage
  figure.

Nothing checked that the two agreed. The failure mode is quiet and flatters the
metric: export a new operation, forget the ``.coveragerc`` entry, and it becomes
public, untested and invisible to the coverage number — which may even *rise*,
because the new untested lines are outside the measured scope.

This test resolves every exported symbol to the module that defines it and
asserts that module is inside the declared coverage scope.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import re

import pytest

import pamola_core

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COVERAGERC = REPO_ROOT / ".coveragerc"

# Exported names that intentionally have no owning module inside the package:
# dunders and re-exported constants.
NOT_MODULE_BACKED = {"__version__"}


def _declared_scope() -> set[str]:
    """Parse ``[run] include`` from .coveragerc into a set of POSIX paths."""
    text = COVERAGERC.read_text(encoding="utf-8", errors="replace")

    # Take everything from `include =` up to the next section or top-level key.
    match = re.search(r"^include\s*=\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert match, ".coveragerc has no '[run] include' block"

    paths = set()
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.add(line.replace("\\", "/"))
    return paths


def _defining_module(name: str) -> str | None:
    """Return the module path that defines the exported symbol, or None."""
    obj = getattr(pamola_core, name)
    module_name = getattr(obj, "__module__", None)
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    file = inspect.getsourcefile(module) or getattr(module, "__file__", None)
    if file is None:
        return None
    return pathlib.Path(file).resolve().relative_to(REPO_ROOT).as_posix()


def test_coveragerc_is_readable_and_non_empty():
    scope = _declared_scope()
    assert len(scope) > 10, f"suspiciously small coverage scope: {len(scope)}"
    for path in scope:
        assert (REPO_ROOT / path).is_file(), f".coveragerc lists a missing file: {path}"


@pytest.mark.parametrize(
    "name", sorted(n for n in pamola_core.__all__ if n not in NOT_MODULE_BACKED)
)
def test_public_symbol_is_inside_declared_coverage_scope(name: str):
    """Every name in __all__ must be defined in a file listed in .coveragerc."""
    module_path = _defining_module(name)
    assert module_path is not None, (
        f"{name!r} is exported from pamola_core but its defining module could "
        f"not be resolved"
    )

    scope = _declared_scope()
    assert module_path in scope, (
        f"{name!r} is exported from pamola_core.__all__ but its module "
        f"{module_path!r} is not in the .coveragerc '[run] include' scope.\n"
        f"Public API and declared coverage scope have drifted: either add the "
        f"module to .coveragerc, or stop exporting the symbol."
    )
