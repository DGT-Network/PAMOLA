"""
Guarded access to optional third-party dependencies.

Why this exists
---------------
Plotting libraries are heavy — `matplotlib`, `plotly` and their transitive
trees dominate the install — and most of what `pamola-core` does never draws
anything. They therefore belong behind an extra rather than in the base
install.

That is only possible if nothing imports them at module scope on the path
reachable from `import pamola_core`; otherwise the base install cannot even be
imported, let alone used. This module provides the one way to reach such a
dependency: import it at the point of use, and fail with a sentence that says
what to install rather than a bare `ModuleNotFoundError` naming a package the
user never asked for.

Usage
-----
    from pamola_core.utils.optional_deps import require_optional

    def save_figure(figure, path):
        go = require_optional("plotly.graph_objects", extra="viz")
        ...

The import cost is paid once: after the first call the module is in
`sys.modules` and `importlib.import_module` is a dictionary lookup.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from pamola_core.errors.exceptions import DependencyMissingError

__all__ = ["require_optional", "optional_import"]

# Which extra ships which distribution. Used to turn an import failure into an
# actionable instruction; keep in sync with [project.optional-dependencies].
_EXTRA_FOR_MODULE = {
    "matplotlib": "viz",
    "matplotlib_venn": "viz",
    "mpl_toolkits": "viz",
    "plotly": "viz",
    "seaborn": "viz",
    "kaleido": "viz",
    "wordcloud": "viz",
}


def _extra_for(module_name: str) -> str | None:
    return _EXTRA_FOR_MODULE.get(module_name.split(".")[0])


def require_optional(module_name: str, extra: str | None = None) -> ModuleType:
    """Import an optional dependency, or explain how to install it.

    Parameters
    ----------
    module_name
        Fully qualified module to import, e.g. ``"plotly.graph_objects"``.
    extra
        Extra that provides it. Inferred from the module name when omitted.

    Returns
    -------
    The imported module.

    Raises
    ------
    DependencyMissingError
        With a message naming the extra to install. Raised instead of letting
        `ModuleNotFoundError` escape, because the user installed
        `pamola-core` and should be told about `pamola-core[viz]` — not about
        a transitive package name they have never heard of.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        extra = extra or _extra_for(module_name)
        top = module_name.split(".")[0]
        if extra:
            reason = (
                f"{top} is not installed. It ships with the '{extra}' extra: "
                f"pip install 'pamola-core[{extra}]'"
            )
        else:
            reason = f"{top} is not installed: {exc}"
        raise DependencyMissingError(
            dependency_name=top,
            reason=reason,
        ) from exc


def optional_import(module_name: str) -> ModuleType | None:
    """Import an optional dependency, or return None if it is absent.

    For code that genuinely has a fallback — an `isinstance` check that should
    simply be False when the library is missing, say. Callers that cannot
    proceed without the dependency must use `require_optional` instead, so the
    user gets an instruction rather than a mysterious code path.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
