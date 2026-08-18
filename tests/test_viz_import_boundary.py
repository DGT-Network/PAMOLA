"""Guards for the visualisation import boundary.

`import pamola_core` must not pull in a plotting backend. Those libraries and
their transitive trees dominate the install, and most of what this package does
never draws anything — so they ship behind the `viz` extra, which is only
possible while nothing imports them at module scope on the reachable path.

The boundary is easy to breach by accident: one convenience import at the top of
one module, anywhere in a chain several levels deep, restores the dependency for
everyone. It moved once already through
`profiling/analyzers/categorical.py → utils/visualization.py →
utils/vis_helpers/__init__.py`, none of which looks like a plotting module from
the outside. So it is asserted rather than assumed.
"""

import subprocess
import sys

import pytest

VIZ_BACKENDS = ("matplotlib", "plotly", "seaborn", "wordcloud", "matplotlib_venn", "kaleido")


def _modules_loaded_by(statement: str) -> set:
    """Report which viz backends `statement` leaves in sys.modules.

    Run in a fresh interpreter rather than by clearing `sys.modules` in-process:
    popping a name does not unload its submodules or undo the bindings other
    modules already hold, so an in-process check can report a clean boundary
    that a real installation would not have.
    """
    code = (
        "import sys\n"
        f"{statement}\n"
        f"print(','.join(sorted(m for m in {VIZ_BACKENDS!r} if m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    out = result.stdout.strip()
    return set(out.split(",")) if out else set()


def test_import_pamola_core_loads_no_viz_backend():
    loaded = _modules_loaded_by("import pamola_core")
    assert loaded == set(), (
        f"`import pamola_core` pulled in {sorted(loaded)}. These ship with the "
        "`viz` extra, so a base install would fail to import. Move the offending "
        "import into the function that uses it, via "
        "pamola_core.utils.optional_deps."
    )


@pytest.mark.parametrize(
    "statement",
    [
        "from pamola_core.utils import io",
        "from pamola_core.utils import visualization",
        "from pamola_core.utils.vis_helpers import base",
        "import pamola_core.profiling",
        "import pamola_core.anonymization",
    ],
)
def test_core_subpackages_load_no_viz_backend(statement):
    """The entry points that reached a backend before, checked individually.

    `utils.visualization` and `vis_helpers.base` are named deliberately: they
    are about drawing, and it is tempting to consider a plotting import there
    harmless. It is not — importing either is what dragged the whole stack in.
    """
    loaded = _modules_loaded_by(statement)
    assert loaded == set(), f"`{statement}` pulled in {sorted(loaded)}"


def test_missing_optional_dependency_names_the_extra():
    """A missing backend must produce an instruction, not a bare ImportError."""
    from pamola_core.errors.exceptions import DependencyMissingError
    from pamola_core.utils.optional_deps import require_optional

    with pytest.raises(DependencyMissingError) as excinfo:
        require_optional("matplotlib_not_installed_xyz", extra="viz")

    message = str(excinfo.value)
    assert "pamola-core[viz]" in message, (
        f"error must tell the user what to install, got: {message}"
    )


def test_optional_import_returns_none_for_absent_module():
    """The fallback form must not raise — some call sites have a real fallback."""
    from pamola_core.utils.optional_deps import optional_import

    assert optional_import("definitely_not_installed_xyz") is None
