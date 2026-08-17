"""Guards for the test-isolation invariant established by TD-PC-12.

The suite must not write into the repository. That is enforced by autouse
fixtures in `tests/conftest.py`, and enforcement of this kind fails *silently*:
if it stops working, every test still passes and the artifacts simply come
back. The regression would be found by someone noticing dirt in `git status`,
which is how the debt arose in the first place.

So the invariant gets its own tests. They are cheap and they fail loudly.
"""

import os
from pathlib import Path


def test_cwd_is_not_the_repository():
    """Relative-path writes must land in `tmp_path`, not the working tree."""
    cwd = Path.cwd()
    assert not (cwd / "pyproject.toml").exists(), (
        f"cwd is the repository ({cwd}); isolate_working_tree is not in effect, "
        "so any test writing to a relative path will dirty the tree"
    )


def test_project_root_redirected_paths_are_set():
    """Overrides for paths that `chdir` cannot reach must be in place.

    `execution_log` resolves against `find_project_root()` and the FastText
    cache against the package directory, so neither follows the working
    directory. Both are redirected by environment variable instead.
    """
    for name in ("PAMOLA_EXECUTION_LOG_PATH", "PAMOLA_MODEL_DIR"):
        value = os.environ.get(name)
        assert value, f"{name} is not set; writes would reach the repository"
        assert Path(value).is_absolute(), f"{name} must be absolute, got {value!r}"


def test_resolved_project_root_is_outside_the_repository():
    """The subtlest leak of all, and the last one found.

    `get_project_root` falls back to markers found from the *package location*
    when the cwd yields none. Running from a source checkout, the package and
    the repository are the same directory — so resolution walks back into the
    working tree regardless of the cwd, and regardless of `PAMOLA_PROJECT_ROOT`
    being unset. `configs/`, `my_task/` and `output/` kept reappearing through
    this path after every cwd-relative write had been fixed.

    The fixture defeats it by making the cwd a marked project, so the
    package-location fallback is never reached.
    """
    from pamola_core.utils.paths import get_project_root

    repo_root = Path(__file__).resolve().parent.parent
    resolved = get_project_root().resolve()
    assert repo_root not in [resolved, *resolved.parents], (
        f"project root resolved into the repository ({resolved}); writes routed "
        "through get_project_root() will land in the working tree"
    )


def test_ambient_pamola_configuration_is_neutralised():
    """A developer's exported PAMOLA_* must not steer the suite."""
    assert "PAMOLA_PROJECT_ROOT" not in os.environ


def test_redirects_survive_the_deletion_fixture():
    """Ordering guard.

    `isolate_pamola_env` deletes every `PAMOLA_*` variable and
    `isolate_working_tree` sets two of them back. Both are autouse, and pytest
    does not order independent autouse fixtures — the second declares the first
    as a dependency to force it. If that dependency is ever dropped, the
    deletion may run last and the redirects vanish. This test is what notices.
    """
    model_dir = os.environ.get("PAMOLA_MODEL_DIR")
    assert model_dir is not None, (
        "PAMOLA_MODEL_DIR was deleted after being set — the fixture ordering "
        "dependency in isolate_working_tree has been lost"
    )
    # The redirect must point somewhere disposable, never into the package.
    package_root = Path(__file__).resolve().parent.parent / "pamola_core"
    assert package_root not in Path(model_dir).resolve().parents, (
        f"PAMOLA_MODEL_DIR points inside the package tree: {model_dir}"
    )
