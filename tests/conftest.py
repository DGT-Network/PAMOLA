"""
Root pytest configuration for the pamola-core test suite.

Purpose
-------
Make the suite hermetic with respect to the developer's environment.

Several PAMOLA components resolve configuration from ambient state by design —
`pamola_core.utils.paths.get_project_root` gives `PAMOLA_PROJECT_ROOT` top
priority, which is correct product behaviour. It is *not* correct for a test
that builds its own configuration under `tmp_path`: such a test must describe
its whole world, or it reports the developer's machine rather than the code.

Before this file existed, a machine with `PAMOLA_PROJECT_ROOT` set (for example
one left over from the predecessor HHR project) failed five tests in
`tests/utils/tasks/` with assertions comparing a `tmp_path` against an unrelated
absolute path. CI never saw it, because CI has no such variable — which is
exactly what makes this class of failure expensive: it only ever appears on a
developer's machine, and it looks like a code defect.

Tests that need to exercise the ambient-configuration behaviour set the
variables explicitly via `monkeypatch.setenv`; the autouse fixture below runs
first, so an explicit `setenv` inside a test always wins.
"""

import os

import pytest

# --- Matplotlib must never reach a GUI backend during tests (TD-PC-14) -------
#
# Symptom: `TestDataWriter` failed a *different* test on each run, with
# `_tkinter.TclError`. Cause: on a developer machine matplotlib auto-selects
# `TkAgg`, and Tk objects are not safe across the threads the suite uses.
#
# The library already ships `matplotlib_agg_context()`
# (`pamola_core/utils/vis_helpers/context.py:175`), but it deliberately
# *restores* the previous backend on exit — correct for a library that must not
# flip the host application's backend, and precisely wrong here: every guarded
# block hands the process back to Tk.
#
# Setting the backend through the environment fixes both paths at once: the
# ambient backend becomes `Agg`, so the context manager sees no difference,
# never switches, and therefore never restores. This must run before the first
# `import matplotlib.pyplot` anywhere, which the root conftest guarantees.
#
# `setdefault`, not assignment: a developer debugging a plot can still export
# MPLBACKEND themselves and have it respected.
os.environ.setdefault("MPLBACKEND", "Agg")

# Environment variables that steer PAMOLA path/config resolution. Any variable
# added here is neutralised for every test unless the test sets it itself.
#
# Keep this list in sync with the resolution order documented in
# pamola_core/utils/paths.py and pamola_core/utils/tasks/project_config_loader.py.
PAMOLA_ENV_VARS = (
    "PAMOLA_PROJECT_ROOT",
    "PAMOLA_DATA_REPOSITORY",
    "PAMOLA_CONFIG_PATH",
    "PAMOLA_LOG_LEVEL",
    "PAMOLA_MASTER_KEY",
    # Redirected into a temporary workspace by `isolate_working_tree` below. Listed here so an
    # ambient value from the developer's shell is neutralised *first* — the
    # fixture sets them again immediately afterwards.
    "PAMOLA_EXECUTION_LOG_PATH",
    "PAMOLA_MODEL_DIR",
)


@pytest.fixture(autouse=True)
def isolate_pamola_env(monkeypatch):
    """Remove ambient PAMOLA configuration for the duration of each test.

    Autouse, so it applies to the whole suite without opt-in. `monkeypatch`
    restores the previous values afterwards, so the developer's shell is not
    modified beyond the test session.
    """
    for name in PAMOLA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def isolate_working_tree(tmp_path_factory, monkeypatch, isolate_pamola_env):
    """Keep every test's file output out of the repository (TD-PC-12).

    `isolate_pamola_env` is requested as an argument, not merely relied upon.
    Both fixtures are autouse, and pytest does not order independent autouse
    fixtures — so without this dependency the deletion could run *after* the
    assignments below and silently undo them. The suite would stay green while
    the isolation quietly stopped working, which is exactly the class of defect
    this file exists to prevent.

    The suite used to write ~43 files into the working tree at fixed relative
    paths — `test_task_dir/`, `temp_task_dir/`, `test_vis_dir/`, `my_task/`,
    `output/`, a root `config.json`, `pamola_processing.log`. `.gitignore`
    hid them; it did not stop them.

    That is worse than untidy. Artifacts left in the tree make later runs
    depend on earlier ones, so a test can pass because a previous run left the
    right file behind — and it blocks `pytest -n auto`, since parallel workers
    would collide on the same fixed paths.

    Four mechanisms, four fixes, applied here rather than in 21 test files:

    1. **Relative paths** — `chdir` into a private workspace. This covers every
       cwd-relative write at once, including ones nobody has enumerated, and
       keeps future tests honest by construction.
    2. **`find_project_root()`** — `pamola_core.utils.tasks.execution_log`
       resolves against the project root, so `chdir` cannot reach it. Its
       documented override points into the workspace instead.
    3. **Package-relative paths** — `utils/nlp/language.py` cached a 125 MB
       FastText model *inside the installed package*. `PAMOLA_MODEL_DIR`
       redirects it. (The same call also created an empty `resources/models/`
       unconditionally; that is fixed in the source.)
    4. **Root resolution falling back to the package location** — the subtlest
       of the four, and reachable by neither `chdir` nor an environment
       variable. See the comment on the marker file below.

    A fresh workspace per test, not a session-wide one, so tests cannot leak
    state into each other through the filesystem either.
    """
    # Make the workspace a real, minimal PAMOLA project.
    #
    # `pamola_core.utils.paths.get_project_root` resolves in four steps: the
    # environment variable, markers found upward from the cwd, markers found
    # upward from the *package location*, and finally the package root. Step
    # three is the one that defeats `chdir`: when the library runs from a
    # source checkout, the package and the repository are the same directory,
    # so resolution walks straight back into the working tree no matter where
    # the cwd points. That is how `configs/`, `my_task/` and `output/` kept
    # reappearing after the cwd-relative writes were fixed.
    #
    # Writing the marker makes step *two* succeed at the cwd, so step three is
    # never reached. This uses the documented discovery mechanism rather than
    # overriding it: `PAMOLA_PROJECT_ROOT` stays deleted, so tests that exercise
    # ambient-configuration behaviour are unaffected (see TD-PC-11, which is
    # precisely why that variable must not be repurposed here).
    #
    # `configs/prj_config.json` satisfies both resolvers — `_find_root` in
    # paths.py and `find_project_root` in project_config_loader.py.
    # The workspace is a directory of its own, deliberately *not* inside
    # `tmp_path`. `tmp_path` is the test's own sandbox: tests legitimately
    # assert on its exact contents (`tests/utils/io_helpers/` counts entries in
    # it), so anything this fixture leaves there is a foreign object in someone
    # else's fixture. Nor can the marker live at `tmp_path` itself — root
    # resolution walks *upward*, so it would shadow projects that the
    # `find_project_root` tests build beneath it.
    #
    # `mktemp` gives a fresh directory per test, outside both.
    workspace = tmp_path_factory.mktemp("pamola_ws")
    configs_dir = workspace / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "prj_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PAMOLA_EXECUTION_LOG_PATH", str(workspace / "execution_log.json"))
    monkeypatch.setenv("PAMOLA_MODEL_DIR", str(workspace / "models"))