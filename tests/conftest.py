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

import pytest

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
