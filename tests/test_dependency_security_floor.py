"""
SEC-06 / DEP-01: declared runtime dependencies keep the security fixes.

``pyarrow`` is reachable from untrusted input through ``pamola_core.io.parquet``;
releases below 17.0.0 carry CVE-2024-52338 (deserialization of untrusted
data) and releases 15.0.0 - 23.0.0 carry CVE-2026-25087 (use-after-free when
reading IPC files). The declared pin must not slide back under 23.0.1, and the
environment under test must satisfy it.
"""

import re
from pathlib import Path

import pytest
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PYARROW_FLOOR = Version("23.0.1")


def _runtime_dependencies():
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert block, "dependencies list not found in pyproject.toml"
    lines = (ln.split("#", 1)[0].strip() for ln in block.group(1).splitlines())
    return [ln.strip(",").strip('"') for ln in lines if ln.startswith('"')]


def test_declared_pyarrow_pin_is_at_or_above_the_fixed_release():
    pins = [d for d in _runtime_dependencies() if d.lower().startswith("pyarrow")]
    assert len(pins) == 1, pins
    match = re.fullmatch(r"pyarrow==([0-9][^\s;]*)", pins[0])
    assert match, f"pyarrow must stay exactly pinned, got {pins[0]!r}"
    assert Version(match.group(1)) >= PYARROW_FLOOR


def test_installed_pyarrow_is_at_or_above_the_fixed_release():
    pyarrow = pytest.importorskip("pyarrow")
    assert Version(pyarrow.__version__) >= PYARROW_FLOOR


def test_parquet_round_trip_with_the_pinned_pyarrow(tmp_path):
    import pandas as pd

    from pamola_core.io.parquet import read_parquet

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})
    path = tmp_path / "t.parquet"
    df.to_parquet(path, engine="pyarrow")
    # pamola_core.io defaults to the nullable dtype backend (Int64, string).
    pd.testing.assert_frame_equal(read_parquet(path), df, check_dtype=False)


def test_pydantic_email_is_not_a_declared_dependency():
    """DEP-01: nothing imports pydantic / email_validator; do not reintroduce it."""
    deps = [d.lower() for d in _runtime_dependencies()]
    assert not [d for d in deps if d.startswith(("pydantic", "email-validator"))]
