"""Verify version consistency across package artifacts."""
from pathlib import Path

import tomllib


def _read_version():
    """Read version from pyproject.toml, Python 3.11+ (tomllib)."""
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_version_format_pep440():
    """Version must be canonical PEP 440 and public - the same rule scripts/release_version.py applies
    (so ``1.0.0.post1`` is as acceptable here as it is to the release workflow)."""
    from packaging.version import Version

    version = _read_version()
    parsed = Version(version)
    assert str(parsed) == version, f"Version {version} is not in canonical PEP 440 form ({parsed})"
    assert parsed.local is None, "a local version (+...) cannot be published"


def test_version_in_changelog():
    """The current version must have its own heading '## [<version>]' in CHANGELOG.md."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_changelog", Path(__file__).parent.parent / "scripts" / "check_changelog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = _read_version()
    changelog = Path(__file__).parent.parent / "CHANGELOG.md"
    assert module.has_release_heading(changelog.read_text(encoding="utf-8"), version), (
        f"Version {version} has no '## [{version}]' heading in CHANGELOG.md")
