"""Verify version consistency across package artifacts."""
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def _read_version():
    """Read version from pyproject.toml, supporting Python 3.10+."""
    if tomllib is None:
        # Fallback: parse version line directly
        text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
        return match.group(1) if match else "0.0.0"
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
