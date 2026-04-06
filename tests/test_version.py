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
    """Version must follow PEP 440."""
    version = _read_version()
    # PEP 440 regex
    pattern = r"^\d+\.\d+\.\d+(\.dev\d+|a\d+|b\d+|rc\d+)?$"
    assert re.match(pattern, version), f"Version {version} doesn't match PEP 440"


def test_version_in_changelog():
    """Current version must be documented in CHANGELOG.md."""
    version = _read_version()
    changelog = Path(__file__).parent.parent / "CHANGELOG.md"
    if changelog.exists():
        content = changelog.read_text()
        assert version in content, f"Version {version} not found in CHANGELOG.md"
