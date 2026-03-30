"""Verify version consistency across package artifacts."""
import re
import tomllib
from pathlib import Path


def test_version_format_pep440():
    """Version must follow PEP 440."""
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    # PEP 440 regex
    pattern = r"^\d+\.\d+\.\d+(a\d+|b\d+|rc\d+)?$"
    assert re.match(pattern, version), f"Version {version} doesn't match PEP 440"


def test_version_in_changelog():
    """Current version must be documented in CHANGELOG.md."""
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    changelog = Path(__file__).parent.parent / "CHANGELOG.md"
    if changelog.exists():
        content = changelog.read_text()
        assert version in content, f"Version {version} not found in CHANGELOG.md"
