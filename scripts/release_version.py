"""Classify the package version for the release workflow (PEP 440, via `packaging`).

    python scripts/release_version.py <ref_type> <ref_name>

Reads the version from pyproject.toml, checks it against the tag, and writes
`version=` and `prerelease=` to $GITHUB_OUTPUT (or stdout). Exits non-zero for a
version that must not be published to a public index.

The classification is `packaging.version.Version.is_prerelease`, not a regex:
`1.0.0rc1.post1` and `1.0.0a1.post2` are pre-releases and must never be routed
to the production index.
"""

import os
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version


def classify(version_string):
    """Return ``(canonical_version, is_prerelease)`` or raise ``ValueError``."""
    try:
        version = Version(version_string)
    except InvalidVersion as exc:
        raise ValueError(f"{version_string!r} is not a valid PEP 440 version") from exc
    if version.local is not None:
        raise ValueError(
            f"{version_string!r} has a local version segment ('+...'); "
            "public indexes reject local versions"
        )
    return str(version), version.is_prerelease


def check_tag(tag_name, package_version):
    """The tag (without the leading ``v``) must be the package version."""
    tag_version = tag_name[1:] if tag_name.startswith("v") else tag_name
    if tag_version != package_version:
        raise ValueError(
            f"tag {tag_name} does not match pyproject.toml version {package_version}"
        )


def main(argv):
    import tomllib

    ref_type, ref_name = argv[1], argv[2]
    with open("pyproject.toml", "rb") as handle:
        package_version = tomllib.load(handle)["project"]["version"]
    try:
        canonical, prerelease = classify(package_version)
        if canonical != package_version:
            raise ValueError(
                f"pyproject.toml version {package_version!r} is not canonical "
                f"(expected {canonical!r}); the built filenames would differ"
            )
        if ref_type == "tag":
            check_tag(ref_name, package_version)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    lines = [f"version={package_version}", f"prerelease={str(prerelease).lower()}"]
    print("\n".join(lines))
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
