"""The CHANGELOG must have a section heading for exactly this version.

    python scripts/check_changelog.py CHANGELOG.md <version>

A substring search is not enough: ``1.0.0`` occurs inside ``1.0.0.dev4``. The heading has to be
``## [<version>]`` (optionally followed by `` - <date>``), with the version matched in full.
"""

import re
import sys
from pathlib import Path


def has_release_heading(text, version):
    pattern = r"^## \[" + re.escape(version) + r"\](?:\s|$)"
    return re.search(pattern, text, re.MULTILINE) is not None


def main(argv):
    path, version = Path(argv[1]), argv[2]
    if has_release_heading(path.read_text(encoding="utf-8"), version):
        print(f"CHANGELOG: section '## [{version}]' found")
        return 0
    print(f"ERROR: {path} has no section heading '## [{version}]'", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
