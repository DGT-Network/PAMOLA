"""Fail if TestPyPI already serves different bytes for a file we are about to upload.

    python scripts/check_testpypi_digests.py <dist-dir> <package> <version>

`skip-existing: true` lets an upload "succeed" while the index keeps an OLDER file
with the same name (manual runs may rehearse different refs with one version).
Skipping is only acceptable for byte-identical files, so every file that already
exists on TestPyPI must have the same sha256 as the file in <dist-dir>.
"""

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def local_digests(dist_dir):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(dist_dir).iterdir())
        if path.is_file()
    }


def mismatches(local, release_json):
    """Names present on the index whose sha256 differs from the local file."""
    remote = {
        entry["filename"]: entry["digests"]["sha256"]
        for entry in (release_json or {}).get("urls", [])
    }
    return sorted(
        name for name, digest in local.items() if name in remote and remote[name] != digest
    )


def fetch_release(package, version):
    url = f"https://test.pypi.org/pypi/{package}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None  # nothing published yet
        raise


def main(argv):
    dist_dir, package, version = argv[1:4]
    local = local_digests(dist_dir)
    bad = mismatches(local, fetch_release(package, version))
    if bad:
        print(
            "ERROR: TestPyPI already serves different bytes for: "
            + ", ".join(bad)
            + "\nUse a new version for this rehearsal instead of re-uploading.",
            file=sys.stderr,
        )
        return 1
    print("TestPyPI: no conflicting files; identical existing files will be skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
