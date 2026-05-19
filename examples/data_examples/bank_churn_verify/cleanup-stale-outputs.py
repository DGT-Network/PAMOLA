"""Cleanup stale timestamped outputs in bank_churn_verify/.

Keep only the latest file per (directory, base_name) group.
Group key = path with timestamp `_YYYYMMDD_HHMMSS` stripped from filename.

Usage:
    python cleanup-stale-outputs.py           # dry run — print what would be deleted
    python cleanup-stale-outputs.py --apply   # actually delete
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
TS_RE = re.compile(r"_(\d{8}_\d{6})(?=\.[a-z]+$)")

KEEP_FILES = {"verification_results.json", "cleanup-stale-outputs.py", "config.json"}


def main(apply: bool) -> int:
    # Group files by (parent_dir, name_without_timestamp)
    groups: dict[tuple[Path, str], list[tuple[str, Path]]] = defaultdict(list)
    untouched = 0

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name in KEEP_FILES:
            untouched += 1
            continue
        m = TS_RE.search(path.name)
        if not m:
            untouched += 1
            continue
        ts = m.group(1)
        base = TS_RE.sub("", path.name)
        groups[(path.parent, base)].append((ts, path))

    to_delete: list[Path] = []
    to_keep: list[Path] = []
    for (parent, base), entries in groups.items():
        entries.sort(key=lambda x: x[0], reverse=True)  # latest first
        to_keep.append(entries[0][1])
        for _, p in entries[1:]:
            to_delete.append(p)

    print(f"Total groups: {len(groups)}")
    print(f"Files to KEEP (latest per group): {len(to_keep)}")
    print(f"Files to DELETE (stale): {len(to_delete)}")
    print(f"Files untouched (no timestamp / kept always): {untouched}")

    if not to_delete:
        print("Nothing to delete.")
        return 0

    if apply:
        for p in to_delete:
            p.unlink()
        print(f"\nDeleted {len(to_delete)} files.")
    else:
        print("\n[dry-run] First 10 files that would be deleted:")
        for p in to_delete[:10]:
            print(f"  - {p.relative_to(ROOT)}")
        print("\nRun with --apply to delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv))
