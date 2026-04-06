"""
Path resolution helpers for PAMOLA.

Avoids relying on the current working directory for core path discovery.
"""

import os
from pathlib import Path
from typing import Optional


_ROOT_MARKERS = (
    "pyproject.toml",
    ".git",
)


def _find_root(start: Path) -> Optional[Path]:
    for candidate in [start, *start.parents]:
        for marker in _ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate
        if (candidate / "configs" / "prj_config.json").exists():
            return candidate
        if (candidate / "configs" / "prj_config.yaml").exists():
            return candidate
    return None


def get_package_root() -> Path:
    """Return the installed package root directory (pamola_core)."""
    return Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """
    Resolve the project root without assuming the working directory.

    Resolution order:
    1. PAMOLA_PROJECT_ROOT environment variable
    2. Root markers found from current working directory
    3. Root markers found from package location
    4. Package root
    """
    env_root = os.environ.get("PAMOLA_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cwd_root = _find_root(Path.cwd().resolve())
    if cwd_root:
        return cwd_root

    pkg_root = get_package_root()
    pkg_root_candidate = _find_root(pkg_root)
    if pkg_root_candidate:
        return pkg_root_candidate

    return pkg_root
