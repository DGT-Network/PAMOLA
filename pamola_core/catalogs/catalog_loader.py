"""
pamola_core/catalogs/catalog_loader.py

Load and expose operations_catalog.yaml.
NFR-EP3-CORE-120: single source of truth for operation catalog.
NFR-EP3-CORE-124: Studio/Processing consume catalogs via CORE API.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

_CATALOG_DIR = Path(__file__).parent
_OPS_CATALOG_PATH = _CATALOG_DIR / "operations_catalog.yaml"


@lru_cache(maxsize=1)
def get_operations_catalog() -> List[Dict[str, Any]]:
    """
    Return all entries from operations_catalog.yaml.

    Returns
    -------
    list of dict
        Each entry has: name, category, module, version, description.

    Raises
    ------
    FileNotFoundError
        If operations_catalog.yaml is missing.
    """
    try:
        import yaml  # PyYAML — already in deps via other modules
    except ImportError:
        # Fallback: minimal YAML parser for simple list format
        return _load_catalog_fallback(_OPS_CATALOG_PATH)

    if not _OPS_CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"operations_catalog.yaml not found at {_OPS_CATALOG_PATH}. "
            "Run catalog generation or check package installation."
        )

    with _OPS_CATALOG_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    operations = data.get("operations", [])
    logger.debug(f"Loaded {len(operations)} operations from catalog.")
    return operations


def get_operation_entry(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up a single operation entry by class name.

    Parameters
    ----------
    name : str
        Operation class name (e.g. 'FullMaskingOperation').

    Returns
    -------
    dict or None
    """
    for entry in get_operations_catalog():
        if entry.get("name") == name:
            return entry
    return None


def _load_catalog_fallback(path: Path) -> List[Dict[str, Any]]:
    """Minimal YAML list parser for environments without PyYAML."""
    operations: List[Dict[str, Any]] = []
    if not path.exists():
        return operations

    current: Optional[Dict[str, Any]] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            if current:
                operations.append(current)
            current = {"name": stripped.split(":", 1)[1].strip()}
        elif current and ":" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition(":")
            current[key.strip()] = value.strip().strip('"')

    if current:
        operations.append(current)

    return operations
