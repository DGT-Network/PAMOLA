"""
pamola_core/catalogs/__init__.py

Public API for catalog access.
NFR-EP3-CORE-120/121/124: single source of truth, consumed via CORE API.
"""

from pamola_core.catalogs.catalog_loader import get_operations_catalog, get_operation_entry

__all__ = [
    "get_operations_catalog",
    "get_operation_entry",
]
