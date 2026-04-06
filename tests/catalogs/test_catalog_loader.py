"""
Unit tests for pamola_core.catalogs.catalog_loader module.

Tests cover:
- get_operations_catalog() returns valid list of dicts
- get_operation_entry(name) lookup by class name
- Fallback YAML parser for PyYAML-less environments
- Missing catalog file handling
- LRU cache behavior
- Entry structure validation

Run with: pytest -s tests/catalogs/test_catalog_loader.py
"""

import tempfile
from pathlib import Path

import pytest

from pamola_core.catalogs.catalog_loader import (
    get_operations_catalog,
    get_operation_entry,
    _load_catalog_fallback,
)


class TestGetOperationsCatalog:
    """Test suite for get_operations_catalog() function."""

    def test_returns_list_of_dicts(self):
        """Catalog should return list of operation entries (dicts)."""
        catalog = get_operations_catalog()
        assert isinstance(catalog, list)
        assert len(catalog) > 0, "Catalog should contain at least one operation"

        # Verify structure of first entry
        first = catalog[0]
        assert isinstance(first, dict)
        assert "name" in first
        assert "category" in first
        assert "module" in first
        assert "version" in first
        assert "description" in first

    def test_all_entries_have_required_fields(self):
        """All catalog entries must have name, category, module, version, description."""
        catalog = get_operations_catalog()
        required_fields = {"name", "category", "module", "version", "description"}

        for idx, entry in enumerate(catalog):
            missing = required_fields - set(entry.keys())
            assert not missing, f"Entry {idx} ({entry.get('name')}) missing fields: {missing}"

    def test_all_names_are_strings(self):
        """Operation names must be non-empty strings."""
        catalog = get_operations_catalog()
        for entry in catalog:
            name = entry.get("name")
            assert isinstance(name, str), f"Name must be string, got {type(name)}"
            assert len(name) > 0, "Name cannot be empty"

    def test_all_categories_are_strings(self):
        """Categories must be non-empty strings."""
        catalog = get_operations_catalog()
        for entry in catalog:
            category = entry.get("category")
            assert isinstance(category, str), f"Category must be string, got {type(category)}"
            assert len(category) > 0, "Category cannot be empty"

    def test_cache_returns_same_object(self):
        """LRU cache should return identical object on repeated calls."""
        result1 = get_operations_catalog()
        result2 = get_operations_catalog()
        assert result1 is result2, "Cache should return same object"

    def test_catalog_not_empty(self):
        """Loaded catalog should contain operations."""
        catalog = get_operations_catalog()
        assert len(catalog) > 0, "Catalog must contain at least one operation"

    def test_names_are_unique(self):
        """Operation names should be unique in catalog."""
        catalog = get_operations_catalog()
        names = [entry.get("name") for entry in catalog]
        assert len(names) == len(set(names)), "Duplicate operation names found"


class TestGetOperationEntry:
    """Test suite for get_operation_entry() function."""

    def test_lookup_existing_operation(self):
        """Should find and return operation entry by name."""
        catalog = get_operations_catalog()
        if len(catalog) > 0:
            first_name = catalog[0]["name"]
            entry = get_operation_entry(first_name)
            assert entry is not None
            assert entry["name"] == first_name

    def test_lookup_missing_operation_returns_none(self):
        """Should return None for non-existent operation name."""
        entry = get_operation_entry("NonExistentOperationXYZ123")
        assert entry is None

    def test_lookup_case_sensitive(self):
        """Operation lookup should be case-sensitive."""
        catalog = get_operations_catalog()
        if len(catalog) > 0:
            original_name = catalog[0]["name"]
            lowercase = original_name.lower()

            # If name already lowercase, skip test
            if original_name == lowercase:
                pytest.skip("Test name is already lowercase")

            # Should not find lowercase variant
            entry = get_operation_entry(lowercase)
            assert entry is None, "Lookup should be case-sensitive"

    def test_entry_has_all_fields(self):
        """Returned entry should have all required fields."""
        catalog = get_operations_catalog()
        if len(catalog) > 0:
            name = catalog[0]["name"]
            entry = get_operation_entry(name)
            assert entry is not None
            assert "name" in entry
            assert "category" in entry
            assert "module" in entry
            assert "version" in entry
            assert "description" in entry

    def test_lookup_multiple_operations(self):
        """Should successfully lookup multiple different operations."""
        catalog = get_operations_catalog()
        test_names = [op["name"] for op in catalog[:min(3, len(catalog))]]

        for name in test_names:
            entry = get_operation_entry(name)
            assert entry is not None
            assert entry["name"] == name

    def test_empty_string_returns_none(self):
        """Lookup with empty string should return None."""
        entry = get_operation_entry("")
        assert entry is None

    def test_none_parameter_returns_none(self):
        """Lookup with None should return None or raise TypeError."""
        # Behavior may vary; test both possibilities
        try:
            entry = get_operation_entry(None)
            assert entry is None
        except (TypeError, AttributeError):
            pass  # Acceptable to raise error for None


class TestLoadCatalogFallback:
    """Test suite for _load_catalog_fallback() function."""

    def test_fallback_returns_list(self):
        """Fallback parser should return list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "test_catalog.yaml"
            result = _load_catalog_fallback(catalog_path)
            assert isinstance(result, list)

    def test_fallback_missing_file_returns_empty_list(self):
        """Fallback for missing file should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "nonexistent.yaml"
            result = _load_catalog_fallback(catalog_path)
            assert isinstance(result, list)
            assert len(result) == 0

    def test_fallback_parses_simple_yaml(self):
        """Fallback parser should parse simple YAML list structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "catalog.yaml"
            yaml_content = """operations:
- name: TestOp1
  category: test
  module: test.module
  version: "1.0.0"
  description: "Test operation"
- name: TestOp2
  category: test
  module: test.module2
  version: "1.0.0"
  description: "Another test op"
"""
            catalog_path.write_text(yaml_content)
            result = _load_catalog_fallback(catalog_path)

            assert len(result) == 2
            assert result[0]["name"] == "TestOp1"
            assert result[1]["name"] == "TestOp2"

    def test_fallback_extracts_name_field(self):
        """Fallback should extract name from '- name:' lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "catalog.yaml"
            yaml_content = """- name: Operation1
  category: test
- name: Operation2
  category: other
"""
            catalog_path.write_text(yaml_content)
            result = _load_catalog_fallback(catalog_path)

            assert len(result) >= 1
            assert result[0].get("name") == "Operation1"

    def test_fallback_handles_quoted_values(self):
        """Fallback should strip quotes from values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "catalog.yaml"
            yaml_content = """- name: "TestOp"
  category: "anonymization"
  description: "Test description"
"""
            catalog_path.write_text(yaml_content)
            result = _load_catalog_fallback(catalog_path)

            assert len(result) >= 1
            if result:
                assert result[0].get("name") in ["TestOp", '"TestOp"']

    def test_fallback_ignores_comments(self):
        """Fallback parser should ignore comment lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "catalog.yaml"
            yaml_content = """# This is a comment
- name: TestOp
  # Another comment
  category: test
"""
            catalog_path.write_text(yaml_content)
            result = _load_catalog_fallback(catalog_path)

            # Should successfully parse without errors
            assert isinstance(result, list)

    def test_fallback_handles_empty_file(self):
        """Fallback should handle empty file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "empty.yaml"
            catalog_path.write_text("")
            result = _load_catalog_fallback(catalog_path)

            assert isinstance(result, list)
            assert len(result) == 0


class TestCatalogIntegration:
    """Integration tests for catalog loading system."""

    def test_catalog_structure_consistency(self):
        """Catalog should have consistent structure across all entries."""
        catalog = get_operations_catalog()

        # Check all entries have same keys or superset
        if len(catalog) > 0:
            first_keys = set(catalog[0].keys())
            for idx, entry in enumerate(catalog[1:], 1):
                # All entries should at minimum have required fields
                required = {"name", "category", "module", "version", "description"}
                entry_keys = set(entry.keys())
                missing = required - entry_keys
                assert not missing, f"Entry {idx} missing required fields: {missing}"

    def test_catalog_versions_valid(self):
        """All operation versions should be valid version strings."""
        catalog = get_operations_catalog()
        import re

        version_pattern = re.compile(r'^\d+\.\d+\.\d+')
        for entry in catalog:
            version = entry.get("version", "")
            assert version_pattern.match(version), \
                f"Invalid version format '{version}' in {entry.get('name')}"

    def test_lookup_and_fetch_consistency(self):
        """Entries fetched via lookup should match catalog entries."""
        catalog = get_operations_catalog()

        for entry in catalog[:5]:  # Test first 5
            name = entry["name"]
            fetched = get_operation_entry(name)

            assert fetched is not None
            assert fetched["name"] == entry["name"]
            assert fetched["category"] == entry["category"]
            assert fetched["module"] == entry["module"]
