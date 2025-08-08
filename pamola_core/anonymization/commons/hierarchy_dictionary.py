"""
AMOLA.CORE - Privacy-Preserving AI Data Processors
------------------------------------------------------------
Module:        Hierarchy Dictionary Management for Categorical Generalization
Package:       pamola_core.anonymization.commons
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01
Updated:       2025-01-23
License:       BSD 3-Clause

Description:
  This module provides a unified interface for loading and managing hierarchical
  dictionaries used in categorical generalization operations. It supports both
  JSON and CSV formats with multi-level hierarchies, aliases, and metadata.

Purpose:
  Serves as the central component for hierarchy-based categorical generalization,
  enabling consistent value-to-category mappings across different data sources
  and hierarchy structures.

Key Features:
  - Unified loading interface for JSON and CSV dictionary formats
  - Multi-level hierarchy support (up to 5 levels)
  - Case-insensitive lookups with normalization
  - Alias resolution for alternative value names
  - Metadata preservation and validation
  - Memory-efficient storage with indexing
  - Basic structure validation
  - Thread-safe LRU caching for lookups
  - File hash calculation for cache invalidation

Design Principles:
  - Format agnostic: Handles various dictionary structures transparently
  - Performance: O(1) lookups with normalized key indexing and LRU cache
  - Thread-safety: Concurrent access support with RLock
  - Extensibility: Easy to add new format parsers or validation rules
  - Integration: Uses pamola_core.utils.io for all file operations

Dependencies:
  - pandas: For CSV parsing and data manipulation
  - pamola_core.utils.io: For file loading with encryption support
  - pamola_core.anonymization.commons.text_processing_utils: For text normalization
  - functools: For LRU caching
  - threading: For thread-safe operations
  - hashlib: For file hash calculation
  - logging: For operation tracking

Changelog:
  2.0.0 - Added thread-safe LRU caching and file hash support (REQ-CATGEN-009)
  1.0.0 - Initial implementation with JSON/CSV support
"""

import hashlib
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import pandas as pd

from pamola_core.anonymization.commons.text_processing_utils import normalize_text

# Import core utilities
from pamola_core.utils.io import read_json, read_full_csv, get_file_metadata

logger = logging.getLogger(__name__)

# Constants
MAX_HIERARCHY_LEVELS = 5
MAX_DICTIONARY_SIZE_MB = 100
MAX_ENTRIES = 1_000_000
SUPPORTED_FORMATS = ["json", "csv"]
CACHE_SIZE = 10000  # LRU cache size for lookups


class HierarchyDictionary:
    """
    Manages hierarchical mappings for categorical generalization.

    Supports multiple dictionary formats (JSON/CSV) with unified access interface.
    Handles multi-level hierarchies, aliases, and metadata while providing
    efficient lookups and validation.

    Thread-safe implementation with LRU caching for improved performance.

    Attributes:
        _data: Main storage mapping values to hierarchy information
        _format: Source file format ('json' or 'csv')
        _metadata: Dictionary metadata (version, type, etc.)
        _levels: Hierarchy level names
        _normalized_index: Lowercase key to original key mapping
        _alias_index: Alias to primary value mapping
        _level_values: Cached unique values per hierarchy level
        _lock: Threading lock for thread-safe operations
        _file_hash: Hash of the loaded file for cache invalidation
    """

    def __init__(self):
        """Initialize empty hierarchy dictionary."""
        self._data: Dict[str, Dict[str, Any]] = {}
        self._format: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        self._levels: List[str] = []
        self._normalized_index: Dict[str, str] = {}
        self._alias_index: Dict[str, str] = {}
        self._level_values: Dict[int, Set[str]] = {}
        self._file_path: Optional[Path] = None
        self._file_hash: Optional[str] = None

        # Thread safety
        self._lock = threading.RLock()

        # Clear LRU cache on new instance
        self._clear_caches()

    def load_from_file(
        self,
        filepath: Union[str, Path],
        format_type: str = "auto",
        encryption_key: Optional[str] = None,
    ) -> None:
        """
        Load hierarchy dictionary from file.

        Thread-safe method that loads and indexes hierarchy data.

        Parameters:
        -----------
        filepath : Union[str, Path]
            Path to dictionary file
        format_type : str
            File format ('json', 'csv', or 'auto' for detection)
        encryption_key : Optional[str]
            Encryption key if file is encrypted

        Raises:
        -------
        ValueError
            If file format is unsupported or file is too large
        FileNotFoundError
            If file doesn't exist
        """
        with self._lock:
            filepath = Path(filepath)
            self._file_path = filepath

            # Check file exists
            if not filepath.exists():
                raise FileNotFoundError(f"Dictionary file not found: {filepath}")

            # Check file size
            file_info = get_file_metadata(filepath)
            size_mb = file_info.get("size_mb", 0)
            if size_mb > MAX_DICTIONARY_SIZE_MB:
                raise ValueError(
                    f"Dictionary file too large ({size_mb:.1f}MB). "
                    f"Maximum supported size is {MAX_DICTIONARY_SIZE_MB}MB"
                )

            # Calculate file hash for cache invalidation
            self._file_hash = self._calculate_file_hash(filepath)

            # Detect format if auto
            if format_type == "auto":
                format_type = self._detect_format(filepath)

            if format_type not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format '{format_type}'. "
                    f"Supported formats: {SUPPORTED_FORMATS}"
                )

            self._format = format_type
            logger.info(f"Loading {format_type} dictionary from {filepath}")

            try:
                if format_type == "json":
                    data = read_json(filepath, encryption_key=encryption_key)
                    self._parse_json_hierarchy(data)
                else:  # csv
                    df = read_full_csv(
                        filepath, encryption_key=encryption_key, show_progress=False
                    )
                    self._parse_csv_hierarchy(df)

                # Build indices
                self._build_normalized_index()
                self._build_alias_index()
                self._cache_level_values()

                # Clear LRU caches after loading new data
                self._clear_caches()

                logger.info(
                    f"Loaded dictionary with {len(self._data)} entries, "
                    f"{len(self._levels)} hierarchy levels"
                )

            except Exception as e:
                logger.error(f"Failed to load dictionary: {e}")
                raise

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load hierarchy from a pandas DataFrame.

        Thread-safe method for loading pre-processed DataFrames.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with hierarchy data
        """
        with self._lock:
            self._format = "dataframe"
            self._parse_csv_hierarchy(df)
            self._build_normalized_index()
            self._build_alias_index()
            self._cache_level_values()
            self._clear_caches()

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load hierarchy from a dictionary.

        Thread-safe method for loading pre-processed dictionaries.

        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary with hierarchy data
        """
        with self._lock:
            self._format = "dict"
            self._parse_json_hierarchy(data)
            self._build_normalized_index()
            self._build_alias_index()
            self._cache_level_values()
            self._clear_caches()

    @lru_cache(maxsize=CACHE_SIZE)
    def get_hierarchy(
        self, value: str, level: int = 1, normalize: bool = True
    ) -> Optional[str]:
        """
        Get generalized value at specified hierarchy level.

        Cached method for improved performance on repeated lookups.

        Parameters:
        -----------
        value : str
            Value to look up
        level : int
            Hierarchy level (1-based, 1 = first level of generalization)
        normalize : bool
            Whether to normalize the lookup value

        Returns:
        --------
        Optional[str]
            Generalized value or None if not found
        """
        with self._lock:
            if not self._data:
                logger.warning("Dictionary is empty")
                return None

            if level < 1 or level > len(self._levels):
                logger.warning(
                    f"Invalid hierarchy level {level}. Valid range: 1-{len(self._levels)}"
                )
                return None

            # Prepare lookup key
            lookup_key = value
            if normalize:
                normalized = normalize_text(value, level="basic").lower()
                # Check normalized index
                if normalized in self._normalized_index:
                    lookup_key = self._normalized_index[normalized]
                # Check alias index
                elif normalized in self._alias_index:
                    lookup_key = self._alias_index[normalized]

            # Direct lookup
            if lookup_key in self._data:
                hierarchy = self._data[lookup_key]
                # Handle different level key formats
                level_key = f"level_{level}"
                if level_key in hierarchy:
                    return hierarchy[level_key]
                # Try with level names
                if level <= len(self._levels):
                    level_name = self._levels[level - 1]
                    return hierarchy.get(level_name)

            return None

    def file_hash(self) -> Optional[str]:
        """
        Get hash of the loaded file.

        Public method for cache key generation.

        Returns:
        --------
        Optional[str]
            SHA256 hash of the file or None if no file loaded
        """
        return self._file_hash

    def validate_structure(self) -> Tuple[bool, List[str]]:
        """
        Validate dictionary structure and consistency.

        Thread-safe validation of the loaded hierarchy.

        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues)
        """
        with self._lock:
            issues = []

            # Check if dictionary is empty
            if not self._data:
                issues.append("Dictionary is empty")
                return False, issues

            # Check entry count
            if len(self._data) > MAX_ENTRIES:
                issues.append(
                    f"Dictionary has {len(self._data)} entries. "
                    f"Maximum supported is {MAX_ENTRIES}"
                )

            # Check hierarchy consistency
            level_counts = {}
            for value, hierarchy in self._data.items():
                # Count actual hierarchy levels (excluding metadata and aliases)
                hierarchy_keys = [
                    k
                    for k in hierarchy.keys()
                    if k not in ["metadata", "aliases", "tags", "skills"]
                ]

                # Count level_N keys
                level_keys = [k for k in hierarchy_keys if k.startswith("level_")]
                if level_keys:
                    level_count = len(level_keys)
                else:
                    # Count keys matching level names
                    level_count = sum(1 for k in hierarchy_keys if k in self._levels)

                level_counts[level_count] = level_counts.get(level_count, 0) + 1

            if len(level_counts) > 1:
                issues.append(
                    f"Inconsistent hierarchy depths found: {dict(level_counts)}"
                )

            # Check for empty values
            empty_values = [
                value
                for value, hierarchy in self._data.items()
                if not any(v for k, v in hierarchy.items() if k != "metadata")
            ]
            if empty_values:
                issues.append(
                    f"Found {len(empty_values)} entries with no hierarchy values"
                )

            # Basic circular reference check (simple one-level check)
            for value in list(self._data.keys())[:100]:  # Check first 100 only
                if self._check_circular_reference(value):
                    issues.append(f"Potential circular reference detected for: {value}")

            return len(issues) == 0, issues

    def get_coverage(self, values: List[str], normalize: bool = True) -> Dict[str, Any]:
        """
        Calculate dictionary coverage for a set of values.

        Thread-safe coverage calculation.

        Parameters:
        -----------
        values : List[str]
            Values to check coverage for
        normalize : bool
            Whether to normalize values before checking

        Returns:
        --------
        Dict[str, Any]
            Coverage statistics including percentage and missing values
        """
        with self._lock:
            total = len(values)
            if total == 0:
                return {
                    "total_values": 0,
                    "found_values": 0,
                    "coverage_percent": 0.0,
                    "missing_values": [],
                }

            found = 0
            missing = []

            for value in values:
                if self.get_hierarchy(value, level=1, normalize=normalize) is not None:
                    found += 1
                else:
                    missing.append(value)

            return {
                "total_values": total,
                "found_values": found,
                "coverage_percent": (found / total) * 100,
                "missing_values": missing[:100],  # Limit to first 100
            }

    def get_all_values_at_level(self, level: int) -> Set[str]:
        """
        Get all unique values at a specific hierarchy level.

        Thread-safe access to cached level values.

        Parameters:
        -----------
        level : int
            Hierarchy level (1-based)

        Returns:
        --------
        Set[str]
            Unique values at that level
        """
        with self._lock:
            if level < 1 or level > len(self._levels):
                return set()

            if level in self._level_values:
                return self._level_values[level].copy()

            return set()

    # Private methods

    def _calculate_file_hash(self, filepath: Path) -> str:
        """
        Calculate SHA256 hash of file.

        Parameters:
        -----------
        filepath : Path
            Path to file

        Returns:
        --------
        str
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _clear_caches(self) -> None:
        """Clear all LRU caches."""
        self.get_hierarchy.cache_clear()

    def _detect_format(self, filepath: Path) -> str:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix == ".csv":
            return "csv"
        else:
            raise ValueError(f"Cannot detect format from extension: {suffix}")

    def _parse_json_hierarchy(self, data: Dict[str, Any]) -> None:
        """Parse JSON hierarchy format."""
        # Extract metadata if present
        if "format_version" in data:
            self._metadata["format_version"] = data.get("format_version")
        if "hierarchy_type" in data:
            self._metadata["hierarchy_type"] = data.get("hierarchy_type")
        if "levels" in data:
            self._levels = data["levels"]

        # Get actual data
        if "data" in data:
            hierarchy_data = data["data"]
        else:
            # Assume entire object is the hierarchy
            hierarchy_data = {
                k: v
                for k, v in data.items()
                if k
                not in ["format_version", "hierarchy_type", "levels", "description"]
            }

        # Parse entries
        for value, hierarchy in hierarchy_data.items():
            if isinstance(hierarchy, dict):
                # Complex hierarchy with multiple levels
                entry = {}

                # Handle different level key formats
                for i, level_name in enumerate(self._levels):
                    if level_name in hierarchy:
                        entry[f"level_{i + 1}"] = hierarchy[level_name]

                # Preserve additional fields
                for key, val in hierarchy.items():
                    if key not in self._levels:
                        entry[key] = val

                self._data[value] = entry

            elif isinstance(hierarchy, str):
                # Simple value mapping
                self._data[value] = {"level_1": hierarchy}

            else:
                logger.warning(
                    f"Skipping invalid hierarchy for '{value}': {type(hierarchy)}"
                )

    def _parse_csv_hierarchy(self, df: pd.DataFrame) -> None:
        """Parse CSV hierarchy format."""
        if df.empty:
            return

        columns = df.columns.tolist()

        # First column is the key
        if not columns:
            raise ValueError("CSV file has no columns")

        key_column = columns[0]

        # Detect hierarchy columns (exclude metadata columns)
        metadata_columns = ["aliases", "tags", "skills", "metadata", "common_names"]
        hierarchy_columns = [col for col in columns[1:] if col not in metadata_columns]

        # Set levels from hierarchy columns
        self._levels = hierarchy_columns

        # Parse each row
        for _, row in df.iterrows():
            value = str(row[key_column])
            if pd.isna(value) or value == "":
                continue

            entry = {}

            # Add hierarchy levels
            for i, col in enumerate(hierarchy_columns):
                if col in row and pd.notna(row[col]):
                    entry[f"level_{i + 1}"] = str(row[col])

            # Add metadata fields
            for col in metadata_columns:
                if col in row and pd.notna(row[col]):
                    entry[col] = str(row[col])

            self._data[value] = entry

    def _build_normalized_index(self) -> None:
        """Build normalized key index for case-insensitive lookups."""
        self._normalized_index.clear()

        for value in self._data.keys():
            normalized = normalize_text(value, level="basic").lower()
            if normalized != value.lower():
                self._normalized_index[normalized] = value

    def _build_alias_index(self) -> None:
        """Build alias to primary value mapping."""
        self._alias_index.clear()

        for value, hierarchy in self._data.items():
            # Check for aliases field
            if "aliases" in hierarchy:
                aliases = hierarchy["aliases"]

                # Handle different alias formats
                if isinstance(aliases, list):
                    # JSON array format
                    alias_list = aliases
                elif isinstance(aliases, str):
                    # CSV delimited format
                    alias_list = [a.strip() for a in aliases.split(";") if a.strip()]
                else:
                    continue

                # Add each alias to index
                for alias in alias_list:
                    normalized = normalize_text(alias, level="basic").lower()
                    self._alias_index[normalized] = value

            # Also check 'common_names' field (medical dictionaries)
            if "common_names" in hierarchy:
                common_names = hierarchy["common_names"]
                if isinstance(common_names, str):
                    names = [n.strip() for n in common_names.split(";") if n.strip()]
                    for name in names:
                        normalized = normalize_text(name, level="basic").lower()
                        self._alias_index[normalized] = value

    def _cache_level_values(self) -> None:
        """Cache unique values for each hierarchy level."""
        self._level_values.clear()

        for i in range(1, len(self._levels) + 1):
            values = set()
            level_key = f"level_{i}"

            for hierarchy in self._data.values():
                if level_key in hierarchy and hierarchy[level_key]:
                    values.add(hierarchy[level_key])

            self._level_values[i] = values

    def _check_circular_reference(
        self, value: str, visited: Optional[Set[str]] = None
    ) -> bool:
        """
        Simple circular reference check.

        Note: This is a basic implementation checking if a value
        appears in its own hierarchy path.
        """
        if visited is None:
            visited = set()

        if value in visited:
            return True

        visited.add(value)

        # Check if value appears in any of its hierarchy levels
        if value in self._data:
            hierarchy = self._data[value]
            for key, val in hierarchy.items():
                if key.startswith("level_") and val == value:
                    return True

        return False


# Module metadata
__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main class
__all__ = ["HierarchyDictionary"]
