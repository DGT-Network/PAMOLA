"""
AMOLA.CORE - Privacy-Preserving AI Data Processors
------------------------------------------------------------
Module:        Category Mapping Engine for Categorical Operations
Package:       pamola_core.anonymization.commons
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-23
Updated:       2025-01-23
License:       BSD 3-Clause

Description:
   This module provides a universal thread-safe engine for managing and applying
   category mappings in anonymization operations. It supports both simple
   value-to-category mappings and complex conditional rules with caching.

Purpose:
   Centralizes all category mapping logic for consistent application across
   different anonymization strategies (hierarchy, frequency-based, etc.) with
   optimal performance through thread-safe LRU caching.

Key Features:
   - Fully thread-safe operations with RLock protection
   - Thread-safe LRU caching for mapping lookups
   - Simple and conditional mapping rules
   - Batch application with pandas Series support
   - Unknown value handling with template support
   - Mapping statistics and coverage metrics
   - Export/import for all mapping types
   - Memory-efficient storage
   - Vectorized operations for large datasets

Design Principles:
   - Thread-safety: All operations are thread-safe for concurrent access
   - Performance: O(1) lookups with thread-safe LRU caching
   - Flexibility: Supports various mapping strategies and conditions
   - Integration: Seamless integration with pandas operations

Dependencies:
   - pandas: For Series operations
   - numpy: For vectorized operations
   - cachetools: For thread-safe LRU cache
   - threading: For thread safety
   - logging: For operation tracking
   - json: For serialization

Usage Example:
   ```python
   engine = CategoryMappingEngine(
       unknown_value="OTHER",
       unknown_template="OTHER_{n}",
       cache_size=10000
   )

   # Add simple mapping
   engine.add_mapping("Toronto", "Ontario")

   # Add conditional mapping
   engine.add_conditional_mapping(
       original="Manager",
       replacement="Executive",
       condition={"department": "Sales"}
   )

   # Apply to series
   result = engine.apply_to_series(data_series, context_df)
   ```

Changelog:
   2.0.0 - Fixed thread-safety issues, added template support, improved vectorization
   1.0.0 - Initial implementation with thread-safe caching
"""

import logging
import threading
from typing import Any, Dict, List, Optional, cast, Callable

import pandas as pd
from cachetools import LRUCache, cachedmethod

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_SIZE = 10000
DEFAULT_UNKNOWN_VALUE = "OTHER"
DEFAULT_UNKNOWN_TEMPLATE = "OTHER_{n}"
MAX_MAPPING_SIZE = 1_000_000
VECTORIZATION_THRESHOLD = 1000  # Use vectorized ops above this size


class CategoryMappingEngine:
    """
    Thread-safe engine for applying category mappings.

    Manages both simple value-to-category mappings and conditional mappings
    with efficient caching and batch processing support.

    Attributes:
        unknown_value: Default value for unmapped entries
        unknown_template: Template for numbered unknown values (e.g., "OTHER_{n}")
        cache_size: Size of the LRU cache
        _mappings: Simple value-to-category mappings
        _conditional_mappings: List of conditional mapping rules
        _cache: Thread-safe LRU cache for mapping lookups
        _lock: Threading lock for thread-safe operations
        _stats: Mapping application statistics
        _unknown_counter: Counter for numbered unknown values
        _hierarchy_version: Version/hash of source hierarchy dictionary
    """

    def __init__(self,
                 unknown_value: str = DEFAULT_UNKNOWN_VALUE,
                 unknown_template: Optional[str] = None,
                 cache_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the category mapping engine.

        Parameters:
        -----------
        unknown_value : str
            Value to use for unmapped entries
        unknown_template : Optional[str]
            Template for numbered unknown values (must contain {n})
        cache_size : int
            Maximum size of the LRU cache
        """
        self.unknown_value = unknown_value
        self.unknown_template = unknown_template or DEFAULT_UNKNOWN_TEMPLATE
        self.cache_size = cache_size

        # Validate template
        if "{n}" not in self.unknown_template:
            raise ValueError("unknown_template must contain {n} placeholder")

        # Storage
        self._mappings: Dict[str, str] = {}
        self._conditional_mappings: List[ConditionalMapping] = []

        # Thread-safe cache
        self._cache = LRUCache(maxsize=cache_size)
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            'total_lookups': 0,
            'cache_hits': 0,
            'unknown_count': 0,
            'conditional_matches': 0,
            'vectorized_ops': 0,
            'row_by_row_ops': 0
        }

        # Unknown value generation
        self._unknown_counter = 0
        self._unknown_mapping_cache: Dict[str, str] = {}

        # Source tracking
        self._hierarchy_version: Optional[str] = None

        logger.info(f"Initialized CategoryMappingEngine with cache_size={cache_size}")

    def add_mapping(self, original: str, replacement: str) -> None:
        """
        Add a simple value-to-category mapping.

        Thread-safe method to add a direct mapping rule.

        Parameters:
        -----------
        original : str
            Original value to map
        replacement : str
            Replacement category value
        """
        with self._lock:
            if len(self._mappings) >= MAX_MAPPING_SIZE:
                raise ValueError(
                    f"Maximum mapping size ({MAX_MAPPING_SIZE}) exceeded"
                )

            self._mappings[original] = replacement
            # Clear cache entry if exists
            self._invalidate_cache_for_value(original)

            logger.debug(f"Added mapping: '{original}' -> '{replacement}'")

    def add_conditional_mapping(self,
                                original: str,
                                replacement: str,
                                condition: Dict[str, Any],
                                priority: int = 0) -> None:
        """
        Add a conditional mapping rule.

        Thread-safe method to add a mapping that applies only when
        certain conditions are met.

        Parameters:
        -----------
        original : str
            Original value to map
        replacement : str
            Replacement category value
        condition : Dict[str, Any]
            Conditions that must be met for mapping to apply
        priority : int
            Priority for rule ordering (higher = higher priority)
        """
        with self._lock:
            mapping = ConditionalMapping(
                original=original,
                replacement=replacement,
                condition=condition,
                priority=priority
            )

            self._conditional_mappings.append(mapping)
            # Sort by priority (descending)
            self._conditional_mappings.sort(key=lambda x: x.priority, reverse=True)

            # Clear relevant cache entries
            self._invalidate_cache_for_value(original)

            logger.debug(
                f"Added conditional mapping: '{original}' -> '{replacement}' "
                f"with conditions {condition}"
            )

    def apply_to_series(self,
                        series: pd.Series,
                        context_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Apply mappings to a pandas Series.

        Efficiently applies all mapping rules to a Series, considering
        conditional rules if context is provided.

        Parameters:
        -----------
        series : pd.Series
            Series to transform
        context_df : Optional[pd.DataFrame]
            DataFrame with context for conditional rules

        Returns:
        --------
        pd.Series
            Transformed series with mappings applied
        """
        # Update total lookups
        with self._lock:
            self._stats['total_lookups'] += len(series)

        if context_df is not None and len(self._conditional_mappings) > 0:
            # Apply with context
            if len(series) > VECTORIZATION_THRESHOLD:
                return self._apply_with_context_vectorized(series, context_df)
            else:
                return self._apply_with_context(series, context_df)
        else:
            # Simple mapping (fully vectorized)
            return self._apply_simple_vectorized(series)

    @cachedmethod(lambda self: self._cache, lock=lambda self: self._lock)
    def apply_to_value(self,
                       value: str,
                       context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply mapping to a single value with caching.

        Thread-safe method with caching for single value transformation.

        Parameters:
        -----------
        value : str
            Value to transform
        context : Optional[Dict[str, Any]]
            Context for conditional rules

        Returns:
        --------
        str
            Mapped value or unknown_value if no mapping found
        """
        # Cache key is created by cachedmethod decorator
        with self._lock:
            self._stats['cache_hits'] += 1  # Hit if we got here from cache

        return self._apply_mapping_logic(value, context)

    def get_mapping_dict(self) -> Dict[str, str]:
        """
        Get a copy of simple mappings dictionary.

        Thread-safe method to export simple mappings.

        Returns:
        --------
        Dict[str, str]
            Dictionary of original -> replacement mappings
        """
        with self._lock:
            return self._mappings.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get mapping engine statistics.

        Thread-safe method to retrieve performance statistics.

        Returns:
        --------
        Dict[str, Any]
            Statistics including cache hit rate and unknown count
        """
        with self._lock:
            stats: Dict[str, Any] = self._stats.copy()
            if stats['total_lookups'] > 0:
                # Adjust cache hits for vectorized operations
                # Ensure we're working with integers
                cache_hits = int(stats.get('cache_hits', 0))
                vectorized_ops = int(stats.get('vectorized_ops', 0))
                total_lookups = int(stats['total_lookups'])

                effective_cache_hits = cache_hits + vectorized_ops
                stats['cache_hit_rate'] = float(effective_cache_hits) / float(total_lookups)
            else:
                stats['cache_hit_rate'] = 0.0

            stats['mapping_count'] = len(self._mappings)
            stats['conditional_mapping_count'] = len(self._conditional_mappings)
            stats['cache_size'] = len(self._cache)
            stats['unknown_generated'] = self._unknown_counter

            return stats

    def get_coverage(self, values: pd.Series) -> Dict[str, Any]:
        """
        Calculate mapping coverage for a set of values.

        Thread-safe method to analyze how well mappings cover the data.

        Parameters:
        -----------
        values : pd.Series
            Values to check coverage for

        Returns:
        --------
        Dict[str, Any]
            Coverage statistics
        """
        with self._lock:
            unique_values = values.dropna().unique()
            total = len(unique_values)

            if total == 0:
                return {
                    'total_unique': 0,
                    'mapped': 0,
                    'unmapped': 0,
                    'coverage_percent': 0.0,
                    'unmapped_values': []
                }

            mapped = 0
            unmapped_values = []

            for value in unique_values:
                str_value = str(value)
                if str_value in self._mappings:
                    mapped += 1
                else:
                    unmapped_values.append(str_value)

            return {
                'total_unique': total,
                'mapped': mapped,
                'unmapped': total - mapped,
                'coverage_percent': (mapped / total) * 100,
                'unmapped_values': unmapped_values[:100]  # Limit to first 100
            }

    def clear(self) -> None:
        """
        Clear all mappings and reset statistics.

        Thread-safe method to reset the engine state.
        """
        with self._lock:
            self._mappings.clear()
            self._conditional_mappings.clear()
            self._cache.clear()
            self._unknown_mapping_cache.clear()
            self._unknown_counter = 0
            self._hierarchy_version = None
            self._stats = {
                'total_lookups': 0,
                'cache_hits': 0,
                'unknown_count': 0,
                'conditional_matches': 0,
                'vectorized_ops': 0,
                'row_by_row_ops': 0
            }
            logger.info("Cleared all mappings and cache")

    def reset_state(self) -> None:
        """
        Reset internal state while keeping mappings.

        Used by operations to reset state between executions.
        """
        with self._lock:
            self._cache.clear()
            self._unknown_mapping_cache.clear()
            self._unknown_counter = 0
            # Reset stats
            for key in self._stats:
                self._stats[key] = 0
            logger.debug("Reset engine state")

    def import_mappings(self, mappings: Dict[str, str], check_duplicates: bool = True) -> None:
        """
        Import simple mappings from a dictionary.

        Thread-safe bulk import of mappings.

        Parameters:
        -----------
        mappings : Dict[str, str]
            Dictionary of mappings to import
        check_duplicates : bool
            Whether to check for duplicate mappings
        """
        with self._lock:
            if len(mappings) > MAX_MAPPING_SIZE:
                raise ValueError(
                    f"Import size ({len(mappings)}) exceeds maximum ({MAX_MAPPING_SIZE})"
                )

            if check_duplicates:
                duplicates = set(mappings.keys()) & set(self._mappings.keys())
                if duplicates:
                    logger.warning(
                        f"Overwriting {len(duplicates)} existing mappings: "
                        f"{list(duplicates)[:10]}"
                    )

            self._mappings.update(mappings)
            self._cache.clear()  # Clear cache after bulk update

            logger.info(f"Imported {len(mappings)} mappings")

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all mappings to a serializable dictionary.

        Thread-safe export of all mapping data.

        Returns:
        --------
        Dict[str, Any]
            Complete mapping configuration
        """
        with self._lock:
            return {
                'version': '2.0.0',
                'unknown_value': self.unknown_value,
                'unknown_template': self.unknown_template,
                'hierarchy_version': self._hierarchy_version,
                'simple_mappings': self._mappings.copy(),
                'conditional_mappings': [
                    {
                        'original': m.original,
                        'replacement': m.replacement,
                        'condition': m.condition,
                        'priority': m.priority
                    }
                    for m in self._conditional_mappings
                ],
                'statistics': self.get_statistics()
            }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import mappings from a serialized dictionary.

        Thread-safe import of complete mapping configuration.

        Parameters:
        -----------
        data : Dict[str, Any]
            Mapping configuration to import
        """
        with self._lock:
            # Clear existing
            self.clear()

            # Import settings
            self.unknown_value = data.get('unknown_value', DEFAULT_UNKNOWN_VALUE)
            self.unknown_template = data.get('unknown_template', DEFAULT_UNKNOWN_TEMPLATE)
            self._hierarchy_version = data.get('hierarchy_version')

            # Import simple mappings
            if 'simple_mappings' in data:
                self.import_mappings(data['simple_mappings'], check_duplicates=False)

            # Import conditional mappings
            if 'conditional_mappings' in data:
                for cm in data['conditional_mappings']:
                    self.add_conditional_mapping(
                        original=cm['original'],
                        replacement=cm['replacement'],
                        condition=cm['condition'],
                        priority=cm.get('priority', 0)
                    )

            logger.info(f"Imported complete mapping configuration")

    def set_hierarchy_version(self, version: Optional[str]) -> None:
        """
        Set the hierarchy dictionary version/hash.

        Used for cache invalidation and tracking.

        Parameters:
        -----------
        version : Optional[str]
            Version or hash of source hierarchy
        """
        with self._lock:
            self._hierarchy_version = version

    # Private methods

    def _apply_simple_vectorized(self, series: pd.Series) -> pd.Series:
        """Apply simple mappings using vectorized operations."""
        with self._lock:
            # Create a copy of mappings for thread safety
            mapping_copy = self._mappings.copy()
            unknown_val = self.unknown_value
            self._stats['vectorized_ops'] += len(series)

        # Vectorized mapping with unknown handling
        def safe_map(val):
            if pd.isna(val):
                return val
            str_val = str(val)
            result = mapping_copy.get(str_val, unknown_val)
            return result

        # Apply mapping
        result = series.map(safe_map)

        # Count unknowns
        with self._lock:
            unknown_mask = result == unknown_val
            self._stats['unknown_count'] += unknown_mask.sum()

        return result

    def _apply_with_context(self, series: pd.Series, context_df: pd.DataFrame) -> pd.Series:
        """Apply mappings with context for conditional rules (row-by-row)."""
        result = series.copy()

        with self._lock:
            self._stats['row_by_row_ops'] += len(series)

        for idx in series.index:
            if idx not in context_df.index:
                continue

            value = series.loc[idx]
            if pd.isna(value):
                continue

            # Get context for this row
            row_context = context_df.loc[idx].to_dict()

            # Apply mapping with context (uses cached method)
            func = cast(Callable[[str, Optional[Dict[str, Any]]], str], self.apply_to_value)
            result.loc[idx] = func(str(value), row_context)

        return result

    def _apply_with_context_vectorized(self, series: pd.Series,
                                       context_df: pd.DataFrame) -> pd.Series:
        """Apply mappings with context using vectorized operations where possible."""
        # Group by unique context combinations
        context_cols = list(set(
            key for mapping in self._conditional_mappings
            for key in mapping.condition.keys()
        ))

        if not context_cols:
            # No actual context needed
            return self._apply_simple_vectorized(series)

        # Ensure context columns exist
        context_cols = [col for col in context_cols if col in context_df.columns]
        if not context_cols:
            return self._apply_simple_vectorized(series)

        # Create result series
        result = series.copy()

        # Group by context values
        grouped = context_df.groupby(context_cols, sort=False)
        apply_fn: Callable[[str, Optional[Dict[str, Any]]], str] = cast(
            Callable[[str, Optional[Dict[str, Any]]], str],
            self.apply_to_value
        )

        with self._lock:
            self._stats['vectorized_ops'] += len(series)

        for context_values, group_indices in grouped.groups.items():
            # Get subset of series for this context
            subset = series.loc[group_indices]

            # Build context dict
            if len(context_cols) == 1:
                context = {context_cols[0]: context_values}
            else:
                context = dict(zip(context_cols, context_values))  # type: ignore[arg-type]

            # Apply mappings for this context group
            for value in subset.dropna().unique():
                str_value = str(value)
                mapped_value = apply_fn(str(value), context)

                # Update all occurrences in this group
                mask = (subset == value).values
                result.loc[group_indices[mask]] = mapped_value

        return result

    def _apply_mapping_logic(self, value: str,
                             context: Optional[Dict[str, Any]]) -> str:
        """Core mapping logic with conditional support."""
        # Check conditional mappings first (if context provided)
        if context and self._conditional_mappings:
            for mapping in self._conditional_mappings:
                if mapping.matches(value, context):
                    with self._lock:
                        self._stats['conditional_matches'] += 1
                    return mapping.replacement

        # Check simple mappings
        if value in self._mappings:
            return self._mappings[value]

        # No mapping found - generate unknown value
        with self._lock:
            self._stats['unknown_count'] += 1

            # Check if we've seen this unknown before
            if value in self._unknown_mapping_cache:
                return self._unknown_mapping_cache[value]

            # Generate new unknown value
            if "{n}" in self.unknown_template:
                self._unknown_counter += 1
                unknown_val = self.unknown_template.format(n=self._unknown_counter)
                self._unknown_mapping_cache[value] = unknown_val
                return unknown_val
            else:
                return self.unknown_value

    def _invalidate_cache_for_value(self, value: str) -> None:
        """Invalidate cache entries for a specific value."""
        # Since we're using cachedmethod, we need to clear the entire cache
        # when mappings change. In a future version, we could implement
        # more granular cache invalidation.
        self._cache.clear()


class ConditionalMapping:
    """
    Represents a conditional mapping rule.

    A mapping that applies only when specified conditions are met.
    """

    def __init__(self,
                 original: str,
                 replacement: str,
                 condition: Dict[str, Any],
                 priority: int = 0):
        """
        Initialize conditional mapping.

        Parameters:
        -----------
        original : str
            Original value to match
        replacement : str
            Replacement value
        condition : Dict[str, Any]
            Conditions to check
        priority : int
            Rule priority
        """
        self.original = original
        self.replacement = replacement
        self.condition = condition
        self.priority = priority

    def matches(self, value: str, context: Dict[str, Any]) -> bool:
        """
        Check if this mapping matches the value and context.

        Parameters:
        -----------
        value : str
            Value to check
        context : Dict[str, Any]
            Context to evaluate conditions against

        Returns:
        --------
        bool
            True if mapping applies
        """
        # First check if value matches
        if value != self.original:
            return False

        # Then check all conditions
        for field, expected in self.condition.items():
            if field not in context:
                return False

            actual = context[field]

            # Handle different condition types
            if isinstance(expected, dict):
                # Complex condition with operator
                operator = expected.get('op', 'eq')
                expected_value = expected.get('value')  # Fixed variable name

                if not self._evaluate_operator(actual, operator, expected_value):
                    return False
            else:
                # Simple equality check
                if actual != expected:
                    return False

        return True

    def _evaluate_operator(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate conditional operator."""
        try:
            if operator == 'eq':
                return actual == expected
            elif operator == 'ne':
                return actual != expected
            elif operator == 'gt':
                return actual > expected
            elif operator == 'gte':
                return actual >= expected
            elif operator == 'lt':
                return actual < expected
            elif operator == 'lte':
                return actual <= expected
            elif operator == 'in':
                return actual in expected
            elif operator == 'not_in':
                return actual not in expected
            elif operator == 'contains':
                return expected in str(actual)
            elif operator == 'not_contains':
                return expected not in str(actual)
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            logger.warning(f"Error evaluating condition: {e}")
            return False


# Utility functions

def create_mapping_from_hierarchy(hierarchy_dict: Dict[str, str],
                                  level: int = 1,
                                  unknown_template: Optional[str] = None,
                                  hierarchy_version: Optional[str] = None) -> CategoryMappingEngine:
    """
    Create a mapping engine from a hierarchy dictionary.

    Utility function to quickly create an engine from hierarchy data.

    Parameters:
    -----------
    hierarchy_dict : Dict[str, str]
        Dictionary of value -> category mappings
    level : int
        Hierarchy level (for logging)
    unknown_template : Optional[str]
        Template for unknown values
    hierarchy_version : Optional[str]
        Version/hash of hierarchy dictionary

    Returns:
    --------
    CategoryMappingEngine
        Configured mapping engine
    """
    engine = CategoryMappingEngine(unknown_template=unknown_template)

    if hierarchy_version:
        engine.set_hierarchy_version(hierarchy_version)

    for original, category in hierarchy_dict.items():
        engine.add_mapping(original, category)

    logger.info(
        f"Created mapping engine with {len(hierarchy_dict)} mappings "
        f"for hierarchy level {level}"
    )

    return engine


def merge_mapping_engines(engines: List[CategoryMappingEngine],
                          unknown_value: str = DEFAULT_UNKNOWN_VALUE,
                          unknown_template: Optional[str] = None) -> CategoryMappingEngine:
    """
    Merge multiple mapping engines into one.

    Utility function to combine mappings from multiple sources.

    Parameters:
    -----------
    engines : List[CategoryMappingEngine]
        List of engines to merge
    unknown_value : str
        Unknown value for merged engine
    unknown_template : Optional[str]
        Template for unknown values

    Returns:
    --------
    CategoryMappingEngine
        Merged mapping engine
    """
    merged = CategoryMappingEngine(
        unknown_value=unknown_value,
        unknown_template=unknown_template
    )

    # Merge all data from engines
    for engine in engines:
        # Export complete configuration
        config = engine.export_to_dict()

        # Import simple mappings
        if 'simple_mappings' in config:
            for original, replacement in config['simple_mappings'].items():
                merged.add_mapping(original, replacement)

        # Import conditional mappings
        if 'conditional_mappings' in config:
            for cm in config['conditional_mappings']:
                merged.add_conditional_mapping(
                    original=cm['original'],
                    replacement=cm['replacement'],
                    condition=cm['condition'],
                    priority=cm.get('priority', 0)
                )

    logger.info(f"Merged {len(engines)} mapping engines")

    return merged


# Module metadata
__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main classes and functions
__all__ = [
    'CategoryMappingEngine',
    'ConditionalMapping',
    'create_mapping_from_hierarchy',
    'merge_mapping_engines'
]