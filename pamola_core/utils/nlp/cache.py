"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        NLP Cache Utilities
Package:       pamola_core.utils.nlp.cache
Version:       1.3.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
 This module provides unified caching mechanisms for resources, models, and other
 expensive objects used throughout the NLP package. It implements multiple cache
 strategies with thread-safe operations and automatic expiration.

Key Features:
 - Multiple cache implementations (Memory, File, Model)
 - Thread-safe operations with RLock
 - Multiple eviction policies (LRU, LFU, FIFO, TTL, TLRU)
 - Automatic expiration based on time-to-live
 - File modification tracking for file-based cache
 - Memory-aware eviction for model cache
 - Bulk operations support for improved performance
 - Cache statistics and monitoring
 - Decorator for caching function results
 - Global cache instances with environment configuration
 - Text canonicalization for consistent cache keys

Framework:
 Part of PAMOLA.CORE infrastructure providing caching support for all
 NLP operations and transformations.

Changelog:
 1.3.0 - Added text canonicalization support for consistent cache keys,
         fixed handling of processing markers, improved key generation
 1.2.0 - Added bulk operations (set_many, get_many), improved memory management,
         enhanced statistics tracking, fixed TTL handling in set method
 1.1.0 - Added model cache with memory pressure monitoring
 1.0.0 - Initial implementation with memory and file caches

TODO:
 - Add Redis backend support for distributed caching
 - Implement cache warming strategies
 - Add cache persistence across sessions
 - Support for cache clustering
"""

import hashlib
import logging
import os
import threading
import time
import psutil
from collections import OrderedDict
from typing import (
    Dict,
    Any,
    Optional,
    TypeVar,
    Generic,
    OrderedDict as OrderedDictType,
    List,
    Tuple,
    Callable,
)

from pamola_core.utils.nlp.base import CacheBase

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic typing
T = TypeVar("T")

# Cache policy constants
POLICY_LRU = "lru"  # Least Recently Used
POLICY_LFU = "lfu"  # Least Frequently Used
POLICY_FIFO = "fifo"  # First In First Out
POLICY_TTL = "ttl"  # Time To Live
POLICY_TLRU = "tlru"  # Time-aware Least Recently Used

# Global cache settings from environment
MAX_CACHE_SIZE = int(os.environ.get("PAMOLA_MAX_CACHE_SIZE", "100"))
DEFAULT_CACHE_TTL = int(os.environ.get("PAMOLA_CACHE_TTL", "3600"))  # 1 hour
CACHE_ENABLED = os.environ.get("PAMOLA_DISABLE_CACHE", "0") != "1"

# Default processing marker for text canonicalization
DEFAULT_PROCESSING_MARKER = "~"


def canonicalize_text(
    text: str, processing_marker: str = DEFAULT_PROCESSING_MARKER
) -> str:
    """
    Canonicalize text for consistent cache key generation.

    This function normalizes text by:
    - Removing processing markers from the beginning
    - Normalizing line endings
    - Stripping leading/trailing whitespace

    Parameters
    ----------
    text : str
        Input text to canonicalize
    processing_marker : str
        Processing marker to remove (default: "~")

    Returns
    -------
    str
        Canonicalized text
    """
    if text is None or (hasattr(text, "isna") and text.isna()):
        return ""

    # Convert to string if not already
    text = str(text)

    # Remove processing marker if at the beginning
    if text.startswith(processing_marker):
        text = text[len(processing_marker) :]

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


class MemoryCache(CacheBase, Generic[T]):
    """
    In-memory cache implementation with support for different eviction policies.

    This is a thread-safe cache with configurable eviction policies and supports
    automatic expiration based on time-to-live (TTL).
    """

    def __init__(
        self,
        max_size: int = MAX_CACHE_SIZE,
        ttl: int = DEFAULT_CACHE_TTL,
        policy: str = POLICY_TLRU,
    ):
        """
        Initialize the memory cache.

        Parameters
        ----------
        max_size : int
            Maximum number of items in the cache.
        ttl : int
            Default time-to-live in seconds.
        policy : str
            Cache eviction policy: 'lru', 'lfu', 'fifo', 'ttl', 'tlru'.
        """
        # Use OrderedDict so we can leverage move_to_end() for LRU-like operations
        self._cache: OrderedDictType[str, T] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._ttls: Dict[str, int] = {}
        self._hit_counts: Dict[str, int] = {}
        self._lock = threading.RLock()

        self._max_size = max(1, max_size)
        self._default_ttl = ttl
        self._policy = policy.lower()

        # Stats counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._bulk_operations = 0

        logger.debug(
            f"Initialized {self.__class__.__name__} with "
            f"max_size={max_size}, ttl={ttl}, policy={policy}"
        )

    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        T or None
            Cached value or None if not found or expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            now = time.time()
            timestamp = self._timestamps.get(key, 0)
            ttl = self._ttls.get(key, self._default_ttl)

            # Check for expiration
            if ttl > 0 and now - timestamp > ttl:
                # Key has expired, remove it
                self._remove_item(key)
                self._expirations += 1
                self._misses += 1
                return None

            # Update eviction policy usage
            if self._policy in (POLICY_LRU, POLICY_TLRU):
                # Move to end as most recently used
                value = self._cache.pop(key)
                self._cache[key] = value
                self._timestamps[key] = now

            if self._policy == POLICY_LFU:
                # Increment hit count
                self._hit_counts[key] = self._hit_counts.get(key, 0) + 1

            self._hits += 1
            return self._cache[key]

    def get_many(self, keys: List[str]) -> Dict[str, Optional[T]]:
        """
        Get multiple values from the cache in a single operation.

        Parameters
        ----------
        keys : List[str]
            List of cache keys

        Returns
        -------
        Dict[str, Optional[T]]
            Dictionary mapping keys to values (None if not found or expired)
        """
        results = {}
        with self._lock:
            now = time.time()

            for key in keys:
                if key not in self._cache:
                    self._misses += 1
                    results[key] = None
                    continue

                # Check expiration
                timestamp = self._timestamps.get(key, 0)
                ttl = self._ttls.get(key, self._default_ttl)

                if ttl > 0 and now - timestamp > ttl:
                    # Key has expired
                    self._remove_item(key)
                    self._expirations += 1
                    self._misses += 1
                    results[key] = None
                    continue

                # Update eviction policy usage
                if self._policy in (POLICY_LRU, POLICY_TLRU):
                    # Move to end as most recently used
                    value = self._cache.pop(key)
                    self._cache[key] = value
                    self._timestamps[key] = now

                if self._policy == POLICY_LFU:
                    # Increment hit count
                    self._hit_counts[key] = self._hit_counts.get(key, 0) + 1

                self._hits += 1
                results[key] = self._cache[key]

        self._bulk_operations += 1
        return results

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        If the key exists and ttl=None, we reset it to the default TTL
        (instead of leaving the old TTL in place). This ensures consistency
        with how new entries are handled.

        Parameters
        ----------
        key : str
            Cache key
        value : T
            Value to cache
        ttl : int, optional
            Time-to-live in seconds, overrides default if provided
        """
        if not CACHE_ENABLED:
            return

        with self._lock:
            # If key already exists
            if key in self._cache:
                self._cache[key] = value
                self._timestamps[key] = time.time()

                # If ttl is explicitly given, use it; otherwise reset to default
                if ttl is not None:
                    self._ttls[key] = ttl
                else:
                    self._ttls[key] = self._default_ttl

                if self._policy in (POLICY_LRU, POLICY_TLRU):
                    self._cache.move_to_end(key)

                if self._policy == POLICY_LFU:
                    # Reset hit count for updated item
                    self._hit_counts[key] = 0

                return

            # If it's a new key, check capacity
            if len(self._cache) >= self._max_size:
                self._evict_item()

            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._ttls[key] = ttl if ttl is not None else self._default_ttl

            if self._policy == POLICY_LFU:
                self._hit_counts[key] = 0

    def set_many(self, mapping: Dict[str, T], ttl: Optional[int] = None) -> None:
        """
        Set multiple key-value pairs in the cache in a single operation.

        Parameters
        ----------
        mapping : Dict[str, T]
            Dictionary of key-value pairs to cache
        ttl : int, optional
            Time-to-live in seconds for all entries
        """
        if not CACHE_ENABLED:
            return

        with self._lock:
            now = time.time()
            effective_ttl = ttl if ttl is not None else self._default_ttl

            for key, value in mapping.items():
                # If key already exists
                if key in self._cache:
                    self._cache[key] = value
                    self._timestamps[key] = now
                    self._ttls[key] = effective_ttl

                    if self._policy in (POLICY_LRU, POLICY_TLRU):
                        self._cache.move_to_end(key)

                    if self._policy == POLICY_LFU:
                        self._hit_counts[key] = 0
                else:
                    # New key - check capacity
                    if len(self._cache) >= self._max_size:
                        self._evict_item()

                    # Add new entry
                    self._cache[key] = value
                    self._timestamps[key] = now
                    self._ttls[key] = effective_ttl

                    if self._policy == POLICY_LFU:
                        self._hit_counts[key] = 0

        self._bulk_operations += 1

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        bool
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self._hit_counts.clear()
            logger.debug(f"Cache cleared: {self.__class__.__name__}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache usage.

        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = self._hits / total if total > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "bulk_operations": self._bulk_operations,
                "policy": self._policy,
            }

    def _evict_item(self) -> None:
        """
        Evict an item based on the current cache policy.
        """
        if not self._cache:
            return

        if self._policy == POLICY_FIFO:
            # Remove first item (oldest insertion)
            key = next(iter(self._cache))
            self._remove_item(key)

        elif self._policy == POLICY_LRU:
            # Remove the least recently used item (first item in OrderedDict)
            key = next(iter(self._cache))
            self._remove_item(key)

        elif self._policy == POLICY_LFU:
            # Remove the least frequently used item
            if self._hit_counts:
                key = min(self._hit_counts, key=self._hit_counts.get)
                self._remove_item(key)
            else:
                # If no hit counts, fallback to LRU
                key = next(iter(self._cache))
                self._remove_item(key)

        elif self._policy == POLICY_TTL:
            # Remove the oldest item by timestamp
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            self._remove_item(oldest_key)

        elif self._policy == POLICY_TLRU:
            # First check for expired items
            now = time.time()
            expired_keys = [
                k
                for k, ts in self._timestamps.items()
                if now - ts > self._ttls.get(k, self._default_ttl)
            ]
            if expired_keys:
                # Remove the oldest expired item
                key = min(expired_keys, key=lambda k: self._timestamps.get(k, 0))
                self._remove_item(key)
            else:
                # If no expired items, fallback to LRU
                key = next(iter(self._cache))
                self._remove_item(key)

        self._evictions += 1

    def _remove_item(self, key: str) -> None:
        """
        Remove an item from all internal structures.

        Parameters
        ----------
        key : str
            Cache key to remove
        """
        if key in self._cache:
            self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._ttls.pop(key, None)
        self._hit_counts.pop(key, None)


class FileCache(CacheBase):
    """
    Cache implementation for file-based resources with timestamp validation.

    This cache keeps track of file modification times to detect changes.
    Supports bulk operations for improved I/O efficiency.
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        """
        Initialize the file cache.

        Parameters
        ----------
        max_size : int
            Maximum number of items in the cache
        """
        self._cache: Dict[str, Any] = {}
        self._file_paths: Dict[str, str] = {}
        self._mtimes: Dict[str, float] = {}
        self._lock = threading.RLock()

        self._max_size = max(1, max_size)

        # Stats counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bulk_operations = 0

        # Batch write buffer for improved I/O
        self._write_buffer: List[Tuple[str, Any, Optional[str]]] = []
        self._buffer_size = 10  # Flush after this many writes

        logger.debug(f"Initialized {self.__class__.__name__} with max_size={max_size}")

    def get(self, key: str) -> Any:
        """
        Get a value from the cache, checking file modification time.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any
            Cached value or None if not found or file has changed
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            file_path = self._file_paths.get(key)
            if file_path and os.path.exists(file_path):
                try:
                    current_mtime = os.path.getmtime(file_path)
                    if current_mtime > self._mtimes.get(key, 0):
                        # File has changed, invalidate cache entry
                        self._remove_item(key)
                        self._misses += 1
                        return None
                except OSError:
                    # If we can't check mtime, assume file changed
                    self._remove_item(key)
                    self._misses += 1
                    return None

            self._hits += 1
            return self._cache[key]

    def get_many(self, keys: List[str]) -> Dict[str, Optional[Any]]:
        """
        Get multiple values from the cache in a single operation.

        Parameters
        ----------
        keys : List[str]
            List of cache keys

        Returns
        -------
        Dict[str, Optional[Any]]
            Dictionary mapping keys to values (None if not found or file changed)
        """
        results = {}
        with self._lock:
            for key in keys:
                if key not in self._cache:
                    self._misses += 1
                    results[key] = None
                    continue

                file_path = self._file_paths.get(key)
                if file_path and os.path.exists(file_path):
                    try:
                        current_mtime = os.path.getmtime(file_path)
                        if current_mtime > self._mtimes.get(key, 0):
                            # File has changed
                            self._remove_item(key)
                            self._misses += 1
                            results[key] = None
                            continue
                    except OSError:
                        self._remove_item(key)
                        self._misses += 1
                        results[key] = None
                        continue

                self._hits += 1
                results[key] = self._cache[key]

        self._bulk_operations += 1
        return results

    def set(self, key: str, value: Any, file_path: Optional[str] = None) -> None:
        """
        Set a value in the cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        file_path : str, optional
            Path to the associated file for modification tracking
        """
        if not CACHE_ENABLED:
            return

        with self._lock:
            # Add to write buffer for batch processing
            self._write_buffer.append((key, value, file_path))

            # Flush buffer if it's full
            if len(self._write_buffer) >= self._buffer_size:
                self._flush_write_buffer()

    def set_many(
        self, mapping: Dict[str, Any], file_paths: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set multiple key-value pairs in the cache with batch write optimization.

        Parameters
        ----------
        mapping : Dict[str, Any]
            Dictionary of key-value pairs to cache
        file_paths : Dict[str, str], optional
            Dictionary mapping keys to file paths
        """
        if not CACHE_ENABLED:
            return

        with self._lock:
            for key, value in mapping.items():
                file_path = file_paths.get(key) if file_paths else None
                self._write_buffer.append((key, value, file_path))

            # Always flush for bulk operations
            self._flush_write_buffer()
            self._bulk_operations += 1

    def _flush_write_buffer(self) -> None:
        """Flush the write buffer to cache storage."""
        if not self._write_buffer:
            return

        for key, value, file_path in self._write_buffer:
            if key not in self._cache and len(self._cache) >= self._max_size:
                self._evict_item()

            self._cache[key] = value

            if file_path:
                self._file_paths[key] = file_path
                try:
                    if os.path.exists(file_path):
                        self._mtimes[key] = os.path.getmtime(file_path)
                except OSError:
                    self._mtimes[key] = time.time()

        self._write_buffer.clear()

    def is_valid(self, key: str) -> bool:
        """
        Check if a cached value is still valid.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        bool
            True if key exists and is valid, False otherwise
        """
        with self._lock:
            if key not in self._cache:
                return False

            file_path = self._file_paths.get(key)
            if not file_path or not os.path.exists(file_path):
                return False

            try:
                current_mtime = os.path.getmtime(file_path)
                return current_mtime <= self._mtimes.get(key, 0)
            except OSError:
                return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        bool
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            # Flush any pending writes first
            self._flush_write_buffer()

            if key in self._cache:
                self._remove_item(key)
                return True
            return False

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        with self._lock:
            self._write_buffer.clear()
            self._cache.clear()
            self._file_paths.clear()
            self._mtimes.clear()
            logger.debug(f"Cache cleared: {self.__class__.__name__}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache usage.

        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = self._hits / total if total > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "evictions": self._evictions,
                "file_tracked": len(self._file_paths),
                "bulk_operations": self._bulk_operations,
                "pending_writes": len(self._write_buffer),
            }

    def _evict_item(self) -> None:
        """
        Evict an item from the cache (oldest mtime first).
        """
        if not self._cache:
            return

        if self._mtimes:
            oldest_key = min(self._mtimes, key=self._mtimes.get)
        else:
            oldest_key = next(iter(self._cache))

        self._remove_item(oldest_key)
        self._evictions += 1

    def _remove_item(self, key: str) -> None:
        """
        Remove an item from all internal structures.

        Parameters
        ----------
        key : str
            Cache key to remove
        """
        self._cache.pop(key, None)
        self._file_paths.pop(key, None)
        self._mtimes.pop(key, None)


class ModelCache(CacheBase):
    """
    Cache for managing NLP models with memory-aware eviction.

    This specialized cache monitors memory usage and can proactively
    unload models when memory pressure is high.
    """

    def __init__(
        self,
        max_size: int = 5,
        memory_threshold: float = 0.75,  # 75% memory usage
        check_memory: bool = True,
    ):
        """
        Initialize the model cache.

        Parameters
        ----------
        max_size : int
            Maximum number of models to keep in cache
        memory_threshold : float
            Memory usage threshold (0-1) for eviction
        check_memory : bool
            Whether to check system memory when adding models
        """
        self._cache: OrderedDictType[str, Any] = OrderedDict()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._last_used: Dict[str, float] = {}
        self._lock = threading.RLock()

        self._max_size = max(1, max_size)
        self._memory_threshold = min(max(0.1, memory_threshold), 0.95)
        self._check_memory = check_memory

        # Stats counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._memory_evictions = 0

        logger.debug(
            f"Initialized {self.__class__.__name__} with "
            f"max_size={max_size}, memory_threshold={memory_threshold}"
        )

    def get(self, key: str) -> Any:
        """
        Get a model from the cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any
            Cached model or None if not found
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            self._last_used[key] = time.time()

            # Move to end for LRU ordering
            model = self._cache.pop(key)
            self._cache[key] = model

            self._hits += 1
            return model

    def set(
        self, key: str, model: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a model in the cache.

        Parameters
        ----------
        key : str
            Cache key
        model : Any
            Model to cache
        metadata : Dict[str, Any], optional
            Metadata about the model
        """
        if not CACHE_ENABLED:
            return

        with self._lock:
            # Check memory pressure first if enabled
            if self._check_memory and self._is_memory_pressure():
                self._reduce_memory_pressure()

            # Check if cache is full and this is a new key
            if key not in self._cache and len(self._cache) >= self._max_size:
                self._evict_item()

            self._cache[key] = model
            self._last_used[key] = time.time()

            if metadata:
                self._metadata[key] = metadata.copy()
            else:
                self._metadata[key] = {"created": time.time()}

    def delete(self, key: str) -> bool:
        """
        Delete a model from the cache.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        bool
            True if model was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False

    def clear(self) -> None:
        """
        Clear all models from the cache.
        """
        with self._lock:
            # Help garbage collection
            for model in self._cache.values():
                del model

            self._cache.clear()
            self._metadata.clear()
            self._last_used.clear()

            import gc

            gc.collect()

            logger.debug(f"Model cache cleared: {self.__class__.__name__}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache usage.

        Returns
        -------
        Dict[str, Any]
            Cache statistics including memory info if psutil is available.
            Memory-related keys will have None values if psutil is not installed.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = self._hits / total if total > 0 else 0

            # Initialize memory info with None values as defaults
            # This ensures these keys always exist in the return dict,
            # even if psutil is not available
            memory_info: Dict[str, Optional[float]] = {
                "memory_percent": None,
                "memory_available_gb": None,
                "memory_total_gb": None,
            }

            # Try to get actual memory stats if psutil is available
            try:
                # Import psutil here as it's an optional dependency
                # If import fails, we'll keep the None values initialized above

                # Get virtual memory stats
                # This line will only execute if psutil import succeeded
                memory = psutil.virtual_memory()

                # Update the dictionary with actual values
                # Using update() preserves the original structure
                memory_info.update(
                    {
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / (1024**3),
                        "memory_total_gb": memory.total / (1024**3),
                    }
                )
            except ImportError:
                # psutil is not installed - this is fine for optional dependency
                # The memory_info dict already has None values, so nothing to do
                pass

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "evictions": self._evictions,
                "memory_evictions": self._memory_evictions,
                "memory_threshold": self._memory_threshold,
                **memory_info,  # Spread operator merges memory stats
            }

    def get_model_info(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about cached models.

        Parameters
        ----------
        key : Optional[str]
            Specific model key to get info for

        Returns
        -------
        Dict[str, Any]
            Information about cached models
        """
        with self._lock:
            if key is not None:
                if key in self._metadata:
                    return {
                        "loaded": key in self._cache,
                        "last_used": self._last_used.get(key, 0),
                        **self._metadata[key],
                    }
                return {}

            # Return info for all models
            result = {}
            for k in self._metadata:
                result[k] = {
                    "loaded": k in self._cache,
                    "last_used": self._last_used.get(k, 0),
                    **self._metadata[k],
                }
            return result

    def _evict_item(self) -> None:
        """
        Evict an item from the cache (least recently used).
        """
        if not self._cache:
            return

        oldest_key = min(self._last_used, key=self._last_used.get)
        self._remove_item(oldest_key)
        self._evictions += 1

        logger.debug(f"Evicted model from cache: {oldest_key}")

    def _remove_item(self, key: str) -> None:
        """
        Remove an item from all internal structures.

        Parameters
        ----------
        key : str
            Cache key to remove
        """
        if key in self._cache:
            model = self._cache.pop(key)
            model = None

        self._last_used.pop(key, None)
        # We keep self._metadata because it can store historical info

        import gc

        gc.collect()

    def _is_memory_pressure(self) -> bool:
        """
        Check if system memory is under pressure.

        Returns
        -------
        bool
            True if memory usage is above threshold
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            return memory.percent / 100 > self._memory_threshold
        except ImportError:
            # If psutil is not available, fallback to cache size check
            return len(self._cache) >= self._max_size

    def _reduce_memory_pressure(self) -> None:
        """
        Reduce memory pressure by removing the least recently used models,
        until memory usage drops below threshold or only one model remains.
        """
        if not self._cache or len(self._cache) <= 1:
            return

        try:
            import psutil

            while len(self._cache) > 1:
                memory = psutil.virtual_memory()
                if memory.percent / 100 <= self._memory_threshold:
                    break

                oldest_key = min(self._last_used, key=self._last_used.get)
                self._remove_item(oldest_key)
                self._memory_evictions += 1

                logger.debug(f"Memory pressure: evicted model {oldest_key}")

        except ImportError:
            # If psutil is not available, remove half of the models
            models_to_remove = max(1, len(self._cache) // 2)
            sorted_keys = sorted(self._last_used.items(), key=lambda x: x[1])
            for k, _ in sorted_keys[:models_to_remove]:
                self._remove_item(k)
                self._memory_evictions += 1


class TextCache(MemoryCache[str]):
    """
    Specialized cache for text content with built-in canonicalization.

    This cache automatically canonicalizes text keys to ensure consistent
    caching regardless of processing markers or whitespace variations.
    """

    def __init__(
        self,
        max_size: int = MAX_CACHE_SIZE,
        ttl: int = DEFAULT_CACHE_TTL,
        policy: str = POLICY_TLRU,
        processing_marker: str = DEFAULT_PROCESSING_MARKER,
        canonicalize_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the text cache.

        Parameters
        ----------
        max_size : int
            Maximum number of items in the cache
        ttl : int
            Default time-to-live in seconds
        policy : str
            Cache eviction policy
        processing_marker : str
            Processing marker to handle during canonicalization
        canonicalize_func : Callable, optional
            Custom canonicalization function
        """
        super().__init__(max_size, ttl, policy)
        self._processing_marker = processing_marker
        self._canonicalize_func = canonicalize_func or self._default_canonicalize

        # Track original keys for reverse lookup
        self._canonical_to_original: Dict[str, str] = {}

    def _default_canonicalize(self, text: str) -> str:
        """Default canonicalization using the module function."""
        return canonicalize_text(text, self._processing_marker)

    def _generate_key(self, text: str) -> str:
        """
        Generate a cache key from canonicalized text.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        str
            Cache key (hash of canonicalized text)
        """
        canonical = self._canonicalize_func(text)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[str]:
        """
        Get a value from the cache using canonicalized key.

        Parameters
        ----------
        text : str
            Text to look up (will be canonicalized)

        Returns
        -------
        str or None
            Cached value or None if not found
        """
        key = self._generate_key(text)
        return super().get(key)

    def set(self, text: str, value: str, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache using canonicalized key.

        Parameters
        ----------
        text : str
            Text key (will be canonicalized)
        value : str
            Value to cache
        ttl : int, optional
            Time-to-live in seconds
        """
        key = self._generate_key(text)
        canonical = self._canonicalize_func(text)

        # Track the mapping
        with self._lock:
            self._canonical_to_original[canonical] = text

        super().set(key, value, ttl)

    def delete(self, text: str) -> bool:
        """
        Delete a key from the cache using canonicalized lookup.

        Parameters
        ----------
        text : str
            Text key to delete

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        key = self._generate_key(text)
        canonical = self._canonicalize_func(text)

        with self._lock:
            self._canonical_to_original.pop(canonical, None)

        return super().delete(key)

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._canonical_to_original.clear()
        super().clear()


# Global cache instances
_memory_cache = MemoryCache()  # Default memory cache
_file_cache = FileCache()  # File-based cache for resources
_model_cache = ModelCache()  # Specialized model cache
_text_cache = TextCache()  # Specialized text cache with canonicalization


def get_cache(cache_type: str = "memory") -> CacheBase:
    """
    Get a global cache instance.

    Parameters
    ----------
    cache_type : str
        Type of cache: 'memory', 'file', 'model', or 'text'

    Returns
    -------
    CacheBase
        Cache instance
    """
    if cache_type == "file":
        return _file_cache
    elif cache_type == "model":
        return _model_cache
    elif cache_type == "text":
        return _text_cache
    else:
        return _memory_cache


def cache_function(ttl: int = DEFAULT_CACHE_TTL, cache_type: str = "memory"):
    """
    Decorator to cache function results.

    Parameters
    ----------
    ttl : int
        Time-to-live in seconds.
    cache_type : str
        Type of cache to use.

    Returns
    -------
    Callable
        The decorated function.
    """

    def decorator(func):
        cache = get_cache(cache_type)

        def wrapper(*args, **kwargs):
            if not CACHE_ENABLED:
                return func(*args, **kwargs)

            # Build a cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = "func:" + ":".join(key_parts)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Otherwise compute and store
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def detect_file_encoding(file_path: str, fallback_encoding: str = "utf-8") -> str:
    """
    Detect encoding of a file with caching.

    Parameters
    ----------
    file_path : str
        Path to the file
    fallback_encoding : str
        Fallback encoding if detection fails

    Returns
    -------
    str
        Detected encoding
    """
    cache_key = f"file_encoding:{file_path}"
    cache = get_cache("memory")

    encoding = cache.get(cache_key)
    if encoding is not None:
        return encoding

    try:
        import chardet

        with open(file_path, "rb") as f:
            result = chardet.detect(f.read())
        encoding = result["encoding"] if result["encoding"] else fallback_encoding
    except (ImportError, IOError):
        encoding = fallback_encoding

    cache.set(cache_key, encoding)
    return encoding
