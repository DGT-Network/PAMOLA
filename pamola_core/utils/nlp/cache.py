"""
Caching utilities for the NLP module.

This module provides unified caching mechanisms for resources, models, and other
expensive objects used throughout the NLP package.
"""

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, TypeVar, Generic, OrderedDict as OrderedDictType

from pamola_core.utils.nlp.base import CacheBase

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic typing
T = TypeVar('T')

# Cache policy constants
POLICY_LRU = 'lru'   # Least Recently Used
POLICY_LFU = 'lfu'   # Least Frequently Used
POLICY_FIFO = 'fifo' # First In First Out
POLICY_TTL = 'ttl'   # Time To Live
POLICY_TLRU = 'tlru' # Time-aware Least Recently Used

# Global cache settings from environment
MAX_CACHE_SIZE = int(os.environ.get('PAMOLA_MAX_CACHE_SIZE', '100'))
DEFAULT_CACHE_TTL = int(os.environ.get('PAMOLA_CACHE_TTL', '3600'))  # 1 hour
CACHE_ENABLED = os.environ.get('PAMOLA_DISABLE_CACHE', '0') != '1'


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
        policy: str = POLICY_TLRU
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
                "policy": self._policy
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
                k for k, ts in self._timestamps.items()
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
                "file_tracked": len(self._file_paths)
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
        check_memory: bool = True
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

    def set(self, key: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
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
                self._metadata[key] = {'created': time.time()}

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
                "memory_evictions": self._memory_evictions,
                "memory_threshold": self._memory_threshold
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
                        'loaded': key in self._cache,
                        'last_used': self._last_used.get(key, 0),
                        **self._metadata[key]
                    }
                return {}

            # Return info for all models
            result = {}
            for k in self._metadata:
                result[k] = {
                    'loaded': k in self._cache,
                    'last_used': self._last_used.get(k, 0),
                    **self._metadata[k]
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
        # we keep self._metadata because it can store historical info

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


# Global cache instances
_memory_cache = MemoryCache()  # Default memory cache
_file_cache = FileCache()      # File-based cache for resources
_model_cache = ModelCache()    # Specialized model cache


def get_cache(cache_type: str = 'memory') -> CacheBase:
    """
    Get a global cache instance.

    Parameters
    ----------
    cache_type : str
        Type of cache: 'memory', 'file', or 'model'

    Returns
    -------
    CacheBase
        Cache instance
    """
    if cache_type == 'file':
        return _file_cache
    elif cache_type == 'model':
        return _model_cache
    else:
        return _memory_cache


def cache_function(ttl: int = DEFAULT_CACHE_TTL, cache_type: str = 'memory'):
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


def detect_file_encoding(file_path: str, fallback_encoding: str = 'utf-8') -> str:
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
    cache = get_cache('memory')

    encoding = cache.get(cache_key)
    if encoding is not None:
        return encoding

    try:
        import chardet
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding'] if result['encoding'] else fallback_encoding
    except (ImportError, IOError):
        encoding = fallback_encoding

    cache.set(cache_key, encoding)
    return encoding
