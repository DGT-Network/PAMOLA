"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Caching
Description: Caching system for operation results
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides caching functionalities for operation results,
allowing computationally intensive operations to store and reuse results
when inputs haven't changed.

Key features:
- Configurable cache directory location and limits
- Automatic aging and size management
- Hash-based cache key generation
- Thread-safe cache operations
- Synchronous and asynchronous API support

Note: Asynchronous methods require Python 3.9+ for asyncio.to_thread()
If you're using an older Python version, please see alternative implementations
in the comments or use only the synchronous methods.

TODO:
- Add support for distributed cache backends (Redis, Memcached)
- Consider migrating from functional API to class-based in v2
- Move filesystem operations to pamola_core.utils.io helpers
"""

import asyncio
import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

from pamola_core.utils.io import read_json
from pamola_core.utils.ops.op_data_reader import DataReader
from pamola_core.utils.ops.op_data_writer import DataWriter

# Configure logger
logger = logging.getLogger(__name__)


class OpsError(Exception):
    """Base class for all operation-related errors."""
    pass


class CacheError(OpsError):
    """Error raised when cache operations fail."""
    pass


class OperationCache:
    """
    Cache manager for operation results.

    This class provides methods for storing, retrieving, and managing
    cached operation results to improve performance of repeated operations.
    Both synchronous and asynchronous methods are supported.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None,
                 max_age_days: float = 7.0,
                 max_size_mb: float = 500.0):
        """
        Initialize the operation cache manager.

        Parameters:
        -----------
        cache_dir : str or Path, optional
            Directory to store cache files. If None, uses {user_home}/.pamolacache
        max_age_days : float
            Maximum age of cache files in days before they're considered stale
        max_size_mb : float
            Maximum size of cache directory in megabytes
        """
        # Set default cache directory if not provided
        if cache_dir is None:
            home_dir = os.path.expanduser("~")
            cache_dir = Path(home_dir) / ".pamolacache"
        else:
            cache_dir = Path(cache_dir)

        # Create a DataWriter for I/O operations
        self.writer = DataWriter(task_dir=cache_dir, logger=logger)

        # Create a DataReader for reading operations
        self.reader = DataReader(logger=logger)

        self.cache_dir = cache_dir
        self.max_age_seconds = max_age_days * 24 * 3600
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()  # Thread safety

        # TODO: Consider adding stats counters for hits/misses
        # self.stats = {"hits": 0, "misses": 0, "saves": 0}

        # Check and clean cache if needed
        self._check_cache_health()

    def __enter__(self):
        """
        Enter context manager protocol.

        This allows using the cache with 'with' statements:

        with OperationCache() as cache:
            result = cache.get_cache(key)

        Returns:
        --------
        OperationCache
            The cache instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager protocol.

        This ensures cache health is checked when the context is exited.

        Parameters:
        -----------
        exc_type : Type[BaseException] or None
            Exception type if an exception was raised, None otherwise
        exc_val : BaseException or None
            Exception instance if an exception was raised, None otherwise
        exc_tb : traceback or None
            Traceback if an exception was raised, None otherwise

        Returns:
        --------
        bool
            False to propagate exceptions
        """
        self._check_cache_health()
        return False  # Don't suppress exceptions

    def get_cache(self, cache_key: str, operation_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached results for a given key.

        Satisfies REQ-OPS-005: Provides cached operation results retrieval.

        Parameters:
        -----------
        cache_key : str
            Unique identifier for the cache item
        operation_type : str, optional
            Type of operation, used to organize cache files

        Returns:
        --------
        Dict[str, Any] or None
            Cached results or None if not found or stale
        """
        # Create cache file path
        cache_file = self._get_cache_file_path(cache_key, operation_type)

        # Thread-safe operation
        with self._lock:
            # Check if cache file exists
            if not cache_file.exists():
                # self.stats["misses"] += 1  # Update stats if enabled
                return None

            try:
                # Check if cache is stale
                if self._is_cache_stale(cache_file):
                    logger.info(f"Cache is stale: {cache_file}")
                    # self.stats["misses"] += 1  # Update stats if enabled
                    return None

                # Load cache file using io.py functions
                cached_data = read_json(cache_file)

                # Update last access time
                os.utime(cache_file, None)

                # self.stats["hits"] += 1  # Update stats if enabled
                logger.info(f"Cache hit: {cache_key}")
                return cached_data.get('data')

            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                # self.stats["misses"] += 1  # Update stats if enabled
                return None

    async def async_get_cache(self, cache_key: str, operation_type: str = None) -> Optional[Dict[str, Any]]:
        """
        Asynchronous version of get_cache.

        Retrieve cached results for a given key without blocking the event loop.

        Parameters:
        -----------
        cache_key : str
            Unique identifier for the cache item
        operation_type : str, optional
            Type of operation, used to organize cache files

        Returns:
        --------
        Dict[str, Any] or None
            Cached results or None if not found or stale
        """
        return await asyncio.to_thread(self.get_cache, cache_key, operation_type)

    def save_cache(self, data: Dict[str, Any], cache_key: str,
                   operation_type: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Save data to cache.

        Satisfies REQ-OPS-005: Provides operation result caching.

        Parameters:
        -----------
        data : Dict[str, Any]
            Data to cache
        cache_key : str
            Unique identifier for the cache item
        operation_type : str, optional
            Type of operation, used to organize cache files
        metadata : Dict[str, Any], optional
            Additional metadata to store with the cache

        Returns:
        --------
        bool
            True if cache was saved successfully, False otherwise
        """
        # Thread-safe operation
        with self._lock:
            # Check if we need to clean cache before saving
            self._check_cache_size()

            try:
                # Prepare cache data with metadata
                cache_data = {
                    'key': cache_key,
                    'operation_type': operation_type,
                    'timestamp': time.time(),
                    'metadata': metadata or {},
                    'data': data
                }

                # Determine the appropriate subdirectory based on operation_type
                subdir = operation_type if operation_type else None

                # Use DataWriter to write the cache file
                result = self.writer.write_json(
                    data=cache_data,
                    name=f"{cache_key}",
                    subdir=subdir,
                    timestamp_in_name=False,
                    overwrite=True
                )

                # self.stats["saves"] += 1  # Update stats if enabled
                logger.info(f"Saved to cache: {cache_key}")
                return True

            except Exception as e:
                logger.warning(f"Error saving to cache file: {e}")
                return False

    async def async_save_cache(self, data: Dict[str, Any], cache_key: str,
                               operation_type: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Asynchronous version of save_cache.

        Save data to cache without blocking the event loop.

        Parameters:
        -----------
        data : Dict[str, Any]
            Data to cache
        cache_key : str
            Unique identifier for the cache item
        operation_type : str, optional
            Type of operation, used to organize cache files
        metadata : Dict[str, Any], optional
            Additional metadata to store with the cache

        Returns:
        --------
        bool
            True if cache was saved successfully, False otherwise
        """
        return await asyncio.to_thread(self.save_cache, data, cache_key, operation_type, metadata)

    def clear_cache(self, cache_key: Optional[str] = None,
                    operation_type: Optional[str] = None) -> int:
        """
        Clear cache files.

        Parameters:
        -----------
        cache_key : str, optional
            Specific cache key to clear. If None, clears based on operation_type
        operation_type : str, optional
            Type of operation to clear. If None and cache_key is None, clears all cache

        Returns:
        --------
        int
            Number of cache files cleared
        """
        files_cleared = 0

        # Thread-safe operation
        with self._lock:
            try:
                if cache_key is not None:
                    # Clear specific cache file
                    cache_file = self._get_cache_file_path(cache_key, operation_type)
                    if cache_file.exists():
                        os.remove(cache_file)
                        files_cleared = 1
                        logger.info(f"Cleared cache for key: {cache_key}")

                elif operation_type is not None:
                    # Clear all cache files for operation type
                    op_dir = self.cache_dir / operation_type if operation_type else self.cache_dir
                    if op_dir.exists():
                        for file in op_dir.glob("*.json"):
                            os.remove(file)
                            files_cleared += 1
                        logger.info(f"Cleared {files_cleared} cache files for operation type: {operation_type}")

                else:
                    # Clear all cache files
                    for file in self.cache_dir.glob("**/*.json"):
                        os.remove(file)
                        files_cleared += 1
                    logger.info(f"Cleared {files_cleared} cache files")

                return files_cleared

            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
                return files_cleared

    async def async_clear_cache(self, cache_key: Optional[str] = None,
                                operation_type: Optional[str] = None) -> int:
        """
        Asynchronous version of clear_cache.

        Clear cache files without blocking the event loop.

        Parameters:
        -----------
        cache_key : str, optional
            Specific cache key to clear. If None, clears based on operation_type
        operation_type : str, optional
            Type of operation to clear. If None and cache_key is None, clears all cache

        Returns:
        --------
        int
            Number of cache files cleared
        """
        return await asyncio.to_thread(self.clear_cache, cache_key, operation_type)

    def generate_cache_key(self, operation_name: str, parameters: Dict[str, Any],
                           data_hash: Optional[str] = None) -> str:
        """
        Generate a unique cache key for an operation.

        Parameters:
        -----------
        operation_name : str
            Name of the operation
        parameters : Dict[str, Any]
            Operation parameters that affect the result
        data_hash : str, optional
            Hash of input data if applicable

        Returns:
        --------
        str
            Unique cache key
        """
        # TODO: Consider migrating from MD5 to a more secure hashing algorithm
        # like SHA-256 in the next major version. MD5 is maintained here for
        # backward compatibility with existing cache files.

        # Create a hash of operation parameters
        hasher = hashlib.md5()  # nosec B324 - MD5 is used for non-security purposes

        # Add operation name
        hasher.update(operation_name.encode('utf-8'))

        # Add parameters (sorted for consistency)
        for key, value in sorted(parameters.items()):
            # Skip parameters that don't affect the result
            if key in ['use_cache', 'cache_dir', 'reporter', 'progress_tracker']:
                continue

            param_str = f"{key}={value}"
            hasher.update(param_str.encode('utf-8'))

        # Add data hash if provided
        if data_hash:
            hasher.update(data_hash.encode('utf-8'))

        # Generate key
        return f"{operation_name}_{hasher.hexdigest()}"

    async def async_generate_cache_key(self, operation_name: str, parameters: Dict[str, Any],
                                       data_hash: Optional[str] = None) -> str:
        """
        Asynchronous version of generate_cache_key.

        Generate a unique cache key for an operation without blocking the event loop.

        Parameters:
        -----------
        operation_name : str
            Name of the operation
        parameters : Dict[str, Any]
            Operation parameters that affect the result
        data_hash : str, optional
            Hash of input data if applicable

        Returns:
        --------
        str
            Unique cache key
        """
        return await asyncio.to_thread(self.generate_cache_key, operation_name, parameters, data_hash)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
        --------
        Dict[str, Any]
            Dictionary with cache statistics
        """
        # Calculate some useful stats
        cache_size = self._get_cache_size()
        file_count = sum(1 for _ in self.cache_dir.glob("**/*.json"))

        stats = {
            "cache_size_mb": cache_size / (1024 * 1024),
            "file_count": file_count,
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "max_age_days": self.max_age_seconds / (24 * 3600),
            # Uncomment if stats tracking is enabled
            # "hits": self.stats.get("hits", 0),
            # "misses": self.stats.get("misses", 0),
            # "hit_ratio": self.stats.get("hits", 0) / (self.stats.get("hits", 0) + self.stats.get("misses", 0))
            #     if (self.stats.get("hits", 0) + self.stats.get("misses", 0)) > 0 else 0,
        }

        return stats

    def _get_cache_file_path(self, cache_key: str, operation_type: Optional[str] = None) -> Path:
        """
        Get the file path for a cache item.

        Parameters:
        -----------
        cache_key : str
            Unique identifier for the cache item
        operation_type : str, optional
            Type of operation, used to organize cache files

        Returns:
        --------
        Path
            Path to the cache file
        """
        if operation_type:
            # Create subdirectory for operation type
            op_dir = self.cache_dir / operation_type
            # Use DataWriter to ensure directory exists
            self.writer._ensure_directories()
            return op_dir / f"{cache_key}.json"
        else:
            return self.cache_dir / f"{cache_key}.json"

    def _is_cache_stale(self, cache_file: Path) -> bool:
        """
        Check if a cache file is older than the maximum age.

        Parameters:
        -----------
        cache_file : Path
            Path to the cache file

        Returns:
        --------
        bool
            True if cache is stale, False otherwise
        """
        # Get file modification time
        mod_time = os.path.getmtime(cache_file)

        # Compare with current time
        current_time = time.time()

        return (current_time - mod_time) > self.max_age_seconds

    def _check_cache_health(self) -> None:
        """Check cache directory health and clean if necessary."""
        # Check for stale files
        self._clean_stale_cache()

        # Check cache size
        self._check_cache_size()

    async def _async_check_cache_health(self) -> None:
        """
        Asynchronous version of _check_cache_health.
        Check cache directory health and clean if necessary without blocking the event loop.
        """
        await asyncio.to_thread(self._check_cache_health)

    def _clean_stale_cache(self) -> int:
        """
        Remove stale cache files.

        Returns:
        --------
        int
            Number of files removed
        """
        files_removed = 0
        current_time = time.time()

        for file in self.cache_dir.glob("**/*.json"):
            try:
                # Get file modification time
                mod_time = os.path.getmtime(file)

                # Check if file is stale
                if (current_time - mod_time) > self.max_age_seconds:
                    os.remove(file)
                    files_removed += 1
            except Exception as e:
                logger.warning(f"Error checking cache file {file}: {e}")

        if files_removed > 0:
            logger.info(f"Removed {files_removed} stale cache files")

        return files_removed

    def _check_cache_size(self) -> None:
        """Check cache directory size and remove oldest files if too large."""
        # Calculate total cache size
        total_size = self._get_cache_size()

        # Check if cache is too large
        if total_size > self.max_size_bytes:
            logger.info(
                f"Cache size ({total_size / 1024 / 1024:.2f} MB) exceeds limit ({self.max_size_bytes / 1024 / 1024:.2f} MB)")
            self._reduce_cache_size(total_size)

    def _get_cache_size(self) -> int:
        """
        Calculate total size of cache directory.

        Returns:
        --------
        int
            Size in bytes
        """
        total_size = 0

        for file in self.cache_dir.glob("**/*.json"):
            try:
                total_size += os.path.getsize(file)
            except Exception:
                pass

        return total_size

    def _reduce_cache_size(self, current_size: int) -> None:
        """
        Reduce cache size by removing oldest files.

        Parameters:
        -----------
        current_size : int
            Current cache size in bytes
        """
        # Get all cache files with their modification times and sizes in a single pass
        cache_files = []

        for file in self.cache_dir.glob("**/*.json"):
            try:
                mod_time = os.path.getmtime(file)
                size = os.path.getsize(file)
                cache_files.append((file, mod_time, size))
            except Exception as e:
                logger.warning(f"Error checking cache file {file}: {e}")

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        # Remove files until we're under the limit
        target_size = self.max_size_bytes * 0.8  # Aim for 80% of max size
        removed_size = 0
        files_removed = 0

        for file, _, size in cache_files:
            if current_size - removed_size <= target_size:
                break

            try:
                os.remove(file)
                removed_size += size
                files_removed += 1
            except Exception as e:
                logger.warning(f"Error removing cache file {file}: {e}")

        if files_removed > 0:
            logger.info(f"Removed {files_removed} cache files to reduce cache size")


# Create a global cache manager
# This maintains backward compatibility while allowing custom instances if needed
operation_cache = OperationCache()

# For Python < 3.9, use the following implementation of async methods:
'''
async def async_get_cache(self, cache_key: str, operation_type: str = None) -> Optional[Dict[str, Any]]:
    """
    Asynchronous version of get_cache.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.get_cache, cache_key, operation_type)

async def async_save_cache(self, data: Dict[str, Any], cache_key: str,
                          operation_type: str = None, metadata: Dict[str, Any] = None) -> bool:
    """
    Asynchronous version of save_cache.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.save_cache, data, cache_key, operation_type, metadata)

async def async_clear_cache(self, cache_key: Optional[str] = None,
                           operation_type: Optional[str] = None) -> int:
    """
    Asynchronous version of clear_cache.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.clear_cache, cache_key, operation_type)

async def async_generate_cache_key(self, operation_name: str, parameters: Dict[str, Any],
                                  data_hash: Optional[str] = None) -> str:
    """
    Asynchronous version of generate_cache_key.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.generate_cache_key, operation_name, parameters, data_hash)

async def _async_check_cache_health(self) -> None:
    """
    Asynchronous version of _check_cache_health.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, self._check_cache_health)
'''