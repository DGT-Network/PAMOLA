"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Shared Pseudonymization Utilities
Package:       pamola_core.anonymization.commons
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-06-10
Modified:      2025-06-15
License:       BSD 3-Clause

Description:
    This module provides shared utilities for pseudonymization operations within
    the PAMOLA.CORE anonymization framework. It offers common functionality used
    by both hash-based and mapping-based pseudonymization operations, including
    caching, salt configuration loading, pepper generation, and output formatting.

Key Features:
    - Thread-safe LRU cache for pseudonym lookups
    - Flexible salt configuration loading (parameter or file-based)
    - Secure session pepper generation with automatic cleanup
    - Pseudonym output formatting with prefixes and suffixes
    - Performance tracking and statistics collection

Security Considerations:
    - All pepper values are stored in SecureBytes for automatic memory cleanup
    - Salt values are validated and securely processed
    - Thread-safe operations for concurrent processing environments

Dependencies:
    - pamola_core.utils.crypto_helpers.pseudonymization: Core cryptographic utilities
    - threading: For thread synchronization
    - json: For salt file parsing
    - pathlib: For file path handling
    - secrets: For secure random generation
"""

import json
import logging
import secrets
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from pamola_core.utils.crypto_helpers.pseudonymization import (
    SecureBytes
)

# Configure module logger
logger = logging.getLogger(__name__)


class PseudonymizationCache:
    """
    Thread-safe LRU cache for pseudonyms.

    This cache stores original values mapped to their pseudonyms to avoid
    recomputing expensive cryptographic operations. It uses an LRU eviction
    policy to maintain a bounded memory footprint.

    Attributes:
        max_size: Maximum number of entries to cache

    Thread Safety:
        All methods are thread-safe and can be called concurrently.
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialize the pseudonymization cache.

        Args:
            max_size: Maximum number of entries to cache (default: 100000)
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        logger.debug(f"Initialized PseudonymizationCache with max_size={max_size}")

    def get(self, key: str) -> Optional[str]:
        """
        Get pseudonym from cache.

        Args:
            key: Original value to look up

        Returns:
            Cached pseudonym if found, None otherwise
        """
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end to mark as recently used
                self._cache.move_to_end(key)
                return self._cache[key]

            self._misses += 1
            return None

    def put(self, key: str, value: str) -> None:
        """
        Add pseudonym to cache.

        Args:
            key: Original value
            value: Pseudonymized value
        """
        with self._lock:
            # Remove if already exists to update position
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                # Remove least recently used item (first item)
                self._cache.popitem(last=False)
                logger.debug("Evicted LRU entry from cache")

            # Add to end (most recently used)
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing:
                - size: Current number of cached entries
                - max_size: Maximum cache size
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Cache hit rate (0.0 to 1.0)
        """
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0.0,
                "total_requests": total_requests
            }

    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache


def load_salt_configuration(config: Dict[str, Any],
                            salt_file: Optional[Path] = None) -> bytes:
    """
    Load salt based on configuration.

    This function supports two salt sources:
    1. Parameter: Salt provided directly as hex string or bytes
    2. File: Salt loaded from a JSON file with field-specific salts

    Args:
        config: Salt configuration dictionary with:
            - source: "parameter" or "file"
            - value: Salt value (for parameter source)
            - field_name: Field name (for file source)
        salt_file: Optional path to salts file (required for file source)

    Returns:
        Salt as bytes

    Raises:
        ValueError: If configuration is invalid or salt cannot be loaded
    """
    source = config.get('source', 'parameter')

    if source == 'parameter':
        # Salt provided directly
        salt_value = config.get('value')
        if not salt_value:
            raise ValueError("salt_value required when source is 'parameter'")

        if isinstance(salt_value, str):
            # Assume hex encoded
            try:
                salt_bytes = bytes.fromhex(salt_value)
                logger.debug(f"Loaded {len(salt_bytes)}-byte salt from parameter")
                return salt_bytes
            except ValueError as e:
                raise ValueError(f"Invalid hex salt value: {e}")
        elif isinstance(salt_value, bytes):
            logger.debug(f"Loaded {len(salt_value)}-byte salt from parameter")
            return salt_value
        else:
            raise ValueError("salt_value must be hex string or bytes")

    elif source == 'file':
        # Load from JSON file
        if not salt_file:
            raise ValueError("salt_file required when source is 'file'")

        if not salt_file.exists():
            raise ValueError(f"Salt file not found: {salt_file}")

        try:
            with open(salt_file, 'r') as f:
                data = json.load(f)

            # Support both versioned and legacy formats
            if isinstance(data, dict) and "salts" in data:
                # Versioned format
                salts = data["salts"]
                logger.debug(f"Loading from versioned salt file (v{data.get('_version', 'unknown')})")
            else:
                # Legacy format
                salts = data
                logger.debug("Loading from legacy salt file format")

            field_name = config.get('field_name')
            if not field_name:
                raise ValueError("field_name required when loading salt from file")

            if field_name not in salts:
                raise ValueError(f"Salt for field '{field_name}' not found in {salt_file}")

            salt_hex = salts[field_name]
            salt_bytes = bytes.fromhex(salt_hex)
            logger.debug(f"Loaded {len(salt_bytes)}-byte salt for field '{field_name}'")
            return salt_bytes

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in salt file {salt_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading salt from file {salt_file}: {e}")

    else:
        raise ValueError(f"Unknown salt source: {source}. Must be 'parameter' or 'file'")


def generate_session_pepper(length: int = 32) -> SecureBytes:
    """
    Generate pepper for current session.

    Generates a cryptographically secure random pepper value that is
    automatically cleared from memory when no longer needed. The pepper
    provides an additional layer of security beyond salting.

    Args:
        length: Pepper length in bytes (default: 32)

    Returns:
        SecureBytes containing pepper

    Raises:
        ValueError: If length is not positive
    """
    if length <= 0:
        raise ValueError("Pepper length must be positive")

    pepper = secrets.token_bytes(length)
    logger.info(f"Generated {length}-byte session pepper")
    return SecureBytes(pepper)


def format_pseudonym_output(pseudonym: str,
                            prefix: Optional[str] = None,
                            suffix: Optional[str] = None,
                            separator: str = "") -> str:
    """
    Format pseudonym with optional prefix/suffix.

    This function adds optional prefix and/or suffix to a pseudonym,
    useful for creating domain-specific or type-indicated pseudonyms.

    Args:
        pseudonym: Base pseudonym value
        prefix: Optional prefix to prepend
        suffix: Optional suffix to append
        separator: Separator between prefix/suffix and pseudonym (default: "")

    Returns:
        Formatted pseudonym string

    Examples:
        >>> format_pseudonym_output("abc123", prefix="USER_")
        "USER_abc123"

        >>> format_pseudonym_output("abc123", prefix="ID", suffix="X", separator="-")
        "ID-abc123-X"
    """
    result = pseudonym

    if prefix:
        result = f"{prefix}{separator}{result}"
    if suffix:
        result = f"{result}{separator}{suffix}"

    return result


def validate_pseudonym_format(pseudonym: str,
                              expected_format: str,
                              expected_length: Optional[int] = None) -> bool:
    """
    Validate that a pseudonym matches expected format.

    Args:
        pseudonym: Pseudonym to validate
        expected_format: Expected format ("hex", "base64", "base58", "uuid", "alphanumeric")
        expected_length: Expected length (optional)

    Returns:
        True if pseudonym matches expected format
    """
    if expected_length and len(pseudonym) != expected_length:
        return False

    if expected_format == "hex":
        try:
            int(pseudonym, 16)
            return True
        except ValueError:
            return False

    elif expected_format == "base64":
        import base64
        try:
            # URL-safe base64 without padding
            base64.urlsafe_b64decode(pseudonym + "==")
            return True
        except Exception:
            return False

    elif expected_format == "base58":
        # Base58 uses alphanumeric minus confusing characters (0, O, I, l)
        import string
        base58_chars = set(string.ascii_letters + string.digits) - set("0OIl")
        return all(c in base58_chars for c in pseudonym)

    elif expected_format == "uuid":
        import uuid
        try:
            uuid.UUID(pseudonym)
            return True
        except ValueError:
            return False

    elif expected_format == "alphanumeric":
        return pseudonym.isalnum()

    else:
        logger.warning(f"Unknown pseudonym format: {expected_format}")
        return False


def create_compound_identifier(values: Dict[str, Any],
                               separator: str = "|",
                               null_handling: str = "skip") -> str:
    """
    Create a compound identifier from multiple values.

    This is useful for pseudonymizing combinations of fields that together
    form a unique identifier (e.g., first_name + last_name + birthdate).

    Args:
        values: Dictionary of field names to values
        separator: Separator between values (default: "|")
        null_handling: How to handle null values:
            - "skip": Skip null values
            - "empty": Use empty string for nulls
            - "null": Use string "NULL" for nulls

    Returns:
        Compound identifier string

    Example:
        >>> create_compound_identifier({"first": "John", "last": "Doe", "id": None})
        "John|Doe"
    """
    parts = []

    for field_name, value in values.items():
        if value is None:
            if null_handling == "skip":
                continue
            elif null_handling == "empty":
                parts.append("")
            elif null_handling == "null":
                parts.append("NULL")
            else:
                raise ValueError(f"Unknown null_handling: {null_handling}")
        else:
            parts.append(str(value))

    return separator.join(parts)


def estimate_collision_probability(n_values: int,
                                   hash_bits: int = 256) -> float:
    """
    Estimate hash collision probability using birthday paradox.

    Args:
        n_values: Number of unique values to be hashed
        hash_bits: Number of bits in hash output (default: 256 for SHA3-256)

    Returns:
        Estimated collision probability (0.0 to 1.0)
    """
    import math

    # Using birthday paradox approximation
    # P(collision) ≈ 1 - e^(-n^2 / 2^(hash_bits + 1))

    if n_values <= 0:
        return 0.0

    hash_space = 2 ** hash_bits
    if n_values >= hash_space:
        return 1.0

    # For better precision with large numbers
    exponent = -(n_values ** 2) / (2 * hash_space)
    probability = 1 - math.exp(exponent)

    return probability


# Module metadata
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Define explicit exports
__all__ = [
    'PseudonymizationCache',
    'load_salt_configuration',
    'generate_session_pepper',
    'format_pseudonym_output',
    'validate_pseudonym_format',
    'create_compound_identifier',
    'estimate_collision_probability'
]