"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Base NLP Utilities
Package:       pamola_core.utils.nlp
Version:       1.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2024
License:       BSD 3-Clause
Description:
This module provides fundamental utilities, constants, and base classes
for all NLP package components. It includes error hierarchies, dependency
management, resource path handling, and common utility functions.

Key Features:
- Centralized error hierarchy for NLP operations
- Dependency checking and version management
- Resource directory management
- Common utility functions for NLP tasks
- Base cache interface definition

Framework:
Foundation module for PAMOLA.CORE NLP utilities, providing shared
functionality used across all NLP submodules.

Changelog:
1.2.0 - Added LLM error hierarchy for new LLM subsystem
1.1.0 - Enhanced dependency management with version checking
1.0.0 - Initial implementation

Dependencies:
- Standard library only for core functionality
- Optional: packaging for version comparison
"""

import importlib
import logging
import os
import sys
from typing import Dict, Any, Optional, List, Callable, Tuple

try:
    from packaging.version import Version, InvalidVersion

    PACKAGING_AVAILABLE = True
except ImportError:
    PACKAGING_AVAILABLE = False
    Version = None
    InvalidVersion = None

# Configure logger
logger = logging.getLogger(__name__)

# Package resource base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

# Resource subdirectories
STOPWORDS_DIR = os.environ.get(
    "PAMOLA_STOPWORDS_DIR", os.path.join(RESOURCES_DIR, "stopwords")
)
TOKENIZATION_DIR = os.environ.get(
    "PAMOLA_TOKENIZATION_DIR", os.path.join(RESOURCES_DIR, "tokenization")
)
DICTIONARIES_DIR = os.environ.get(
    "PAMOLA_DICTIONARIES_DIR", os.path.join(RESOURCES_DIR, "dictionaries")
)

# Ensure resource directories exist
for directory in [RESOURCES_DIR, STOPWORDS_DIR, TOKENIZATION_DIR, DICTIONARIES_DIR]:
    os.makedirs(directory, exist_ok=True)


# -----------------------------------------------------------------------------
# Error classes for package-specific exceptions
# -----------------------------------------------------------------------------


class NLPError(Exception):
    """
    Base class for all NLP module exceptions.

    This serves as the root of the exception hierarchy for all NLP-related
    errors in PAMOLA.CORE. All custom exceptions in the NLP module should
    inherit from this class.
    """

    pass


class ResourceNotFoundError(NLPError):
    """
    Exception raised when a required resource is not found.

    This typically occurs when looking for resource files like stopwords,
    tokenization rules, or dictionaries.
    """

    pass


class ModelNotAvailableError(NLPError):
    """
    Exception raised when a required model is not available.

    This can occur when a specific NLP model (e.g., spaCy model) is not
    installed or cannot be loaded.
    """

    pass


class UnsupportedLanguageError(NLPError):
    """
    Exception raised when the requested language is not supported.

    This occurs when operations are requested for a language that the
    system does not have resources or models for.
    """

    pass


class ConfigurationError(NLPError):
    """
    Exception raised for configuration-related errors.

    This includes invalid configuration values, missing required
    configuration parameters, or configuration conflicts.
    """

    pass


# -----------------------------------------------------------------------------
# LLM-specific error hierarchy
# -----------------------------------------------------------------------------


class LLMError(NLPError):
    """
    Base exception for LLM operations.

    This serves as the root for all Large Language Model related errors,
    providing a way to catch all LLM-specific exceptions at once.
    """

    pass


class LLMConnectionError(LLMError):
    """
    Exception raised when LLM connection fails.

    This includes:
    - Failed to connect to LLM service (network issues)
    - Authentication failures
    - Service unavailable errors
    - Timeout during connection establishment
    """

    pass


class LLMGenerationError(LLMError):
    """
    Exception raised during text generation.

    This includes:
    - Model failures during generation
    - Token limit exceeded
    - Invalid generation parameters
    - Timeout during generation
    - Model-specific errors
    """

    pass


class LLMResponseError(LLMError):
    """
    Exception raised when response is invalid.

    This includes:
    - Empty or null responses
    - Malformed response format
    - Response parsing failures
    - Response validation failures
    - Unexpected response structure
    """

    pass


# -----------------------------------------------------------------------------
# Dependency management
# -----------------------------------------------------------------------------


class DependencyManager:
    """
    Manager for checking and handling external dependencies.

    This centralizes dependency checks that were previously scattered across
    modules, providing a single point of control for optional dependencies.
    """

    # Cache for dependency checks to avoid repeated imports
    _dependency_cache: Dict[str, bool] = {}

    @classmethod
    def check_dependency(cls, module_name: str) -> bool:
        """
        Check if a dependency (Python module) is available.

        Parameters
        ----------
        module_name : str
            The name of the module to check.

        Returns
        -------
        bool
            True if the module is available, False otherwise.
        """
        if module_name in cls._dependency_cache:
            return cls._dependency_cache[module_name]

        try:
            importlib.import_module(module_name)
            cls._dependency_cache[module_name] = True
            return True
        except ImportError:
            cls._dependency_cache[module_name] = False
            return False

    @classmethod
    def get_module(cls, module_name: str) -> Optional[Any]:
        """
        Get a module if available, otherwise return None.

        Parameters
        ----------
        module_name : str
            The name of the module to import.

        Returns
        -------
        Optional[Any]
            The imported module or None if not available.
        """
        if cls.check_dependency(module_name):
            return importlib.import_module(module_name)
        return None

    @classmethod
    def check_version(
        cls,
        module_name: str,
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a module's version meets specified requirements.

        Parameters
        ----------
        module_name : str
            The name of the module to check.
        min_version : Optional[str]
            Minimum required version.
        max_version : Optional[str]
            Maximum allowed version.

        Returns
        -------
        (bool, Optional[str])
            A tuple of (requirements_met, current_version).
            If the module is unavailable, returns (False, None).
        """
        if not cls.check_dependency(module_name):
            return (False, None)

        try:
            module = importlib.import_module(module_name)
            version_str = getattr(module, "__version__", None) or getattr(
                module, "version", None
            )

            # Try various methods to get version
            if version_str is None and cls.check_dependency(f"{module_name}.version"):
                try:
                    version_module = importlib.import_module(f"{module_name}.version")
                    version_str = getattr(version_module, "__version__", None)
                except (ImportError, AttributeError):
                    pass

            # Use importlib.metadata for Python 3.8+
            if version_str is None and sys.version_info >= (3, 8):
                from importlib import metadata

                try:
                    version_str = metadata.version(module_name)
                except metadata.PackageNotFoundError:
                    pass

            if version_str is None:
                return True, None  # Module is available but version is unknown

            # Check version requirements if specified
            if min_version or max_version:
                if PACKAGING_AVAILABLE:
                    try:
                        current_version = Version(version_str)
                        if min_version and current_version < Version(min_version):
                            return False, version_str
                        if max_version and current_version > Version(max_version):
                            return False, version_str
                    except InvalidVersion:
                        logger.warning(
                            f"Could not parse version string '{version_str}' for module '{module_name}'."
                        )
                        return (
                            True,
                            version_str,
                        )  # Treat as meeting requirements if parsing fails
                else:
                    logger.warning(
                        "The 'packaging' library is not installed. Version comparison might be unreliable."
                    )
                    # Fallback to simple string comparison
                    if min_version and version_str < min_version:
                        return (False, version_str)
                    if max_version and version_str > max_version:
                        return (False, version_str)

            return True, version_str

        except ImportError as e:
            logger.debug(f"Module '{module_name}' not found: {e}")
            return (False, None)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while checking version for '{module_name}': {e}"
            )
            return True, None  # Assume present but version check failed on error

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the dependency check cache.

        This forces re-checking of dependencies on next access.
        """
        cls._dependency_cache.clear()
        logger.debug("Dependency cache cleared")

    @classmethod
    def get_nlp_status(cls) -> Dict[str, bool]:
        """
        Get the status (availability) of various NLP dependencies.

        Returns
        -------
        Dict[str, bool]
            Mapping from module names to booleans indicating availability.
        """
        dependencies = {
            "nltk": cls.check_dependency("nltk"),
            "spacy": cls.check_dependency("spacy"),
            "pymorphy2": cls.check_dependency("pymorphy2"),
            "langdetect": cls.check_dependency("langdetect"),
            "fasttext": cls.check_dependency("fasttext"),
            "transformers": cls.check_dependency("transformers"),
            "wordcloud": cls.check_dependency("wordcloud"),
            # Add LLM-related dependencies
            "lmstudio": cls.check_dependency("lmstudio"),
            "tiktoken": cls.check_dependency("tiktoken"),
            "openai": cls.check_dependency("openai"),
            "anthropic": cls.check_dependency("anthropic"),
        }
        return dependencies

    @staticmethod
    def get_best_available_module(module_preferences: List[str]) -> Optional[str]:
        """
        Get the best available module from a list of preferences.

        Parameters
        ----------
        module_preferences : List[str]
            List of module names in order of preference.

        Returns
        -------
        str or None
            Name of the best available module or None if none is available.
        """
        for module_name in module_preferences:
            if DependencyManager.check_dependency(module_name):
                return module_name
        return None


# -----------------------------------------------------------------------------
# Common utility functions
# -----------------------------------------------------------------------------


def normalize_language_code(lang_code: str) -> str:
    """
    Normalize a language code to a standard format.

    Parameters
    ----------
    lang_code : str
        The language code or name to normalize.

    Returns
    -------
    str
        A normalized 2-letter language code (default 'en' if unknown).
    """
    language_map = {
        # English
        "english": "en",
        "eng": "en",
        "en_us": "en",
        "en_gb": "en",
        "en-us": "en",
        "en-gb": "en",
        # Russian
        "russian": "ru",
        "rus": "ru",
        "ru_ru": "ru",
        "ru-ru": "ru",
        # German
        "german": "de",
        "deu": "de",
        "ger": "de",
        "de_de": "de",
        "de-de": "de",
        # French
        "french": "fr",
        "fra": "fr",
        "fre": "fr",
        "fr_fr": "fr",
        "fr-fr": "fr",
        # Spanish
        "spanish": "es",
        "spa": "es",
        "es_es": "es",
        "es-es": "es",
    }

    # Already normalized
    if len(lang_code) == 2 and lang_code.islower() and lang_code.isalpha():
        return lang_code

    # Check mapping
    normalized = language_map.get(lang_code.lower())
    if normalized:
        return normalized

    # Try to normalize unknown codes
    if len(lang_code) == 2 and lang_code.isalpha():
        return lang_code.lower()

    # Default to English
    return "en"


def batch_process(
    items: List[Any], process_func: Callable, processes: Optional[int] = None, **kwargs
) -> List[Any]:
    """
    Process multiple items in parallel using the specified function.

    Parameters
    ----------
    items : List[Any]
        The list of items to process.
    process_func : Callable
        The function to apply to each item.
    processes : Optional[int]
        Number of processes to use (defaults to up to 8 cores).
    **kwargs
        Additional arguments to pass to `process_func`.

    Returns
    -------
    List[Any]
        The list of processing results.
    """
    if not items:
        return []

    # For small datasets, don't use multiprocessing
    if len(items) < 10:
        return [process_func(item, **kwargs) for item in items]

    # For larger datasets, use multiprocessing
    if processes is None:
        from multiprocessing import cpu_count

        processes = min(cpu_count(), 8)

    def worker(item):
        return process_func(item, **kwargs)

    from multiprocessing import Pool

    with Pool(processes) as pool:
        results = pool.map(worker, items)
    return results


# -----------------------------------------------------------------------------
# Sentinel object to represent missing values
# -----------------------------------------------------------------------------


class _Missing:
    """Internal sentinel class for representing missing values."""

    def __repr__(self):
        return "MISSING"

    def __bool__(self):
        return False


MISSING = _Missing()


# -----------------------------------------------------------------------------
# Cache base class
# -----------------------------------------------------------------------------


class CacheBase:
    """
    Base class for all cache implementations in the package.

    Defines the common interface for caches. All cache implementations
    should inherit from this class and implement the abstract methods.
    """

    def get(self, key: str) -> Any:
        """
        Get a value from the cache by key.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        Any
            The cached value, or None if not found.
        """
        raise NotImplementedError

    def set(self, key: str, value: Any, **kwargs) -> None:
        """
        Set a value in the cache.

        Parameters
        ----------
        key : str
            The cache key.
        value : Any
            The value to store.
        **kwargs
            Implementation-specific arguments (e.g., ttl, tags).
        """
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Parameters
        ----------
        key : str
            The cache key.

        Returns
        -------
        bool
            True if the key was found and deleted, False otherwise.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """
        Clear all items from the cache.
        """
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache usage.

        Returns
        -------
        Dict[str, Any]
            Cache statistics such as hits, misses, size, etc.
        """
        raise NotImplementedError

    def get_or_set(self, key: str, default_func: Callable[[], Any], **kwargs) -> Any:
        """
        Get a value from the cache, or set it if not present by calling `default_func`.

        Parameters
        ----------
        key : str
            The cache key.
        default_func : Callable[[], Any]
            A function that returns the default value if the key is not in cache.
        **kwargs
            Additional arguments passed to the `set` method.

        Returns
        -------
        Any
            The cached or newly computed value.
        """
        value = self.get(key)
        if value is None:
            value = default_func()
            self.set(key, value, **kwargs)
        return value

    def get_model_info(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Return metadata or info about cached items.

        This is a stub method that can be overridden in subclasses if needed.

        Parameters
        ----------
        key : Optional[str]
            If provided, return info for this specific cache key. Otherwise,
            return info about all items.

        Returns
        -------
        Dict[str, Any]
            A dictionary of metadata or an empty dict if not implemented.
        """
        raise NotImplementedError("This cache does not support get_model_info.")


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    # Base error classes
    "NLPError",
    "ResourceNotFoundError",
    "ModelNotAvailableError",
    "UnsupportedLanguageError",
    "ConfigurationError",
    # LLM error classes
    "LLMError",
    "LLMConnectionError",
    "LLMGenerationError",
    "LLMResponseError",
    # Utility classes
    "DependencyManager",
    "CacheBase",
    "MISSING",
    # Utility functions
    "normalize_language_code",
    "batch_process",
    # Constants
    "BASE_DIR",
    "RESOURCES_DIR",
    "STOPWORDS_DIR",
    "TOKENIZATION_DIR",
    "DICTIONARIES_DIR",
]

# Module metadata
__version__ = "1.2.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"
__maintainer__ = "PAMOLA Core Team"
__status__ = "stable"
