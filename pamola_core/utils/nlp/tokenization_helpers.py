"""
Helper utilities for tokenization.

This module provides supporting functions for loading configuration,
dictionaries, and other resources needed for tokenization.
"""

import json
import logging
import os
from typing import Dict, List, Set, Any, Optional, Union, Callable

from pamola_core.utils.nlp.base import TOKENIZATION_DIR
from pamola_core.utils.nlp.cache import get_cache, cache_function

# Configure logger
logger = logging.getLogger(__name__)

# Get caches
file_cache = get_cache('file')
memory_cache = get_cache('memory')


@cache_function(ttl=3600, cache_type='file')
def load_tokenization_config(config_sources: Optional[Union[str, List[str]]] = None,
                             language: Optional[str] = None) -> Dict[str, Any]:
    """
    Load tokenization configuration from files.

    Parameters
    ----------
    config_sources : str or List[str], optional
        Path(s) to configuration files. If None, use default.
    language : str, optional
        Language code for language-specific configurations.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    # Start with empty config
    config = {}

    # Default config path if not specified
    if not config_sources:
        config_sources = []
        # Try to load default config
        default_path = os.path.join(TOKENIZATION_DIR, 'default_config.json')
        if os.path.exists(default_path):
            config_sources.append(default_path)

        # Try to load language-specific config
        if language:
            lang_path = os.path.join(TOKENIZATION_DIR, f"{language}_config.json")
            if os.path.exists(lang_path):
                config_sources.append(lang_path)

    # Normalize to list
    if isinstance(config_sources, str):
        config_sources = [config_sources]

    # Load and merge configs
    for source in config_sources:
        try:
            if not os.path.exists(source):
                logger.warning(f"Configuration file not found: {source}")
                continue

            with open(source, 'r', encoding='utf-8') as f:
                src_config = json.load(f)

            # Merge into main config
            config.update(src_config)
            logger.debug(f"Loaded tokenization config from {source}")

        except Exception as e:
            logger.error(f"Error loading config from {source}: {e}")

    return config


@cache_function(ttl=3600, cache_type='file')
def load_synonym_dictionary(sources: Optional[Union[str, List[str]]] = None,
                            language: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load synonym dictionary from files.

    Parameters
    ----------
    sources : str or List[str], optional
        Path(s) to synonym dictionary files. If None, use default.
    language : str, optional
        Language code for language-specific dictionaries.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary of {canonical_form: [synonyms]}
    """
    synonyms = {}

    # Set default sources if not provided
    if not sources:
        sources = []
        # Try to load default synonyms
        default_path = os.path.join(TOKENIZATION_DIR, 'synonyms.json')
        if os.path.exists(default_path):
            sources.append(default_path)

        # Try to load language-specific synonyms
        if language:
            lang_path = os.path.join(TOKENIZATION_DIR, f"{language}_synonyms.json")
            if os.path.exists(lang_path):
                sources.append(lang_path)

    # Normalize to list
    if isinstance(sources, str):
        sources = [sources]

    # Load and merge synonym dictionaries
    for source in sources:
        try:
            if not os.path.exists(source):
                logger.warning(f"Synonym dictionary not found: {source}")
                continue

            with open(source, 'r', encoding='utf-8') as f:
                src_synonyms = json.load(f)

            # Handle different formats
            if isinstance(src_synonyms, dict):
                # Format: {canonical: [synonyms]}
                for canonical, variants in src_synonyms.items():
                    if canonical in synonyms:
                        synonyms[canonical].extend([v for v in variants if v not in synonyms[canonical]])
                    else:
                        synonyms[canonical] = list(variants)
            elif isinstance(src_synonyms, list) and all(isinstance(item, dict) for item in src_synonyms):
                # Format: [{canonical: "term", synonyms: []}]
                for item in src_synonyms:
                    canonical = item.get('canonical', '')
                    variants = item.get('synonyms', [])
                    if canonical:
                        if canonical in synonyms:
                            synonyms[canonical].extend([v for v in variants if v not in synonyms[canonical]])
                        else:
                            synonyms[canonical] = list(variants)

            logger.debug(f"Loaded synonym dictionary from {source}")

        except Exception as e:
            logger.error(f"Error loading synonyms from {source}: {e}")

    return synonyms


@cache_function(ttl=3600, cache_type='file')
def load_ngram_dictionary(sources: Optional[Union[str, List[str]]] = None,
                          language: Optional[str] = None) -> Set[str]:
    """
    Load n-gram dictionary from files.

    Parameters
    ----------
    sources : str or List[str], optional
        Path(s) to n-gram dictionary files. If None, use default.
    language : str, optional
        Language code for language-specific dictionaries.

    Returns
    -------
    Set[str]
        Set of n-grams
    """
    ngrams = set()

    # Set default sources if not provided
    if not sources:
        sources = []
        # Try to load default ngrams
        default_path = os.path.join(TOKENIZATION_DIR, 'ngrams.txt')
        if os.path.exists(default_path):
            sources.append(default_path)

        # Try to load language-specific ngrams
        if language:
            lang_path = os.path.join(TOKENIZATION_DIR, f"{language}_ngrams.txt")
            if os.path.exists(lang_path):
                sources.append(lang_path)

    # Normalize to list
    if isinstance(sources, str):
        sources = [sources]

    # Load and merge ngram dictionaries
    for source in sources:
        try:
            if not os.path.exists(source):
                logger.warning(f"N-gram dictionary not found: {source}")
                continue

            splitext_result = os.path.splitext(source)
            if len(splitext_result) > 1:
                file_ext = splitext_result[1].lower()
            else:
                file_ext = ""  # Или другое значение по умолчанию, если расширение отсутствует

            if file_ext == '.json':
                with open(source, 'r', encoding='utf-8') as f:
                    src_ngrams = json.load(f)

                if isinstance(src_ngrams, list):
                    ngrams.update(src_ngrams)
                elif isinstance(src_ngrams, dict) and 'ngrams' in src_ngrams:
                    ngrams.update(src_ngrams['ngrams'])

            else:  # Default to text file with one n-gram per line
                with open(source, 'r', encoding='utf-8') as f:
                    src_ngrams = [line.strip() for line in f if line.strip()]
                ngrams.update(src_ngrams)

            logger.debug(f"Loaded n-gram dictionary from {source}")

        except Exception as e:
            logger.error(f"Error loading n-grams from {source}: {e}")

    return ngrams


def batch_process(items: List[Any], process_func: Callable, processes: Optional[int] = None, **kwargs) -> List[Any]:
    """
    Process items in parallel using the specified function.
    This is a thin wrapper around base.batch_process for backward compatibility.

    Parameters
    ----------
    items : List[Any]
        Items to process
    process_func : callable
        Function to apply to each item
    processes : int, optional
        Number of processes to use
    **kwargs
        Additional parameters to pass to process_func

    Returns
    -------
    List[Any]
        Processed items
    """
    from pamola_core.utils.nlp.base import batch_process as base_batch_process
    return base_batch_process(items, process_func, processes, **kwargs)


class ProgressTracker:
    """
    Track progress of operations, with optional display.

    This class provides a unified way to track progress for various operations,
    with support for console output or integration with progress bars.
    """

    def __init__(self, total: int, description: str = "", show: bool = False):
        """
        Initialize progress tracker.

        Parameters
        ----------
        total : int
            Total number of items to process
        description : str
            Description of the operation
        show : bool
            Whether to display progress
        """
        self.total = total
        self.description = description
        self.show = show
        self.current = 0
        self._tqdm = None

        # Initialize progress bar if showing
        if self.show:
            try:
                from tqdm import tqdm
                self._tqdm = tqdm(total=total, desc=description)
            except ImportError:
                self._tqdm = None

    def update(self, increment: int = 1) -> None:
        """
        Update progress.

        Parameters
        ----------
        increment : int
            Number of items completed
        """
        self.current += increment

        if self._tqdm:
            self._tqdm.update(increment)
        elif self.show and self.current % max(1, self.total // 100) == 0:
            percent = int(self.current / self.total * 100)
            logger.info(f"{self.description}: {percent}% ({self.current}/{self.total})")

    def close(self) -> None:
        """
        Close the progress tracker.
        """
        if self._tqdm:
            self._tqdm.close()
            self._tqdm = None

        if self.show:
            logger.info(f"{self.description}: Completed ({self.current}/{self.total})")