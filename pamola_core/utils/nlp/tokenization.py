"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Text Tokenization Utilities
Description: Classes and functions for tokenizing text across multiple libraries
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides classes and functions for tokenizing text with support for
multiple libraries (spaCy, NLTK, transformers), lemmatization, n-gram extraction,
and a fallback simple tokenizer.

Key features:
- Multiple tokenization backends (spaCy, NLTK, transformers) with fallbacks
- Customizable tokenization pipeline with preprocessing and filtering
- N-gram extraction with configurable parameters
- Lemmatization across multiple languages
- Token normalization and synonym mapping
- Caching for improved performance
"""

import logging
import re
import string
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple, Set, Iterator

import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

# Import from base module for pamola core functionality
from pamola_core.utils.nlp.base import (
    DependencyManager,
    normalize_language_code,
    batch_process
)
# Import from cache module for caching functionality
from pamola_core.utils.nlp.cache import get_cache, cache_function

# Configure logger
logger = logging.getLogger(__name__)

# Use memory cache for efficient lookup
memory_cache = get_cache('memory')

# Check for external libraries using DependencyManager
_NLTK_AVAILABLE = DependencyManager.check_dependency('nltk')
_PYMORPHY_AVAILABLE = DependencyManager.check_dependency('pymorphy2')
_SPACY_AVAILABLE = DependencyManager.check_dependency('spacy')
_TRANSFORMERS_AVAILABLE = DependencyManager.check_dependency('transformers')

# Global dictionary of lemmatizers or stemmers
_LEMMATIZERS = {}

# If pymorphy2 is available, set up Russian lemmatizer
if _PYMORPHY_AVAILABLE:
    try:
        pymorphy2 = DependencyManager.get_module('pymorphy2')
        if pymorphy2:
            try:
                _LEMMATIZERS['ru'] = pymorphy2.MorphAnalyzer()
                logger.debug("Initialized pymorphy2 MorphAnalyzer for Russian")
            except Exception as e:
                # Catch ANY exception during MorphAnalyzer initialization
                # and downgrade to INFO level to avoid failing the whole process
                logger.info(f"Could not initialize pymorphy2 MorphAnalyzer: {e}")
                # Optional: set a fallback if needed
                # _LEMMATIZERS['ru'] = SomeFallbackStemmer()
    except Exception as e:
        logger.warning(f"Error loading pymorphy2 module: {e}")

# If NLTK is available, set up WordNetLemmatizer for English and optional SnowballStemmer
if _NLTK_AVAILABLE:
    try:
        nltk = DependencyManager.get_module('nltk')
        if nltk:
            # Import required NLTK components
            from nltk.stem import WordNetLemmatizer, SnowballStemmer

            # Ensure NLTK resources (punkt, wordnet) are available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.info("Downloading NLTK 'punkt' resource.")
                nltk.download('punkt', quiet=True)

            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading NLTK 'wordnet' resource.")
                nltk.download('wordnet', quiet=True)

            _LEMMATIZERS['en'] = WordNetLemmatizer()
            logger.debug("Initialized NLTK WordNetLemmatizer for English")

            # Initialize Snowball stemmers for various languages
            for lang in [
                'danish', 'dutch', 'english', 'finnish', 'french', 'german',
                'hungarian', 'italian', 'norwegian', 'portuguese', 'romanian',
                'russian', 'spanish', 'swedish'
            ]:
                try:
                    stemmer = SnowballStemmer(lang)
                    lang_code = {
                        'english': 'en',
                        'russian': 'ru',
                        'german': 'de',
                        'french': 'fr',
                        'spanish': 'es',
                        'italian': 'it',
                        'portuguese': 'pt'
                    }.get(lang, lang[:2])
                    if lang_code not in _LEMMATIZERS:
                        _LEMMATIZERS[lang_code] = stemmer
                        logger.debug(f"Initialized SnowballStemmer for {lang}")
                except Exception as e:
                    logger.debug(f"Could not initialize stemmer for {lang}: {e}")
    except Exception as e:
        logger.warning(f"Error initializing NLTK-based lemmatizers: {e}")

if _SPACY_AVAILABLE:
    logger.debug("spaCy is available (SpacyTokenizer can be used).")


class NGramExtractor:
    """
    Class for extracting n-grams (sequences of n consecutive items) from text or tokens.

    This class provides efficient methods for extracting character or token n-grams
    with various configuration options for handling overlap, padding, and filtering.
    """

    def __init__(
            self,
            n: int = 3,
            lowercase: bool = True,
            min_length: int = 1,
            pad_text: bool = True,
            pad_char: str = "_",
            skip_punctuation: bool = True,
            skip_stopwords: bool = False,
            language: Optional[str] = None,
            stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize the n-gram extractor.

        Parameters:
        -----------
        n : int
            Size of n-grams to extract (default: 3)
        lowercase : bool
            Whether to convert text to lowercase (default: True)
        min_length : int
            Minimum length of tokens to consider (default: 1)
        pad_text : bool
            Whether to pad text with pad_char when extracting character n-grams (default: True)
        pad_char : str
            Character to use for padding (default: "_")
        skip_punctuation : bool
            Whether to skip punctuation when extracting token n-grams (default: True)
        skip_stopwords : bool
            Whether to skip stopwords when extracting token n-grams (default: False)
        language : str, optional
            Language code for stopwords (default: None)
        stopwords : Set[str], optional
            Set of stopwords to skip (default: None)
        """
        self.n = n
        self.lowercase = lowercase
        self.min_length = min_length
        self.pad_text = pad_text
        self.pad_char = pad_char
        self.skip_punctuation = skip_punctuation
        self.skip_stopwords = skip_stopwords

        self.language = language
        self._stopwords: Optional[Set[str]] = stopwords

        if self.skip_stopwords and self._stopwords is None:
            self._load_stopwords()

    def _load_stopwords(self) -> None:
        """
        Load stopwords for the specified language.
        """
        if self.language is None:
            return

        # Try to load stopwords for the language
        try:
            if _NLTK_AVAILABLE:
                nltk = DependencyManager.get_module('nltk')
                if nltk:
                    try:
                        from nltk.corpus import stopwords
                        try:
                            nltk.data.find('corpora/stopwords')
                        except LookupError:
                            logger.info("Downloading NLTK 'stopwords' resource.")
                            nltk.download('stopwords', quiet=True)

                        normalized_lang = normalize_language_code(self.language)
                        lang_map = {
                            'en': 'english',
                            'ru': 'russian',
                            'de': 'german',
                            'fr': 'french',
                            'es': 'spanish',
                            'it': 'italian',
                            'pt': 'portuguese'
                        }
                        nltk_lang = lang_map.get(normalized_lang, normalized_lang)

                        if nltk_lang in stopwords.fileids():
                            self._stopwords = set(stopwords.words(nltk_lang))
                            logger.debug(f"Loaded {len(self._stopwords)} stopwords for {nltk_lang}")
                    except Exception as e:
                        logger.debug(f"Error loading NLTK stopwords: {e}")

            # If still no stopwords, try to load from local resources
            if not self._stopwords:
                # Attempt to load from language resources
                normalized_lang = normalize_language_code(self.language)
                from pamola_core.utils.nlp.language import get_language_resources_path

                resources_path = get_language_resources_path(normalized_lang, 'stopwords')
                if resources_path:
                    try:
                        stopwords_file = f"{resources_path}/stopwords.txt"
                        import os
                        if os.path.exists(stopwords_file):
                            with open(stopwords_file, 'r', encoding='utf-8') as f:
                                self._stopwords = set(line.strip() for line in f if line.strip())
                            logger.debug(f"Loaded {len(self._stopwords)} stopwords from {stopwords_file}")
                    except Exception as e:
                        logger.debug(f"Error loading stopwords from language resources: {e}")

        except Exception as e:
            logger.warning(f"Could not load stopwords for language '{self.language}': {e}")

    def extract_character_ngrams(self, text: str) -> List[str]:
        """
        Extract character n-grams from text.

        Parameters:
        -----------
        text : str
            Text to extract n-grams from

        Returns:
        --------
        List[str]
            List of character n-grams
        """
        if not text:
            return []

        if self.lowercase:
            text = text.lower()

        if self.pad_text:
            # Pad text to generate n-grams that include start and end
            text = self.pad_char * (self.n - 1) + text + self.pad_char * (self.n - 1)

        # Extract n-grams
        ngrams = []
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            ngrams.append(ngram)

        return ngrams

    def extract_token_ngrams(self, tokens: List[str]) -> List[List[str]]:
        """
        Extract token n-grams from a list of tokens.

        Parameters:
        -----------
        tokens : List[str]
            List of tokens to extract n-grams from

        Returns:
        --------
        List[List[str]]
            List of token n-grams, where each n-gram is a list of tokens
        """
        if not tokens or len(tokens) < self.n:
            return []

        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        # Filter tokens by length and type if needed
        filtered_tokens = []
        for token in tokens:
            if len(token) < self.min_length:
                continue

            if self.skip_punctuation and all(c in string.punctuation for c in token):
                continue

            if self.skip_stopwords and self._stopwords and token.lower() in self._stopwords:
                continue

            filtered_tokens.append(token)

        # Extract n-grams
        ngrams = []
        for i in range(len(filtered_tokens) - self.n + 1):
            ngram = filtered_tokens[i:i + self.n]
            ngrams.append(ngram)

        return ngrams

    def extract_token_ngrams_as_strings(self, tokens: List[str], separator: str = ' ') -> List[str]:
        """
        Extract token n-grams and join them with a separator.

        Parameters:
        -----------
        tokens : List[str]
            List of tokens to extract n-grams from
        separator : str
            Separator to join tokens in each n-gram (default: ' ')

        Returns:
        --------
        List[str]
            List of token n-grams as strings
        """
        ngrams = self.extract_token_ngrams(tokens)
        return [separator.join(ngram) for ngram in ngrams]

    def extract_ngrams_from_text(self, text: str, tokenizer=None) -> List[str]:
        """
        Extract token n-grams from text, first tokenizing then extracting n-grams.

        Parameters:
        -----------
        text : str
            Text to extract n-grams from
        tokenizer : callable, optional
            Function to tokenize the text (default: None, uses simple whitespace splitting)

        Returns:
        --------
        List[str]
            List of token n-grams as strings
        """
        if not text:
            return []

        # Tokenize the text
        if tokenizer:
            tokens = tokenizer(text)
        else:
            # Simple whitespace tokenization
            tokens = text.split()

        return self.extract_token_ngrams_as_strings(tokens)

    def iter_character_ngrams(self, text: str) -> Iterator[str]:
        """
        Iterate over character n-grams from text (memory-efficient).

        Parameters:
        -----------
        text : str
            Text to extract n-grams from

        Yields:
        -------
        str
            Character n-grams
        """
        if not text:
            return

        if self.lowercase:
            text = text.lower()

        if self.pad_text:
            # Pad text to generate n-grams that include start and end
            text = self.pad_char * (self.n - 1) + text + self.pad_char * (self.n - 1)

        for i in range(len(text) - self.n + 1):
            yield text[i:i + self.n]

    def get_unique_ngrams(self, text_or_tokens: Union[str, List[str]],
                          is_tokens: bool = False) -> Set[str]:
        """
        Get a set of unique n-grams from text or tokens.

        Parameters:
        -----------
        text_or_tokens : str or List[str]
            Text or list of tokens to extract n-grams from
        is_tokens : bool
            Whether input is already tokenized (default: False)

        Returns:
        --------
        Set[str]
            Set of unique n-grams
        """
        if is_tokens:
            if isinstance(text_or_tokens, list):
                ngrams = self.extract_token_ngrams_as_strings(text_or_tokens)
            else:
                raise ValueError("is_tokens=True but input is not a list")
        else:
            if isinstance(text_or_tokens, str):
                ngrams = self.extract_character_ngrams(text_or_tokens)
            else:
                raise ValueError("is_tokens=False but input is not a string")

        return set(ngrams)

    def count_ngrams(self, text_or_tokens: Union[str, List[str]],
                     is_tokens: bool = False) -> Dict[str, int]:
        """
        Count occurrences of n-grams in text or tokens.

        Parameters:
        -----------
        text_or_tokens : str or List[str]
            Text or list of tokens to extract n-grams from
        is_tokens : bool
            Whether input is already tokenized (default: False)

        Returns:
        --------
        Dict[str, int]
            Dictionary mapping n-grams to their counts
        """
        if is_tokens:
            if isinstance(text_or_tokens, list):
                ngrams = self.extract_token_ngrams_as_strings(text_or_tokens)
            else:
                raise ValueError("is_tokens=True but input is not a list")
        else:
            if isinstance(text_or_tokens, str):
                ngrams = self.extract_character_ngrams(text_or_tokens)
            else:
                raise ValueError("is_tokens=False but input is not a string")

        # Count occurrences
        counts = {}
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1

        return counts

    @staticmethod
    def create_ngram_vocabulary(texts: List[str], n: int,
                                min_count: int = 2,
                                max_vocab_size: Optional[int] = None) -> Dict[str, int]:
        """
        Create a vocabulary of n-grams from multiple texts.

        Parameters:
        -----------
        texts : List[str]
            List of texts to analyze
        n : int
            Size of n-grams to extract
        min_count : int
            Minimum count to include n-gram in vocabulary (default: 2)
        max_vocab_size : int, optional
            Maximum vocabulary size (default: None, no limit)

        Returns:
        --------
        Dict[str, int]
            Dictionary mapping n-grams to their counts in the corpus
        """
        extractor = NGramExtractor(n=n)

        # Extract n-grams from all texts and count them
        ngram_counts = {}
        for text in texts:
            if not text:
                continue

            text_ngrams = extractor.extract_character_ngrams(text)
            for ngram in text_ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Filter by minimum count
        ngram_counts = {ng: count for ng, count in ngram_counts.items() if count >= min_count}

        # Sort by frequency (descending)
        sorted_counts = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)

        # Limit vocabulary size if specified
        if max_vocab_size and len(sorted_counts) > max_vocab_size:
            sorted_counts = sorted_counts[:max_vocab_size]

        return dict(sorted_counts)


class LemmatizerRegistry:
    """
    Registry that allows adding or removing custom lemmatizers for specific languages.
    """
    _lemmatizers: Dict[str, Any] = {}

    @classmethod
    def register(cls, language: str, lemmatizer, overwrite: bool = False):
        """
        Register a custom lemmatizer for a language.

        Parameters
        ----------
        language : str
            Language code
        lemmatizer : Any
            Lemmatizer object with a method to lemmatize words
        overwrite : bool
            Whether to overwrite an existing lemmatizer
        """
        if language in cls._lemmatizers and not overwrite:
            logger.warning(f"Lemmatizer for '{language}' already exists. Use overwrite=True to replace.")
            return
        cls._lemmatizers[language] = lemmatizer
        logger.debug(f"Registered custom lemmatizer for '{language}'")

    @classmethod
    def get(cls, language: str):
        """
        Get a registered lemmatizer for a language.

        Parameters
        ----------
        language : str
            Language code

        Returns
        -------
        Any
            Registered lemmatizer or None if not found
        """
        return cls._lemmatizers.get(language)

    @classmethod
    def has_lemmatizer(cls, language: str) -> bool:
        """
        Check if a lemmatizer is registered for a language.

        Parameters
        ----------
        language : str
            Language code

        Returns
        -------
        bool
            True if a lemmatizer is registered, False otherwise
        """
        return language in cls._lemmatizers

    @classmethod
    def remove(cls, language: str) -> bool:
        """
        Remove a registered lemmatizer.

        Parameters
        ----------
        language : str
            Language code

        Returns
        -------
        bool
            True if a lemmatizer was removed, False otherwise
        """
        if language in cls._lemmatizers:
            del cls._lemmatizers[language]
            logger.debug(f"Removed custom lemmatizer for '{language}'")
            return True
        return False

    @classmethod
    def clear(cls):
        """
        Clear all registered lemmatizers.
        """
        cls._lemmatizers.clear()
        logger.debug("Cleared all custom lemmatizers.")


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers. It includes methods for config loading,
    text preprocessing, stats gathering, and a pipeline for post-processing.
    """

    def __init__(self, language: Optional[str] = None):
        """
        Initialize the tokenizer.

        Parameters
        ----------
        language : str, optional
            Language code. If None, auto-detection will be used.
        """
        self.language = language
        self._config: Optional[Dict[str, Any]] = None
        self._stats: Dict[str, int] = {
            "tokenized_texts": 0,
            "total_tokens": 0
        }

    def load_config(self, config_sources: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Load tokenization config as a dictionary from the given file(s).

        Parameters
        ----------
        config_sources : str or List[str], optional
            Path(s) to configuration files

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        from pamola_core.utils.nlp.tokenization_helpers import load_tokenization_config

        if self._config is None or config_sources is not None:
            self._config = load_tokenization_config(config_sources, self.language)
        return self._config or {}

    def detect_language(self, text: str) -> str:
        """
        Detect language from text or return the predefined language.

        Parameters
        ----------
        text : str
            Text to analyze

        Returns
        -------
        str
            Language code
        """
        if self.language is not None:
            return self.language

        # Import language detection lazily to avoid circular imports
        from pamola_core.utils.nlp.language import detect_language as detect_lang
        return detect_lang(text)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tokenization statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary with statistics
        """
        return dict(self._stats)

    def reset_stats(self):
        """
        Reset tokenization statistics.
        """
        self._stats = {
            "tokenized_texts": 0,
            "total_tokens": 0
        }

    @abstractmethod
    def _core_tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Pamola Core tokenization logic to be implemented by subclasses.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        pass

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Full tokenization pipeline: preprocess, tokenize, apply filters and post-processing.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        if not text:
            return []

        config_sources = kwargs.get('config_sources')
        config = self.load_config(config_sources)

        preserve_case = kwargs.get('preserve_case', config.get('preserve_case', False))
        preserve_patterns = kwargs.get('preserve_patterns', config.get('preserve_patterns'))

        # Preprocess text
        processed_text, protected_map = self.preprocess_text(text, preserve_case, preserve_patterns)

        # Pamola Core tokenize
        tokens = self._core_tokenize(processed_text, **kwargs)

        # Restore placeholders
        tokens = self.restore_protected_patterns(tokens, protected_map)

        # Optional post-processing from config
        if 'token_processing' in config:
            tokens = self._apply_token_processing(tokens, config['token_processing'])

        # min_length filter
        min_length = kwargs.get('min_length', 2)
        tokens = [t for t in tokens if len(t) >= min_length]

        self._stats["tokenized_texts"] += 1
        self._stats["total_tokens"] += len(tokens)

        return tokens

    def preprocess_text(
            self,
            text: str,
            preserve_case: bool = False,
            preserve_patterns: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Preprocess text by replacing patterns with placeholders and adjusting case.

        Parameters
        ----------
        text : str
            Text to preprocess
        preserve_case : bool
            Whether to preserve letter case
        preserve_patterns : List[Dict[str, str]], optional
            Patterns to preserve

        Returns
        -------
        Tuple[str, Dict[str, str]]
            Preprocessed text and map of placeholders to original text
        """
        if not text:
            return "", {}

        protected_map: Dict[str, str] = {}

        if preserve_patterns:
            for i, pattern_def in enumerate(preserve_patterns):
                patt = pattern_def.get('pattern', '')
                if not patt:
                    continue
                found = re.findall(patt, text)
                for j, fragment in enumerate(found):
                    placeholder = f"__PROTECTED_{i}_{j}__"
                    text = text.replace(fragment, placeholder, 1)
                    protected_map[placeholder] = fragment

        if not preserve_case:
            text = text.lower()

        return text, protected_map

    def restore_protected_patterns(
            self,
            tokens: List[str],
            protected_map: Dict[str, str]
    ) -> List[str]:
        """
        Restore placeholders with original fragments.

        Parameters
        ----------
        tokens : List[str]
            Tokens with placeholders
        protected_map : Dict[str, str]
            Map of placeholders to original text

        Returns
        -------
        List[str]
            Tokens with restored patterns
        """
        if not protected_map:
            return tokens

        restored = []
        for tok in tokens:
            restored.append(protected_map.get(tok, tok))
        return restored

    def _apply_token_processing(
            self,
            tokens: List[str],
            processing_config: Dict[str, Any]
    ) -> List[str]:
        """
        Apply post-processing to tokens according to the configuration.

        Parameters
        ----------
        tokens : List[str]
            List of tokens
        processing_config : Dict[str, Any]
            Processing configuration

        Returns
        -------
        List[str]
            Processed tokens
        """
        # 1) Split compound words
        if processing_config.get('split_compound_words'):
            separators = processing_config.get('compound_word_separators', ['-', '/', '_'])
            new_tokens = []
            for t in tokens:
                parts = [t]
                for sep in separators:
                    temp = []
                    for piece in parts:
                        temp.extend(piece.split(sep))
                    parts = temp
                new_tokens.extend(parts)
            tokens = new_tokens

        # 2) preserve_tokens
        if 'preserve_tokens' in processing_config:
            preserve_list = processing_config['preserve_tokens']
            preserve_map = {p.lower(): p for p in preserve_list}
            for i, t in enumerate(tokens):
                low = t.lower()
                if low in preserve_map:
                    tokens[i] = preserve_map[low]

        return tokens


class SimpleTokenizer(BaseTokenizer):
    """
    A whitespace-based fallback tokenizer.
    """

    def _core_tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text using simple whitespace and punctuation rules.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        # Replace punctuation with spaces and split by whitespace
        for punct in string.punctuation:
            text = text.replace(punct, ' ')
        return text.split()


class NLTKTokenizer(BaseTokenizer):
    """
    Uses NLTK's word_tokenize if available, or falls back to SimpleTokenizer.
    """

    def __init__(self, language: Optional[str] = None):
        """
        Initialize the NLTK tokenizer.

        Parameters
        ----------
        language : str, optional
            Language code
        """
        super().__init__(language)
        if not _NLTK_AVAILABLE:
            logger.warning("NLTK unavailable; falling back to SimpleTokenizer.")
            self._fallback = SimpleTokenizer(language)
        else:
            self._fallback: Optional[SimpleTokenizer] = None

    def _core_tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text using NLTK's word_tokenize.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        if self._fallback:
            return self._fallback._core_tokenize(text, **kwargs)

        nltk = DependencyManager.get_module('nltk')
        if not nltk:
            logger.warning("NLTK unavailable; falling back to SimpleTokenizer.")
            if self._fallback is None:
                self._fallback = SimpleTokenizer(self.language)
            return self._fallback._core_tokenize(text, **kwargs)

        from nltk.tokenize import word_tokenize as nltk_word_tokenize
        detected_lang = self.detect_language(text) if self.language is None else self.language
        nltk_lang_map = {'en': 'english', 'ru': 'russian'}
        nltk_lang = nltk_lang_map.get(detected_lang, detected_lang)

        try:
            tokens = nltk_word_tokenize(text, language=nltk_lang)
            return tokens
        except Exception as e:
            logger.debug(f"NLTK tokenization error: {e}, falling back to SimpleTokenizer.")
            if self._fallback is None:
                self._fallback = SimpleTokenizer(self.language)
            return self._fallback._core_tokenize(text, **kwargs)


class SpacyTokenizer(BaseTokenizer):
    """
    Uses spaCy if available, or falls back to SimpleTokenizer.
    """

    def __init__(self, language: Optional[str] = None):
        """
        Initialize the spaCy tokenizer.

        Parameters
        ----------
        language : str, optional
            Language code
        """
        super().__init__(language)
        if not _SPACY_AVAILABLE:
            logger.warning("spaCy unavailable; falling back to SimpleTokenizer.")
            self._fallback = SimpleTokenizer(language)
            self._nlp_models: Dict[str, Any] = {}
        else:
            self._fallback: Optional[SimpleTokenizer] = None
            self._nlp_models: Dict[str, Any] = {}

    def _load_spacy_model(self, lang: str, disable_components: List[str]) -> Any:
        """
        Load a spaCy model for the specified language.

        Parameters
        ----------
        lang : str
            Language code
        disable_components : List[str]
            Components to disable

        Returns
        -------
        Any
            spaCy language model or None if not available
        """
        lang = lang.lower()
        if lang in self._nlp_models:
            return self._nlp_models[lang]

        # Get spaCy module
        spacy = DependencyManager.get_module('spacy')
        if not spacy:
            logger.warning("spaCy unavailable; falling back to SimpleTokenizer.")
            return None

        model_map = {
            'en': 'en_core_web_sm',
            'ru': 'ru_core_news_sm',
            'de': 'de_core_news_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm'
        }
        model_name = model_map.get(lang)
        if not model_name:
            logger.warning(f"No spaCy model known for language '{lang}', fallback to SimpleTokenizer.")
            return None

        try:
            nlp = spacy.load(model_name, disable=disable_components)
            self._nlp_models[lang] = nlp
            return nlp
        except OSError:
            logger.info(f"Downloading spaCy model {model_name}")
            try:
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name, disable=disable_components)
                self._nlp_models[lang] = nlp
                return nlp
            except Exception as e:
                logger.error(f"Failed to download spaCy model '{model_name}': {e}")
                return None

    def _core_tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text using spaCy.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        if self._fallback:
            return self._fallback._core_tokenize(text, **kwargs)

        spacy_model = kwargs.get('spacy_model')
        disable_comps = kwargs.get('disable_spacy_components', ['ner', 'parser'])
        lang = self.language or self.detect_language(text)

        spacy = DependencyManager.get_module('spacy')
        if not spacy:
            if self._fallback is None:
                self._fallback = SimpleTokenizer(self.language)
            return self._fallback._core_tokenize(text, **kwargs)

        if spacy_model:
            try:
                nlp = spacy.load(spacy_model, disable=disable_comps)
            except Exception as e:
                logger.error(f"Failed to load spaCy model '{spacy_model}': {e}")
                if self._fallback is None:
                    self._fallback = SimpleTokenizer(self.language)
                return self._fallback._core_tokenize(text, **kwargs)
        else:
            nlp = self._load_spacy_model(lang, disable_comps)
            if not nlp:
                if self._fallback is None:
                    self._fallback = SimpleTokenizer(self.language)
                return self._fallback._core_tokenize(text, **kwargs)

        try:
            doc = nlp(text)
            return [t.text for t in doc]
        except Exception as e:
            logger.debug(f"SpaCy error: {e}; falling back to SimpleTokenizer.")
            if self._fallback is None:
                self._fallback = SimpleTokenizer(self.language)
            return self._fallback._core_tokenize(text, **kwargs)


class TransformersTokenizer(BaseTokenizer):
    """
    Uses a Hugging Face transformers tokenizer if available, or falls back to SimpleTokenizer.
    """

    def __init__(self, language: Optional[str] = None, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize the transformers tokenizer.

        Parameters
        ----------
        language : str, optional
            Language code
        model_name : str
            Name of the transformers model to use
        """
        super().__init__(language)
        if not _TRANSFORMERS_AVAILABLE:
            logger.warning("transformers unavailable; falling back to SimpleTokenizer.")
            self._fallback = SimpleTokenizer(language)
            self._tokenizer = None
        else:
            self._fallback: Optional[SimpleTokenizer] = None
            self._tokenizer = None
            self._model_name = model_name

    def _init_tokenizer(self, model_name: Optional[str] = None):
        """
        Initialize a transformers tokenizer.

        Parameters
        ----------
        model_name : str, optional
            Name of the model to use

        Returns
        -------
        Any
            Transformers tokenizer or None if not available
        """
        if self._tokenizer is not None:
            return self._tokenizer

        model_name = model_name or self._model_name

        # Get transformers module
        transformers = DependencyManager.get_module('transformers')
        if not transformers:
            logger.warning("transformers unavailable; falling back to SimpleTokenizer.")
            return None

        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            return self._tokenizer
        except Exception as e:
            logger.error(f"Failed to init Transformers tokenizer '{model_name}': {e}")
            return None

    def _core_tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text using a transformers tokenizer.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        if self._fallback:
            return self._fallback._core_tokenize(text, **kwargs)

        model_name = kwargs.get('model_name', self._model_name)
        add_special_tokens = kwargs.get('add_special_tokens', False)
        subword_prefix = kwargs.get('subword_prefix', "##")
        keep_subwords = kwargs.get('keep_subwords', False)
        preserve_case = kwargs.get('preserve_case', False)

        # Convert to lowercase if not preserving case
        if not preserve_case:
            text = text.lower()

        tokenizer = self._init_tokenizer(model_name)
        if not tokenizer:
            logger.debug("No valid Transformers tokenizer found; fallback to SimpleTokenizer.")
            if self._fallback is None:
                self._fallback = SimpleTokenizer(self.language)
            return self._fallback._core_tokenize(text, **kwargs)

        try:
            tokens = tokenizer.tokenize(text, add_special_tokens=add_special_tokens)
            if subword_prefix and not keep_subwords:
                tokens = [t for t in tokens if not t.startswith(subword_prefix)]
            return tokens
        except Exception as e:
            logger.debug(f"Transformers tokenization error: {e}, fallback to SimpleTokenizer.")
            if self._fallback is None:
                self._fallback = SimpleTokenizer(self.language)
            return self._fallback._core_tokenize(text, **kwargs)


class TokenizerFactory:
    """
    A factory for creating tokenizers with an internal cache.
    """

    _tokenizer_cache: Dict[str, BaseTokenizer] = {}

    @classmethod
    def create_tokenizer(
            cls,
            tokenizer_type: str = 'auto',
            language: Optional[str] = None,
            **kwargs
    ) -> BaseTokenizer:
        """
        Create a tokenizer instance with the desired type.
        Caches it unless 'no_cache' is True in kwargs.

        Parameters
        ----------
        tokenizer_type : str
            Type of tokenizer to create
        language : str, optional
            Language code
        **kwargs
            Additional parameters

        Returns
        -------
        BaseTokenizer
            Tokenizer instance
        """
        cache_key = f"{tokenizer_type}_{language or 'auto'}"
        if 'model_name' in kwargs:
            cache_key += f"_{kwargs['model_name']}"

        if kwargs.get('no_cache', False):
            return cls._create_new_tokenizer(tokenizer_type, language, **kwargs)

        if cache_key in cls._tokenizer_cache:
            return cls._tokenizer_cache[cache_key]

        tok = cls._create_new_tokenizer(tokenizer_type, language, **kwargs)
        cls._tokenizer_cache[cache_key] = tok
        return tok

    @classmethod
    def _create_new_tokenizer(
            cls,
            tokenizer_type: str,
            language: Optional[str],
            **kwargs
    ) -> BaseTokenizer:
        """
        Create a new tokenizer instance.

        Parameters
        ----------
        tokenizer_type : str
            Type of tokenizer to create
        language : str, optional
            Language code
        **kwargs
            Additional parameters

        Returns
        -------
        BaseTokenizer
            Tokenizer instance
        """
        if tokenizer_type == 'auto':
            if _SPACY_AVAILABLE:
                return SpacyTokenizer(language)
            elif _NLTK_AVAILABLE:
                return NLTKTokenizer(language)
            elif _TRANSFORMERS_AVAILABLE:
                model_name = kwargs.get('model_name', 'bert-base-multilingual-cased')
                return TransformersTokenizer(language, model_name)
            else:
                return SimpleTokenizer(language)

        if tokenizer_type == 'simple':
            return SimpleTokenizer(language)
        if tokenizer_type == 'nltk':
            return NLTKTokenizer(language)
        if tokenizer_type == 'spacy':
            return SpacyTokenizer(language)
        if tokenizer_type == 'transformers':
            model_name = kwargs.get('model_name', 'bert-base-multilingual-cased')
            return TransformersTokenizer(language, model_name)

        logger.warning(f"Unknown tokenizer_type='{tokenizer_type}', fallback to SimpleTokenizer.")
        return SimpleTokenizer(language)

    @classmethod
    def clear_cache(cls):
        """
        Clear the tokenizer cache.
        """
        cls._tokenizer_cache.clear()
        logger.debug("TokenizerFactory cache cleared.")


@cache_function(ttl=3600, cache_type='memory')
def lemmatize(
        tokens: List[str],
        language: Optional[str] = None,
        dict_sources: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """
    Lemmatize tokens using custom dictionaries or built-in lemmatizers.

    Parameters
    ----------
    tokens : List[str]
        Tokens to lemmatize
    language : str, optional
        Language code
    dict_sources : str or List[str], optional
        Path(s) to lemma dictionaries

    Returns
    -------
    List[str]
        Lemmatized tokens
    """
    if not tokens:
        return []

    if language is None:
        language = 'en'
    language = normalize_language_code(language)

    # 1) Load custom synonyms (like override or expansions)
    custom_lemmas: Dict[str, List[str]] = {}
    if dict_sources:
        from pamola_core.utils.nlp.tokenization_helpers import load_synonym_dictionary
        custom_lemmas = load_synonym_dictionary(dict_sources, language)

    # 2) Apply custom lemma dictionary
    if custom_lemmas:
        adjusted = []
        for token in tokens:
            t_low = token.lower()
            replaced = False
            for lemma_key, variants in custom_lemmas.items():
                if t_low == lemma_key or t_low in variants:
                    adjusted.append(lemma_key)
                    replaced = True
                    break
            if not replaced:
                adjusted.append(token)
        tokens = adjusted

    # 3) Use known lemmatizer or stemmer from _LEMMATIZERS
    if language in _LEMMATIZERS:
        lemma_obj = _LEMMATIZERS[language]
        if language == 'ru' and _PYMORPHY_AVAILABLE:
            return [lemma_obj.parse(t)[0].normal_form for t in tokens]
        elif language == 'en' and _NLTK_AVAILABLE:
            # WordNetLemmatizer
            return [lemma_obj.lemmatize(t) for t in tokens]
        else:
            # Possibly a SnowballStemmer or custom callable
            if hasattr(lemma_obj, 'stem'):
                return [lemma_obj.stem(t) for t in tokens]
            if callable(lemma_obj):
                return [lemma_obj(t) for t in tokens]

    # 4) Check custom lemmatizers in registry
    custom_lemmatizer = LemmatizerRegistry.get(language)
    if custom_lemmatizer:
        if hasattr(custom_lemmatizer, 'lemmatize'):
            return [custom_lemmatizer.lemmatize(t) for t in tokens]
        elif callable(custom_lemmatizer):
            return [custom_lemmatizer(t) for t in tokens]

    logger.debug(f"No lemmatizer available for language '{language}', returning tokens as-is.")
    return tokens


def normalize_text(
        text: str,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        remove_digits: bool = False,
        replacement_rules: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None
) -> str:
    """
    Normalize raw text by applying optional regex replacements, case folding,
    punctuation/digit removal, etc.

    Parameters
    ----------
    text : str
        Text to normalize
    remove_punctuation : bool
        Whether to remove punctuation
    lowercase : bool
        Whether to convert to lowercase
    remove_digits : bool
        Whether to remove digits
    replacement_rules : Dict[str, str] or List[Dict[str, str]], optional
        Regex replacement rules

    Returns
    -------
    str
        Normalized text
    """
    if not text:
        return ""

    # 1) Apply custom replacements
    if replacement_rules:
        if isinstance(replacement_rules, dict):
            for patt, repl in replacement_rules.items():
                text = re.sub(patt, repl, text)
        elif isinstance(replacement_rules, list):
            for rule in replacement_rules:
                pat = rule.get('pattern')
                rep = rule.get('replacement', '')
                if pat:
                    text = re.sub(pat, rep, text)

    # 2) Lowercase
    if lowercase:
        text = text.lower()

    # 3) Remove punctuation
    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

    # 4) Remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # 5) Remove extra whitespace
    text = ' '.join(text.split())

    return text


@cache_function(ttl=3600, cache_type='memory')
def normalize_tokens(
        tokens: List[str],
        synonym_sources: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None
) -> List[str]:
    """
    Normalize tokens to a canonical form using synonyms.

    Parameters
    ----------
    tokens : List[str]
        Tokens to normalize
    synonym_sources : str or List[str], optional
        Path(s) to synonym dictionaries
    language : str, optional
        Language code

    Returns
    -------
    List[str]
        Normalized tokens
    """
    if not tokens:
        return []

    synonyms: Dict[str, List[str]] = {}
    if synonym_sources or language:
        from pamola_core.utils.nlp.tokenization_helpers import load_synonym_dictionary
        synonyms = load_synonym_dictionary(synonym_sources, language)

    if not synonyms:
        return tokens

    normalized = []
    for tk in tokens:
        tk_low = tk.lower()
        replaced = False
        for canonical, variants in synonyms.items():
            if tk_low == canonical or tk_low in variants:
                normalized.append(canonical)
                replaced = True
                break
        if not replaced:
            normalized.append(tk)

    return normalized


def _get_or_detect_language(text: str, language: Optional[str] = None) -> str:
    """
    Get the specified language or detect it from text.

    Parameters
    ----------
    text : str
        Text to analyze
    language : str, optional
        Language code

    Returns
    -------
    str
        Language code
    """
    if language is not None:
        return normalize_language_code(language)

    # Import language detection lazily to avoid circular imports
    from pamola_core.utils.nlp.language import detect_language as detect_lang
    return detect_lang(text)


def tokenize(
        text: str,
        language: Optional[str] = None,
        min_length: int = 2,
        config_sources: Optional[Union[str, List[str]]] = None,
        preserve_case: bool = False,
        preserve_patterns: Optional[List[Dict[str, str]]] = None,
        tokenizer_type: str = 'auto'
) -> List[str]:
    """
    A simple top-level function for tokenizing text.

    Parameters
    ----------
    text : str
        The input text to be tokenized.
    language : str, optional
        Language code (if None, auto-detect).
    min_length : int
        Minimum token length to keep (default = 2).
    config_sources : str or List[str], optional
        Path(s) to config files.
    preserve_case : bool
        Whether to preserve letter case.
    preserve_patterns : List[Dict[str, str]], optional
        Patterns to preserve (e.g., regex placeholders).
    tokenizer_type : str
        'auto', 'nltk', 'spacy', 'simple', etc.

    Returns
    -------
    List[str]
        List of tokens.
    """
    # 1) Create or reuse a tokenizer
    tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, language)

    # 2) Actually tokenize using the tokenizer's .tokenize() method
    tokens = tokenizer.tokenize(
        text,
        min_length=min_length,
        config_sources=config_sources,
        preserve_case=preserve_case,
        preserve_patterns=preserve_patterns
    )
    return tokens


def calculate_word_frequencies(texts: List[str], stop_words: Optional[Set[str]] = None,
                               min_word_length: int = 3, max_words: Optional[int] = None) -> Dict[str, int]:
    """
    Calculate word frequencies across multiple texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings
    stop_words : Set[str], optional
        Set of stop words to exclude
    min_word_length : int
        Minimum word length to include
    max_words : int, optional
        Maximum number of words to include in the result

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping words to their frequencies
    """
    word_counts = {}

    # Process each text
    for text in texts:
        if not text:
            continue

        # Tokenize the text
        tokens = tokenize(text, min_length=min_word_length)

        # Remove stop words if provided
        if stop_words:
            tokens = [t for t in tokens if t.lower() not in stop_words]

        # Count word frequencies
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

    # Sort by frequency (descending)
    sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Limit to max_words if specified
    if max_words and len(sorted_counts) > max_words:
        sorted_counts = sorted_counts[:max_words]

    return dict(sorted_counts)


def calculate_term_frequencies(texts: List[str], language: str = "auto",
                               stop_words: Optional[Set[str]] = None,
                               min_word_length: int = 3, max_terms: Optional[int] = None) -> Dict[str, int]:
    """
    Calculate term frequencies with optional lemmatization.

    Parameters:
    -----------
    texts : List[str]
        List of text strings
    language : str
        Language code or "auto" for detection
    stop_words : Set[str], optional
        Set of stop words to exclude
    min_word_length : int
        Minimum word length to include
    max_terms : int, optional
        Maximum number of terms to include in the result

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping terms to their frequencies
    """
    term_counts = {}

    # Process each text
    for text in texts:
        if not text:
            continue

        # Tokenize and lemmatize the text
        tokens = tokenize_and_lemmatize(
            text,
            language=language,
            min_length=min_word_length
        )

        # Remove stop words if provided
        if stop_words:
            tokens = [t for t in tokens if t.lower() not in stop_words]

        # Count term frequencies
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

    # Sort by frequency (descending)
    sorted_counts = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)

    # Limit to max_terms if specified
    if max_terms and len(sorted_counts) > max_terms:
        sorted_counts = sorted_counts[:max_terms]

    return dict(sorted_counts)


@cache_function(ttl=3600, cache_type='memory')
def tokenize_and_lemmatize(
        text: str,
        language: Optional[str] = None,
        min_length: int = 2,
        config_sources: Optional[Union[str, List[str]]] = None,
        lemma_dict_sources: Optional[Union[str, List[str]]] = None,
        tokenizer_type: str = 'auto'
) -> List[str]:
    """
    Tokenize and then lemmatize text in one step (no separate TokenizerFactory required).

    Parameters
    ----------
    text : str
        The input text to process
    language : str, optional
        Language code (if None, auto-detect)
    min_length : int
        Minimum token length to keep
    config_sources : str or List[str], optional
        Paths to config file(s) for tokenization
    lemma_dict_sources : str or List[str], optional
        Paths to lemma dictionary file(s)
    tokenizer_type : str
        'auto', 'nltk', 'spacy', or 'simple'

    Returns
    -------
    List[str]
        Lemmatized tokens
    """
    if not text:
        return []

    # Detect language if needed
    if language is None:
        language = _get_or_detect_language(text)

    # Get tokens
    tokens = tokenize(
        text,
        language=language,
        min_length=min_length,
        config_sources=config_sources,
        tokenizer_type=tokenizer_type
    )

    # Lemmatize tokens
    return lemmatize(
        tokens=tokens,
        language=language,
        dict_sources=lemma_dict_sources
    )


def batch_tokenize(
        texts: List[str],
        language: Optional[str] = None,
        min_length: int = 2,
        config_sources: Optional[Union[str, List[str]]] = None,
        processes: Optional[int] = None,
        tokenizer_type: str = 'auto'
) -> List[List[str]]:
    """
    Tokenize multiple texts in parallel.

    Parameters
    ----------
    texts : List[str]
        Texts to tokenize
    language : str, optional
        Language code
    min_length : int
        Minimum token length
    config_sources : str or List[str], optional
        Path(s) to config files
    processes : int, optional
        Number of processes to use
    tokenizer_type : str
        Tokenizer type

    Returns
    -------
    List[List[str]]
        List of token lists
    """
    return batch_process(
        texts,
        tokenize,
        processes=processes,
        language=language,
        min_length=min_length,
        config_sources=config_sources,
        tokenizer_type=tokenizer_type
    )


def batch_tokenize_and_lemmatize(
        texts: List[str],
        language: Optional[str] = None,
        min_length: int = 2,
        config_sources: Optional[Union[str, List[str]]] = None,
        lemma_dict_sources: Optional[Union[str, List[str]]] = None,
        processes: Optional[int] = None,
        tokenizer_type: str = 'auto'
) -> List[List[str]]:
    """
    Tokenize and lemmatize multiple texts in parallel.

    Parameters
    ----------
    texts : List[str]
        Texts to process
    language : str, optional
        Language code
    min_length : int
        Minimum token length
    config_sources : str or List[str], optional
        Path(s) to config files
    lemma_dict_sources : str or List[str], optional
        Path(s) to lemma dictionaries
    processes : int, optional
        Number of processes to use
    tokenizer_type : str
        Tokenizer type

    Returns
    -------
    List[List[str]]
        List of lemmatized token lists
    """
    return batch_process(
        texts,
        tokenize_and_lemmatize,
        processes=processes,
        language=language,
        min_length=min_length,
        config_sources=config_sources,
        lemma_dict_sources=lemma_dict_sources,
        tokenizer_type=tokenizer_type
    )


class TextProcessor:
    """
    High-level class that ties together tokenization, lemmatization, n-gram extraction, etc.
    """

    def __init__(
            self,
            language: Optional[str] = None,
            tokenizer_type: str = 'auto',
            config_sources: Optional[Union[str, List[str]]] = None,
            lemma_dict_sources: Optional[Union[str, List[str]]] = None,
            min_token_length: int = 2,
            preserve_case: bool = False
    ):
        """
        Initialize the text processor.

        Parameters
        ----------
        language : str, optional
            Language code
        tokenizer_type : str
            Type of tokenizer to use
        config_sources : str or List[str], optional
            Path(s) to config files
        lemma_dict_sources : str or List[str], optional
            Path(s) to lemma dictionaries
        min_token_length : int
            Minimum token length
        preserve_case : bool
            Whether to preserve case
        """
        self.language = language
        self.tokenizer_type = tokenizer_type
        self.config_sources = config_sources
        self.lemma_dict_sources = lemma_dict_sources
        self.min_token_length = min_token_length
        self.preserve_case = preserve_case
        # We can create a tokenizer up front
        self.tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type, language)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text using the configured tokenizer.

        Parameters
        ----------
        text : str
            Text to tokenize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            List of tokens
        """
        # Merge default arguments with user-provided ones
        actual_kwargs = {
            'min_length': self.min_token_length,
            'preserve_case': self.preserve_case,
            'config_sources': self.config_sources
        }
        actual_kwargs.update(kwargs)
        return self.tokenizer.tokenize(text, **actual_kwargs)

    def lemmatize(self, tokens: List[str], **kwargs) -> List[str]:
        """
        Lemmatize tokens using an optional lemma dictionary.

        Parameters
        ----------
        tokens : List[str]
            Tokens to lemmatize
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            Lemmatized tokens
        """
        lang = kwargs.get('language', self.language)
        lemma_srcs = kwargs.get('lemma_dict_sources', self.lemma_dict_sources)
        return lemmatize(tokens, lang, dict_sources=lemma_srcs)

    def tokenize_and_lemmatize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize and lemmatize text in one call.

        Parameters
        ----------
        text : str
            Text to process
        **kwargs
            Additional parameters

        Returns
        -------
        List[str]
            Lemmatized tokens
        """
        tokens = self.tokenize(text, **kwargs)
        return self.lemmatize(tokens, **kwargs)

    def process_text(
            self,
            text: str,
            lemmatize_tokens: bool = True,
            extract_ngrams_flag: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Full pipeline: tokenize -> (optional) lemmatize

        Parameters
        ----------
        text : str
            Text to process
        lemmatize_tokens : bool
            Whether to lemmatize tokens
        extract_ngrams_flag : bool
            Whether to extract n-grams (will be imported from tokenization_ext)
        **kwargs
            Additional parameters

        Returns
        -------
        Dict[str, Any]
            Processing results
        """
        result: Dict[str, Any] = {"text": text, "length": len(text)}

        if not text:
            result["tokens"] = []
            result["token_count"] = 0
            return result

        # Determine language
        lang = kwargs.get('language', self.language)
        if lang is None:
            # Import language detection lazily to avoid circular imports
            from pamola_core.utils.nlp.language import detect_language as detect_lang
            lang = detect_lang(text)
            result['detected_language'] = lang
        else:
            result['language'] = lang

        # 1) Tokenize
        tokens = self.tokenize(text, language=lang, **kwargs)
        result['tokens'] = tokens
        result['token_count'] = len(tokens)

        # 2) Lemmatize if requested
        if lemmatize_tokens and tokens:
            lemmas = self.lemmatize(tokens, language=lang, **kwargs)
            result['lemmas'] = lemmas
            result['lemma_count'] = len(lemmas)

        # 3) Extract n-grams if requested
        if extract_ngrams_flag and tokens:
            # Extract n-grams using our internal NGramExtractor
            ngram_sizes = kwargs.get('ngram_sizes', [2, 3])
            result['ngrams'] = {}

            for n in ngram_sizes:
                # Create extractor with the right size
                extractor = NGramExtractor(
                    n=n,
                    language=lang,
                    lowercase=not kwargs.get('preserve_case', False),
                    skip_stopwords=kwargs.get('skip_stopwords', False)
                )

                # Get n-grams as strings
                ngrams = extractor.extract_token_ngrams_as_strings(tokens)
                result['ngrams'][n] = ngrams

        return result


# Helper functions for NGram extraction that can be used directly without creating an extractor

def extract_ngrams(
       tokens: List[str],
       n: int = 3,
       language: Optional[str] = None,
       lowercase: bool = True,
       min_length: int = 1,
       as_strings: bool = True,
       separator: str = ' ',
       skip_stopwords: bool = False
) -> Union[List[str], List[List[str]]]:
   """
   Extract n-grams from a list of tokens.

   This is a convenience function that uses NGramExtractor internally.

   Parameters:
   -----------
   tokens : List[str]
       List of tokens to extract n-grams from
   n : int
       Size of n-grams to extract (default: 3)
   language : str, optional
       Language code for stopwords (default: None)
   lowercase : bool
       Whether to convert tokens to lowercase (default: True)
   min_length : int
       Minimum length of tokens to consider (default: 1)
   as_strings : bool
       Whether to return n-grams as strings or token lists (default: True)
   separator : str
       Separator to join tokens in each n-gram if as_strings=True (default: ' ')
   skip_stopwords : bool
       Whether to skip stopwords when extracting n-grams (default: False)

   Returns:
   --------
   Union[List[str], List[List[str]]]
       List of n-grams (as strings or token lists)
   """
   extractor = NGramExtractor(
       n=n,
       lowercase=lowercase,
       min_length=min_length,
       skip_stopwords=skip_stopwords,
       language=language
   )

   if as_strings:
       return extractor.extract_token_ngrams_as_strings(tokens, separator)
   else:
       return extractor.extract_token_ngrams(tokens)


def extract_character_ngrams(text: str, n: int = 3, lowercase: bool = True,
                            pad_text: bool = True) -> List[str]:
   """
   Extract character n-grams from text.

   This is a convenience function that uses NGramExtractor internally.

   Parameters:
   -----------
   text : str
       Text to extract n-grams from
   n : int
       Size of n-grams to extract (default: 3)
   lowercase : bool
       Whether to convert text to lowercase (default: True)
   pad_text : bool
       Whether to pad text with underscore characters (default: True)

   Returns:
   --------
   List[str]
       List of character n-grams
   """
   extractor = NGramExtractor(
       n=n,
       lowercase=lowercase,
       pad_text=pad_text
   )

   return extractor.extract_character_ngrams(text)