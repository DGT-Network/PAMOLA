"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Language Detection and Processing
Description: Utilities for language detection, normalization, and multilingual text analysis
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides enhanced functions for detecting text language with
graceful degradation when specialized libraries are not available.

Key features:
- Multi-layer language detection (FastText, langdetect, heuristics)
- Confidence scoring for detected languages
- Mixed language analysis and multilingual text detection
- Script detection (Latin, Cyrillic, CJK, etc.)
- Language code normalization and standardization
- Cached results for improved performance
"""

import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

# Import from base to avoid circular dependencies
from pamola_core.utils.nlp.base import DependencyManager
from pamola_core.utils.nlp.cache import get_cache, cache_function

# Configure logger
logger = logging.getLogger(__name__)

# Try to import langdetect with graceful degradation
_LANGDETECT_AVAILABLE = DependencyManager.check_dependency('langdetect')
if _LANGDETECT_AVAILABLE:
    from langdetect import detect, DetectorFactory

    # Set seed for consistent language detection
    DetectorFactory.seed = 0

# Check for FastText availability
_FASTTEXT_AVAILABLE = DependencyManager.check_dependency('fasttext')

# Simplified language identification patterns
_LANGUAGE_PATTERNS = {
    'ru': re.compile(r'[а-яА-ЯёЁ]'),  # Russian Cyrillic characters
    'en': re.compile(r'[a-zA-Z]'),  # English Latin characters
    'de': re.compile(r'[äöüßÄÖÜ]'),  # German specific characters
    'fr': re.compile(r'[éèêëàâçîïôùûüÿæœÉÈÊËÀÂÇÎÏÔÙÛÜŸÆŒ]'),  # French specific characters
    'es': re.compile(r'[áéíóúüñÁÉÍÓÚÜÑ]'),  # Spanish specific characters
    'zh': re.compile(r'[\u4e00-\u9fff]'),  # Chinese characters
    'ja': re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),  # Japanese characters
}

# Words unique to specific languages
_LANGUAGE_WORDS = {
    'ru': {'и', 'в', 'не', 'на', 'я', 'быть', 'с', 'он', 'что', 'а', 'по', 'это'},
    'en': {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'it', 'for', 'with'},
    'de': {'der', 'die', 'das', 'und', 'ist', 'von', 'mit', 'den', 'für', 'nicht'},
    'fr': {'le', 'la', 'les', 'des', 'un', 'une', 'et', 'est', 'pour', 'dans'},
    'es': {'el', 'la', 'que', 'de', 'y', 'a', 'en', 'un', 'ser', 'por'}
}

# Language code mappings for normalization
_LANGUAGE_CODE_MAP = {
    # ISO 639-1 to standard two-letter codes
    'eng': 'en', 'rus': 'ru', 'deu': 'de', 'fra': 'fr', 'spa': 'es',
    'ita': 'it', 'jpn': 'ja', 'zho': 'zh', 'kor': 'ko', 'ara': 'ar',
    'hin': 'hi', 'por': 'pt', 'ben': 'bn', 'nld': 'nl', 'tur': 'tr',

    # Common alternative codes
    'english': 'en', 'russian': 'ru', 'german': 'de', 'french': 'fr', 'spanish': 'es',
    'italian': 'it', 'japanese': 'ja', 'chinese': 'zh', 'korean': 'ko', 'arabic': 'ar',
    'hindi': 'hi', 'portuguese': 'pt', 'bengali': 'bn', 'dutch': 'nl', 'turkish': 'tr',

    # Handle regional variants
    'en_us': 'en', 'en_gb': 'en', 'en_ca': 'en', 'en_au': 'en',
    'fr_ca': 'fr', 'fr_be': 'fr', 'fr_ch': 'fr',
    'de_at': 'de', 'de_ch': 'de',
    'pt_br': 'pt', 'pt_pt': 'pt',
    'zh_cn': 'zh', 'zh_tw': 'zh', 'zh_hk': 'zh'
}


def normalize_language_code(language_code: str, default_code: str = 'en') -> str:
    """
    Normalize a language code to a standard ISO 639-1 format.

    Handles various input formats including ISO 639-1, ISO 639-2, country-specific
    codes (en-US, fr-CA), and converts them to standard two-letter codes.

    Parameters:
    -----------
    language_code : str
        Language code to normalize (e.g., 'en-US', 'en_us', 'eng', etc.)
    default_code : str
        Default language code to return if input is invalid

    Returns:
    --------
    str
        Normalized language code (e.g., 'en', 'ru', 'de')
    """
    if not language_code:
        return default_code

    # Convert to lowercase and strip whitespace
    code = language_code.lower().strip()

    # Check direct mapping first
    if code in _LANGUAGE_CODE_MAP:
        return _LANGUAGE_CODE_MAP[code]

    # Remove country code if present using standard separators
    for separator in ['-', '_']:
        if separator in code:
            code = code.split(separator)[0]
            break

    # Now check if the base code is in our map
    if code in _LANGUAGE_CODE_MAP:
        return _LANGUAGE_CODE_MAP[code]

    # If it's already a standard two-letter code, return it
    if len(code) == 2 and code.isalpha():
        return code

    # If code is in ISO 639-2 format (3 letters), try to convert
    if len(code) == 3 and code.isalpha():
        for iso2, iso3 in [('en', 'eng'), ('ru', 'rus'), ('de', 'deu'),
                           ('fr', 'fra'), ('es', 'spa')]:
            if code == iso3:
                return iso2

    # Return as-is if we couldn't normalize, or use default
    return code if code else default_code


# Function to load FastText model when needed
def _get_fasttext_model():
    """Get a cached FastText model for language detection."""
    if not _FASTTEXT_AVAILABLE:
        return None

    cache = get_cache('model')
    model_key = 'fasttext_language_detection'

    # Try to get from cache
    model = cache.get(model_key)
    if model is not None:
        return model

    # Load the model
    try:
        import fasttext

        # Check if the model exists, if not download it
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'resources', 'models', 'lid.176.bin'
        )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if not os.path.exists(model_path):
            # Download model from official repository
            import urllib.request
            logger.info("Downloading FastText language identification model...")
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin',
                model_path
            )

        # Load the model
        model = fasttext.load_model(model_path)

        # Cache the model
        cache.set(model_key, model, metadata={'type': 'fasttext', 'task': 'language-detection'})
        return model
    except Exception as e:
        logger.warning(f"Error loading FastText model: {e}")
        return None


@cache_function(ttl=3600, cache_type='memory')
def detect_language_with_confidence(text: str, default_language: str = 'en') -> Tuple[str, float]:
    """
    Detect the language of a text with confidence score.

    Uses FastText if available, then langdetect, and falls back to character-based heuristics.

    Parameters:
    -----------
    text : str
        Text to analyze
    default_language : str
        Default language to return if detection fails

    Returns:
    --------
    Tuple[str, float]
        Detected language code ('en', 'ru', etc.) and confidence score (0-1)
    """
    if not text:
        return default_language, 0.0

    # Try to use FastText for detection if available
    if _FASTTEXT_AVAILABLE:
        fasttext_model = _get_fasttext_model()
        if fasttext_model:
            try:
                # FastText requires newlines to be replaced
                processed_text = text.replace('\n', ' ')
                predictions = fasttext_model.predict(processed_text)
                lang_code = predictions[0][0].replace('__label__', '')
                confidence = predictions[1][0]
                return lang_code, float(confidence)
            except Exception as e:
                logger.debug(f"FastText language detection failed: {e}")
                # Fall back to other methods

    # Try langdetect if available
    if _LANGDETECT_AVAILABLE:
        try:
            return detect(text), 0.8  # langdetect doesn't provide confidence, using estimated value
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            # Fall back to heuristic detection

    # Basic heuristic for multiple languages based on character sets
    lang_chars = {}
    for lang, pattern in _LANGUAGE_PATTERNS.items():
        lang_chars[lang] = len(re.findall(pattern, text))

    # If we have matches
    if sum(lang_chars.values()) > 0:
        # Find language with most character matches
        best_lang = max(lang_chars.items(), key=lambda x: x[1])
        total_chars = sum(lang_chars.values())
        confidence = best_lang[1] / total_chars if total_chars > 0 else 0

        if best_lang[1] > 0:
            return best_lang[0], confidence

    # If character analysis failed, try word matching
    words = text.lower().split()
    if words:
        matches = {}
        for lang, lang_words in _LANGUAGE_WORDS.items():
            matches[lang] = sum(1 for word in words if word in lang_words)

        if sum(matches.values()) > 0:
            best_match = max(matches.items(), key=lambda x: x[1])
            total_matches = sum(matches.values())
            confidence = best_match[1] / len(words) if words else 0

            if best_match[1] > 0:
                return best_match[0], confidence

    return default_language, 0.0


@cache_function(ttl=3600, cache_type='memory')
def detect_language(text: str, default_language: str = 'en') -> str:
    """
    Detect the language of a text.

    Uses langdetect if available, otherwise falls back to character-based heuristics.
    This is a backward compatible function that maintains the original interface.

    Parameters:
    -----------
    text : str
        Text to analyze
    default_language : str
        Default language to return if detection fails

    Returns:
    --------
    str
        Detected language code ('en', 'ru', etc.)
    """
    lang, _ = detect_language_with_confidence(text, default_language)
    return lang


def detect_mixed_language(text: str, min_segment_length: int = 10) -> Dict[str, float]:
    """
    Analyze text with potentially mixed languages and determine language proportions.

    Parameters:
    -----------
    text : str
        Text to analyze
    min_segment_length : int
        Minimum length of text segments to analyze

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping language codes to their proportions
    """
    if not text:
        return {}

    # Split text into sentences or segments for analysis
    segments = [s.strip() for s in re.split(r'[.!?।\n]+', text) if len(s.strip()) >= min_segment_length]

    if not segments:
        # If no valid segments, analyze the whole text
        lang, conf = detect_language_with_confidence(text)
        return {lang: 1.0}

    # Detect language for each segment
    segment_langs = []
    for segment in segments:
        lang, conf = detect_language_with_confidence(segment)
        # Only include confident detections
        if conf > 0.3:
            segment_langs.append((lang, len(segment)))

    if not segment_langs:
        # Fallback if no confident detections
        lang, _ = detect_language_with_confidence(text)
        return {lang: 1.0}

    # Calculate proportions based on text length
    total_length = sum(length for _, length in segment_langs)
    proportions = {}

    for lang, length in segment_langs:
        proportions[lang] = proportions.get(lang, 0) + (length / total_length)

    return proportions


def get_primary_language(text: str, threshold: float = 0.6, default_language: str = 'en') -> str:
    """
    Determine primary language of possibly mixed-language text.

    Parameters:
    -----------
    text : str
        Text to analyze
    threshold : float
        Threshold proportion for considering a language primary
    default_language : str
        Default language if no language meets the threshold

    Returns:
    --------
    str
        Primary language code
    """
    if not text:
        return default_language

    proportions = detect_mixed_language(text)

    # Find language with highest proportion
    if proportions:
        primary_lang, proportion = max(proportions.items(), key=lambda x: x[1])

        # Return primary language if it meets threshold
        if proportion >= threshold:
            return primary_lang

    # If no language meets threshold or no proportions, use standard detection
    return detect_language(text, default_language)


def is_multilingual(text: str, threshold: float = 0.2) -> bool:
    """
    Determine if text is multilingual.

    Parameters:
    -----------
    text : str
        Text to analyze
    threshold : float
        Minimum proportion for a language to be considered significant

    Returns:
    --------
    bool
        True if text contains multiple significant languages
    """
    proportions = detect_mixed_language(text)

    # Count languages with proportion above threshold
    significant_langs = sum(1 for prop in proportions.values() if prop >= threshold)

    return significant_langs > 1


def analyze_language_structure(text: str) -> Dict[str, Any]:
    """
    Perform comprehensive language analysis of text.

    Parameters:
    -----------
    text : str
        Text to analyze

    Returns:
    --------
    Dict[str, Any]
        Comprehensive language analysis including:
        - primary_language: Primary language code
        - is_multilingual: Whether text is multilingual
        - language_proportions: Proportions of different languages
        - confidence: Overall confidence in the analysis
        - script_info: Information about writing scripts used
    """
    if not text:
        return {
            "primary_language": "unknown",
            "is_multilingual": False,
            "language_proportions": {},
            "confidence": 0.0,
            "script_info": {}
        }

    # Detect language proportions
    proportions = detect_mixed_language(text)

    # Determine if multilingual
    is_multi = sum(1 for prop in proportions.values() if prop >= 0.2) > 1

    # Get primary language
    primary_lang = "unknown"
    max_prop = 0.0

    if proportions:
        primary_lang, max_prop = max(proportions.items(), key=lambda x: x[1])

    # Analyze scripts (writing systems)
    scripts = {}
    for script, pattern in {
        "latin": re.compile(r'[a-zA-Z]'),
        "cyrillic": re.compile(r'[а-яА-ЯёЁ]'),
        "cjk": re.compile(r'[\u4e00-\u9fff\u3040-\u30FF\u3400-\u4DBF]'),
        "arabic": re.compile(r'[\u0600-\u06FF]'),
        "devanagari": re.compile(r'[\u0900-\u097F]')
    }.items():
        match_count = len(re.findall(pattern, text))
        if match_count > 0:
            scripts[script] = match_count

    # Calculate script proportions
    total_script_chars = sum(scripts.values())
    script_proportions = {k: v / total_script_chars for k, v in scripts.items()} if total_script_chars > 0 else {}

    return {
        "primary_language": primary_lang,
        "is_multilingual": is_multi,
        "language_proportions": proportions,
        "confidence": max_prop,
        "script_info": {
            "scripts": list(scripts.keys()),
            "script_proportions": script_proportions
        }
    }


def get_language_resources_path(language: str, resource_type: str = 'general') -> Optional[str]:
    """
    Get the path to language-specific resources.

    Parameters:
    -----------
    language : str
        Language code ('en', 'ru', etc.)
    resource_type : str
        Type of resource ('general', 'stopwords', 'dictionaries', etc.)

    Returns:
    --------
    str or None
        Path to the resource directory or None if not available
    """
    # Base directory for language resources
    base_dir = os.environ.get(
        'PAMOLA_LANGUAGE_RESOURCES_DIR',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'languages')
    )

    # Language-specific directory
    lang_dir = os.path.join(base_dir, language.lower())

    # Resource type directory
    resource_dir = os.path.join(lang_dir, resource_type)

    if os.path.exists(resource_dir):
        return resource_dir
    elif os.path.exists(lang_dir):
        return lang_dir

    return None


def get_supported_languages() -> List[str]:
    """
    Get list of languages with support in the system.

    Returns:
    --------
    List[str]
        List of supported language codes
    """
    # Basic supported languages
    languages = ['en', 'ru', 'de', 'fr', 'es']

    # Check for additional languages with resources
    base_dir = os.environ.get(
        'PAMOLA_LANGUAGE_RESOURCES_DIR',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'languages')
    )

    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            # Only include directories that represent language codes
            if os.path.isdir(os.path.join(base_dir, item)) and len(item) <= 5:
                if item.lower() not in languages:
                    languages.append(item.lower())

    return languages


def is_cyrillic(text: str) -> bool:
    """
    Check if text contains Cyrillic characters.

    Parameters:
    -----------
    text : str
        Text to check

    Returns:
    --------
    bool
        True if text contains Cyrillic characters
    """
    return bool(re.search(_LANGUAGE_PATTERNS['ru'], text))


def is_latin(text: str) -> bool:
    """
    Check if text contains Latin characters.

    Parameters:
    -----------
    text : str
        Text to check

    Returns:
    --------
    bool
        True if text contains Latin characters
    """
    return bool(re.search(_LANGUAGE_PATTERNS['en'], text))


def get_language_from_file(file_path: str) -> str:
    """
    Detect language of a text file.

    Parameters:
    -----------
    file_path : str
        Path to the text file

    Returns:
    --------
    str
        Detected language code
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read(1000)  # Read first 1000 chars for detection
        return detect_language(text)
    except Exception as e:
        logger.error(f"Error detecting language from file: {e}")
        return 'en'  # Default to English


def detect_languages(texts: List[str], sample_size: int = 100, default_language: str = 'en') -> Dict[str, float]:
    """
    Detect languages used in a sample of texts.

    Parameters:
    -----------
    texts : List[str]
        List of text strings to analyze
    sample_size : int
        Number of texts to sample for language detection (0 for all)
    default_language : str
        Default language to use if detection fails

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping language codes to their frequencies
    """
    # Filter out None and empty texts
    valid_texts = [text for text in texts if text]

    if not valid_texts:
        return {default_language: 1.0}

    # Sample texts if there are too many
    if sample_size and len(valid_texts) > sample_size:
        import random
        random.seed(42)  # For reproducibility
        sampled_texts = random.sample(valid_texts, sample_size)
    else:
        sampled_texts = valid_texts

    # Detect languages in batches for efficiency
    max_batch_size = 100  # Process in batches to avoid memory issues
    languages = []

    for i in range(0, len(sampled_texts), max_batch_size):
        batch = sampled_texts[i:min(i + max_batch_size, len(sampled_texts))]

        # Process each text in the batch
        for text in batch:
            lang = detect_language(text, default_language)
            languages.append(lang)

    # Calculate frequencies
    language_counter = Counter(languages)
    total_count = len(languages)

    return {lang: count / total_count for lang, count in language_counter.items()}