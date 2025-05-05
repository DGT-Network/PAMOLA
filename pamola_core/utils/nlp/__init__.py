"""
NLP utilities for the HHR project.

This package provides language processing utilities with graceful degradation
when specialized NLP libraries are not available.
"""

from pamola_core.utils.nlp.language import (
    detect_language,
    detect_language_with_confidence,
    detect_mixed_language,
    get_primary_language,
    is_multilingual,
    analyze_language_structure,
    get_supported_languages,
    # normalize_language_code,
    is_cyrillic,
    is_latin,
    detect_languages
)

from pamola_core.utils.nlp.stopwords import (
    get_stopwords,
    remove_stopwords,
    load_stopwords_from_file,
    save_stopwords_to_file,
    combine_stopwords_files,
    setup_nltk
)

from pamola_core.utils.nlp.tokenization import (
    tokenize,
    lemmatize,
    tokenize_and_lemmatize,
    normalize_text,
    normalize_tokens,
    TextProcessor,
    TokenizerFactory,
    # NGramExtractor,
    batch_tokenize,
    batch_tokenize_and_lemmatize
)

from pamola_core.utils.nlp.compatibility import (
    check_dependency,
    # get_module,
    # get_nlp_status,
    log_nlp_status,
    dependency_info,
    get_best_available_module
)

__all__ = [
    # Language detection
    'detect_language',
    'detect_language_with_confidence',
    'detect_mixed_language',
    'get_primary_language',
    'is_multilingual',
    'analyze_language_structure',
    'get_supported_languages',
    # 'normalize_language_code',
    'is_cyrillic',
    'is_latin',
    'detect_languages',

    # Stopwords
    'get_stopwords',
    'remove_stopwords',
    'load_stopwords_from_file',
    'save_stopwords_to_file',
    'combine_stopwords_files',
    'setup_nltk',

    # Tokenization
    'tokenize',
    'lemmatize',
    'tokenize_and_lemmatize',
    'normalize_text',
    'normalize_tokens',
    'TextProcessor',
    'TokenizerFactory',
    # 'NGramExtractor',
    'batch_tokenize',
    'batch_tokenize_and_lemmatize',

    # Compatibility
    'check_dependency',
    # 'get_module',
    # 'get_nlp_status',
    'log_nlp_status',
    'dependency_info',
    'get_best_available_module'
]

# Package version
__version__ = '1.1.0'

# Log NLP status at package import if debugging is enabled
import logging

logger = logging.getLogger(__name__)
if logger.isEnabledFor(logging.DEBUG):
    log_nlp_status()