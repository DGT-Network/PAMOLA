"""
NLP utilities for the project.

This package provides language processing utilities with graceful degradation
when specialized NLP libraries are not available.
"""

# Instead of trying to mock the compatibility module, let's modify the imports
# to directly bypass the problematic import
import importlib
import logging

from pamola_core.utils.nlp.language import (
    detect_language,
    detect_language_with_confidence,
    detect_mixed_language,
    get_primary_language,
    is_multilingual,
    analyze_language_structure,
    get_supported_languages,
    normalize_language_code,
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
    NGramExtractor,
    batch_tokenize,
    batch_tokenize_and_lemmatize
)


# Define the compatibility functions directly
def check_dependency(module_name):
    """Check if a dependency is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_module(module_name):
    """Import a module dynamically with error handling."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def get_nlp_status():
    """Get status of NLP dependencies."""
    status = {}
    for module in ['nltk', 'spacy', 'transformers', 'pymorphy2', 'datasketch']:
        status[module] = check_dependency(module)
    return status


def log_nlp_status():
    """Log the status of NLP dependencies."""
    logger = logging.getLogger(__name__)
    status = get_nlp_status()
    for module, available in status.items():
        logger.debug(f"NLP dependency '{module}': {'Available' if available else 'Not available'}")


def dependency_info(module_name):
    """Get detailed information about a dependency."""
    module = get_module(module_name)
    info = {}

    if module is None:
        info['available'] = False
        return info

    info['available'] = True

    # Get version
    try:
        ver = getattr(module, '__version__', 'unknown')
        info['version'] = str(ver)
    except:
        info['version'] = 'unknown'

    return info


def get_best_available_module(module_names):
    """Get the best available module from a list of alternatives."""
    for name in module_names:
        module = get_module(name)
        if module:
            return module
    return None


# Export all functions
__all__ = [
    # Language detection
    'detect_language',
    'detect_language_with_confidence',
    'detect_mixed_language',
    'get_primary_language',
    'is_multilingual',
    'analyze_language_structure',
    'get_supported_languages',
    'normalize_language_code',
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
    'NGramExtractor',
    'batch_tokenize',
    'batch_tokenize_and_lemmatize',

    # Compatibility
    'check_dependency',
    'get_module',
    'get_nlp_status',
    'log_nlp_status',
    'dependency_info',
    'get_best_available_module'
]

# Package version
__version__ = '1.1.0'

# Log NLP status at package import if debugging is enabled
logger = logging.getLogger(__name__)
if logger.isEnabledFor(logging.DEBUG):
    log_nlp_status()