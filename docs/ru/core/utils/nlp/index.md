# PAMOLA.CORE NLP Package Documentation

## Overview

The `pamola_core.utils.nlp` package provides a comprehensive set of natural language processing (NLP) utilities designed for the PAMOLA.CORE (Privacy-Preserving AI Data Processors) project. This package enables robust text analysis with graceful degradation when specialized NLP libraries are unavailable, making it ideal for processing multilingual resume and job posting data in diverse environments.

The package is designed with flexibility, extensibility, and resilience in mind, allowing it to work effectively even in constrained environments while leveraging advanced NLP capabilities when available.

## Package Architecture

The package is structured into four primary modules:

1. **compatibility.py**: Manages dependency checking and graceful degradation
2. **language.py**: Provides language detection and multilingual text analysis
3. **stopwords.py**: Handles stopword management across multiple languages
4. **tokenization.py**: Offers flexible text tokenization and lemmatization

Supporting modules include:
- **tokenization_helpers.py**: Utilities for tokenization resource management

The architecture follows a layered approach, where each module can operate independently while also working together seamlessly:

```
┌────────────────────────────────────────────────────────────────┐
│                      pamola_core.utils.nlp                             │
└────────────────────────────────────────────────────────────────┘
          │                │                │               │
┌─────────▼──────┐  ┌─────▼─────┐  ┌───────▼────────┐ ┌────▼─────┐
│ compatibility  │  │  language  │  │    stopwords   │ │tokenization│
└────────────────┘  └────────────┘  └────────────────┘ └────────────┘
          ▲                                              │
          │                                              ▼
          │                                     ┌────────────────────┐
          └─────────────────────────────────────┤tokenization_helpers│
                                                └────────────────────┘
```

## Key Features

### 1. Graceful Degradation

All modules implement multiple fallback strategies to ensure functionality even when advanced NLP libraries are unavailable:

- **Primary Strategy**: Utilize specialized libraries (NLTK, spaCy, langdetect, FastText, etc.)
- **Secondary Strategy**: Fall back to simpler methods when primary libraries are unavailable
- **Baseline Strategy**: Ensure minimal functionality using only the Python standard library

### 2. Multilingual Support

The package provides comprehensive support for multiple languages:

- **Language Detection**: Accurate identification of text language with confidence scores
- **Mixed Language Analysis**: Detection and analysis of texts containing multiple languages
- **Language-specific Resources**: Management of language-specific stopwords and tokenization rules
- **Script Analysis**: Detection and quantification of different writing systems (Latin, Cyrillic, CJK, etc.)

### 3. Flexible Text Processing

The tokenization module offers multiple approaches to text processing:

- **Multiple Tokenizer Types**: SimpleTokenizer, NLTKTokenizer, SpacyTokenizer, TransformersTokenizer
- **Lemmatization**: Language-specific lemmatization with customizable dictionaries
- **N-gram Extraction**: Generation and filtering of n-grams of various sizes
- **Pattern Preservation**: Preservation of special patterns (URLs, emails, etc.) during tokenization

### 4. Resource Management

The package includes sophisticated resource management capabilities:

- **Resource Discovery**: Automatic discovery of language resources in configured directories
- **Multiple Formats**: Support for resources in various formats (JSON, CSV, text)
- **Resource Combination**: Merging of resources from multiple sources
- **Caching**: Efficient caching of resources and results for performance optimization

## Dependency Management

The compatibility module serves as the foundation for the package's resilience:

- **Dependency Checking**: Verification of NLP library availability at runtime
- **Graceful Fallbacks**: Selection of appropriate alternatives when preferred libraries are unavailable
- **Version Compatibility**: Verification of compatible library versions
- **Status Reporting**: Comprehensive reporting of available NLP capabilities

## Module Highlights

### 1. Language Module

The language module provides:

- **Multi-method Detection**: Neural, statistical, and heuristic-based language identification
- **Confidence Scoring**: Quantitative assessment of detection reliability
- **Multilingual Analysis**: Segmentation and proportion analysis for mixed-language texts
- **Script Analysis**: Identification of writing systems used in texts

### 2. Stopwords Module

The stopwords module offers:

- **Multi-source Loading**: Loading from files, directories, in-memory sets, and NLTK
- **Flexible Language Support**: Language-specific stopword collection
- **File Management**: Automatic discovery and combination of stopword sources
- **Efficient Filtering**: Fast removal of stopwords from token lists

### 3. Tokenization Module

The tokenization module provides:

- **Tokenization Interface**: Common interface across multiple tokenizer implementations
- **Advanced Processing**: Lemmatization, normalization, and n-gram extraction
- **Resource Efficiency**: Caching and parallel processing optimizations
- **High-level API**: TextProcessor class for streamlined text analysis

## Usage Examples

### Basic Usage

```python
from pamola_core.utils.nlp import detect_language, tokenize, get_stopwords, remove_stopwords

# Language detection
text = "The quick brown fox jumps over the lazy dog."
language = detect_language(text)
print(f"Detected language: {language}")

# Tokenization
tokens = tokenize(text, language=language)
print(f"Tokens: {tokens}")

# Stopword removal
stopwords = get_stopwords([language])
filtered_tokens = remove_stopwords(tokens, stopwords)
print(f"Filtered tokens: {filtered_tokens}")
```

### Advanced Language Analysis

```python
from pamola_core.utils.nlp import analyze_language_structure

mixed_text = """First paragraph in English.
Второй абзац на русском языке.
Third paragraph in English again."""

analysis = analyze_language_structure(mixed_text)
print(f"Primary language: {analysis['primary_language']}")
print(f"Is multilingual: {analysis['is_multilingual']}")
print(f"Language proportions: {analysis['language_proportions']}")
```

### Comprehensive Text Processing

```python
from pamola_core.utils.nlp import TextProcessor

processor = TextProcessor(tokenizer_type="auto")

text = "Senior Software Engineer with 5+ years of experience in Python and JavaScript"

# Process text with all available features
result = processor.process_text(
    text,
    lemmatize_tokens=True,
    extract_ngrams=True,
    ngram_sizes=[2, 3]
)

print(f"Detected language: {result.get('detected_language')}")
print(f"Tokens: {result['tokens']}")
print(f"Lemmas: {result['lemmas']}")
print(f"Bigrams: {result['ngrams'][2]}")
```

### Dependency Checking

```python
from pamola_core.utils.nlp import get_nlp_status, dependency_info

# Check basic NLP library availability
status = get_nlp_status()
for module, available in status.items():
    print(f"{module}: {'Available' if available else 'Not available'}")

# Get detailed dependency information
details = dependency_info(verbose=True)
print(f"Python version: {details['python_version']}")
print(f"Available libraries: {details['count']['available']}/{details['count']['total']}")
```

## Package Dependencies

### Pamola Core Dependencies

- Python 3.6+
- Standard library modules (re, os, json, logging, etc.)

### Optional Dependencies

For enhanced functionality, the package leverages:

- **NLTK**: For tokenization and English lemmatization
- **spaCy**: For high-quality linguistics-aware tokenization
- **pymorphy2**: For Russian morphological analysis
- **langdetect**: For statistical language detection
- **FastText**: For neural network-based language identification
- **transformers**: For advanced neural tokenization

## Resource Configuration

The package supports flexible resource configuration via environment variables:

- `PAMOLA_STOPWORDS_DIR`: Custom path to stopwords resources
- `PAMOLA_TOKENIZATION_DIR`: Custom path to tokenization resources
- `PAMOLA_LANGUAGE_RESOURCES_DIR`: Custom path to language resources
- `PAMOLA_FASTTEXT_MODEL_PATH`: Custom path to FastText language model

## Performance Considerations

The package implements several optimizations:

1. **Caching**: Resources and results are cached for efficient reuse
2. **Lazy Loading**: Models and resources are loaded only when needed
3. **Parallel Processing**: Batch operations support parallel processing
4. **Resource Timestamps**: Change detection prevents using stale cached resources

## Conclusion

The `pamola_core.utils.nlp` package provides a robust, flexible foundation for natural language processing in the PAMOLA.CORE (Privacy-Preserving AI Data Processors) project. Its architecture ensures reliable operation across varied environments while offering advanced capabilities when specialized libraries are available. The package's multilingual support, graceful degradation, and comprehensive text processing make it ideal for analyzing diverse resume and job posting content.