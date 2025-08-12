# PAMOLA Core: Extended Tokenization Utilities Module

## Overview

The `tokenization_ext.py` module extends the core tokenization capabilities of the PAMOLA Core framework, providing advanced utilities for natural language processing (NLP) tasks. It builds upon the base tokenization module to offer specialized features such as n-gram extraction, advanced token filtering, keyphrase extraction, collocation detection, and sentiment word identification. This module is designed for flexible integration into NLP pipelines, supporting both single-text and batch processing scenarios.

---

## Key Features

- **N-gram Extraction**: Extracts n-grams and multi-ngrams from tokenized text, with optional dictionary filtering.
- **Keyphrase Extraction**: Identifies key phrases using n-gram frequency and stopword filtering.
- **Collocation Detection**: Finds word collocations using NLTK or fallback logic.
- **POS-based Token Filtering**: Filters tokens by part-of-speech (POS) tags using spaCy.
- **Sentiment Word Extraction**: Identifies positive and negative sentiment words using lexicons.
- **Batch Processing**: Parallelized batch extraction of keyphrases.
- **Advanced Text Processing**: The `AdvancedTextProcessor` class provides a high-level interface for tokenization, lemmatization, n-gram/keyphrase extraction, and more.

---

## Dependencies

### Standard Library
- `logging`
- `typing` (`Dict`, `List`, `Any`, `Optional`, `Union`, `Set`)
- `collections.Counter`
- `os`, `json` (for sentiment lexicon loading)

### Internal Modules
- `pamola_core.utils.nlp.base` (`batch_process`, `DependencyManager`, `RESOURCES_DIR`)
- `pamola_core.utils.nlp.cache` (`get_cache`, `cache_function`)
- `pamola_core.utils.nlp.tokenization` (`tokenize`, `tokenize_and_lemmatize`, `_get_or_detect_language`, `TextProcessor`)
- `pamola_core.utils.nlp.tokenization_helpers` (`load_ngram_dictionary`)
- `pamola_core.utils.nlp.stopwords` (`get_stopwords`)

### Optional Third-Party
- `spaCy` (for POS filtering)
- `NLTK` (for collocation extraction)

---

## Exception Classes

This module does not define custom exception classes directly. Instead, it handles exceptions from dependencies and logs errors. Common exceptions encountered include:

- **ImportError**: Raised if required third-party libraries (e.g., spaCy, NLTK) are not installed.
- **FileNotFoundError**: Raised when sentiment lexicon files are missing.
- **TypeError/ValueError**: Raised for invalid input types or parameters.

### Example: Handling spaCy Dependency
```python
from pamola_core.utils.nlp.base import DependencyManager
if not DependencyManager.check_dependency('spacy'):
    logger.warning("spaCy not available, cannot filter by POS tags")
    # Fallback to basic tokenization
    tokens = tokenize(text, language=language)
```
**When raised**: If spaCy is not installed, POS filtering will not be available and a warning is logged.

### Example: Handling Missing Sentiment Lexicons
```python
try:
    with open(pos_path, 'r', encoding='utf-8') as f:
        positive_words = {line.strip().lower() for line in f if line.strip()}
except FileNotFoundError:
    logger.error(f"Sentiment lexicon not found: {pos_path}")
    positive_words = set()
```
**When raised**: If the sentiment lexicon file does not exist for the specified language.

---

## Main Classes and Functions

### NGramExtractor

Utility class for extracting n-grams from a list of tokens, optionally filtered by a dictionary.

#### Methods
- `extract_ngrams(tokens: List[str], n: int = 2, ngram_sources: Optional[Union[str, List[str]]] = None, language: Optional[str] = None) -> List[str]`
    - **Parameters:**
        - `tokens`: List of tokens to extract n-grams from
        - `n`: Size of n-grams
        - `ngram_sources`: Optional n-gram dictionary sources
        - `language`: Optional language code
    - **Returns:** List of n-grams
    - **Raises:** None (logs errors)

- `extract_multi_ngrams(tokens: List[str], min_n: int = 1, max_n: int = 3, ngram_sources: Optional[Union[str, List[str]]] = None, language: Optional[str] = None) -> List[str]`
    - **Parameters:**
        - `tokens`: List of tokens
        - `min_n`: Minimum n-gram size
        - `max_n`: Maximum n-gram size
        - `ngram_sources`: Optional n-gram dictionary sources
        - `language`: Optional language code
    - **Returns:** List of n-grams
    - **Raises:** None (logs errors)

### AdvancedTextProcessor

Extended text processor with additional capabilities beyond the base `TextProcessor`.

#### Constructor
```python
def __init__(
    self,
    language: Optional[str] = None,
    tokenizer_type: str = 'auto',
    config_sources: Optional[Union[str, List[str]]] = None,
    lemma_dict_sources: Optional[Union[str, List[str]]] = None,
    ngram_sources: Optional[Union[str, List[str]]] = None,
    min_token_length: int = 2,
    preserve_case: bool = False
)
```
- **Parameters:**
    - `language`: Language code
    - `tokenizer_type`: Tokenizer type
    - `config_sources`: Config file paths
    - `lemma_dict_sources`: Lemma dictionary paths
    - `ngram_sources`: N-gram dictionary paths
    - `min_token_length`: Minimum token length
    - `preserve_case`: Whether to preserve case

#### Key Attributes
- `processor`: Underlying `TextProcessor` instance
- `ngram_sources`: N-gram dictionary sources

#### Public Methods
- `tokenize(text: str, **kwargs) -> List[str]`
    - Tokenizes text using the underlying processor.
- `lemmatize(tokens: List[str], **kwargs) -> List[str]`
    - Lemmatizes tokens.
- `tokenize_and_lemmatize(text: str, **kwargs) -> List[str]`
    - Tokenizes and lemmatizes text.
- `extract_ngrams(tokens: List[str], n: int = 2, **kwargs) -> List[str]`
    - Extracts n-grams from tokens.
- `extract_multi_ngrams(tokens: List[str], min_n: int = 1, max_n: int = 3, **kwargs) -> List[str]`
    - Extracts n-grams of varying sizes.
- `extract_keyphrases(text: str, min_length: int = 2, max_length: int = 5, top_n: int = 10, **kwargs) -> List[Dict[str, Any]]`
    - Extracts key phrases from text.
- `process_text_advanced(text: str, lemmatize_tokens: bool = True, extract_ngrams_flag: bool = True, extract_keyphrases_flag: bool = False, pos_filter: Optional[List[str]] = None, ngram_sizes: List[int] = None, **kwargs) -> Dict[str, Any]`
    - Full advanced pipeline for text processing.

#### Example Usage
```python
# Create advanced text processor
processor = AdvancedTextProcessor(language='en')

# Tokenize and lemmatize text
lemmas = processor.tokenize_and_lemmatize("The quick brown fox jumps over the lazy dog.")

# Extract bigrams
bigrams = processor.extract_ngrams(lemmas, n=2)

# Extract keyphrases
keyphrases = processor.extract_keyphrases("Natural language processing is fun and useful.")
```

---

## Dependency Resolution and Completion Validation

- **DependencyManager**: Used to check for optional dependencies (e.g., spaCy, NLTK) before using advanced features. If a dependency is missing, the module logs a warning and falls back to basic logic.
- **Completion Validation**: Functions and methods check for required input (e.g., non-empty text or tokens) and return early if not satisfied.

---

## Usage Examples

### Extracting N-grams and Keyphrases
```python
from pamola_core.utils.nlp.tokenization_ext import AdvancedTextProcessor

# Tokenize and extract n-grams
processor = AdvancedTextProcessor(language='en')
tokens = processor.tokenize("Data science is awesome!")
bigrams = processor.extract_ngrams(tokens, n=2)

# Extract keyphrases
keyphrases = processor.extract_keyphrases("Machine learning enables new possibilities.")
```

### Filtering Tokens by POS
```python
from pamola_core.utils.nlp.tokenization_ext import filter_tokens_by_pos

# Filter for nouns and adjectives
filtered = filter_tokens_by_pos("The quick brown fox jumps over the lazy dog.", pos_tags=['NOUN', 'ADJ'])
```

### Handling Failed Dependencies
```python
from pamola_core.utils.nlp.tokenization_ext import filter_tokens_by_pos

# spaCy not installed: falls back to basic tokenization
filtered = filter_tokens_by_pos("Text to analyze", pos_tags=['NOUN'])
```

### Using in a Pipeline (with BaseTask)
```python
# Example integration with a task pipeline
class MyTask(BaseTask):
    def run(self, text):
        processor = AdvancedTextProcessor(language='en')
        result = processor.process_text_advanced(text, extract_ngrams_flag=True)
        return result
```

### Continue-on-Error with Logging
```python
try:
    keyphrases = processor.extract_keyphrases("Some text")
except Exception as e:
    logger.error(f"Keyphrase extraction failed: {e}")
    # Continue pipeline execution
```

---

## Error Handling and Exception Hierarchy

- **ImportError**: Raised if a required library is missing. The module logs a warning and uses fallback logic.
- **FileNotFoundError**: Raised if a sentiment lexicon file is missing. The module logs an error and returns empty results.
- **TypeError/ValueError**: Raised for invalid input types or parameters. The module logs errors and returns safe defaults.

---

## Configuration Requirements

- **Config Sources**: Paths to configuration files can be provided for custom tokenization or lemmatization.
- **Lemma Dictionary Sources**: Optional paths for lemmatization dictionaries.
- **N-gram Sources**: Optional paths for n-gram dictionaries to filter extracted n-grams.

---

## Security Considerations and Best Practices

- **Path Security**: Only use absolute paths for external data. Internal dependencies should use logical task IDs.
- **Risks of Disabling Path Security**: Allowing arbitrary absolute paths can expose sensitive data or allow unauthorized access.

### Example: Security Failure
```python
# BAD: Using unvalidated absolute path from user input
ngram_sources = [user_supplied_path]  # Risk: may point to sensitive files
processor = AdvancedTextProcessor(ngram_sources=ngram_sources)
```
**How it is handled**: Always validate and sanitize external paths before use. Prefer internal references when possible.

---

## Best Practices

1. **Use Task IDs for Internal Dependencies**: For data flows within your project, use task IDs as dependencies to maintain logical connections.
2. **Use Absolute Paths Judiciously**: Only use absolute paths for truly external data that isn't produced by your task pipeline.
3. **Check for Optional Dependencies**: Use `DependencyManager` to check for spaCy or NLTK before using advanced features.
4. **Handle Missing Resources Gracefully**: Always provide fallback logic and log errors for missing files or dependencies.
5. **Validate Input**: Ensure text and token inputs are non-empty and of correct type before processing.
6. **Log Errors and Warnings**: Use the logger to record issues for debugging and monitoring.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical references (e.g., task IDs) for data produced within the pipeline.
- **External Dependencies**: Use absolute paths only for data outside the pipeline, and validate paths for security.

---

## Integration Notes

- The module is designed for seamless integration with PAMOLA Core's `BaseTask` and other pipeline components.
- Batch processing functions (`batch_extract_keyphrases`) support parallel execution for large datasets.
- All advanced features are optional and gracefully degrade if dependencies are missing.
