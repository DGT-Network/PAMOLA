# Tokenization Module Documentation

## Overview

The `tokenization.py` module is a core component of the PAMOLA CORE natural language processing system, designed for efficient text tokenization, lemmatization, and n-gram extraction across multiple languages. It provides adaptable text processing capabilities with graceful degradation when specialized NLP libraries are unavailable, making it essential for text analysis tasks within resume and job posting processing.

## Architecture

The module implements an object-oriented architecture focused on:

1. **Flexible Tokenization**: Multiple tokenizer implementations with a common interface
2. **Graceful Degradation**: Automatic fallback to simpler methods when advanced libraries are unavailable
3. **Resource Efficiency**: Caching mechanisms and performance optimizations
4. **Extensibility**: Easy addition of new tokenizer types and custom processors

This architecture aligns with the broader NLP utilities framework of the PAMOLA CORE system:

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Text Source   │──────▶  Tokenization  │──────▶  Lemmatization │
└────────────────┘      └────────────────┘      └────────────────┘
                                │
                                ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  N-gram        │◀─────▶  Stopwords     │◀─────▶  Language      │
│  Extraction    │      │  Removal       │      │  Detection     │
└────────────────┘      └────────────────┘      └────────────────┘
```

The module includes the following key components:

1. **BaseTokenizer**: Abstract base class defining the tokenizer interface
2. **SimpleTokenizer**: Basic whitespace-based tokenizer (always available)
3. **NLTKTokenizer**: NLTK-based tokenizer with advanced capabilities
4. **SpacyTokenizer**: spaCy-based tokenizer with linguistic features
5. **TransformersTokenizer**: Modern neural tokenizer using Hugging Face transformers
6. **TokenizerFactory**: Factory for creating appropriate tokenizers
7. **TextProcessor**: High-level interface for combined text processing
8. **NGramExtractor**: Utility for n-gram extraction and processing
9. **LemmatizerRegistry**: Registry for custom lemmatization implementations

## Key Capabilities

The tokenization module provides the following key capabilities:

1. **Multi-method tokenization**:
   - Simple whitespace-based tokenization as fallback
   - NLTK tokenization with language-specific rules
   - spaCy tokenization with linguistic awareness
   - Transformer-based tokenization with subword units

2. **Lemmatization across languages**:
   - English lemmatization via NLTK's WordNet
   - Russian lemmatization via pymorphy2
   - Fallback to Snowball stemmers for other languages
   - Custom lemmatizer registration

3. **Advanced text processing**:
   - Pattern preservation (URLs, emails, etc.)
   - Case preservation options
   - Compound word handling
   - Text normalization

4. **N-gram extraction**:
   - Fixed-size n-gram generation
   - Multi-size n-gram generation
   - Optional filtering by dictionary

5. **Performance optimizations**:
   - Tokenizer caching
   - Resource caching
   - Parallel processing for batch operations

6. **Backward compatibility**:
   - Functional interfaces for legacy code
   - Consistent API across tokenizer implementations

## Tokenizer Types

The module provides four types of tokenizers, each with specific capabilities:

### 1. SimpleTokenizer

A basic tokenizer that splits text on whitespace and removes punctuation. It's always available as it uses only Python standard library.

**Key features:**
- Handles basic tokenization with whitespace splitting
- Removes punctuation
- Provides customizable minimum token length filtering
- Can preserve case if needed
- Supports custom pattern preservation

### 2. NLTKTokenizer

A tokenizer based on NLTK's word_tokenize function, providing language-specific tokenization rules.

**Key features:**
- Uses NLTK's language-specific tokenization
- Handles contractions and special cases
- Supports multiple languages
- Gracefully degrades to SimpleTokenizer if NLTK is unavailable

### 3. SpacyTokenizer

A tokenizer that leverages spaCy's linguistic models for high-quality tokenization.

**Key features:**
- Linguistic-aware tokenization
- Automatic loading of appropriate language models
- Disabling of unnecessary pipeline components for efficiency
- Tokenization with rich linguistic metadata
- Gracefully degrades when spaCy or specific models are unavailable

### 4. TransformersTokenizer

A modern tokenizer based on neural transformer models from Hugging Face.

**Key features:**
- Handles subword tokenization
- Supports multilingual models
- Can filter subword tokens
- Special token handling options
- Degrades gracefully when transformers library is unavailable

## Resource Management

The module relies on the `tokenization_helpers.py` module for efficient resource management:

1. **Resource Caching**: Implements a caching system for dictionaries, configurations, and tokenizers
2. **File Discovery**: Provides automatic discovery of resources in configurable directories
3. **Multiple Formats**: Supports loading from JSON, CSV, and plain text files
4. **Resource Combination**: Allows merging of resources from multiple sources

## Usage Examples

### Basic Tokenization

```python
from pamola_core.utils.nlp.tokenization import tokenize

# English text tokenization
text_en = "The quick brown fox jumps over the lazy dog."
tokens_en = tokenize(text_en, language="en", min_length=2)
print(f"English tokens: {tokens_en}")
# Output: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

# Russian text tokenization
text_ru = "Съешь ещё этих мягких французских булок."
tokens_ru = tokenize(text_ru, language="ru", min_length=2)
print(f"Russian tokens: {tokens_ru}")
# Output: ['Съешь', 'ещё', 'этих', 'мягких', 'французских', 'булок']

# Using different tokenizer type
tokens_spacy = tokenize(text_en, tokenizer_type="spacy", min_length=2)
print(f"spaCy tokens: {tokens_spacy}")
```

### Tokenization and Lemmatization

```python
from pamola_core.utils.nlp.tokenization import tokenize, lemmatize, tokenize_and_lemmatize

# Separate tokenization and lemmatization
text = "Running quickly through the forests and jumping over rivers"
tokens = tokenize(text, language="en")
lemmas = lemmatize(tokens, language="en")
print(f"Tokens: {tokens}")
# Output: ['Running', 'quickly', 'through', 'the', 'forests', 'and', 'jumping', 'over', 'rivers']
print(f"Lemmas: {lemmas}")
# Output: ['run', 'quickly', 'through', 'the', 'forest', 'and', 'jump', 'over', 'river']

# Combined tokenization and lemmatization
lemmatized_tokens = tokenize_and_lemmatize(text, language="en", min_length=3)
print(f"Lemmatized tokens (min length 3): {lemmatized_tokens}")
# Output: ['run', 'quickly', 'through', 'forest', 'jumping', 'over', 'river']
```

### Using the TextProcessor

```python
from pamola_core.utils.nlp.tokenization import TextProcessor

# Create a processor with configured settings
processor = TextProcessor(
    language="en",
    tokenizer_type="nltk",
    min_token_length=3,
    preserve_case=False
)

# Process text
text = "Senior Software Engineer with 5+ years of experience in Python and JavaScript"
result = processor.process_text(
    text,
    lemmatize_tokens=True,
    extract_ngrams=True,
    ngram_sizes=[2, 3]
)

# Access results
print(f"Tokens: {result['tokens']}")
# Output: ['senior', 'software', 'engineer', 'with', 'years', 'experience', 'python', 'javascript']

print(f"Lemmas: {result['lemmas']}")
# Output: ['senior', 'software', 'engineer', 'with', 'year', 'experience', 'python', 'javascript']

print(f"Bigrams: {result['ngrams'][2][:3]}")
# Output: ['senior software', 'software engineer', 'engineer with']
```

### N-gram Extraction

```python
from pamola_core.utils.nlp.tokenization import NGramExtractor, tokenize

# Tokenize text
text = "Data scientists analyze and interpret complex digital data"
tokens = tokenize(text, min_length=3)
print(f"Tokens: {tokens}")
# Output: ['Data', 'scientists', 'analyze', 'and', 'interpret', 'complex', 'digital', 'data']

# Extract bigrams
bigrams = NGramExtractor.extract_ngrams(tokens, n=2)
print(f"Bigrams: {bigrams}")
# Output: ['Data scientists', 'scientists analyze', 'analyze and', 'and interpret', 'interpret complex', 'complex digital', 'digital data']

# Extract multiple n-gram sizes
multi_ngrams = NGramExtractor.extract_multi_ngrams(tokens, min_n=1, max_n=3)
print(f"Total multi-ngrams: {len(multi_ngrams)}")
# Output: Total multi-ngrams: 22
```

### Pattern Preservation

```python
from pamola_core.utils.nlp.tokenization import TokenizerFactory

# Create a tokenizer
tokenizer = TokenizerFactory.create_tokenizer("nltk", language="en")

# Define patterns to preserve
preserve_patterns = [
    {"pattern": r'\S+@\S+\.\S+', "type": "email"},  # Email pattern
    {"pattern": r'https?://\S+', "type": "url"}  # URL pattern
]

# Text with patterns to preserve
text = "Send your resume to careers@example.com or visit https://careers.example.com"

# Tokenize with pattern preservation
tokens = tokenizer.tokenize(
    text,
    min_length=1,
    preserve_patterns=preserve_patterns
)

print(f"Tokens with preserved patterns: {tokens}")
# Output: ['Send', 'your', 'resume', 'to', 'careers@example.com', 'or', 'visit', 'https://careers.example.com']
```

### Text Normalization

```python
from pamola_core.utils.nlp.tokenization import normalize_text

# Text with various elements to normalize
text = "The PRICE is $19.99 and the delivery date is 2023-03-15!"

# Basic normalization
normalized = normalize_text(text, lowercase=True, remove_punctuation=True)
print(f"Basic normalization: {normalized}")
# Output: "the price is 19 99 and the delivery date is 2023 03 15"

# Advanced normalization with replacements
replacement_rules = {
    r'\$\d+\.\d+': '[PRICE]',
    r'\d{4}-\d{2}-\d{2}': '[DATE]'
}

normalized = normalize_text(
    text,
    lowercase=True,
    remove_punctuation=True,
    replacement_rules=replacement_rules
)

print(f"Advanced normalization: {normalized}")
# Output: "the price is [PRICE] and the delivery date is [DATE]"
```

### Batch Processing

```python
from pamola_core.utils.nlp.tokenization import batch_tokenize, batch_tokenize_and_lemmatize

# List of texts
texts = [
    "Senior Software Engineer with Python experience",
    "Product Manager for e-commerce platforms",
    "Data Scientist specializing in natural language processing",
    "UI/UX Designer for mobile applications"
]

# Batch tokenization
tokens_list = batch_tokenize(
    texts,
    language="en",
    min_length=3,
    processes=2
)

print(f"Number of processed texts: {len(tokens_list)}")
# Output: Number of processed texts: 4

# Batch tokenization and lemmatization
lemmatized_tokens_list = batch_tokenize_and_lemmatize(
    texts,
    language="en",
    min_length=3,
    processes=2
)

# Show results for the first text
print(f"Original: {texts[0]}")
print(f"Tokens: {tokens_list[0]}")
print(f"Lemmatized: {lemmatized_tokens_list[0]}")
```

### Custom Lemmatizer Registration

```python
from pamola_core.utils.nlp.tokenization import register_lemmatizer, lemmatize


# Define a custom lemmatizer for a specific domain
def tech_lemmatizer(tokens, **kwargs):
    """Custom lemmatizer for technology terms."""
    replacements = {
        'js': 'javascript',
        'py': 'python',
        'react.js': 'react',
        'node.js': 'node',
        'typescript': 'typescript',
        'angular.js': 'angular'
    }

    result = []
    for token in tokens:
        lower_token = token.lower()
        if lower_token in replacements:
            result.append(replacements[lower_token])
        else:
            # Use the token as is
            result.append(token)

    return result


# Register the custom lemmatizer
register_lemmatizer('tech', tech_lemmatizer)

# Use the custom lemmatizer
tokens = ['Python', 'JS', 'React.js', 'Angular.js']
lemmas = lemmatize(tokens, language='tech')
print(f"Tech lemmas: {lemmas}")
# Output: Tech lemmas: ['python', 'javascript', 'react', 'angular']
```

## Parameters

### Tokenize Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to tokenize |
| `language` | str | No | None | Language code. If None, language is auto-detected |
| `min_length` | int | No | 2 | Minimum length for tokens to keep |
| `config_sources` | str or List[str] | No | None | Tokenization configuration source(s) |
| `preserve_case` | bool | No | False | Whether to preserve token case |
| `preserve_patterns` | List[Dict[str, str]] | No | None | List of regex patterns to preserve |
| `tokenizer_type` | str | No | 'auto' | Tokenizer type: 'auto', 'simple', 'nltk', 'spacy', 'transformers' |

### Lemmatize Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tokens` | List[str] | Yes | - | List of tokens to lemmatize |
| `language` | str | No | None | Language code. If None, defaults to 'en' |
| `lemma_dict_sources` | str or List[str] | No | None | Custom lemmatization dictionary source(s) |

### TextProcessor Constructor

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `language` | str | No | None | Language code. If None, language will be auto-detected |
| `tokenizer_type` | str | No | 'auto' | Type of tokenizer to use |
| `config_sources` | str or List[str] | No | None | Tokenization configuration source(s) |
| `lemma_dict_sources` | str or List[str] | No | None | Custom lemmatization dictionary source(s) |
| `min_token_length` | int | No | 2 | Minimum length for tokens to keep |
| `preserve_case` | bool | No | False | Whether to preserve token case |

### NGramExtractor Methods

**extract_ngrams Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tokens` | List[str] | Yes | - | List of tokens |
| `n` | int | No | 2 | Size of n-grams to extract |
| `ngram_sources` | str or List[str] | No | None | Path(s) to n-gram dictionary file(s) |
| `language` | str | No | None | Language code for language-specific dictionaries |

**extract_multi_ngrams Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tokens` | List[str] | Yes | - | List of tokens |
| `min_n` | int | No | 1 | Minimum size of n-grams to extract |
| `max_n` | int | No | 3 | Maximum size of n-grams to extract |
| `ngram_sources` | str or List[str] | No | None | Path(s) to n-gram dictionary file(s) |
| `language` | str | No | None | Language code for language-specific dictionaries |

## Return Values

### Tokenize Function

Returns a `List[str]` containing extracted tokens from the text.

### Lemmatize Function

Returns a `List[str]` containing lemmatized forms of the input tokens.

### TextProcessor.process_text Method

Returns a `Dict[str, Any]` containing:
- `text`: Original text
- `length`: Length of the original text
- `tokens`: List of tokens
- `token_count`: Number of tokens
- `lemmas`: List of lemmatized tokens (if requested)
- `lemma_count`: Number of lemmas (if requested)
- `ngrams`: Dictionary mapping n-gram sizes to lists of n-grams (if requested)
- `language` or `detected_language`: Language code (provided or detected)

## Working with Different Languages

The module provides robust support for multiple languages:

### English

```python
from pamola_core.utils.nlp.tokenization import tokenize_and_lemmatize

text_en = "The quick brown foxes are jumping over the lazy dogs"
lemmas_en = tokenize_and_lemmatize(text_en, language="en")
print(lemmas_en)
# Output: ['the', 'quick', 'brown', 'fox', 'are', 'jump', 'over', 'the', 'lazy', 'dog']
```

### Russian

```python
from pamola_core.utils.nlp.tokenization import tokenize_and_lemmatize

text_ru = "Быстрые коричневые лисы прыгают через ленивых собак"
lemmas_ru = tokenize_and_lemmatize(text_ru, language="ru")
print(lemmas_ru)
# Output: ['быстрый', 'коричневый', 'лиса', 'прыгать', 'через', 'ленивый', 'собака']
```

### Automatic Language Detection

```python
from pamola_core.utils.nlp.tokenization import TextProcessor

processor = TextProcessor()  # No language specified, will auto-detect

text_en = "The quick brown fox jumps over the lazy dog"
result_en = processor.process_text(text_en, lemmatize_tokens=True)
print(f"Detected language: {result_en.get('detected_language')}")
print(f"Lemmas: {result_en.get('lemmas')}")

text_ru = "Быстрая коричневая лиса прыгает через ленивую собаку"
result_ru = processor.process_text(text_ru, lemmatize_tokens=True)
print(f"Detected language: {result_ru.get('detected_language')}")
print(f"Lemmas: {result_ru.get('lemmas')}")
```

## Tokenizer Selection Strategy

The `TokenizerFactory` selects tokenizers based on availability and preference:

1. If a specific tokenizer type is requested, it attempts to use that type
2. If that tokenizer is unavailable, it falls back to simpler alternatives
3. With 'auto' selection, it picks the best available tokenizer:
   - First choice: SpacyTokenizer (most advanced)
   - Second choice: NLTKTokenizer (good capabilities)
   - Third choice: TransformersTokenizer (specialized for neural models)
   - Last choice: SimpleTokenizer (always available)

```python
from pamola_core.utils.nlp.tokenization import TokenizerFactory, get_available_tokenizers

# Check what's available
available = get_available_tokenizers()
print(f"Available tokenizers: {available}")

# Create best available tokenizer
tokenizer = TokenizerFactory.create_tokenizer("auto", language="en")
print(f"Selected tokenizer: {type(tokenizer).__name__}")

# Try to create specific tokenizer with fallback
spacy_tokenizer = TokenizerFactory.create_tokenizer("spacy", language="en")
# If spaCy is not available, this will return a different tokenizer type
```

## Dependencies

The module has the following dependencies:

1. **Required**:
   - `re`, `string`, `logging`, `typing` (Python standard library)
   - `pamola_core.utils.nlp.language` (Language detection)
   - `pamola_core.utils.nlp.compatibility` (Dependency checking)
   - `pamola_core.utils.nlp.tokenization_helpers` (Resource management)

2. **Optional**:
   - `nltk` - For enhanced tokenization and English lemmatization
   - `pymorphy2` - For Russian lemmatization
   - `spacy` - For high-quality linguistics-aware tokenization
   - `transformers` - For neural tokenization models

## Resource Configuration

Tokenization resources can be managed in several ways:

1. **Default Location**: The module looks for configuration files in the default resources directory (`/path/to/pamola_core/resources/tokenization/`)

2. **Environment Variable**: Override the default path by setting the `PAMOLA_TOKENIZATION_DIR` environment variable:
   ```bash
   export PAMOLA_TOKENIZATION_DIR=/custom/path/to/tokenization
   ```

3. **Direct Specification**: Provide explicit file paths when calling tokenization functions

## Performance Considerations

### Caching

The module implements several caching mechanisms:

1. **Tokenizer Caching**: The `TokenizerFactory` caches tokenizer instances by type and language
2. **Resource Caching**: Configuration files, dictionaries, and n-gram lists are cached
3. **Resource Timestamps**: File change detection prevents using stale cached resources

### Batch Processing

For processing large amounts of text, use the batch functions:

```python
from pamola_core.utils.nlp.tokenization import batch_tokenize

# Process multiple texts in parallel
results = batch_tokenize(
    texts,
    language="en",
    min_length=2,
    processes=4  # Number of parallel processes
)
```

### Memory Efficiency

To control memory usage with caching:

```python
from pamola_core.utils.nlp.tokenization_helpers import configure_cache

# Set cache size limit
configure_cache(enable=True, max_size=500)

# Disable caching for memory-constrained environments
configure_cache(enable=False)
```

## Conclusion

The `tokenization.py` module provides a comprehensive, flexible foundation for text tokenization in the PAMOLA CORE system. By supporting multiple languages, diverse tokenization methods, and graceful degradation, it ensures reliable text processing across various NLP tasks in resume and job posting analysis. The module's architecture allows for easy extension and customization while maintaining high performance, making it an essential component of the PAMOLA CORE natural language processing pipeline.