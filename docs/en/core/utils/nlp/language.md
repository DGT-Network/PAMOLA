# Language Module Documentation

## Overview

The `language.py` module is a core component of the PAMOLA CORE natural language processing system, designed for robust language detection with advanced capabilities for analyzing multilingual and mixed-language texts. It provides reliable language identification with graceful degradation when specialized NLP libraries are unavailable, making it essential for preprocessing diverse resume and job posting content across multiple languages.

## Architecture

The module implements a flexible architecture focused on:

1. **Multi-method Detection**: Layered approach with multiple detection techniques
2. **Confidence Scoring**: Quantitative reliability assessments for detection results
3. **Graceful Degradation**: Fallback mechanisms when specialized libraries are unavailable
4. **Mixed-language Analysis**: Capability to analyze texts containing multiple languages

This architecture aligns with the broader NLP utilities framework of the PAMOLA CORE system:

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Text Source   │──────▶  Language      │──────▶  Tokenization  │
└────────────────┘      │  Detection     │      └────────────────┘
                        └────────────────┘              │
                                │                        ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Script        │◀─────▶  Multilingual  │◀─────▶  Stopwords     │
│  Analysis      │      │  Analysis      │      │  Removal       │
└────────────────┘      └────────────────┘      └────────────────┘
```

The module depends on the `compatibility.py` module for checking library availability and leverages optional integration with external libraries like `langdetect` and `fasttext` for enhanced language detection.

## Key Capabilities

The language module provides the following key capabilities:

1. **Multi-method Language Detection**:
   - Neural-based detection using FastText (if available)
   - Statistical detection using langdetect (if available)
   - Character-set based heuristics (always available)
   - Common word matching (always available)

2. **Confidence Metrics**:
   - Detection confidence scoring (0.0-1.0)
   - Proportional analysis of mixed languages
   - Script (writing system) distribution analysis

3. **Mixed-language Analysis**:
   - Segmentation-based language proportion detection
   - Primary language identification
   - Multilingualism detection
   - Script distribution analysis

4. **Language Resource Management**:
   - Language-specific resource path discovery
   - Resource organization by language and type
   - Language code normalization
   - Supported language enumeration

5. **Script Detection**:
   - Latin script detection
   - Cyrillic script detection
   - CJK (Chinese, Japanese, Korean) script detection
   - Other writing systems (Arabic, Devanagari, etc.)

## Language Detection Methods

The module employs multiple language detection strategies in order of preference:

1. **FastText Neural Detection**: Uses the compact but highly accurate FastText language identification model (when available)
2. **langdetect Statistical Detection**: Employs the Naive Bayes-based langdetect library (when available)
3. **Character-set Heuristics**: Analyzes character distributions against language-specific patterns
4. **Common Word Matching**: Identifies frequent words specific to particular languages

This layered approach ensures high accuracy when specialized libraries are available while maintaining basic functionality when they're not.

## Usage Examples

### Basic Language Detection

```python
from pamola_core.utils.nlp.language import detect_language

# Simple language detection
text_en = "The quick brown fox jumps over the lazy dog."
lang_en = detect_language(text_en)
print(f"Detected language: {lang_en}")
# Output: Detected language: en

# Russian text detection
text_ru = "Быстрая коричневая лиса прыгает через ленивую собаку."
lang_ru = detect_language(text_ru)
print(f"Detected language: {lang_ru}")
# Output: Detected language: ru

# Default language when detection fails
empty_text = ""
lang_empty = detect_language(empty_text, default_language="en")
print(f"Empty text language: {lang_empty}")
# Output: Empty text language: en
```

### Language Detection with Confidence

```python
from pamola_core.utils.nlp.language import detect_language_with_confidence

# Detection with confidence score
text = "The quick brown fox jumps over the lazy dog."
lang, confidence = detect_language_with_confidence(text)
print(f"Language: {lang}, Confidence: {confidence:.2f}")
# Output: Language: en, Confidence: 0.85

# Short text may have lower confidence
short_text = "Hello world"
lang, confidence = detect_language_with_confidence(short_text)
print(f"Language: {lang}, Confidence: {confidence:.2f}")
# Output: Language: en, Confidence: 0.65
```

### Analyzing Mixed-language Text

```python
from pamola_core.utils.nlp.language import detect_mixed_language

# Mixed language text
mixed_text = """The conference features presentations in multiple languages.
Конференция включает презентации на нескольких языках.
La conferencia incluye presentaciones en varios idiomas."""

# Detect language proportions
proportions = detect_mixed_language(mixed_text)
print("Language proportions:")
for lang, prop in proportions.items():
    print(f"  - {lang}: {prop:.2f}")
# Output:
# Language proportions:
#   - en: 0.33
#   - ru: 0.33
#   - es: 0.33

# Check if text is multilingual
from pamola_core.utils.nlp.language import is_multilingual

is_multi = is_multilingual(mixed_text)
print(f"Is multilingual: {is_multi}")
# Output: Is multilingual: True

# Get primary language
from pamola_core.utils.nlp.language import get_primary_language

primary = get_primary_language(mixed_text, threshold=0.5)
print(f"Primary language: {primary}")
# Output: Primary language: en
```

### Comprehensive Language Analysis

```python
from pamola_core.utils.nlp.language import analyze_language_structure

# Mixed script text
text = "Python программирование с 日本語 elements."

# Full language structure analysis
analysis = analyze_language_structure(text)
print("Language analysis:")
print(f"  Primary language: {analysis['primary_language']}")
print(f"  Is multilingual: {analysis['is_multilingual']}")
print(f"  Confidence: {analysis['confidence']:.2f}")
print("  Language proportions:", analysis['language_proportions'])
print("  Scripts detected:", analysis['script_info']['scripts'])
print("  Script proportions:", analysis['script_info']['script_proportions'])
# Output:
# Language analysis:
#   Primary language: en
#   Is multilingual: True
#   Confidence: 0.45
#   Language proportions: {'en': 0.45, 'ru': 0.35, 'ja': 0.20}
#   Scripts detected: ['latin', 'cyrillic', 'cjk']
#   Script proportions: {'latin': 0.45, 'cyrillic': 0.35, 'cjk': 0.20}
```

### Language Resource Management

```python
from pamola_core.utils.nlp.language import get_language_resources_path

# Get path to language-specific resources
ru_stopwords_path = get_language_resources_path('ru', 'stopwords')
print(f"Russian stopwords path: {ru_stopwords_path}")
# Output: Russian stopwords path: /path/to/pamola_core/resources/languages/ru/stopwords

# Get supported languages
from pamola_core.utils.nlp.language import get_supported_languages

languages = get_supported_languages()
print(f"Supported languages: {languages}")
# Output: Supported languages: ['en', 'ru', 'de', 'fr', 'es']

# Normalize language codes
from pamola_core.utils.nlp.language import normalize_language_code

norm = normalize_language_code('english')
print(f"Normalized 'english': {norm}")
# Output: Normalized 'english': en

norm = normalize_language_code('ru_RU')
print(f"Normalized 'ru_RU': {norm}")
# Output: Normalized 'ru_RU': ru
```

### Script Detection

```python
from pamola_core.utils.nlp.language import is_cyrillic, is_latin

# Check for Cyrillic characters
text_ru = "Привет, мир!"
print(f"'{text_ru}' contains Cyrillic: {is_cyrillic(text_ru)}")
# Output: 'Привет, мир!' contains Cyrillic: True

# Check for Latin characters
text_en = "Hello, world!"
print(f"'{text_en}' contains Latin: {is_latin(text_en)}")
# Output: 'Hello, world!' contains Latin: True

# Mixed script text
mixed = "Hello Привет"
print(f"'{mixed}' contains Cyrillic: {is_cyrillic(mixed)}")
print(f"'{mixed}' contains Latin: {is_latin(mixed)}")
# Output:
# 'Hello Привет' contains Cyrillic: True
# 'Hello Привет' contains Latin: True
```

### Batch Language Detection

```python
from pamola_core.utils.nlp.language import detect_languages

# Batch of texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Der schnelle braune Fuchs springt über den faulen Hund."
]

# Detect languages with sampling
lang_distribution = detect_languages(texts, sample_size=4)
print("Language distribution:")
for lang, freq in lang_distribution.items():
    print(f"  - {lang}: {freq:.2f}")
# Output:
# Language distribution:
#   - en: 0.25
#   - ru: 0.25
#   - fr: 0.25
#   - de: 0.25
```

## Parameters

### detect_language Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to analyze |
| `default_language` | str | No | 'en' | Default language to return if detection fails |

**Returns:** str - Detected language code ('en', 'ru', etc.)

### detect_language_with_confidence Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to analyze |
| `default_language` | str | No | 'en' | Default language to return if detection fails |

**Returns:** Tuple[str, float] - Detected language code and confidence score (0-1)

### detect_mixed_language Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to analyze |
| `min_segment_length` | int | No | 10 | Minimum length of text segments to analyze |

**Returns:** Dict[str, float] - Dictionary mapping language codes to their proportions

### get_primary_language Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to analyze |
| `threshold` | float | No | 0.6 | Threshold proportion for considering a language primary |
| `default_language` | str | No | 'en' | Default language if no language meets the threshold |

**Returns:** str - Primary language code

### is_multilingual Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to analyze |
| `threshold` | float | No | 0.2 | Minimum proportion for a language to be considered significant |

**Returns:** bool - True if text contains multiple significant languages

### analyze_language_structure Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | str | Yes | - | Text to analyze |

**Returns:** Dict[str, Any] - Comprehensive language analysis including primary language, language proportions, script information, etc.

## Return Values

### detect_language Function

Returns a string containing the detected language code ('en', 'ru', etc.).

### detect_language_with_confidence Function

Returns a tuple containing the detected language code and a confidence score from 0.0 to 1.0.

### detect_mixed_language Function

Returns a dictionary mapping language codes to their proportions in the text (values sum to 1.0).

### get_primary_language Function

Returns a string containing the primary language code of the text.

### is_multilingual Function

Returns a boolean indicating whether the text contains multiple significant languages.

### analyze_language_structure Function

Returns a dictionary containing comprehensive language analysis:
```python
{
    "primary_language": str,         # Primary language code
    "is_multilingual": bool,         # Whether text is multilingual
    "language_proportions": dict,    # Language code to proportion mapping
    "confidence": float,             # Overall confidence in the analysis
    "script_info": {                 # Information about writing scripts used
        "scripts": list,             # List of detected scripts
        "script_proportions": dict   # Script to proportion mapping
    }
}
```

## Language Detection Strategy

The `detect_language` function employs a cascading strategy:

1. Try FastText neural model for language identification (if available)
2. Try langdetect statistical detection (if available)
3. Fall back to character-set based heuristics
4. Use word matching for final attempt
5. Return default language if all methods fail

This approach ensures:
- High accuracy when specialized libraries are available
- Reasonable results when libraries are unavailable
- Degradation in capability rather than complete failure

## Handling Special Cases

### Empty Text

```python
from pamola_core.utils.nlp.language import detect_language

# Empty text returns default language
empty = ""
lang = detect_language(empty, default_language="en")
print(f"Empty text language: {lang}")
# Output: Empty text language: en
```

### Very Short Text

```python
from pamola_core.utils.nlp.language import detect_language_with_confidence

# Very short text has lower confidence
short = "Hi"
lang, conf = detect_language_with_confidence(short)
print(f"Short text: language={lang}, confidence={conf:.2f}")
# Output: Short text: language=en, confidence=0.30
```

### Code Snippets

```python
from pamola_core.utils.nlp.language import detect_language

# Code often detected as English
code = "def hello_world():\n    print('Hello, world!')"
lang = detect_language(code)
print(f"Code language: {lang}")
# Output: Code language: en
```

## Dependencies

The module has the following dependencies:

1. **Required**:
   - `re`, `os`, `logging`, `collections`, `typing` (Python standard library)
   - `pamola_core.utils.nlp.compatibility` (Internal module)

2. **Optional**:
   - `langdetect` - For enhanced statistical language detection
   - `fasttext` - For neural network-based language identification

## Resource Management

Language detection resources can be managed in several ways:

1. **Default Location**: The module looks for language resources in the default resources directory (`/path/to/pamola_core/resources/languages/`)

2. **Environment Variable**: Override the default path by setting the `PAMOLA_LANGUAGE_RESOURCES_DIR` environment variable:
   ```bash
   export PAMOLA_LANGUAGE_RESOURCES_DIR=/custom/path/to/languages
   ```

3. **FastText Model**: Override the default FastText model path with the `PAMOLA_FASTTEXT_MODEL_PATH` environment variable:
   ```bash
   export PAMOLA_FASTTEXT_MODEL_PATH=/path/to/lid.176.bin
   ```

## Conclusion

The `language.py` module provides a robust, flexible foundation for language detection in the PAMOLA CORE system. By supporting multiple detection methods, mixed-language analysis, and graceful degradation, it ensures reliable language identification across various NLP tasks in resume and job posting analysis. The module's architecture allows for easy extension and customization while maintaining high performance, making it an essential component of the PAMOLA CORE natural language processing pipeline.