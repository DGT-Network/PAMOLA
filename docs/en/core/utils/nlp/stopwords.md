# Stopwords Module Documentation

## Overview

The `stopwords.py` module is a pamola core component of the PAMOLA.CORE (Privacy-Preserving AI Data Processors) natural language processing system, designed for efficient management and application of stopwords across multiple languages. It provides flexible loading mechanisms, comprehensive filtering capabilities, and graceful degradation when specialized NLP libraries are unavailable, making it essential for text preprocessing tasks within resume and job posting analysis.

## Architecture

The module implements a flexible architecture focused on:

1. **Resource Management**: Dynamic loading of stopwords from multiple sources and formats
2. **Multi-language Support**: Handling of stopwords for different languages with automatic detection
3. **Graceful Degradation**: Fallback mechanisms when specialized NLP libraries are unavailable

This architecture aligns with the broader NLP utilities framework of the PAMOLA.CORE system:

```
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Text Source   │──────▶  Tokenization  │──────▶  Lemmatization │
└────────────────┘      └────────────────┘      └────────────────┘
                                │
                                ▼
┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│  Final Tokens  │◀─────▶  Stopwords     │◀─────▶  Language      │
└────────────────┘      │  Removal       │      │  Detection     │
                        └────────────────┘      └────────────────┘
```

The module depends on the `compatibility.py` module for checking NLTK availability and leverages optional integration with the NLTK library for enhanced stopword coverage.

## Key Capabilities

The stopwords module provides the following key capabilities:

1. **Multi-source Loading**: Load stopwords from various sources including:
   - Text files (one word per line)
   - JSON files (list or dictionary format)
   - CSV files
   - Directories (automatically processes all supported files)
   - In-memory sets
   - NLTK's comprehensive stopword lists

2. **Flexible Language Support**: 
   - Automatic language detection from filenames
   - Language-specific stopword collection
   - Support for multiple languages simultaneously

3. **File Management**:
   - Automatic discovery of stopword files in resource directories
   - Combining multiple stopword sources
   - Saving combined stopword lists for future use

4. **Efficient Token Filtering**:
   - Fast stopword removal from token lists
   - Customizable filtration based on specific languages or domains

5. **Integration with NLTK**:
   - Seamless integration with NLTK's extensive stopword collections
   - Automatic fallback to file-based stopwords when NLTK is unavailable

## Recommended Stopword Sources for PAMOLA.CORE

For PAMOLA.CORE (Privacy-Preserving AI Data Processors) processing, we recommend using the following stopword sources:

1. **General Language Stopwords**:
   - English: `stop_words_english.txt` - Comprehensive list for general English text
   - Russian: `stop_words_russian.txt` - Extensive stopwords for Russian language processing

2. **Domain-Specific HR Stopwords**:
   - `hr_position_modifiers.txt` - Contains position modifiers (senior, junior, lead, etc.)
   - `hr_employment_terms.txt` - Employment-related terms (remote, full-time, contract, etc.)
   - `hr_resume_terms.txt` - Common resume structural terms (experience, skills, education, etc.)

3. **Technical Fields Stopwords**:
   - `tech_programming.txt` - Common programming terms that may not be useful discriminators 
   - `tech_management.txt` - Management and process terminology

4. **Industry-Specific Stopwords**:
   - Various industry-specific stopword collections based on the sector being analyzed

## Usage Examples

### Basic Usage

Here's a simple example of using the module to remove stopwords from tokens:

```python
from pamola_core.utils.nlp.stopwords import get_stopwords, remove_stopwords
from pamola_core.utils.nlp.tokenization import tokenize

# Example resume text
text = "Senior Software Engineer with 5 years of experience in Python and JavaScript"

# Tokenize the text
tokens = tokenize(text)
print(f"Original tokens: {tokens}")

# Get default English stopwords
stop_words = get_stopwords(['en'])

# Remove stopwords
filtered_tokens = remove_stopwords(tokens, stop_words)
print(f"Filtered tokens: {filtered_tokens}")
```

### Advanced Multi-language Processing

```python
from pamola_core.utils.nlp.stopwords import get_stopwords, remove_stopwords
from pamola_core.utils.nlp.tokenization import tokenize
from pamola_core.utils.nlp.language import detect_language

# Process a multilingual job description
text_ru = "Требуется старший разработчик программного обеспечения с опытом работы от 3 лет"
text_en = "Looking for a senior software developer with 3+ years of experience"

# Detect language and tokenize
lang_ru = detect_language(text_ru)
tokens_ru = tokenize(text_ru, lang_ru)

lang_en = detect_language(text_en)
tokens_en = tokenize(text_en, lang_en)

# Get stopwords for both languages
stop_words = get_stopwords(['ru', 'en'])

# Remove stopwords
filtered_ru = remove_stopwords(tokens_ru, stop_words)
filtered_en = remove_stopwords(tokens_en, stop_words)

print(f"Russian filtered tokens: {filtered_ru}")
print(f"English filtered tokens: {filtered_en}")
```

### Custom Stopwords with Multiple Sources

```python
from pamola_core.utils.nlp.stopwords import get_stopwords, remove_stopwords
from pamola_core.utils.nlp.tokenization import tokenize

# Define custom stopwords specific to the task
custom_hr_terms = {
    'experience', 'experienced', 'work', 'year', 'years',
    'skills', 'ability', 'position', 'job'
}

# Custom file paths
domain_file = '/path/to/domain_specific_stops.txt'
project_dir = '/path/to/project_stopwords/'

# Get comprehensive stopwords combining multiple sources
stop_words = get_stopwords(
    languages=['en'],
    custom_sources=[custom_hr_terms, domain_file, project_dir],
    include_defaults=True,
    use_nltk=True
)

# Process job description
text = "Senior Java Developer with 5+ years of experience in enterprise applications..."
tokens = tokenize(text)
filtered_tokens = remove_stopwords(tokens, stop_words)

print(f"Filtered tokens: {filtered_tokens}")
```

### Creating and Saving Combined Stopword Lists

```python
from pamola_core.utils.nlp.stopwords import get_stopwords, save_stopwords_to_file, combine_stopwords_files
from pathlib import Path

# Example: Create a comprehensive HR stopwords file

# Source files
sources = [
    'resources/stopwords/en_general.txt',
    'resources/stopwords/hr_terms.txt',
    'resources/stopwords/tech_roles.txt'
]

# Combine files into a single comprehensive file
output_file = 'resources/stopwords/comprehensive_hr_en.json'
combine_stopwords_files(sources, output_file, format='json')

# Alternative: Generate and save programmatically
custom_terms = {'applicant', 'candidate', 'recruitment', 'hiring'}
all_stops = get_stopwords(['en'], custom_sources=[custom_terms])
save_stopwords_to_file(all_stops, 'resources/stopwords/enhanced_hr_en.txt')
```

## Parameters

### Get Stopwords Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `languages` | List[str] | No | ['en'] | Language codes for stopwords |
| `custom_sources` | List[Union[str, Set[str]]] | No | None | Additional stopword sources |
| `include_defaults` | bool | No | True | Whether to include default stopwords |
| `use_nltk` | bool | No | True | Whether to use NLTK stopwords |
| `encodings` | Union[str, List[str]] | No | 'utf-8' | File encoding(s) to use |

### Remove Stopwords Function

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tokens` | List[str] | Yes | - | Tokens to filter |
| `stop_words` | Set[str] | No | None | Set of stopwords to exclude |
| `languages` | List[str] | No | None | Languages if stop_words is None |
| `custom_sources` | List[Union[str, Set[str]]] | No | None | Additional sources if stop_words is None |

## Return Values

### Get Stopwords Function

Returns a `Set[str]` containing all combined stopwords from the specified sources.

### Remove Stopwords Function

Returns a `List[str]` containing the original tokens with all stopwords removed.

## Recommended Workflow for PAMOLA.CORE Tasks

For PAMOLA.CORE (Privacy-Preserving AI Data Processors) analysis, we recommend the following workflows:

### 1. Resume Field Extraction and Analysis

```python
from pamola_core.utils.nlp.stopwords import get_stopwords, remove_stopwords
from pamola_core.utils.nlp.tokenization import tokenize, lemmatize
from pamola_core.utils.nlp.language import detect_language


def analyze_job_title(title_text):
    # Detect language
    lang = detect_language(title_text)

    # Get stopwords for the detected language and add HR-specific terms
    stop_words = get_stopwords(
        languages=[lang],
        custom_sources=['resources/stopwords/hr_position_modifiers.txt']
    )

    # Tokenize, remove stopwords, and lemmatize
    tokens = tokenize(title_text, language=lang)
    filtered_tokens = remove_stopwords(tokens, stop_words)
    lemmatized_tokens = lemmatize(filtered_tokens, language=lang)

    # Return normalized job title tokens for further processing
    return lemmatized_tokens
```

### 2. Batch Processing of Resumes

```python
from pamola_core.utils.nlp.stopwords import get_stopwords, remove_stopwords
import pandas as pd


def process_resume_batch(resume_df, text_column):
    # Preload stopwords for efficiency in batch processing
    all_stopwords = get_stopwords(
        languages=['en', 'ru'],
        custom_sources=[
            'resources/stopwords/hr_terms.txt',
            'resources/stopwords/tech_terminology.txt'
        ]
    )

    # Process function to apply to each resume
    def process_text(text):
        if not text:
            return []

        # Here you would typically include language detection, 
        # tokenization, etc., but for brevity we're focusing on stopwords
        tokens = tokenize(text)
        return remove_stopwords(tokens, all_stopwords)

    # Apply to dataframe
    resume_df['processed_tokens'] = resume_df[text_column].apply(process_text)

    return resume_df
```

### 3. Creating Domain-Specific Stopword Lists

For different domains within HR, create specialized stopword lists:

1. **Technical Positions**:
   - Combine general language stopwords with technical terminology
   - Include common programming language names that aren't discriminating

2. **Management Positions**:
   - Combine general language stopwords with management terminology
   - Include common business and leadership terms

3. **Domain-Specific Roles**:
   - For each industry (healthcare, finance, etc.), create specialized lists
   - Include industry jargon that appears frequently but doesn't add discriminative value

## Handling Large Datasets

For large datasets with many resumes:

1. **Preload stopwords** once at the beginning of processing
2. **Cache language-specific stopwords** to avoid repeated loading
3. **Combine multiple sources upfront** rather than on each processing call
4. **Save combined stopword lists** to disk for reuse in future processing

## Dependencies

The module has the following dependencies:

1. **Required**:
   - `os`, `json`, `glob`, `logging` (Python standard library)
   - `pamola_core.utils.nlp.compatibility` (Internal module)

2. **Optional**:
   - `nltk` - For enhanced stopword coverage across multiple languages

## Resource Management

Stopword resources can be managed in several ways:

1. **Default Location**: The module looks for stopword files in the default resources directory (`/path/to/pamola_core/resources/stopwords/`)

2. **Environment Variable**: Override the default path by setting the `PAMOLA_STOPWORDS_DIR` environment variable:
   ```bash
   export PAMOLA_STOPWORDS_DIR=/custom/path/to/stopwords
   ```

3. **Direct Specification**: Provide explicit file paths when calling `get_stopwords()`

## Conclusion

The `stopwords.py` module provides a robust, flexible foundation for handling stopwords in the PAMOLA.CORE system. By supporting multiple languages, diverse sources, and graceful degradation, it ensures reliable text preprocessing across various NLP tasks in resume and job posting analysis. The module's architecture allows for easy extension and customization while maintaining high performance, making it an essential component of the PAMOLA.CORE natural language processing pipeline.