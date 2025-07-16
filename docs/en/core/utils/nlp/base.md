# Module Documentation: `pamola_core.utils.nlp.base`

## Overview

The `base.py` module is the foundational component of the NLP package in PAMOLA.CORE. It provides fundamental utilities, constants, and base classes for all NLP package components, including error hierarchies, dependency management, resource path handling, and common utility functions.

**Version:** 1.2.0  
**Status:** stable  
**License:** BSD 3-Clause

## Key Features

- **Centralized error hierarchy** for NLP operations
- **Dependency checking and version management**
- **Resource directory management**
- **Common utility functions** for NLP tasks
- **Base cache interface** definition

## Module Structure

### 1. Resource Paths

The module defines and manages resource directories:

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')

# Resource subdirectories
STOPWORDS_DIR = os.environ.get('PAMOLA_STOPWORDS_DIR', os.path.join(RESOURCES_DIR, 'stopwords'))
TOKENIZATION_DIR = os.environ.get('PAMOLA_TOKENIZATION_DIR', os.path.join(RESOURCES_DIR, 'tokenization'))
DICTIONARIES_DIR = os.environ.get('PAMOLA_DICTIONARIES_DIR', os.path.join(RESOURCES_DIR, 'dictionaries'))
```

These paths can be overridden via environment variables for deployment flexibility.

### 2. Error Hierarchy

#### Base NLP Errors

```python
class NLPError(Exception)
```
Base class for all NLP module exceptions. All custom exceptions in the NLP module inherit from this class.

```python
class ResourceNotFoundError(NLPError)
```
Raised when a required resource (stopwords, dictionaries, etc.) is not found.

```python
class ModelNotAvailableError(NLPError)
```
Raised when a required NLP model (e.g., spaCy model) is not available or cannot be loaded.

```python
class UnsupportedLanguageError(NLPError)
```
Raised when operations are requested for an unsupported language.

```python
class ConfigurationError(NLPError)
```
Raised for configuration-related errors (invalid values, missing parameters, conflicts).

#### LLM-Specific Errors

```python
class LLMError(NLPError)
```
Base exception for all Large Language Model operations.

```python
class LLMConnectionError(LLMError)
```
Raised when LLM connection fails (network issues, authentication, timeouts).

```python
class LLMGenerationError(LLMError)
```
Raised during text generation (model failures, token limits, invalid parameters).

```python
class LLMResponseError(LLMError)
```
Raised when response is invalid (empty responses, parsing failures, validation errors).

### 3. Dependency Management

The `DependencyManager` class provides centralized dependency checking:

#### Key Methods

```python
@classmethod
def check_dependency(cls, module_name: str) -> bool
```
Checks if a Python module is available. Results are cached for performance.

```python
@classmethod
def get_module(cls, module_name: str) -> Optional[Any]
```
Imports and returns a module if available, otherwise returns None.

```python
@classmethod
def check_version(cls, module_name: str, min_version: Optional[str] = None, 
                  max_version: Optional[str] = None) -> Tuple[bool, Optional[str]]
```
Checks if a module's version meets specified requirements. Returns `(requirements_met, current_version)`.

```python
@classmethod
def get_nlp_status(cls) -> Dict[str, bool]
```
Returns availability status of common NLP dependencies:
- nltk, spacy, pymorphy2, langdetect, fasttext, transformers
- LLM-related: lmstudio, tiktoken, openai, anthropic

```python
@classmethod
def get_best_available_module(cls, module_preferences: List[str]) -> Optional[str]
```
Returns the first available module from a prioritized list.

### 4. Utility Functions

```python
def normalize_language_code(lang_code: str) -> str
```
Normalizes language codes to 2-letter format:
- Handles various formats: 'english' → 'en', 'ru_ru' → 'ru'
- Defaults to 'en' for unknown codes

```python
def batch_process(items: List[Any], process_func: Callable, 
                 processes: Optional[int] = None, **kwargs) -> List[Any]
```
Processes multiple items in parallel:
- Uses multiprocessing for large datasets (≥10 items)
- Automatically determines optimal process count
- Falls back to sequential processing for small datasets

### 5. Special Objects

```python
MISSING = _Missing()
```
A sentinel object representing missing values:
- Returns `False` in boolean context
- Useful for distinguishing between `None` and "truly missing"

### 6. Base Classes

```python
class CacheBase
```
Abstract base class for cache implementations. All cache implementations must implement:

- `get(key: str) -> Any`: Retrieve cached value
- `set(key: str, value: Any, **kwargs) -> None`: Store value
- `delete(key: str) -> bool`: Remove key
- `clear() -> None`: Clear all cache entries
- `get_stats() -> Dict[str, Any]`: Return usage statistics
- `get_or_set(key: str, default_func: Callable, **kwargs) -> Any`: Get or compute and set
- `get_model_info(key: Optional[str]) -> Dict[str, Any]`: Return metadata

## Usage Examples

### Error Handling
```python
from pamola_core.utils.nlp.base import NLPError, ResourceNotFoundError

try:
    load_resource("unknown_resource.txt")
except ResourceNotFoundError as e:
    logger.error(f"Resource not found: {e}")
except NLPError as e:
    logger.error(f"General NLP error: {e}")
```

### Dependency Checking
```python
from pamola_core.utils.nlp.base import DependencyManager

# Check if spaCy is available
if DependencyManager.check_dependency('spacy'):
    spacy = DependencyManager.get_module('spacy')
    
# Check version requirements
is_compatible, version = DependencyManager.check_version('transformers', min_version='4.0.0')
if not is_compatible:
    logger.warning(f"Transformers version {version} is below required 4.0.0")

# Get dependency status
nlp_status = DependencyManager.get_nlp_status()
```

### Batch Processing
```python
from pamola_core.utils.nlp.base import batch_process

def process_text(text):
    return text.lower().strip()

texts = ["Hello World", "PAMOLA Core", "NLP Processing"]
results = batch_process(texts, process_text, processes=4)
```

### Language Normalization
```python
from pamola_core.utils.nlp.base import normalize_language_code

lang1 = normalize_language_code("english")    # Returns: "en"
lang2 = normalize_language_code("ru_RU")      # Returns: "ru"
lang3 = normalize_language_code("unknown")    # Returns: "en" (default)
```

## Integration with PAMOLA.CORE

This module serves as the foundation for all NLP operations in PAMOLA.CORE:

1. **Error Handling**: All NLP components use the standardized error hierarchy
2. **Resource Management**: Centralized paths ensure consistent resource access
3. **Dependency Management**: Graceful handling of optional dependencies
4. **Cache Interface**: Consistent caching across all NLP components
5. **Utilities**: Common functions reduce code duplication

## Best Practices

1. **Always use the error hierarchy** for consistent exception handling
2. **Check dependencies** before using optional features
3. **Use resource paths** from this module rather than hardcoding
4. **Leverage batch_process** for parallel processing of multiple items
5. **Implement CacheBase** for new cache backends
6. **Use MISSING sentinel** instead of None when appropriate

## Thread Safety

- `DependencyManager` uses class-level caching (thread-safe for reads)
- `batch_process` uses multiprocessing (process-based parallelism)
- Resource paths are read-only after initialization

## Performance Considerations

- Dependency checks are cached to avoid repeated imports
- Small datasets (<10 items) bypass multiprocessing overhead
- Resource directories are created only once at import time

## Changelog

- **1.2.0**: Added LLM error hierarchy for new LLM subsystem
- **1.1.0**: Enhanced dependency management with version checking
- **1.0.0**: Initial implementation