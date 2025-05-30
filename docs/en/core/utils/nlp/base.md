# PAMOLA Core: NLP Base Utilities Module

## Overview

The `pamola_core.utils.nlp.base` module provides foundational utilities, constants, and base classes for all NLP components within the PAMOLA Core framework. It centralizes resource management, dependency handling, error definitions, and common utility functions to ensure consistency and reliability across NLP-related modules.

---

## Key Features

- Centralized management of NLP resource directories (stopwords, tokenization, dictionaries)
- Robust dependency management and version checking for external NLP libraries
- Comprehensive exception hierarchy for NLP-specific errors
- Utility functions for language code normalization and batch processing
- Abstract base class for cache implementations
- Sentinel object for representing missing values

---

## Supported Functionality

- **Resource Directory Management**: Ensures required directories exist for NLP resources, with support for environment variable overrides.
- **Dependency Management**: Checks for the presence and version of external libraries (e.g., `nltk`, `spacy`, `fasttext`).
- **Error Handling**: Defines custom exceptions for missing resources, unsupported languages, configuration errors, and more.
- **Utility Functions**: Includes helpers for language code normalization and parallel batch processing.
- **Cache Interface**: Provides a base class for implementing custom cache strategies.

---

## Dependencies

### Standard Library
- `os`, `sys`, `logging`, `importlib`, `multiprocessing`, `typing`

### Third-Party
- `packaging.version` (for version comparison)

### Internal
- None (self-contained within the `pamola_core` package)

---

## Exception Classes

### NLPError
Base class for all NLP module exceptions.

```python
try:
    # Some NLP operation
    ...
except NLPError as e:
    print(f"NLP error occurred: {e}")
```
**When raised:** Any generic NLP-related error not covered by more specific exceptions.

---

### ResourceNotFoundError
Raised when a required resource (e.g., stopwords file, dictionary) is missing.

```python
try:
    # Attempt to load a resource
    ...
except ResourceNotFoundError as e:
    print("Resource not found:", e)
```
**When raised:** Resource files or directories are missing or inaccessible.

---

### ModelNotAvailableError
Raised when a required NLP model is not available.

```python
try:
    # Attempt to load a model
    ...
except ModelNotAvailableError as e:
    print("Model not available:", e)
```
**When raised:** Model files are missing, or the model cannot be loaded.

---

### UnsupportedLanguageError
Raised when an unsupported language is requested.

```python
try:
    # Attempt to process an unsupported language
    ...
except UnsupportedLanguageError as e:
    print("Unsupported language:", e)
```
**When raised:** The requested language is not supported by the module or resource.

---

### ConfigurationError
Raised for configuration-related errors.

```python
try:
    # Invalid configuration detected
    ...
except ConfigurationError as e:
    print("Configuration error:", e)
```
**When raised:** Invalid or missing configuration parameters.

---

## Main Classes and Functions

### DependencyManager
Manages external dependency checks and version validation for NLP modules.

#### Constructor
This class is not instantiated; all methods are class or static methods.

#### Key Attributes
- `_dependency_cache: Dict[str, bool]` — Caches results of dependency checks.

#### Public Methods

##### check_dependency
```python
@classmethod
def check_dependency(cls, module_name: str) -> bool
```
- **Parameters:**
  - `module_name`: Name of the module to check.
- **Returns:** `bool` — True if available, False otherwise.
- **Raises:** None

##### get_module
```python
@classmethod
def get_module(cls, module_name: str) -> Optional[Any]
```
- **Parameters:**
  - `module_name`: Name of the module to import.
- **Returns:** The module object or None.
- **Raises:** None

##### check_version
```python
@classmethod
def check_version(
    cls,
    module_name: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> Tuple[bool, Optional[str]]
```
- **Parameters:**
  - `module_name`: Name of the module.
  - `min_version`: Minimum required version.
  - `max_version`: Maximum allowed version.
- **Returns:** Tuple of (requirements_met: bool, current_version: Optional[str])
- **Raises:** None (logs warnings if version cannot be determined)

##### clear_cache
```python
@classmethod
def clear_cache(cls) -> None
```
- **Parameters:** None
- **Returns:** None
- **Raises:** None

##### get_nlp_status
```python
@classmethod
def get_nlp_status(cls) -> Dict[str, bool]
```
- **Parameters:** None
- **Returns:** Dictionary mapping module names to availability.
- **Raises:** None

##### get_best_available_module
```python
@staticmethod
def get_best_available_module(module_preferences: List[str]) -> Optional[str]
```
- **Parameters:**
  - `module_preferences`: List of module names in order of preference.
- **Returns:** Name of the best available module or None.
- **Raises:** None

---

### Utility Functions

#### normalize_language_code
```python
def normalize_language_code(lang_code: str) -> str
```
- **Parameters:**
  - `lang_code`: Language code or name to normalize.
- **Returns:** Standardized 2-letter language code (default 'en').
- **Raises:** None

#### batch_process
```python
def batch_process(
    items: List[Any],
    process_func: Callable,
    processes: Optional[int] = None,
    **kwargs
) -> List[Any]
```
- **Parameters:**
  - `items`: List of items to process.
  - `process_func`: Function to apply to each item.
  - `processes`: Number of processes (optional).
  - `**kwargs`: Additional arguments for `process_func`.
- **Returns:** List of results.
- **Raises:** None

---

### Sentinel Object

#### MISSING
A singleton object representing a missing value. Used to distinguish between `None` and an unset value.

---

### CacheBase
Abstract base class for cache implementations.

#### Methods

##### get
```python
def get(self, key: str) -> Any
```
- **Parameters:**
  - `key`: Cache key.
- **Returns:** Cached value or None.
- **Raises:** NotImplementedError

##### set
```python
def set(self, key: str, value: Any, **kwargs) -> None
```
- **Parameters:**
  - `key`: Cache key.
  - `value`: Value to store.
  - `**kwargs`: Implementation-specific arguments.
- **Returns:** None
- **Raises:** NotImplementedError

##### delete
```python
def delete(self, key: str) -> bool
```
- **Parameters:**
  - `key`: Cache key.
- **Returns:** True if deleted, False otherwise.
- **Raises:** NotImplementedError

##### clear
```python
def clear(self) -> None
```
- **Parameters:** None
- **Returns:** None
- **Raises:** NotImplementedError

##### get_stats
```python
def get_stats(self) -> Dict[str, Any]
```
- **Parameters:** None
- **Returns:** Cache statistics.
- **Raises:** NotImplementedError

##### get_or_set
```python
def get_or_set(self, key: str, default_func: Callable[[], Any], **kwargs) -> Any
```
- **Parameters:**
  - `key`: Cache key.
  - `default_func`: Function to compute value if not present.
  - `**kwargs`: Arguments for `set`.
- **Returns:** Cached or computed value.
- **Raises:** None

##### get_model_info
```python
def get_model_info(self, key: Optional[str] = None) -> Dict[str, Any]
```
- **Parameters:**
  - `key`: Optional cache key.
- **Returns:** Metadata dictionary.
- **Raises:** NotImplementedError

---

## Dependency Resolution and Validation Logic

- **DependencyManager** checks for module presence and version, caches results, and provides status for common NLP libraries.
- Version checks use `packaging.version` for robust comparison, with fallbacks for missing version info.
- All dependency checks are cached for efficiency and can be cleared with `clear_cache()`.

---

## Usage Examples

### Checking for a Dependency
```python
from pamola_core.utils.nlp.base import DependencyManager

# Check if spaCy is available
if DependencyManager.check_dependency('spacy'):
    print("spaCy is installed!")
else:
    print("spaCy is not available.")
```

### Handling a Missing Resource
```python
from pamola_core.utils.nlp.base import ResourceNotFoundError

try:
    # Attempt to load a stopwords file
    ...
except ResourceNotFoundError:
    # Handle missing resource gracefully
    print("Stopwords resource is missing.")
```

### Batch Processing with Multiprocessing
```python
from pamola_core.utils.nlp.base import batch_process

def process_item(item):
    # Process a single item
    return item * 2

items = [1, 2, 3, 4, 5]
results = batch_process(items, process_item)
print(results)  # [2, 4, 6, 8, 10]
```

### Using the CacheBase Interface
```python
from pamola_core.utils.nlp.base import CacheBase

class MyCache(CacheBase):
    def __init__(self):
        self._store = {}
    def get(self, key):
        return self._store.get(key)
    def set(self, key, value, **kwargs):
        self._store[key] = value
    def delete(self, key):
        return self._store.pop(key, None) is not None
    def clear(self):
        self._store.clear()
    def get_stats(self):
        return {'size': len(self._store)}
    def get_model_info(self, key=None):
        return {}

cache = MyCache()
cache.set('foo', 123)
print(cache.get('foo'))  # 123
```

---

## Integration Notes

- The module is designed to be used as a foundational utility for all NLP-related tasks in PAMOLA Core.
- Exception classes and dependency management are intended for use in higher-level task classes (e.g., `BaseTask`).
- Resource directory paths can be overridden via environment variables for flexible deployment.

---

## Error Handling and Exception Hierarchy

- All exceptions inherit from `NLPError` for unified error handling.
- Specific exceptions provide granular control for resource, model, language, and configuration errors.
- Example hierarchy:
  - `NLPError`
    - `ResourceNotFoundError`
    - `ModelNotAvailableError`
    - `UnsupportedLanguageError`
    - `ConfigurationError`

---

## Configuration Requirements

- Resource directories are set up automatically but can be overridden with environment variables:
  - `PAMOLA_STOPWORDS_DIR`
  - `PAMOLA_TOKENIZATION_DIR`
  - `PAMOLA_DICTIONARIES_DIR`
- Ensure these directories are accessible and writable by the application.

---

## Security Considerations and Best Practices

- **Path Security**: Only use absolute paths for external data. Internal dependencies should use logical task IDs.
- **Risks of Disabling Path Security**: Allowing arbitrary absolute paths can expose sensitive data or allow unauthorized access.

### Security Failure Example
```python
# BAD: Using an unvalidated absolute path from user input
user_path = input("Enter resource path: ")
with open(user_path) as f:
    data = f.read()  # Potential security risk!
```

### Secure Handling Example
```python
# GOOD: Restrict to known resource directories
from pamola_core.utils.nlp.base import STOPWORDS_DIR
import os

filename = 'english.txt'
filepath = os.path.join(STOPWORDS_DIR, filename)
with open(filepath) as f:
    data = f.read()
```

---

## Best Practices

1. **Use Logical Task IDs for Internal Dependencies**: Maintain clear data flows within your pipeline.
2. **Use Absolute Paths Only for External Data**: Avoid using absolute paths for internal resources.
3. **Handle Exceptions Explicitly**: Catch and handle specific exceptions for robust error management.
4. **Validate Configuration**: Ensure all required directories and configuration parameters are set and accessible.
5. **Leverage DependencyManager**: Use the provided methods to check for and validate external dependencies before use.
6. **Implement Custom Caches by Subclassing CacheBase**: Follow the interface for consistent cache behavior.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical identifiers (e.g., task IDs) for resources produced within the pipeline.
- **External Dependencies**: Use absolute paths only for data not managed by the pipeline (e.g., third-party datasets).
