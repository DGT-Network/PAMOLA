# PAMOLA Core: NLP Model Manager Module

## Overview

The **NLP Model Manager** module provides centralized management of NLP models within the PAMOLA Core framework. It offers a unified interface for loading, caching, and managing various NLP models (such as spaCy, HuggingFace Transformers, and NER models), with built-in support for memory optimization, dependency checking, and graceful degradation when models or libraries are unavailable.

This module is designed to simplify the integration and orchestration of NLP capabilities in data pipelines, ensuring efficient resource usage and robust error handling.

---

## Key Features

- **Centralized Model Management**: Unified interface for loading, accessing, and unloading NLP models.
- **Caching and Memory Optimization**: Automatic caching of loaded models with configurable limits and expiry.
- **Dependency Detection**: Checks for required NLP libraries and handles missing dependencies gracefully.
- **Graceful Degradation**: Provides fallback mechanisms when models or libraries are unavailable.
- **Support for Multiple Libraries**: Works with spaCy, HuggingFace Transformers, NLTK, and custom entity extractors.
- **Singleton Pattern**: Ensures a single manager instance throughout the application.
- **Extensible Loader System**: Easily add support for new model types or libraries.
- **Detailed Model Metadata**: Tracks model type, language, parameters, and initialization details.
- **Memory Usage Reporting**: Optionally reports process memory usage if `psutil` is available.

---

## Dependencies

### Standard Library
- `gc`, `logging`, `os`, `sys`
- `typing` (Dict, Any, Optional, List, Set, Callable)

### Internal Modules
- `pamola_core.utils.nlp.base`  
  - `DependencyManager`, `normalize_language_code`, `ModelNotAvailableError`
- `pamola_core.utils.nlp.cache`  
  - `get_cache`, `cache_function`
- `pamola_core.utils.nlp.entity_extraction` (for entity extractors)

---

## Exception Classes

### ModelLoadError
- **Base class**: `ModelNotAvailableError`
- **Description**: Raised when a model fails to load due to missing files, download errors, or runtime issues.
- **When Raised**: During model loading (e.g., spaCy or Transformers) if the model cannot be loaded or downloaded.

**Example:**
```python
from pamola_core.utils.nlp.model_manager import ModelLoadError, NLPModelManager

try:
    model = NLPModelManager().get_model('spacy', 'en')
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
```

### ModelNotAvailableError
- **Base class**: Exception (imported from base)
- **Description**: Raised when a required NLP library is not available in the environment.
- **When Raised**: When attempting to load a model for which the underlying library is not installed.

**Example:**
```python
from pamola_core.utils.nlp.model_manager import ModelNotAvailableError, NLPModelManager

try:
    model = NLPModelManager().get_model('transformers', 'en')
except ModelNotAvailableError as e:
    print(f"Required library not available: {e}")
```

---

## Main Class: `NLPModelManager`

### Constructor
```python
NLPModelManager()
```
**Parameters:** None (singleton pattern; use `NLPModelManager()` to get the instance)

### Key Attributes
- `_model_cache`: Model cache instance for storing loaded models.
- `_max_models`: Maximum number of models to keep in memory.
- `_model_expiry`: Time (seconds) after which unused models may be unloaded.
- `_nlp_libraries`: Set of detected NLP libraries in the environment.
- `_loaders`: Mapping of model type strings to loader methods.

### Public Methods

#### get_model
```python
def get_model(
    self,
    model_type: str,
    language: str,
    **params
) -> Any
```
- **Parameters:**
    - `model_type`: Type of model to retrieve (e.g., 'spacy', 'transformers', 'ner').
    - `language`: Language code (e.g., 'en', 'ru').
    - `**params`: Additional model configuration parameters.
- **Returns:** Loaded model instance.
- **Raises:**
    - `ModelNotAvailableError`: Required library is not available.
    - `ModelLoadError`: Model loading fails at runtime.

#### unload_model
```python
def unload_model(
    self,
    model_key: str
) -> bool
```
- **Parameters:**
    - `model_key`: Unique key for the model in cache.
- **Returns:** `True` if model was removed, `False` otherwise.

#### clear_models
```python
def clear_models(self) -> None
```
- **Description:** Unloads all models and clears the cache. Triggers garbage collection.

#### get_supported_model_types
```python
def get_supported_model_types(self) -> List[str]
```
- **Returns:** List of recognized model types that can be loaded (based on detected libraries).

#### get_model_info
```python
def get_model_info(
    self,
    model_key: Optional[str] = None
) -> Dict[str, Any]
```
- **Parameters:**
    - `model_key`: (Optional) Specific model key for detailed info.
- **Returns:** Dictionary with metadata about the requested model(s).

#### get_memory_stats
```python
def get_memory_stats(self) -> Dict[str, Any]
```
- **Returns:** Memory usage information (RSS, VMS, percent) if `psutil` is available, else fallback message.

#### set_max_models
```python
def set_max_models(
    self,
    max_models: int
) -> None
```
- **Parameters:**
    - `max_models`: Maximum number of models allowed in memory.

#### set_model_expiry
```python
def set_model_expiry(
    self,
    expiry_seconds: int
) -> None
```
- **Parameters:**
    - `expiry_seconds`: Number of seconds to keep a model in the cache.

#### check_model_availability
```python
@cache_function(ttl=300, cache_type='memory')
def check_model_availability(
    self,
    model_type: str,
    language: str
) -> Dict[str, Any]
```
- **Parameters:**
    - `model_type`: Type of model (e.g., 'spacy', 'ner', 'transformers', 'nltk').
    - `language`: Language code.
- **Returns:** Information about the model's potential availability.

---

## Dependency Resolution and Validation Logic

- **DependencyManager** is used to check for the presence of required NLP libraries (e.g., spaCy, transformers, nltk, torch, tensorflow).
- The manager checks for available libraries at initialization and before loading models, raising `ModelNotAvailableError` if a required library is missing.
- Model loading methods attempt to load or download models as needed, raising `ModelLoadError` on failure.
- Caching is used to avoid redundant model loading and to optimize memory usage.

---

## Usage Examples

### Loading a spaCy Model
```python
from pamola_core.utils.nlp.model_manager import NLPModelManager

# Get the singleton manager instance
manager = NLPModelManager()

# Load an English spaCy model
spacy_model = manager.get_model('spacy', 'en')
```

### Handling Model Loading Errors
```python
try:
    model = manager.get_model('spacy', 'xx')  # Nonexistent language code
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
```

### Checking Model Availability
```python
info = manager.check_model_availability('spacy', 'en')
print(info)
```

### Unloading and Clearing Models
```python
# Unload a specific model
key = manager._get_model_key('spacy', 'en', {})
manager.unload_model(key)

# Clear all models
manager.clear_models()
```

### Using with a Pipeline Task
```python
# Example integration with a BaseTask
class MyTask(BaseTask):
    def run(self):
        nlp = NLPModelManager().get_model('spacy', 'en')
        # ... use nlp ...
```

### Continue-on-Error Example
```python
try:
    model = manager.get_model('transformers', 'en', model_name='nonexistent-model')
except ModelLoadError as e:
    logger.warning(f"Model load failed, continuing: {e}")
    # Fallback or skip
```

---

## Integration Notes

- The manager is designed to be used as a singleton (`NLPModelManager()`), ensuring consistent model management across the application.
- Integrates seamlessly with pipeline tasks (e.g., `BaseTask`) for NLP processing.
- Use the `get_model` method to retrieve models as needed; caching and dependency checks are handled automatically.

---

## Error Handling and Exception Hierarchy

- **ModelNotAvailableError**: Raised when a required NLP library is not installed. Inherits from the base exception in `nlp.base`.
- **ModelLoadError**: Raised when a model cannot be loaded or downloaded. Inherits from `ModelNotAvailableError`.

**Handling Example:**
```python
try:
    model = manager.get_model('spacy', 'en')
except ModelNotAvailableError:
    print("spaCy is not installed.")
except ModelLoadError:
    print("spaCy model could not be loaded.")
```

---

## Configuration Requirements

- The manager reads environment variables for configuration:
    - `PAMOLA_MAX_MODELS`: Maximum number of models to keep loaded (default: 5).
    - `PAMOLA_MODEL_EXPIRY`: Model cache expiry in seconds (default: 3600).
- These can be set in your environment or pipeline configuration.

---

## Security Considerations and Best Practices

- **Path Security**: Avoid using absolute paths for model files unless necessary. Prefer logical task IDs and internal references.
- **Risks of Disabling Path Security**: Allowing arbitrary absolute paths can expose the system to path traversal or unauthorized data access.

**Security Failure Example:**
```python
# BAD: Loading a model from an untrusted absolute path
model = manager.get_model('spacy', '/tmp/untrusted_model')  # Risk: arbitrary code/data
```
**Mitigation:**
- Only use absolute paths for trusted, external data sources.
- Validate and sanitize all external paths before use.

---

## Best Practices

1. **Use Logical Model Types**: Prefer using model types like 'spacy', 'ner', or 'transformers' for clarity and maintainability.
2. **Leverage Caching**: Rely on the manager's caching to avoid redundant model loading and optimize memory usage.
3. **Handle Exceptions Explicitly**: Always catch `ModelNotAvailableError` and `ModelLoadError` to provide robust error handling in your pipeline.
4. **Configure Limits Appropriately**: Set `PAMOLA_MAX_MODELS` and `PAMOLA_MODEL_EXPIRY` according to your workload and memory constraints.
5. **Avoid Hardcoding Paths**: Use logical references and let the manager handle model resolution and loading.
6. **Monitor Memory Usage**: Use `get_memory_stats()` to monitor and manage memory consumption in production environments.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical model types and language codes (e.g., 'spacy', 'en') for models managed within the pipeline.
- **External (Absolute Path) Dependencies**: Only use absolute paths for models or data not produced by the pipeline. Ensure paths are trusted and validated.
