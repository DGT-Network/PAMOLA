# PAMOLA.CORE NLP Compatibility Module Technical Documentation

**Module:** `pamola_core.utils.nlp.compatibility`  
**Version:** 1.0.0  
**Last Updated:** December 2024  
**Status:** Stable  

## 1. Overview

### 1.1 Purpose

The compatibility module provides a unified interface for managing NLP dependencies and handling graceful degradation when certain libraries are unavailable. It acts as a lightweight wrapper around `DependencyManager` while providing additional utilities for resource management and system diagnostics.

### 1.2 Key Features

- **Dependency Checking**: Verify availability of NLP libraries
- **Resource Management**: Setup and download NLP resources (NLTK data, spaCy models)
- **System Information**: Comprehensive dependency status reporting
- **Graceful Degradation**: Enable partial functionality when dependencies are missing
- **Caching**: Efficient dependency status caching to minimize overhead

### 1.3 Design Principles

1. **Zero Required Dependencies**: The module itself has no hard dependencies
2. **Lazy Loading**: Dependencies are only imported when needed
3. **Comprehensive Reporting**: Detailed status information for debugging
4. **Resource Automation**: Automatic download of missing resources when possible
5. **Performance Conscious**: Cached results to avoid repeated checks

## 2. Core Functions

### 2.1 Dependency Checking

#### `check_dependency(module_name: str) -> bool`

Checks if a specific Python module is available for import.

```python
# Example usage
if check_dependency('spacy'):
    # Use spaCy functionality
    import spacy
    nlp = spacy.load('en_core_web_sm')
else:
    # Fall back to simpler tokenization
    logger.warning("spaCy not available, using basic tokenization")
```

**Parameters:**
- `module_name`: Name of the Python module to check

**Returns:**
- `bool`: True if module is installed and importable

**Notes:**
- Results are cached internally by `DependencyManager`
- Thread-safe implementation

### 2.2 Status Reporting

#### `log_nlp_status() -> None`

Logs the availability status of all NLP dependencies to the configured logger.

```python
# Example output in logs
# INFO: NLP dependencies status:
# INFO:   - nltk: Available
# INFO:   - spacy: Available
# INFO:   - transformers: Not available
# INFO:   - fasttext: Not available
# INFO: NLP availability: 2/4 dependencies available
```

**Use Cases:**
- Application startup diagnostics
- Debugging dependency issues
- System health checks

#### `dependency_info(verbose: bool = False) -> Dict[str, Any]`

Returns comprehensive information about NLP dependencies and system configuration.

```python
# Basic usage
info = dependency_info()
# Returns: {
#     "available": {"nltk": True, "spacy": True, ...},
#     "count": {"total": 10, "available": 6},
#     "python_version": "3.9.7...",
#     "system": "linux"
# }

# Verbose mode
info = dependency_info(verbose=True)
# Additionally includes:
# - "versions": {"nltk": "3.7", "spacy": "3.4.1", ...}
# - "details": module paths and dependencies (if verbose > 1)
```

**Parameters:**
- `verbose`: Enable detailed information including versions and module paths

**Returns:**
- Dictionary with dependency status, counts, versions, and system information

**Caching:**
- Results cached for 1 hour (3600 seconds)
- Use `clear_dependency_cache()` to force refresh

### 2.3 Module Selection

#### `get_best_available_module(module_preferences: List[str]) -> Optional[str]`

Selects the first available module from a preference list.

```python
# Example: Choose best available tokenizer
tokenizer_module = get_best_available_module([
    'spacy',          # Preferred
    'nltk',           # Second choice
    'regex'           # Fallback
])

if tokenizer_module == 'spacy':
    # Use spaCy tokenizer
    pass
elif tokenizer_module == 'nltk':
    # Use NLTK tokenizer
    pass
else:
    # Use basic regex tokenizer
    pass
```

**Parameters:**
- `module_preferences`: List of module names in order of preference

**Returns:**
- Name of the first available module, or None if none available

### 2.4 Requirements Checking

#### `check_nlp_requirements(requirements: Dict[str, List[str]]) -> Dict[str, bool]`

Checks availability of features based on required modules.

```python
# Define feature requirements
requirements = {
    "entity_extraction": ["spacy", "nltk"],
    "embeddings": ["sentence_transformers", "fasttext"],
    "language_detection": ["langdetect", "fasttext"]
}

# Check what's available
available_features = check_nlp_requirements(requirements)
# Returns: {
#     "entity_extraction": True,    # At least one module available
#     "embeddings": False,         # No modules available
#     "language_detection": True
# }
```

**Parameters:**
- `requirements`: Dictionary mapping feature names to lists of required modules

**Returns:**
- Dictionary mapping feature names to availability status

**Logic:**
- A feature is available if at least ONE of its required modules is installed
- Useful for feature flags and conditional functionality

### 2.5 Resource Management

#### `setup_nlp_resources(download_if_missing: bool = True) -> Dict[str, bool]`

Checks and optionally downloads NLP resources like NLTK data and spaCy models.

```python
# Check and download missing resources
resource_status = setup_nlp_resources(download_if_missing=True)
# Returns: {
#     "nltk_punkt": True,
#     "nltk_wordnet": True,
#     "nltk_stopwords": True,
#     "spacy_en_core_web_sm": True,
#     "spacy_ru_core_news_sm": False  # Failed to download
# }

# Just check without downloading
resource_status = setup_nlp_resources(download_if_missing=False)
```

**Parameters:**
- `download_if_missing`: Whether to attempt downloading missing resources

**Returns:**
- Dictionary mapping resource names to availability status

**Supported Resources:**
- **NLTK**: punkt (tokenizer), wordnet, stopwords
- **spaCy**: en_core_web_sm, ru_core_news_sm

**Notes:**
- Downloads are performed quietly in the background
- Errors during download are logged but don't raise exceptions
- Network connectivity required for downloads

### 2.6 Cache Management

#### `clear_dependency_cache() -> None`

Clears the dependency check cache.

```python
# Force re-check of all dependencies
clear_dependency_cache()

# Useful after installing new packages
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "spacy"])
clear_dependency_cache()
```

**Use Cases:**
- After installing/uninstalling packages
- When troubleshooting dependency issues
- In development environments

## 3. Integration Patterns

### 3.1 Feature Degradation Pattern

```python
from pamola_core.utils.nlp.compatibility import check_dependency, get_best_available_module

class TextProcessor:
    def __init__(self):
        # Determine available features
        self.has_advanced_nlp = check_dependency('spacy')
        self.tokenizer_module = get_best_available_module(['spacy', 'nltk', 'regex'])
        
    def extract_entities(self, text):
        if self.has_advanced_nlp:
            return self._extract_entities_spacy(text)
        else:
            logger.warning("Advanced NLP not available, using pattern matching")
            return self._extract_entities_regex(text)
```

### 3.2 Startup Diagnostics Pattern

```python
def initialize_nlp_system():
    """Initialize NLP system with diagnostics."""
    # Log dependency status
    log_nlp_status()
    
    # Get detailed info for metrics
    dep_info = dependency_info(verbose=True)
    metrics.record("nlp.dependencies.available", dep_info["count"]["available"])
    metrics.record("nlp.dependencies.total", dep_info["count"]["total"])
    
    # Setup resources
    resources = setup_nlp_resources(download_if_missing=True)
    
    # Check critical features
    critical_features = check_nlp_requirements({
        "tokenization": ["nltk", "spacy"],
        "entity_extraction": ["spacy"]
    })
    
    if not critical_features["tokenization"]:
        raise RuntimeError("No tokenization library available")
    
    return dep_info, resources
```

### 3.3 Conditional Import Pattern

```python
from pamola_core.utils.nlp.compatibility import check_dependency

# Conditional imports with fallbacks
if check_dependency('transformers'):
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
else:
    sentiment_analyzer = None
    logger.info("Transformers not available, sentiment analysis disabled")

def analyze_sentiment(text):
    if sentiment_analyzer:
        return sentiment_analyzer(text)
    else:
        return {"label": "NEUTRAL", "score": 0.5, "error": "No sentiment model available"}
```

## 4. Error Handling

### 4.1 Common Scenarios

1. **Missing Dependencies**
   - Functions return False/None for unavailable modules
   - No exceptions raised for missing dependencies
   - Appropriate warnings logged

2. **Download Failures**
   - Errors logged but execution continues
   - Status dictionary reflects failed downloads
   - Network timeouts handled gracefully

3. **Import Errors**
   - Caught and logged internally
   - Safe to call functions even if base dependencies missing

### 4.2 Best Practices

```python
# Always check before using optional features
if check_dependency('optional_lib'):
    from optional_lib import advanced_feature
    result = advanced_feature(data)
else:
    # Provide meaningful fallback
    result = basic_fallback(data)
    logger.info("Using basic implementation due to missing dependencies")

# Use requirements checking for feature flags
features = check_nlp_requirements({
    "advanced_search": ["elasticsearch", "whoosh"],
    "ml_models": ["transformers", "tensorflow", "torch"]
})

app_config = {
    "enable_advanced_search": features["advanced_search"],
    "enable_ml_features": features["ml_models"]
}
```

## 5. Performance Considerations

### 5.1 Caching Strategy

- Dependency checks are cached by `DependencyManager`
- `dependency_info()` results cached for 1 hour
- Resource availability not cached (checked each time)

### 5.2 Optimization Tips

1. **Check dependencies once at startup**
   ```python
   class NLPService:
       def __init__(self):
           self._dependencies = dependency_info()
           self._setup_complete = False
       
       def _ensure_setup(self):
           if not self._setup_complete:
               setup_nlp_resources()
               self._setup_complete = True
   ```

2. **Batch dependency checks**
   ```python
   # Instead of multiple calls
   # if check_dependency('nltk') and check_dependency('spacy'):
   
   # Use requirements checking
   deps_available = check_nlp_requirements({
       "all_required": ["nltk", "spacy"]
   })
   if deps_available["all_required"]:
       # Both available
   ```

## 6. Configuration

### 6.1 Environment Variables

While the module doesn't directly use environment variables, it respects:
- Standard Python module search paths
- pip/conda installation locations
- Virtual environment configurations

### 6.2 Logging Configuration

The module uses the standard Python logging framework:

```python
import logging

# Configure logging level for compatibility checks
logging.getLogger('pamola_core.utils.nlp.compatibility').setLevel(logging.INFO)
```

## 7. Testing

### 7.1 Unit Testing

```python
import unittest
from unittest.mock import patch
from pamola_core.utils.nlp.compatibility import check_dependency, dependency_info

class TestCompatibility(unittest.TestCase):
    def test_check_dependency_builtin(self):
        # Built-in modules should always be available
        self.assertTrue(check_dependency('os'))
        self.assertTrue(check_dependency('sys'))
    
    def test_check_dependency_missing(self):
        # Fake module should not be available
        self.assertFalse(check_dependency('definitely_not_a_real_module_12345'))
    
    @patch('pamola_core.utils.nlp.base.DependencyManager.get_nlp_status')
    def test_dependency_info(self, mock_status):
        mock_status.return_value = {'nltk': True, 'spacy': False}
        info = dependency_info()
        
        self.assertEqual(info['count']['total'], 2)
        self.assertEqual(info['count']['available'], 1)
```

### 7.2 Integration Testing

```python
def test_nlp_system_integration():
    """Test full NLP system initialization."""
    # Clear any cached data
    clear_dependency_cache()
    
    # Check and setup resources
    resources = setup_nlp_resources(download_if_missing=False)
    
    # Verify critical dependencies
    critical_deps = check_nlp_requirements({
        "tokenization": ["nltk", "spacy", "regex"],
        "language_detection": ["langdetect", "fasttext"]
    })
    
    assert critical_deps["tokenization"], "No tokenization library available"
    
    # Log final status
    log_nlp_status()
```

## 8. Troubleshooting

### 8.1 Common Issues

1. **"Module available but import fails"**
   - Check for version conflicts
   - Verify virtual environment activation
   - Clear dependency cache and retry

2. **"Resources fail to download"**
   - Check network connectivity
   - Verify proxy settings
   - Try manual download with library tools

3. **"Dependency status incorrect"**
   - Clear cache with `clear_dependency_cache()`
   - Check Python path configuration
   - Verify no namespace conflicts

### 8.2 Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('pamola_core.utils.nlp.compatibility')

# Get verbose information
debug_info = dependency_info(verbose=2)  # Maximum verbosity
print(json.dumps(debug_info, indent=2))
```

## 9. Future Enhancements

### 9.1 Planned Features

1. **Dependency Version Requirements**
   - Support for minimum version specifications
   - Compatibility matrix checking

2. **Resource Versioning**
   - Track versions of downloaded resources
   - Update notifications for outdated resources

3. **Performance Profiling**
   - Measure import times for each dependency
   - Optimize startup based on usage patterns

4. **Cloud Resource Support**
   - Download resources from cloud storage
   - Support for custom model repositories

### 9.2 API Stability

- Core functions are stable and backward compatible
- New parameters will have sensible defaults
- Deprecation warnings provided before removing features

## 10. Conclusion

The compatibility module provides essential infrastructure for building robust NLP applications that gracefully handle missing dependencies. By following the patterns and best practices outlined in this documentation, developers can create systems that provide the best possible functionality given available resources while maintaining reliability and performance.