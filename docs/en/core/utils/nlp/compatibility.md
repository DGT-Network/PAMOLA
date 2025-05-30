# PAMOLA Core: NLP Compatibility Utilities Module

## Overview

The `compatibility.py` module provides robust utilities for managing and validating NLP-related dependencies within the PAMOLA Core framework. It ensures that NLP features gracefully degrade when optional dependencies are missing, and offers a unified interface for dependency checking, resource setup, and status reporting. This module is essential for building resilient NLP pipelines that can adapt to varying environments and configurations.

---

## Key Features

- **Dependency Checking**: Verify the presence of required Python modules for NLP tasks.
- **Dependency Status Logging**: Log the availability of all NLP dependencies for diagnostics.
- **Dependency Information**: Retrieve detailed status, versions, and metadata for all NLP dependencies.
- **Best-Available Module Selection**: Select the most preferred available module from a list.
- **Dependency Cache Management**: Clear cached dependency checks to force re-evaluation.
- **Requirements Validation**: Check if feature requirements are met based on available modules.
- **NLP Resource Setup**: Check and optionally download required NLP resources (e.g., NLTK data, spaCy models).
- **Graceful Degradation**: Provides fallback logic when dependencies or resources are missing.

---

## Dependencies

### Standard Library
- `logging`
- `typing` (`Dict`, `Any`, `Optional`, `List`)
- `sys`

### Internal Modules
- `pamola_core.utils.nlp.base.DependencyManager`
- `pamola_core.utils.nlp.cache.cache_function`

---

## Exception Classes

> **Note:** This module does not define custom exception classes directly. All exceptions are raised by the underlying `DependencyManager` or standard library modules. Typical exceptions to handle include `ImportError`, `OSError`, and custom exceptions from `DependencyManager`.

### Example: Handling Dependency Errors

```python
try:
    if not check_dependency('spacy'):
        raise ImportError('spaCy is not installed')
except ImportError as e:
    # Handle missing dependency gracefully
    logger.warning(f"Dependency error: {e}")
```

**When Raised:**
- When a required module is not installed or importable.
- When a resource (e.g., NLTK data) is missing and cannot be downloaded.

---

## Main Functions and Their Usage

### check_dependency

```python
def check_dependency(module_name: str) -> bool:
    """
    Check if a specific Python module (dependency) is available.
    """
```
- **Parameters:**
    - `module_name`: Name of the Python module to check.
- **Returns:**
    - `bool`: `True` if the module is installed and importable, else `False`.
- **Raises:**
    - None (returns `False` on failure).

### log_nlp_status

```python
def log_nlp_status() -> None:
    """
    Log the status of all NLP dependencies.
    """
```
- **Parameters:** None
- **Returns:** None
- **Raises:** None

### dependency_info

```python
@cache_function(ttl=3600)
def dependency_info(verbose: bool = False) -> Dict[str, Any]:
    ...
```
- **Parameters:**
    - `verbose`: If `True`, includes version and metadata details.
- **Returns:**
    - `Dict[str, Any]`: Dictionary with dependency status, counts, Python version, and system info.
- **Raises:**
    - None (errors are logged and included in the result dictionary).

### get_best_available_module

```python
def get_best_available_module(module_preferences: List[str]) -> Optional[str]:
    ...
```
- **Parameters:**
    - `module_preferences`: List of module names in order of preference.
- **Returns:**
    - `str` or `None`: Name of the best available module, or `None` if none are installed.
- **Raises:** None

### clear_dependency_cache

```python
def clear_dependency_cache() -> None:
    ...
```
- **Parameters:** None
- **Returns:** None
- **Raises:** None

### check_nlp_requirements

```python
def check_nlp_requirements(requirements: Dict[str, List[str]]) -> Dict[str, bool]:
    ...
```
- **Parameters:**
    - `requirements`: `{ feature_name: [list_of_required_modules] }`
- **Returns:**
    - `Dict[str, bool]`: `{ feature_name: True/False }`
- **Raises:** None

### setup_nlp_resources

```python
def setup_nlp_resources(download_if_missing: bool = True) -> Dict[str, bool]:
    ...
```
- **Parameters:**
    - `download_if_missing`: Whether to attempt auto-downloading missing data.
- **Returns:**
    - `Dict[str, bool]`: Mapping of resource name to availability status.
- **Raises:**
    - Logs errors for failed downloads or missing resources.

---

## Dependency Resolution and Validation Logic

- **Dependency Checking**: Uses `DependencyManager.check_dependency` to determine if a module is importable.
- **Version Validation**: Uses `DependencyManager.check_version` to verify module versions if needed.
- **Best-Available Selection**: Returns the first available module from a preference list.
- **Resource Setup**: Checks for required NLP resources (NLTK, spaCy) and attempts to download them if missing and permitted.
- **Cache Management**: Dependency checks are cached for performance; can be cleared with `clear_dependency_cache()`.

---

## Usage Examples

### 1. Checking and Logging Dependency Status

```python
from pamola_core.utils.nlp.compatibility import log_nlp_status

# Log the status of all NLP dependencies
def main():
    log_nlp_status()
```

### 2. Handling Failed Dependencies

```python
from pamola_core.utils.nlp.compatibility import check_dependency

if not check_dependency('spacy'):
    # spaCy is not available; fallback to another library or skip feature
    print('spaCy not available, using fallback.')
```

### 3. Using Best-Available Module

```python
from pamola_core.utils.nlp.compatibility import get_best_available_module

preferred = ['spacy', 'nltk', 'textblob']
best = get_best_available_module(preferred)
if best:
    print(f'Using {best} for NLP tasks')
else:
    print('No suitable NLP module found')
```

### 4. Setting Up NLP Resources

```python
from pamola_core.utils.nlp.compatibility import setup_nlp_resources

# Ensure all required resources are available (downloads if missing)
resources_status = setup_nlp_resources(download_if_missing=True)
print(resources_status)
```

### 5. Clearing the Dependency Cache

```python
from pamola_core.utils.nlp.compatibility import clear_dependency_cache

# Clear cached dependency checks (useful after installing new packages)
clear_dependency_cache()
```

### 6. Integration with BaseTask

```python
from pamola_core.utils.nlp.compatibility import check_nlp_requirements

requirements = {
    'tokenization': ['spacy', 'nltk'],
    'lemmatization': ['spacy'],
}
status = check_nlp_requirements(requirements)
# Use status dict to enable/disable features in your BaseTask
```

---

## Error Handling and Exception Hierarchy

- **ImportError**: Raised when a required module is missing.
- **OSError**: Raised when a resource (e.g., NLTK data) is missing and cannot be downloaded.
- **Custom Exceptions**: May be raised by `DependencyManager` (see its documentation for details).

**Example:**
```python
try:
    setup_nlp_resources()
except OSError as e:
    logger.error(f"Resource setup failed: {e}")
```

---

## Configuration Requirements

- No explicit configuration is required for this module.
- For advanced use, ensure that the `DependencyManager` is properly configured (see its documentation).
- If using in a pipeline, ensure that all required modules are listed in your environment or requirements file.

---

## Security Considerations and Best Practices

- **Path Security**: Avoid using absolute paths for dependencies unless necessary. Absolute paths may expose sensitive data or allow access to unintended files.
- **Best Practice**: Use logical task IDs for internal dependencies; only use absolute paths for external data.

### Security Failure Example

```python
# BAD: Using an untrusted absolute path
external_data = '/tmp/untrusted/data.csv'
if check_dependency(external_data):
    # This may expose or overwrite sensitive files
    ...

# GOOD: Use only trusted, validated paths or task IDs
trusted_data = get_best_available_module(['my_internal_task', 'external_data'])
```

**Risks of Disabling Path Security:**
- May allow access to files outside the intended project scope.
- Increases risk of data leakage or unauthorized access.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use task IDs or logical names for data produced within the pipeline.
- **External Dependencies**: Use absolute paths only for data not produced by the pipeline and ensure paths are validated.

---

## Best Practices

1. **Use Task IDs for Internal Dependencies**: Maintain logical connections within your project.
2. **Use Absolute Paths Judiciously**: Only for truly external data.
3. **Clear Dependency Cache After Installing New Packages**: Ensures up-to-date status.
4. **Log Dependency Status Regularly**: For easier debugging and diagnostics.
5. **Handle Missing Dependencies Gracefully**: Always provide fallbacks or user warnings.
