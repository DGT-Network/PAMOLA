# PAMOLA Core: Tokenization Helpers Module

## Overview

The `tokenization_helpers` module provides essential utilities for tokenization workflows within the PAMOLA Core framework. It supports loading and managing tokenization configurations, synonym and n-gram dictionaries, and offers batch processing and progress tracking utilities. These tools are designed to streamline the setup and execution of NLP tokenization tasks, ensuring consistency, efficiency, and extensibility across language processing pipelines.

---

## Key Features

- **Configurable Tokenization**: Load and merge tokenization configurations from multiple sources, including language-specific files.
- **Synonym Dictionary Management**: Flexible loading and merging of synonym dictionaries in various formats.
- **N-gram Dictionary Support**: Efficient loading of n-gram lists from text or JSON files.
- **Batch Processing**: Parallel processing of items with customizable worker count.
- **Progress Tracking**: Unified progress tracking with optional console or progress bar display.
- **Caching**: File and memory caching for expensive operations.

---

## Dependencies

### Standard Library
- `json`
- `logging`
- `os`
- `typing`

### Internal Modules
- `pamola_core.utils.nlp.base` (`TOKENIZATION_DIR`, `batch_process`)
- `pamola_core.utils.nlp.cache` (`get_cache`, `cache_function`)

---

## Exception Classes

> **Note:** This module does not define custom exception classes. All errors are logged and handled internally, typically as `Exception`. If you wish to handle errors, catch `Exception` or use custom wrappers as needed.

**Example:**
```python
try:
    config = load_tokenization_config()
except Exception as e:
    # Handle or log the error
    print(f"Failed to load config: {e}")
```

**When exceptions are raised:**
- File not found, invalid JSON, or IO errors during config/dictionary loading.
- All such errors are logged using the module logger.

---

## Main Classes and Functions

### `load_tokenization_config`
```python
def load_tokenization_config(
    config_sources: Optional[Union[str, List[str]]] = None,
    language: Optional[str] = None
) -> Dict[str, Any]:
```
**Parameters:**
- `config_sources`: Path(s) to configuration files. If `None`, uses default and language-specific configs.
- `language`: Language code for language-specific configurations.

**Returns:**
- `Dict[str, Any]`: The merged configuration dictionary.

**Raises:**
- Logs and skips files on error (e.g., file not found, JSON decode error).

---

### `load_synonym_dictionary`
```python
def load_synonym_dictionary(
    sources: Optional[Union[str, List[str]]] = None,
    language: Optional[str] = None
) -> Dict[str, List[str]]:
```
**Parameters:**
- `sources`: Path(s) to synonym dictionary files. If `None`, uses default and language-specific files.
- `language`: Language code for language-specific dictionaries.

**Returns:**
- `Dict[str, List[str]]`: Dictionary mapping canonical forms to lists of synonyms.

**Raises:**
- Logs and skips files on error.

---

### `load_ngram_dictionary`
```python
def load_ngram_dictionary(
    sources: Optional[Union[str, List[str]]] = None,
    language: Optional[str] = None
) -> Set[str]:
```
**Parameters:**
- `sources`: Path(s) to n-gram dictionary files. If `None`, uses default and language-specific files.
- `language`: Language code for language-specific dictionaries.

**Returns:**
- `Set[str]`: Set of n-grams.

**Raises:**
- Logs and skips files on error.

---

### `batch_process`
```python
def batch_process(
    items: List[Any],
    process_func: Callable,
    processes: Optional[int] = None,
    **kwargs
) -> List[Any]:
```
**Parameters:**
- `items`: List of items to process.
- `process_func`: Function to apply to each item.
- `processes`: Number of processes to use (optional).
- `**kwargs`: Additional parameters for `process_func`.

**Returns:**
- `List[Any]`: List of processed items.

---

### `ProgressTracker`
```python
class ProgressTracker:
    def __init__(self, total: int, description: str = "", show: bool = False)
```
**Parameters:**
- `total`: Total number of items to process.
- `description`: Description of the operation.
- `show`: Whether to display progress (console or progress bar).

**Key Attributes:**
- `total`, `description`, `show`, `current`, `_tqdm`

**Public Methods:**
- `update(increment: int = 1) -> None`: Update progress by `increment`.
- `close() -> None`: Close the progress tracker.

---

## Dependency Resolution and Completion Validation

- **Config and Dictionary Loading**: The module attempts to load default and language-specific files. If a file is missing or invalid, it logs a warning and continues.
- **Merging Logic**: When multiple sources are provided, later files override earlier ones for configs, and synonyms/ngrams are merged.
- **Caching**: Results are cached (file or memory) for performance.

---

## Usage Examples

### Loading Tokenization Config
```python
# Load default and English-specific config
token_config = load_tokenization_config(language='en')

# Load from custom config file
custom_config = load_tokenization_config(config_sources='my_config.json')
```

### Loading Synonym Dictionary
```python
# Load default synonyms
synonyms = load_synonym_dictionary()

# Load language-specific synonyms
ru_synonyms = load_synonym_dictionary(language='ru')
```

### Loading N-gram Dictionary
```python
# Load default n-grams
ngrams = load_ngram_dictionary()

# Load from a custom file
ngrams = load_ngram_dictionary(sources='custom_ngrams.txt')
```

### Batch Processing with Progress Tracking
```python
from pamola_core.utils.nlp.tokenization_helpers import batch_process, ProgressTracker

def process_item(item):
    # ... process logic ...
    return result

items = [1, 2, 3, 4, 5]
tracker = ProgressTracker(total=len(items), description="Processing", show=True)

def wrapped_process(item):
    result = process_item(item)
    tracker.update()
    return result

results = batch_process(items, wrapped_process)
tracker.close()
```

---

## Integration Notes

- Designed for use in NLP pipelines and can be integrated with higher-level task classes (e.g., `BaseTask`).
- Caching and progress tracking are compatible with multiprocessing and large-scale data flows.

---

## Error Handling and Exception Hierarchy

- All file and data loading errors are caught and logged.
- No custom exception classes are defined; use `Exception` for broad error handling.
- Example:
```python
try:
    ngrams = load_ngram_dictionary()
except Exception as e:
    logger.error(f"Failed to load ngrams: {e}")
```

---

## Configuration Requirements

- **Config Files**: Should be valid JSON files, optionally language-specific (e.g., `en_config.json`).
- **Synonym Files**: Accepts both dict and list-of-dict formats.
- **N-gram Files**: Accepts plain text (one n-gram per line) or JSON (list or dict with `ngrams` key).
- **TOKENIZATION_DIR**: Must be set in `pamola_core.utils.nlp.base` and point to the directory containing config/dictionary files.

---

## Security Considerations and Best Practices

- **File Path Security**: Only load files from trusted locations. Avoid using untrusted or user-supplied paths.
- **Risks of Disabling Path Security**: If you allow arbitrary file paths, you risk loading malicious or corrupted files, which may compromise your pipeline.

**Security Failure Example:**
```python
# BAD: Loading from an untrusted path
config = load_tokenization_config(config_sources='/tmp/evil_config.json')
# This may result in code injection or data corruption if the file is malicious.
```

**Mitigation:**
- Always validate file paths and restrict to known directories.
- Use language-specific configs only from `TOKENIZATION_DIR`.

---

## Internal vs. External Dependencies

- **Internal**: Default and language-specific files within `TOKENIZATION_DIR`.
- **External**: Absolute paths provided as arguments; use with caution and validate sources.

---

## Best Practices

1. **Use Default and Language-Specific Files**: Prefer using the built-in config and dictionary files for consistency.
2. **Validate Custom Sources**: When specifying custom files, ensure they are well-formed and trusted.
3. **Leverage Caching**: Take advantage of built-in caching to improve performance on repeated loads.
4. **Monitor Logs**: Check logs for warnings or errors during file loading, especially in production.
5. **Progress Tracking**: Use `ProgressTracker` for long-running batch operations to monitor progress and completion.
6. **Handle Errors Gracefully**: Wrap calls in try/except blocks to handle and log errors as needed.
