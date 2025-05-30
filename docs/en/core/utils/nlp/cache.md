# PAMOLA Core: NLP Caching Utilities Module

## Overview

The `pamola_core.utils.nlp.cache` module provides robust, unified caching mechanisms for the NLP components of the PAMOLA Core framework. It is designed to optimize resource usage, speed up repeated operations, and manage memory efficiently for expensive objects such as models, file resources, and function outputs. The module supports multiple cache policies and is thread-safe, making it suitable for high-performance NLP pipelines.

---

## Key Features

- **Unified Caching API**: Consistent interface for memory, file, and model caching.
- **Multiple Eviction Policies**: Supports LRU, LFU, FIFO, TTL, and TLRU policies.
- **Thread-Safe**: All cache operations are protected by locks for safe concurrent access.
- **Automatic Expiration**: Time-to-live (TTL) and file modification tracking for cache invalidation.
- **Memory-Aware Model Cache**: Proactively unloads models under memory pressure.
- **Function Result Caching**: Decorator for easy function-level caching.
- **Global and Per-Instance Caches**: Use global caches or instantiate your own.
- **Statistics and Monitoring**: Access hit/miss/eviction stats for tuning and debugging.

---

## Dependencies

### Standard Library
- `logging`
- `os`
- `threading`
- `time`
- `collections.OrderedDict`
- `typing`

### Internal Modules
- `pamola_core.utils.nlp.base.CacheBase`

### Optional
- `psutil` (for memory monitoring in `ModelCache`)
- `chardet` (for file encoding detection)

---

## Exception Classes

> **Note:** This module does not define custom exception classes. All exceptions raised are standard Python exceptions (e.g., `OSError`, `ImportError`, `IOError`).

### Example: Handling File Cache Errors
```python
try:
    value = file_cache.get('my_key')
except OSError as e:
    # Handle file system errors (e.g., file not found, permission denied)
    logger.error(f"File cache error: {e}")
```
**When Raised:**
- File system errors during file modification time checks or file access.

---

## Main Classes

### MemoryCache

**Purpose:** In-memory cache with configurable eviction policies and TTL support.

#### Constructor
```python
MemoryCache(
    max_size: int = MAX_CACHE_SIZE,
    ttl: int = DEFAULT_CACHE_TTL,
    policy: str = POLICY_TLRU
)
```
**Parameters:**
- `max_size`: Maximum number of items in the cache.
- `ttl`: Default time-to-live (seconds) for cache entries.
- `policy`: Eviction policy (`'lru'`, `'lfu'`, `'fifo'`, `'ttl'`, `'tlru'`).

#### Key Attributes
- `_cache`: OrderedDict storing cached items.
- `_timestamps`, `_ttls`, `_hit_counts`: Track entry metadata.
- `_lock`: Threading lock for concurrency.

#### Public Methods
- `get(key: str) -> Optional[T]`
    - Retrieve a value by key, respecting expiration and policy.
    - **Returns:** Cached value or `None` if not found/expired.
- `set(key: str, value: T, ttl: Optional[int] = None) -> None`
    - Store a value with optional TTL override.
- `delete(key: str) -> bool`
    - Remove a key from the cache.
    - **Returns:** `True` if deleted, `False` otherwise.
- `clear() -> None`
    - Remove all items from the cache.
- `get_stats() -> Dict[str, Any]`
    - Get cache usage statistics.

#### Example Usage
```python
# Create a memory cache with LRU policy
cache = MemoryCache(max_size=50, policy='lru')

# Store and retrieve a value
cache.set('tokenizer', tokenizer_obj)
tokenizer = cache.get('tokenizer')
```

---

### FileCache

**Purpose:** Caches file-based resources, automatically invalidating entries if the underlying file changes.

#### Constructor
```python
FileCache(max_size: int = MAX_CACHE_SIZE)
```
**Parameters:**
- `max_size`: Maximum number of items in the cache.

#### Key Attributes
- `_cache`: Dictionary of cached values.
- `_file_paths`: Maps keys to file paths.
- `_mtimes`: Tracks last known modification times.

#### Public Methods
- `get(key: str) -> Any`
    - Retrieve a value, invalidating if the file has changed.
- `set(key: str, value: Any, file_path: Optional[str] = None) -> None`
    - Store a value, optionally tracking a file.
- `is_valid(key: str) -> bool`
    - Check if a cached value is still valid (file unchanged).
- `delete(key: str) -> bool`
    - Remove a key from the cache.
- `clear() -> None`
    - Remove all items.
- `get_stats() -> Dict[str, Any]`
    - Get cache usage statistics.

#### Example Usage
```python
# Cache a file resource
file_cache = FileCache()
file_cache.set('vocab', vocab_obj, file_path='vocab.txt')

# Retrieve, auto-invalidating if file changed
vocab = file_cache.get('vocab')
```

---

### ModelCache

**Purpose:** Specialized cache for NLP models, with memory pressure monitoring and LRU eviction.

#### Constructor
```python
ModelCache(
    max_size: int = 5,
    memory_threshold: float = 0.75,
    check_memory: bool = True
)
```
**Parameters:**
- `max_size`: Maximum number of models to cache.
- `memory_threshold`: Fraction of system memory usage to trigger evictions.
- `check_memory`: Whether to monitor system memory.

#### Key Attributes
- `_cache`: OrderedDict of models.
- `_metadata`: Model metadata.
- `_last_used`: Last access times.

#### Public Methods
- `get(key: str) -> Any`
    - Retrieve a model by key.
- `set(key: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> None`
    - Store a model with optional metadata.
- `delete(key: str) -> bool`
    - Remove a model.
- `clear() -> None`
    - Remove all models and free memory.
- `get_stats() -> Dict[str, Any]`
    - Get cache usage statistics.
- `get_model_info(key: Optional[str] = None) -> Dict[str, Any]`
    - Get info for one or all models.

#### Example Usage
```python
# Cache a model
model_cache = ModelCache(max_size=3)
model_cache.set('ner', ner_model, metadata={'lang': 'en'})

# Retrieve model
model = model_cache.get('ner')
```

---

## Function Result Caching

### Decorator: `cache_function`

Caches the result of a function call based on its arguments.

```python
@cache_function(ttl=600, cache_type='memory')
def expensive_computation(x, y):
    # ...
    return result
```

---

## Dependency Resolution and Validation Logic

- **MemoryCache**: Evicts items based on the selected policy (LRU, LFU, FIFO, TTL, TLRU). TTL and TLRU policies use timestamps to expire entries.
- **FileCache**: Checks file modification times to ensure cached data is still valid. If the file changes, the cache entry is invalidated.
- **ModelCache**: Monitors system memory (if `psutil` is available) and evicts least recently used models when memory usage exceeds the threshold.

---

## Usage Examples

### Accessing Outputs and Validating Cache
```python
# Access a cached NLP resource
resource = get_cache('memory').get('resource_key')

# Validate a file-based cache entry
if get_cache('file').is_valid('vocab'):
    vocab = get_cache('file').get('vocab')
```

### Handling Failed Dependencies
```python
# Attempt to get a model, handle cache miss
model = get_cache('model').get('ner')
if model is None:
    # Load model from disk or remote
    model = load_model()
    get_cache('model').set('ner', model)
```

### Using the Manager in a Pipeline
```python
from pamola_core.utils.nlp.cache import get_cache

# In a BaseTask or pipeline step
def run(self):
    cache = get_cache('memory')
    result = cache.get('step_output')
    if result is None:
        result = self.compute()
        cache.set('step_output', result)
    return result
```

### Continue-on-Error with Logging
```python
try:
    output = get_cache('file').get('important_file')
except Exception as e:
    logger.warning(f"Failed to retrieve file from cache: {e}")
    # Continue pipeline execution
```

---

## Integration Notes

- The cache module is designed to be used with pipeline tasks (e.g., `BaseTask`).
- Use `get_cache('memory')`, `get_cache('file')`, or `get_cache('model')` to access global caches.
- For custom needs, instantiate your own cache class.

---

## Error Handling and Exception Hierarchy

- All exceptions are standard Python exceptions (e.g., `OSError`, `ImportError`).
- File and model caches handle errors gracefully, invalidating or skipping entries as needed.
- Always check for `None` returns from `get()` methods to handle cache misses or invalidations.

---

## Configuration Requirements

- Environment variables:
    - `PAMOLA_MAX_CACHE_SIZE`: Sets the default maximum cache size.
    - `PAMOLA_CACHE_TTL`: Sets the default TTL for cache entries.
    - `PAMOLA_DISABLE_CACHE`: Set to `'1'` to disable all caching.
- For best results, configure these variables before importing the module.

---

## Security Considerations and Best Practices

- **Path Security**: Avoid using absolute file paths for dependencies unless necessary. Absolute paths may expose sensitive data or break portability.
- **Cache Poisoning**: Ensure cache keys are unique and not user-controlled to prevent cache poisoning attacks.
- **Memory Usage**: Monitor memory usage when caching large models or datasets.

### Example: Security Failure and Handling
```python
# BAD: Using user-supplied absolute path
user_path = get_user_input()
cache.set(user_path, data)  # Risk: may overwrite or expose sensitive cache entries

# GOOD: Use controlled, internal keys
cache.set('user_data', data)
```
**Risks of Disabling Path Security:**
- May allow access to or modification of files outside the intended scope.
- Can lead to data leaks or corruption if external paths are not validated.

---

## Internal vs. External Dependencies

- **Internal Dependencies**: Use logical task IDs or resource names as cache keys for data produced within the pipeline.
- **External (Absolute Path) Dependencies**: Only use for resources not managed by the pipeline. Always validate and sanitize paths.

---

## Best Practices

1. **Use Logical Keys for Internal Data**: Prefer task IDs or resource names for cache keys.
2. **Limit Use of Absolute Paths**: Only use absolute paths for external, immutable resources.
3. **Monitor Cache Statistics**: Use `get_stats()` to tune cache size and policy.
4. **Handle Cache Misses Gracefully**: Always check for `None` and reload resources as needed.
5. **Configure Environment Early**: Set environment variables before importing the module.
6. **Avoid Caching Sensitive Data**: Do not cache secrets or credentials.
7. **Use Decorators for Expensive Functions**: Apply `@cache_function` to cache results of slow computations.
