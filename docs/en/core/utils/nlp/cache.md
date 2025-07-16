# PAMOLA.CORE NLP Cache Module Documentation

## Overview

The `cache.py` module is a critical component of the PAMOLA.CORE NLP utilities package, providing unified caching mechanisms for resources, models, and other expensive objects used throughout the NLP package. It implements multiple cache strategies with thread-safe operations and automatic expiration.

### Module Information
- **Package**: `pamola.pamola_core.utils.nlp.cache`
- **Version**: 1.3.0
- **Status**: stable
- **License**: BSD 3-Clause

## Key Features

### 1. Multiple Cache Implementations
- **MemoryCache**: In-memory cache with configurable eviction policies
- **FileCache**: File-based cache with modification time tracking
- **ModelCache**: Specialized cache for NLP models with memory-aware eviction
- **TextCache**: Text-specific cache with built-in canonicalization

### 2. Eviction Policies
- **LRU** (Least Recently Used)
- **LFU** (Least Frequently Used)
- **FIFO** (First In First Out)
- **TTL** (Time To Live)
- **TLRU** (Time-aware Least Recently Used)

### 3. Advanced Features
- Thread-safe operations with RLock
- Automatic expiration based on time-to-live
- Bulk operations support for improved performance
- Cache statistics and monitoring
- Memory-aware eviction for model cache
- Text canonicalization for consistent cache keys
- Function result caching decorator

## Architecture

### Cache Hierarchy

```
CacheBase (from pamola_core.utils.nlp.base)
├── MemoryCache[T]
│   └── TextCache (specialized for text)
├── FileCache
└── ModelCache
```

### Global Configuration

The module supports environment-based configuration:

```python
MAX_CACHE_SIZE = int(os.environ.get('PAMOLA_MAX_CACHE_SIZE', '100'))
DEFAULT_CACHE_TTL = int(os.environ.get('PAMOLA_CACHE_TTL', '3600'))  # 1 hour
CACHE_ENABLED = os.environ.get('PAMOLA_DISABLE_CACHE', '0') != '1'
```

## Core Components

### 1. Text Canonicalization

```python
def canonicalize_text(text: str, processing_marker: str = "~") -> str:
    """
    Canonicalize text for consistent cache key generation.
    
    Normalizes text by:
    - Removing processing markers from the beginning
    - Normalizing line endings
    - Stripping leading/trailing whitespace
    """
```

This function ensures consistent cache keys regardless of minor text variations.

### 2. MemoryCache

The base in-memory cache implementation with generic typing support:

```python
class MemoryCache(CacheBase, Generic[T]):
    def __init__(self, max_size: int = MAX_CACHE_SIZE, 
                 ttl: int = DEFAULT_CACHE_TTL, 
                 policy: str = POLICY_TLRU):
        """
        Initialize the memory cache.
        
        Parameters:
        - max_size: Maximum number of items in the cache
        - ttl: Default time-to-live in seconds
        - policy: Cache eviction policy
        """
```

#### Key Methods:
- `get(key: str) -> Optional[T]`: Retrieve a value from cache
- `set(key: str, value: T, ttl: Optional[int] = None)`: Store a value
- `get_many(keys: List[str]) -> Dict[str, Optional[T]]`: Bulk retrieve
- `set_many(mapping: Dict[str, T], ttl: Optional[int] = None)`: Bulk store
- `delete(key: str) -> bool`: Remove a specific key
- `clear()`: Clear all cached items
- `get_stats() -> Dict[str, Any]`: Get cache statistics

### 3. FileCache

Specialized cache for file-based resources with timestamp validation:

```python
class FileCache(CacheBase):
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        """
        Cache that tracks file modification times to detect changes.
        Includes write buffer for improved I/O efficiency.
        """
```

#### Special Features:
- Automatic invalidation on file modification
- Batch write buffer for improved I/O performance
- File path tracking and validation

### 4. ModelCache

Memory-aware cache specifically designed for NLP models:

```python
class ModelCache(CacheBase):
    def __init__(self, max_size: int = 5, 
                 memory_threshold: float = 0.75, 
                 check_memory: bool = True):
        """
        Cache for managing NLP models with memory pressure monitoring.
        
        Parameters:
        - max_size: Maximum number of models to keep
        - memory_threshold: Memory usage threshold (0-1) for eviction
        - check_memory: Whether to check system memory
        """
```

#### Key Features:
- Proactive memory pressure monitoring
- Automatic model unloading when memory is high
- Model metadata tracking
- Integration with garbage collection

### 5. TextCache

Specialized cache for text content with automatic canonicalization:

```python
class TextCache(MemoryCache[str]):
    def __init__(self, max_size: int = MAX_CACHE_SIZE,
                 ttl: int = DEFAULT_CACHE_TTL,
                 policy: str = POLICY_TLRU,
                 processing_marker: str = "~",
                 canonicalize_func: Optional[Callable] = None):
        """
        Text-specific cache with built-in canonicalization.
        Ensures consistent caching regardless of processing markers.
        """
```

## Usage Examples

### Basic Cache Usage

```python
from pamola_core.utils.nlp.cache import get_cache

# Get a memory cache instance
cache = get_cache('memory')

# Store a value
cache.set('key1', 'value1', ttl=3600)

# Retrieve a value
value = cache.get('key1')

# Bulk operations
cache.set_many({'key2': 'value2', 'key3': 'value3'})
results = cache.get_many(['key1', 'key2', 'key3'])
```

### File Cache Usage

```python
# Get file cache instance
file_cache = get_cache('file')

# Cache file content with modification tracking
with open('data.json', 'r') as f:
    data = json.load(f)
file_cache.set('data_json', data, file_path='data.json')

# Cache automatically invalidates if file changes
cached_data = file_cache.get('data_json')  # None if file modified
```

### Model Cache Usage

```python
# Get model cache instance
model_cache = get_cache('model')

# Store a model with metadata
model_cache.set('bert_model', model, metadata={
    'type': 'bert',
    'size': 'base',
    'language': 'en'
})

# Memory pressure is automatically handled
cached_model = model_cache.get('bert_model')
```

### Function Caching Decorator

```python
from pamola_core.utils.nlp.cache import cache_function

@cache_function(ttl=3600, cache_type='memory')
def expensive_computation(text: str) -> Dict[str, Any]:
    # Expensive processing
    return results

# First call computes and caches
result1 = expensive_computation("sample text")

# Subsequent calls use cache
result2 = expensive_computation("sample text")  # From cache
```

### Text Cache with Canonicalization

```python
# Get text cache instance
text_cache = get_cache('text')

# These will all hit the same cache entry
text_cache.set("~Hello World", "processed")
value1 = text_cache.get("Hello World")      # Returns "processed"
value2 = text_cache.get("~Hello World")     # Returns "processed"
value3 = text_cache.get("  Hello World  ")  # Returns "processed"
```

## Cache Statistics

All cache implementations provide statistics:

```python
stats = cache.get_stats()
# Returns:
# {
#     "size": 42,
#     "max_size": 100,
#     "hits": 1523,
#     "misses": 234,
#     "hit_ratio": 0.867,
#     "evictions": 12,
#     "expirations": 5,
#     "bulk_operations": 3,
#     "policy": "tlru"
# }
```

## Performance Considerations

### Memory Management
- The module implements efficient memory management with configurable limits
- ModelCache includes memory pressure monitoring (requires psutil)
- Automatic garbage collection integration for model cleanup

### Thread Safety
- All cache operations are thread-safe using RLock
- Bulk operations minimize lock contention
- Write buffering in FileCache reduces I/O operations

### Optimization Tips
1. Use bulk operations (`get_many`, `set_many`) for multiple items
2. Configure appropriate TTL values for your use case
3. Choose the right eviction policy based on access patterns
4. Use TextCache for text data to benefit from canonicalization
5. Monitor cache statistics to tune parameters

## Integration with NLP Operations

The cache module integrates seamlessly with other NLP utilities:

```python
# Category matching with caching
from pamola_core.utils.nlp.category_matching import CategoryDictionary

# Dictionary loading is automatically cached
dict1 = CategoryDictionary.from_file('categories.json')
dict2 = CategoryDictionary.from_file('categories.json')  # From cache

# Text processing with result caching
@cache_function(ttl=3600)
def process_text(text: str) -> List[str]:
    # Expensive NLP processing
    return results
```

## Best Practices

### 1. Cache Key Design
- Use descriptive, unique keys
- Include version information for cache invalidation
- Consider using hash-based keys for long strings

### 2. TTL Configuration
- Set appropriate TTL based on data volatility
- Use shorter TTL for frequently changing data
- Consider infinite TTL (0) for static resources

### 3. Memory Management
- Monitor cache size and hit ratios
- Use ModelCache for large NLP models
- Configure memory thresholds appropriately

### 4. Error Handling
```python
try:
    value = cache.get(key)
    if value is None:
        # Compute value
        value = compute_value()
        cache.set(key, value)
except Exception as e:
    logger.error(f"Cache operation failed: {e}")
    # Fallback to direct computation
    value = compute_value()
```

## Environment Variables

Configure cache behavior via environment variables:

```bash
# Maximum cache size
export PAMOLA_MAX_CACHE_SIZE=200

# Default TTL in seconds
export PAMOLA_CACHE_TTL=7200

# Disable caching (for debugging)
export PAMOLA_DISABLE_CACHE=1
```

## Future Enhancements

The module roadmap includes:
- Redis backend support for distributed caching
- Cache warming strategies
- Persistent cache across sessions
- Cache clustering support
- Advanced compression for cached values

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce cache size limits
   - Enable memory monitoring in ModelCache
   - Use more aggressive eviction policies

2. **Low Hit Ratio**
   - Increase cache size
   - Adjust TTL values
   - Review access patterns and eviction policy

3. **File Cache Invalidation**
   - Ensure file paths are absolute
   - Check file system permissions
   - Monitor file modification times

## Conclusion

The cache module is a fundamental component of PAMOLA.CORE's NLP utilities, providing efficient, thread-safe caching with multiple strategies. Its integration with the broader NLP package ensures optimal performance for text processing, model management, and resource loading operations.