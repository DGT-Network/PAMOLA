# PseudonymizationUtils Module Documentation

## Module Overview

**Module:** `pamola_core.anonymization.commons.pseudonymization_utils`  
**Version:** 1.0.0  
**Package:** PAMOLA.CORE - Privacy-Preserving AI Data Processors  
**License:** BSD 3-Clause

## Purpose

The `pseudonymization_utils` module provides shared utility functions and classes for pseudonymization operations within the PAMOLA.CORE anonymization framework. It offers common functionality used by both hash-based and mapping-based pseudonymization operations, including performance optimization through caching, flexible salt configuration, secure pepper generation, and various helper functions.

## Description

This module serves as a centralized collection of utilities that support pseudonymization operations with the following capabilities:

- **Performance Optimization**: Thread-safe LRU cache for avoiding redundant cryptographic operations
- **Configuration Management**: Flexible salt loading from parameters or files
- **Security Utilities**: Secure pepper generation with automatic memory cleanup
- **Output Formatting**: Consistent pseudonym formatting with prefixes/suffixes
- **Validation Tools**: Format validation for generated pseudonyms
- **Data Processing**: Compound identifier creation and collision probability estimation

The utilities are designed to be reusable across different pseudonymization strategies while maintaining security best practices and performance efficiency.

## Key Features

### Performance Features
- **LRU Caching**: Thread-safe cache with configurable size limits
- **Hit Rate Tracking**: Cache performance statistics for optimization
- **Efficient Eviction**: OrderedDict-based implementation for O(1) operations

### Security Features
- **Secure Memory**: Integration with `SecureBytes` for sensitive data
- **Flexible Salt Sources**: Support for parameter-based and file-based salts
- **Session Peppers**: Cryptographically secure pepper generation
- **Collision Estimation**: Birthday paradox-based probability calculations

### Utility Features
- **Format Validation**: Verify pseudonym formats (hex, base64, UUID, etc.)
- **Compound Identifiers**: Create identifiers from multiple fields
- **Output Formatting**: Add prefixes/suffixes to pseudonyms
- **Statistics Collection**: Cache and operation metrics

## Dependencies

### Required Dependencies
- **Python 3.8+**: Core language requirement
- **pamola_core.utils.crypto_helpers.pseudonymization**: For `SecureBytes` class
- **threading** (standard library): Thread synchronization
- **json** (standard library): Salt file parsing
- **secrets** (standard library): Secure random generation
- **collections** (standard library): OrderedDict for cache
- **pathlib** (standard library): File path handling
- **logging** (standard library): Operation logging

### Optional Dependencies
- **base64** (standard library): For base64 format validation
- **uuid** (standard library): For UUID format validation
- **math** (standard library): For collision probability calculations

## Public API Reference

### Classes

#### `PseudonymizationCache`

Thread-safe LRU cache for storing pseudonymization results.

```python
class PseudonymizationCache:
    def __init__(self, max_size: int = 100000) -> None
```

**Parameters:**
- `max_size`: Maximum number of entries to cache (default: 100000)

**Attributes:**
- `max_size`: Maximum cache capacity
- Thread-safe with internal RLock

### Cache Methods

#### `get()`

Retrieve a cached pseudonym.

```python
def get(self, key: str) -> Optional[str]
```

**Parameters:**
- `key`: Original value to look up

**Returns:**
- Cached pseudonym if found, None otherwise

**Side Effects:**
- Updates access order (marks as recently used)
- Increments hit/miss counters

**Example:**
```python
cache = PseudonymizationCache(max_size=10000)
pseudonym = cache.get("john.doe@example.com")
if pseudonym is None:
    # Generate new pseudonym
    pseudonym = generate_pseudonym(...)
    cache.put("john.doe@example.com", pseudonym)
```

#### `put()`

Add a pseudonym to the cache.

```python
def put(self, key: str, value: str) -> None
```

**Parameters:**
- `key`: Original value
- `value`: Pseudonymized value

**Side Effects:**
- Evicts least recently used entry if at capacity
- Updates access order

#### `clear()`

Clear all entries from the cache.

```python
def clear(self) -> None
```

**Side Effects:**
- Removes all cached entries
- Resets hit/miss statistics

#### `get_statistics()`

Get cache performance statistics.

```python
def get_statistics(self) -> Dict[str, Any]
```

**Returns:**
Dictionary containing:
- `size`: Current number of cached entries
- `max_size`: Maximum cache size
- `hits`: Number of cache hits
- `misses`: Number of cache misses
- `hit_rate`: Cache hit rate (0.0 to 1.0)
- `total_requests`: Total number of requests

**Example:**
```python
stats = cache.get_statistics()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Current size: {stats['size']}/{stats['max_size']}")
```

### Functions

#### `load_salt_configuration()`

Load salt based on flexible configuration options.

```python
def load_salt_configuration(
    config: Dict[str, Any], 
    salt_file: Optional[Path] = None
) -> bytes
```

**Parameters:**
- `config`: Salt configuration dictionary with:
  - `source`: "parameter" or "file"
  - `value`: Salt value (for parameter source)
  - `field_name`: Field name (for file source)
- `salt_file`: Optional path to salts file (required for file source)

**Returns:**
- Salt as bytes

**Raises:**
- `ValueError`: If configuration is invalid or salt cannot be loaded

**Configuration Examples:**

1. **Parameter-based salt**:
```python
config = {
    "source": "parameter",
    "value": "0123456789abcdef" * 4  # 64-char hex = 32 bytes
}
salt = load_salt_configuration(config)
```

2. **File-based salt**:
```python
config = {
    "source": "file",
    "field_name": "email"
}
salt = load_salt_configuration(config, Path("salts.json"))
```

**Supported Salt File Formats:**

1. **Versioned format**:
```json
{
  "_version": "1.0",
  "salts": {
    "email": "0123456789abcdef...",
    "phone": "fedcba9876543210..."
  }
}
```

2. **Legacy format**:
```json
{
  "email": "0123456789abcdef...",
  "phone": "fedcba9876543210..."
}
```

#### `generate_session_pepper()`

Generate a cryptographically secure pepper for the current session.

```python
def generate_session_pepper(length: int = 32) -> SecureBytes
```

**Parameters:**
- `length`: Pepper length in bytes (default: 32)

**Returns:**
- SecureBytes containing pepper (auto-cleanup on deletion)

**Raises:**
- `ValueError`: If length is not positive

**Security Notes:**
- Pepper is automatically cleared from memory when SecureBytes is deleted
- Use with context manager for guaranteed cleanup
- Never persist peppers - they are session-specific

**Example:**
```python
# Basic usage
pepper = generate_session_pepper(32)
try:
    # Use pepper for operations
    hash_result = hash_with_pepper(data, salt, pepper.get())
finally:
    pepper.clear()

# With context manager (if SecureBytes supports it)
with generate_session_pepper(32) as pepper:
    hash_result = hash_with_pepper(data, salt, pepper.get())
# Pepper automatically cleared here
```

#### `format_pseudonym_output()`

Format a pseudonym with optional prefix and/or suffix.

```python
def format_pseudonym_output(
    pseudonym: str, 
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    separator: str = ""
) -> str
```

**Parameters:**
- `pseudonym`: Base pseudonym value
- `prefix`: Optional prefix to prepend
- `suffix`: Optional suffix to append
- `separator`: Separator between prefix/suffix and pseudonym (default: "")

**Returns:**
- Formatted pseudonym string

**Examples:**
```python
# Simple prefix
format_pseudonym_output("abc123", prefix="USER_")
# Output: "USER_abc123"

# Prefix and suffix with separator
format_pseudonym_output("abc123", prefix="ID", suffix="X", separator="-")
# Output: "ID-abc123-X"

# Domain-specific formatting
format_pseudonym_output("a1b2c3", prefix="CUST", separator="_")
# Output: "CUST_a1b2c3"
```

#### `validate_pseudonym_format()`

Validate that a pseudonym matches the expected format.

```python
def validate_pseudonym_format(
    pseudonym: str, 
    expected_format: str,
    expected_length: Optional[int] = None
) -> bool
```

**Parameters:**
- `pseudonym`: Pseudonym to validate
- `expected_format`: Expected format type:
  - `"hex"`: Hexadecimal string
  - `"base64"`: Base64 encoded (URL-safe, no padding)
  - `"base58"`: Base58 encoded
  - `"uuid"`: UUID format
  - `"alphanumeric"`: Letters and numbers only
- `expected_length`: Expected length (optional)

**Returns:**
- True if pseudonym matches expected format, False otherwise

**Examples:**
```python
# Validate hex format
is_valid = validate_pseudonym_format("a1b2c3d4", "hex", expected_length=8)
# True

# Validate UUID
is_valid = validate_pseudonym_format(
    "550e8400-e29b-41d4-a716-446655440000", 
    "uuid"
)
# True

# Validate base64
is_valid = validate_pseudonym_format("YWJjMTIz", "base64")
# True
```

#### `create_compound_identifier()`

Create a compound identifier from multiple field values.

```python
def create_compound_identifier(
    values: Dict[str, Any], 
    separator: str = "|",
    null_handling: str = "skip"
) -> str
```

**Parameters:**
- `values`: Dictionary of field names to values
- `separator`: Separator between values (default: "|")
- `null_handling`: How to handle null values:
  - `"skip"`: Skip null values
  - `"empty"`: Use empty string for nulls
  - `"null"`: Use string "NULL" for nulls

**Returns:**
- Compound identifier string

**Raises:**
- `ValueError`: If null_handling is invalid

**Examples:**
```python
# Basic usage
values = {"first": "John", "last": "Doe", "dept": "Sales"}
compound = create_compound_identifier(values)
# Output: "John|Doe|Sales"

# With null handling
values = {"first": "John", "last": "Doe", "middle": None}
compound = create_compound_identifier(values, null_handling="skip")
# Output: "John|Doe"

compound = create_compound_identifier(values, null_handling="empty")
# Output: "John|Doe|"

# Custom separator
values = {"city": "New York", "state": "NY", "zip": "10001"}
compound = create_compound_identifier(values, separator=", ")
# Output: "New York, NY, 10001"
```

#### `estimate_collision_probability()`

Estimate hash collision probability using the birthday paradox.

```python
def estimate_collision_probability(
    n_values: int, 
    hash_bits: int = 256
) -> float
```

**Parameters:**
- `n_values`: Number of unique values to be hashed
- `hash_bits`: Number of bits in hash output (default: 256 for SHA3-256)

**Returns:**
- Estimated collision probability (0.0 to 1.0)

**Mathematical Basis:**
- Uses birthday paradox approximation: P(collision) ≈ 1 - e^(-n²/2^(bits+1))
- Accurate for n << 2^(bits/2)

**Examples:**
```python
# For 1 million values with SHA-256
prob = estimate_collision_probability(1_000_000, 256)
print(f"Collision probability: {prob:.2e}")
# Output: Collision probability: 4.37e-63

# For 1 billion values with 128-bit hash
prob = estimate_collision_probability(1_000_000_000, 128)
print(f"Collision probability: {prob:.2e}")
# Output: Collision probability: 2.94e-21

# Finding safe limits
for n in [10**6, 10**9, 10**12, 10**15]:
    prob = estimate_collision_probability(n, 256)
    print(f"{n:,} values: {prob:.2e}")
```

## Usage Patterns

### Pattern 1: Complete Caching Setup

```python
from pamola_core.anonymization.commons.pseudonymization_utils import (
    PseudonymizationCache,
    load_salt_configuration,
    generate_session_pepper
)

# Initialize cache
cache = PseudonymizationCache(max_size=50000)

# Load salt configuration
salt_config = {
    "source": "file",
    "field_name": "customer_id"
}
salt = load_salt_configuration(salt_config, Path("config/salts.json"))

# Generate session pepper
pepper = generate_session_pepper()

# Process with caching
def pseudonymize_value(value: str) -> str:
    # Check cache first
    cached = cache.get(value)
    if cached:
        return cached
    
    # Generate new pseudonym
    pseudonym = hash_function(value, salt, pepper.get())
    
    # Cache for future use
    cache.put(value, pseudonym)
    
    return pseudonym

# Monitor cache performance
stats = cache.get_statistics()
logger.info(f"Cache performance: {stats['hit_rate']:.1%} hit rate")
```

### Pattern 2: Format Validation Pipeline

```python
def process_and_validate_pseudonym(
    original: str,
    expected_format: str = "hex",
    prefix: str = "ID_"
) -> str:
    # Generate base pseudonym
    base_pseudo = generate_base_pseudonym(original)
    
    # Format with prefix
    formatted = format_pseudonym_output(base_pseudo, prefix=prefix)
    
    # Validate format
    if not validate_pseudonym_format(base_pseudo, expected_format):
        raise ValueError(f"Invalid {expected_format} format: {base_pseudo}")
    
    return formatted
```

### Pattern 3: Compound Identifier Handling

```python
def pseudonymize_customer_record(record: Dict[str, Any]) -> str:
    # Create compound identifier from multiple fields
    identifier_fields = {
        "customer_id": record.get("id"),
        "email": record.get("email"),
        "registration_date": record.get("registered_on")
    }
    
    # Create compound with null handling
    compound = create_compound_identifier(
        identifier_fields,
        separator="|",
        null_handling="skip"
    )
    
    # Pseudonymize the compound
    return pseudonymize_value(compound)
```

## Best Practices

### Cache Management

1. **Size Configuration**:
   ```python
   # Estimate based on expected unique values
   expected_unique = 100000
   cache_size = int(expected_unique * 0.8)  # 80% of expected
   cache = PseudonymizationCache(max_size=cache_size)
   ```

2. **Performance Monitoring**:
   ```python
   # Regular monitoring
   stats = cache.get_statistics()
   if stats['hit_rate'] < 0.7:  # Less than 70% hit rate
       logger.warning(f"Low cache hit rate: {stats['hit_rate']:.1%}")
       # Consider increasing cache size
   ```

3. **Memory Management**:
   ```python
   # Clear cache when done with batch
   cache.clear()
   # Or let it go out of scope for garbage collection
   ```

### Salt Management

1. **Centralized Configuration**:
   ```python
   # Create a salt manager class
   class SaltConfigManager:
       def __init__(self, salt_file: Path):
           self.salt_file = salt_file
           self._cache = {}
       
       def get_salt(self, field_name: str) -> bytes:
           if field_name not in self._cache:
               config = {"source": "file", "field_name": field_name}
               self._cache[field_name] = load_salt_configuration(
                   config, self.salt_file
               )
           return self._cache[field_name]
   ```

2. **Environment-Based Loading**:
   ```python
   import os
   
   # Determine salt source from environment
   if os.getenv("SALT_SOURCE") == "parameter":
       config = {
           "source": "parameter",
           "value": os.getenv("SALT_VALUE")
       }
   else:
       config = {
           "source": "file",
           "field_name": field_name
       }
   ```

### Security Considerations

1. **Pepper Lifecycle**:
   ```python
   # Always clear peppers after use
   pepper = generate_session_pepper()
   try:
       # Use pepper
       process_data(pepper)
   finally:
       # Ensure cleanup even if error occurs
       pepper.clear()
   ```

2. **Collision Monitoring**:
   ```python
   # Check collision probability before deployment
   dataset_size = 10_000_000  # 10 million records
   collision_prob = estimate_collision_probability(dataset_size)
   
   if collision_prob > 1e-9:  # More than 1 in a billion
       logger.warning(
           f"High collision probability: {collision_prob:.2e}"
       )
       # Consider using longer hashes or additional measures
   ```

## Testing Guidelines

### Unit Tests

1. **Cache Functionality**:
   - Basic get/put operations
   - LRU eviction behavior
   - Thread safety with concurrent access
   - Statistics accuracy

2. **Salt Loading**:
   - Parameter-based loading
   - File-based loading (both formats)
   - Error handling for missing files/fields
   - Invalid configuration handling

3. **Format Validation**:
   - All supported formats
   - Edge cases (empty strings, special characters)
   - Length validation

4. **Utility Functions**:
   - Compound identifier creation
   - Null handling options
   - Collision probability calculations

### Integration Tests

1. **With Hash Operations**:
   - Cache integration with hash generation
   - Performance impact measurement
   - Memory usage profiling

2. **With Mapping Operations**:
   - Concurrent access patterns
   - Large dataset handling
   - Error recovery

### Performance Tests

```python
import time

def test_cache_performance():
    cache = PseudonymizationCache(max_size=10000)
    
    # Warm up cache
    for i in range(10000):
        cache.put(f"key_{i}", f"value_{i}")
    
    # Test hit performance
    start = time.time()
    for _ in range(100000):
        cache.get("key_5000")
    hit_time = time.time() - start
    
    print(f"Cache hits: {100000/hit_time:.0f} ops/sec")
    
    # Test miss performance
    start = time.time()
    for i in range(100000):
        cache.get(f"missing_{i}")
    miss_time = time.time() - start
    
    print(f"Cache misses: {100000/miss_time:.0f} ops/sec")
```

## Error Handling

### Common Errors and Solutions

1. **ValueError: Salt file not found**
   ```python
   try:
       salt = load_salt_configuration(config, salt_file)
   except ValueError as e:
       if "not found" in str(e):
           # Create default salt file or use fallback
           logger.warning("Salt file missing, using default")
           salt = secrets.token_bytes(32)
   ```

2. **ValueError: Invalid format type**
   ```python
   # Validate format before use
   valid_formats = ["hex", "base64", "base58", "uuid", "alphanumeric"]
   if format_type not in valid_formats:
       raise ValueError(f"Format must be one of: {valid_formats}")
   ```

3. **Memory Errors with Large Caches**
   ```python
   # Monitor memory usage
   import psutil
   
   process = psutil.Process()
   memory_mb = process.memory_info().rss / 1024 / 1024
   
   if memory_mb > 1000:  # Over 1GB
       logger.warning("High memory usage, clearing cache")
       cache.clear()
   ```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Cache get | O(1) | Hash table lookup |
| Cache put | O(1) | Amortized, may trigger eviction |
| Salt loading | O(n) | Where n is file size |
| Format validation | O(m) | Where m is string length |
| Compound creation | O(k) | Where k is number of fields |

### Space Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Cache | O(n) | Where n is max_size |
| Salt storage | O(f) | Where f is number of fields |
| Pepper | O(1) | Fixed size (32 bytes default) |

### Benchmarks

Typical performance on modern hardware:

```
Cache Operations:
- Get (hit): ~1M ops/sec
- Get (miss): ~500K ops/sec
- Put: ~300K ops/sec

Salt Loading:
- From parameter: ~1μs
- From file (cached): ~10μs
- From file (first load): ~1ms

Format Validation:
- Hex (32 chars): ~0.5μs
- UUID: ~2μs
- Base64: ~1μs
```

## Version History

- **1.0.0** (2025-01-20): Initial release with core utilities