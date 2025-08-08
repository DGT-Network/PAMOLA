# MappingStorage Module Documentation

## Module Overview

**Module:** `pamola_core.anonymization.commons.mapping_storage`  
**Version:** 1.0.0  
**Package:** PAMOLA.CORE - Privacy-Preserving AI Data Processors  
**License:** BSD 3-Clause

## Purpose

The `mapping_storage` module provides secure, encrypted storage for pseudonymization mappings, enabling reversible data transformations while maintaining strict security guarantees. It handles the persistence of original-to-pseudonym mappings with AES-256-GCM encryption, ensuring that sensitive mapping data is protected at rest.

## Description

This module implements a robust storage system for pseudonymization mappings with the following characteristics:

- **Encrypted Storage**: All mappings are encrypted using AES-256-GCM before being written to disk
- **Format Flexibility**: Supports both CSV and JSON storage formats
- **Data Integrity**: Atomic file operations prevent corruption during writes
- **Backup Management**: Optional automatic backups with configurable retention
- **Thread Safety**: All operations are thread-safe for concurrent access
- **Metadata Tracking**: Maintains metadata about mappings including timestamps and counts

The module is designed to integrate seamlessly with the PAMOLA.CORE pseudonymization operations, providing a secure backend for storing reversible transformations.

## Key Features

### Security Features
- **AES-256-GCM Encryption**: Industry-standard authenticated encryption
- **Secure File Permissions**: Owner-only access (0600) on Unix systems
- **No Plaintext Storage**: Mappings never exist unencrypted on disk
- **Memory Security**: Integration with `SecureBytes` for key handling

### Operational Features
- **Atomic Writes**: Temporary file + rename pattern prevents partial writes
- **Automatic Backups**: Timestamped backups before updates
- **Backup Rotation**: Automatic cleanup of old backups (configurable retention)
- **Format Support**: CSV with headers or JSON with metadata
- **Validation**: Built-in mapping consistency validation

### Performance Features
- **Thread-Safe Operations**: RLock-based synchronization for concurrent access
- **Efficient Updates**: Merge new mappings without full rewrite
- **Lazy Loading**: Mappings loaded only when needed
- **Metadata Queries**: File statistics without decryption

## Dependencies

### Required Dependencies
- **Python 3.8+**: Core language requirement
- **pamola_core.utils.crypto_helpers.pseudonymization**: For `MappingEncryption` class
- **csv** (standard library): CSV format support
- **json** (standard library): JSON format support
- **threading** (standard library): Thread synchronization
- **pathlib** (standard library): Path handling
- **shutil** (standard library): File operations

### Optional Dependencies
None - all functionality is available with standard Python installation.

## Public API Reference

### Classes

#### `MappingStorage`

Main class for managing encrypted mapping storage.

```python
class MappingStorage:
    def __init__(
        self, 
        mapping_file: Path,
        encryption_key: bytes,
        format: str = "csv",
        backup_on_update: bool = True,
        create_if_missing: bool = True
    ) -> None
```

**Parameters:**
- `mapping_file`: Path to the encrypted mapping file
- `encryption_key`: 256-bit encryption key (32 bytes)
- `format`: Storage format - "csv" or "json" (default: "csv")
- `backup_on_update`: Create backups before updates (default: True)
- `create_if_missing`: Create empty file if it doesn't exist (default: True)

**Raises:**
- `ValueError`: If format is invalid or key size is incorrect
- `MappingStorageError`: If initialization fails

### Methods

#### `load()`

Load and decrypt mappings from file.

```python
def load(self) -> Dict[str, str]
```

**Returns:**
- Dictionary mapping original values to pseudonyms

**Raises:**
- `MappingStorageError`: If loading or decryption fails

**Example:**
```python
storage = MappingStorage(Path("mappings.enc"), key)
mappings = storage.load()
print(f"Loaded {len(mappings)} mappings")
```

#### `save()`

Encrypt and save mappings atomically.

```python
def save(self, mapping: Dict[str, str]) -> None
```

**Parameters:**
- `mapping`: Dictionary mapping original values to pseudonyms

**Raises:**
- `MappingStorageError`: If saving fails

**Example:**
```python
mappings = {
    "john.doe@example.com": "USER_123e4567",
    "jane.smith@example.com": "USER_987f6543"
}
storage.save(mappings)
```

#### `update()`

Update existing mappings with new entries.

```python
def update(self, new_mappings: Dict[str, str]) -> Dict[str, str]
```

**Parameters:**
- `new_mappings`: New mappings to add or update

**Returns:**
- Complete updated mapping dictionary

**Raises:**
- `MappingStorageError`: If update fails

**Notes:**
- Existing mappings with the same key will be overwritten
- Conflicts are logged but do not prevent the update

**Example:**
```python
new_entries = {"bob.jones@example.com": "USER_456d7890"}
updated = storage.update(new_entries)
```

#### `get_metadata()`

Get mapping file metadata without decryption.

```python
def get_metadata(self) -> Dict[str, Any]
```

**Returns:**
Dictionary containing:
- `exists`: Whether file exists
- `size_bytes`: File size in bytes
- `modified`: Last modification timestamp (ISO format)
- `created`: Creation timestamp (ISO format)
- `format`: Storage format
- `path`: Full file path

**Example:**
```python
metadata = storage.get_metadata()
print(f"File size: {metadata['size_bytes']} bytes")
print(f"Last modified: {metadata['modified']}")
```

#### `validate_mappings()`

Validate mapping consistency and detect issues.

```python
def validate_mappings(self, mapping: Dict[str, str]) -> Dict[str, Any]
```

**Parameters:**
- `mapping`: Mapping dictionary to validate

**Returns:**
Dictionary containing:
- `valid`: Whether mapping is valid
- `duplicate_values`: List of pseudonyms mapped to multiple originals
- `empty_keys`: Number of empty original values
- `empty_values`: Number of empty pseudonyms
- `total_mappings`: Total number of mappings

**Example:**
```python
validation = storage.validate_mappings(mappings)
if not validation["valid"]:
    print(f"Issues found: {validation}")
```

### Exceptions

#### `MappingStorageError`

Base exception for all mapping storage errors.

```python
class MappingStorageError(Exception):
    """Base exception for mapping storage errors."""
```

## Storage Formats

### CSV Format

The CSV format uses a simple two-column structure:

```csv
original,pseudonym
john.doe@example.com,USER_123e4567-e89b-41d4-a716-446655440000
jane.smith@example.com,USER_987f6543-a21c-32b1-9876-543210fedcba
```

**Characteristics:**
- Headers: "original" and "pseudonym"
- UTF-8 encoding
- Sorted by original value for consistency
- Suitable for simple mappings

### JSON Format

The JSON format includes metadata:

```json
{
  "_metadata": {
    "version": "1.0",
    "created": "2025-01-20T10:30:00",
    "count": 2,
    "format": "json"
  },
  "mappings": {
    "john.doe@example.com": "USER_123e4567-e89b-41d4-a716-446655440000",
    "jane.smith@example.com": "USER_987f6543-a21c-32b1-9876-543210fedcba"
  }
}
```

**Characteristics:**
- Includes creation timestamp and count
- Pretty-printed with indentation
- Sorted keys for consistency
- Extensible with additional metadata

## Security Considerations

### Encryption Details
- **Algorithm**: AES-256-GCM (Galois/Counter Mode)
- **Key Size**: 256 bits (32 bytes)
- **Nonce**: 96-bit random nonce per encryption
- **Authentication**: GCM provides built-in authentication

### File Security
- **Permissions**: Set to 0600 (owner read/write only) on Unix
- **Atomic Writes**: Prevents partial file states
- **No Temporary Plaintext**: Data never exists unencrypted on disk

### Key Management
- Keys must be provided by the caller
- Keys should be derived using secure methods
- Consider using `derive_key_from_password()` for password-based keys
- Never store encryption keys in the same location as encrypted files

## Best Practices

### Performance Optimization

1. **Batch Updates**: Accumulate changes and update once
   ```python
   # Good: Single update
   storage.update({"key1": "val1", "key2": "val2", "key3": "val3"})
   
   # Avoid: Multiple updates
   storage.update({"key1": "val1"})
   storage.update({"key2": "val2"})
   storage.update({"key3": "val3"})
   ```

2. **Format Selection**: 
   - Use CSV for simple mappings (more compact)
   - Use JSON for complex needs or when metadata is important

3. **Backup Management**: Configure retention based on storage constraints
   ```python
   # In _cleanup_old_backups(), adjust keep_count
   self._cleanup_old_backups(keep_count=3)  # Keep only 3 backups
   ```

### Error Handling

1. **Always Handle Exceptions**:
   ```python
   try:
       mappings = storage.load()
   except MappingStorageError as e:
       logger.error(f"Failed to load mappings: {e}")
       # Handle gracefully - maybe start with empty mappings
       mappings = {}
   ```

2. **Validate Before Saving**:
   ```python
   validation = storage.validate_mappings(new_mappings)
   if not validation["valid"]:
       logger.warning(f"Mapping issues: {validation}")
       # Decide whether to proceed
   ```

### Thread Safety

The module is thread-safe, but consider:
- All methods use internal locking
- Long operations (large file I/O) may block other threads
- For high-concurrency scenarios, consider caching loaded mappings

## Complete Example

```python
from pathlib import Path
import secrets
from pamola_core.anonymization.commons.mapping_storage import (
    MappingStorage, MappingStorageError
)

# Generate secure encryption key
encryption_key = secrets.token_bytes(32)

# Initialize storage
storage = MappingStorage(
    mapping_file=Path("data/customer_mappings.enc"),
    encryption_key=encryption_key,
    format="json",
    backup_on_update=True,
    create_if_missing=True
)

# Initial mappings
initial_mappings = {
    "customer_001": "CUST_a1b2c3d4",
    "customer_002": "CUST_e5f6g7h8"
}

try:
    # Save initial mappings
    storage.save(initial_mappings)
    print("Initial mappings saved")
    
    # Load and verify
    loaded = storage.load()
    assert loaded == initial_mappings
    print(f"Verified {len(loaded)} mappings")
    
    # Add new mappings
    new_customers = {
        "customer_003": "CUST_i9j0k1l2",
        "customer_004": "CUST_m3n4o5p6"
    }
    
    # Update with validation
    validation = storage.validate_mappings(new_customers)
    if validation["valid"]:
        updated = storage.update(new_customers)
        print(f"Total mappings: {len(updated)}")
    
    # Check metadata
    metadata = storage.get_metadata()
    print(f"Storage file: {metadata['path']}")
    print(f"Size: {metadata['size_bytes']} bytes")
    print(f"Modified: {metadata['modified']}")
    
except MappingStorageError as e:
    print(f"Storage error: {e}")
    
# Key cleanup (if using SecureBytes)
# encryption_key.clear()  # If wrapped in SecureBytes
```

## Testing Recommendations

### Unit Tests

1. **Basic Operations**:
   - Create, save, load cycle
   - Update with new and existing keys
   - Empty mapping handling

2. **Format Tests**:
   - CSV format parsing and generation
   - JSON format with metadata
   - Format-specific edge cases

3. **Security Tests**:
   - Verify encryption (file should be unreadable)
   - Key size validation
   - Permission checks

4. **Concurrency Tests**:
   - Parallel reads
   - Concurrent updates
   - Race condition handling

5. **Error Scenarios**:
   - Corrupted file handling
   - Invalid encryption key
   - Disk full conditions
   - Permission denied

### Integration Tests

1. **With Pseudonymization Operations**:
   - Consistent mapping operation integration
   - Large dataset handling
   - Performance benchmarks

2. **Backup System**:
   - Backup creation
   - Rotation logic
   - Recovery from backups

## Performance Characteristics

### Time Complexity
- Load: O(n) where n is the number of mappings
- Save: O(n log n) due to sorting
- Update: O(n + m) where m is new mappings
- Validation: O(n)

### Space Complexity
- Memory: O(n) for loaded mappings
- Disk: Encrypted size ≈ 1.1x plaintext size
- Backups: O(k × n) where k is retention count

### Benchmarks
Typical performance on modern hardware:
- 10K mappings: ~100ms load/save
- 100K mappings: ~1s load/save  
- 1M mappings: ~10s load/save

## Version History

- **1.0.0** (2025-01-20): Initial release with full encryption support