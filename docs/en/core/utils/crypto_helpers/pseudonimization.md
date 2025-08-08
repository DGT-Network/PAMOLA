# Cryptographic Pseudonymization Utilities Documentation

## Module Overview

**Module:** `pamola.pamola_core.utils.crypto_helpers.pseudonymization`  
**Version:** 1.1.0  
**Package:** PAMOLA.CORE - Privacy-Preserving AI Data Processors  
**License:** BSD 3-Clause

## Purpose

The `pseudonymization` module provides a comprehensive cryptographic foundation for data pseudonymization operations within the PAMOLA.CORE framework. It implements secure, industry-standard algorithms for transforming personally identifiable information (PII) into pseudonyms while maintaining data utility and optional reversibility.

## Description

This module serves as the cryptographic backbone for privacy-preserving data transformations, offering both irreversible (one-way) and reversible (two-way) pseudonymization capabilities. It implements defense-in-depth security principles through multiple layers of protection including salting, peppering, and encryption.

The module is designed for high-performance, concurrent processing environments while maintaining strict security guarantees. All sensitive data is handled with memory-secure operations, and cryptographic operations are implemented to prevent common attacks such as timing attacks and rainbow table lookups.

## Key Features

### Security Features
- **Keccak-256 (SHA-3) Hashing**: Modern cryptographic hash function for irreversible pseudonymization
- **Salt and Pepper Architecture**: Two-layer defense against rainbow table attacks
- **AES-256-GCM Encryption**: Authenticated encryption for reversible pseudonymization mappings
- **Secure Memory Management**: Automatic memory wiping for sensitive data through `SecureBytes` class
- **Constant-Time Operations**: Protection against timing attacks
- **Thread-Safe Design**: Safe for concurrent processing in multi-threaded environments

### Functional Features
- **Multiple Pseudonym Types**: UUID, sequential, and random string generation
- **Flexible Output Formats**: Hex, Base64, and Base58 encoding options
- **Collision Detection**: Built-in tracking and reporting of hash collisions
- **Persistent Salt Management**: Versioned salt storage with atomic file operations
- **Context Manager Support**: Automatic resource cleanup for secure operations
- **Comprehensive Error Handling**: Custom exception hierarchy for precise error management

### Performance Features
- **Efficient Caching**: Thread-safe caching mechanisms for repeated operations
- **Batch Processing Support**: Optimized for processing large datasets
- **Memory Optimization**: Configurable tracking limits and cleanup strategies

## Dependencies

### Required Dependencies
- **Python 3.8+**: Core language requirement
- **hashlib** (standard library): SHA-3 hashing algorithms
- **secrets** (standard library): Cryptographically secure random generation
- **uuid** (standard library): UUID generation
- **json** (standard library): Configuration and mapping storage
- **threading** (standard library): Thread synchronization
- **logging** (standard library): Operational logging

### Optional Dependencies
- **cryptography** (>=3.0): Required for AES-GCM encryption functionality
  - Install: `pip install cryptography`
  - Used by: `MappingEncryption` class
- **base58** (>=2.0): Required for Base58 encoding
  - Install: `pip install base58`
  - Used by: `HashGenerator.format_output()` method

## Public API Reference

### Classes

#### `SecureBytes`
Secure container for sensitive byte data with automatic memory clearing.

```python
class SecureBytes:
    def __init__(self, data: bytes) -> None
    def get(self) -> bytes
    def clear(self) -> None
    def __enter__(self) -> SecureBytes
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool
    def __len__(self) -> int
```

**Usage Example:**
```python
# Using context manager for automatic cleanup
with SecureBytes(b"sensitive_data") as secure_data:
    processed = process_data(secure_data.get())
# Memory automatically cleared here
```

#### `HashGenerator`
Cryptographic hash generator using Keccak-256 with salt and pepper support.

```python
class HashGenerator:
    def __init__(self, algorithm: str = "sha3_256") -> None
    def hash_with_salt(self, data: Union[str, bytes], salt: bytes) -> bytes
    def hash_with_salt_and_pepper(self, data: Union[str, bytes], 
                                  salt: bytes, pepper: bytes) -> bytes
    def format_output(self, hash_bytes: bytes, output_format: str = "hex",
                      output_length: Optional[int] = None) -> str
    def get_statistics(self) -> Dict[str, int]
```

**Usage Example:**
```python
generator = HashGenerator()
salt = b"random_salt_bytes"
pepper = b"session_pepper"

# Generate hash with salt and pepper
hash_bytes = generator.hash_with_salt_and_pepper("sensitive_id", salt, pepper)

# Format as hex string, truncated to 16 characters
pseudonym = generator.format_output(hash_bytes, "hex", 16)
```

#### `SaltManager`
Manages cryptographic salt generation, storage, and retrieval with versioning.

```python
class SaltManager:
    SALT_FILE_VERSION: str = "1.0"
    
    def __init__(self, salts_file: Optional[Path] = None) -> None
    def generate_salt(self, length: int = 32) -> bytes
    def get_or_create_salt(self, identifier: str, length: int = 32) -> bytes
```

**Usage Example:**
```python
manager = SaltManager(Path("config/salts.json"))

# Get existing salt or create new one
salt = manager.get_or_create_salt("user_id_field")
```

#### `PepperGenerator`
Session-specific pepper generator with secure memory management.

```python
class PepperGenerator:
    def __init__(self) -> None
    def generate(self, length: int = 32) -> bytes
    def get(self) -> Optional[bytes]
    def clear(self) -> None
```

**Usage Example:**
```python
pepper_gen = PepperGenerator()
try:
    # Generate pepper for session
    pepper = pepper_gen.generate(32)
    # Use pepper for operations...
finally:
    # Always clear pepper from memory
    pepper_gen.clear()
```

#### `MappingEncryption`
AES-256-GCM encryption for secure mapping storage.

```python
class MappingEncryption:
    def __init__(self, key: bytes) -> None
    def encrypt(self, plaintext: bytes, 
                associated_data: Optional[bytes] = None) -> bytes
    def decrypt(self, encrypted: bytes, 
                associated_data: Optional[bytes] = None) -> bytes
```

**Usage Example:**
```python
# Generate or load 256-bit key
key = secrets.token_bytes(32)
encryptor = MappingEncryption(key)

# Encrypt mapping data
mapping_json = json.dumps({"id123": "pseudo456"}).encode()
encrypted = encryptor.encrypt(mapping_json)

# Decrypt mapping data
decrypted = encryptor.decrypt(encrypted)
mapping = json.loads(decrypted)
```

#### `PseudonymGenerator`
Generates various types of pseudonyms with uniqueness guarantees.

```python
class PseudonymGenerator:
    def __init__(self, pseudonym_type: str = "uuid") -> None
    def generate(self, prefix: Optional[str] = None, length: int = 36) -> str
    def generate_unique(self, existing: set, prefix: Optional[str] = None,
                        max_attempts: int = 100) -> str
```

**Pseudonym Types:**
- `"uuid"`: UUID v4 format (e.g., "550e8400-e29b-41d4-a716-446655440000")
- `"sequential"`: Sequential numbers (e.g., "00000001", "00000002")
- `"random_string"`: Random alphanumeric (e.g., "a7B9x2K4m8")

**Usage Example:**
```python
# UUID generator
uuid_gen = PseudonymGenerator("uuid")
pseudonym = uuid_gen.generate(prefix="USER_")  # "USER_550e8400-e29b-41d4-a716-446655440000"

# Sequential generator
seq_gen = PseudonymGenerator("sequential")
pseudonym = seq_gen.generate(prefix="ID")  # "ID00000001"

# Random string generator
rand_gen = PseudonymGenerator("random_string")
pseudonym = rand_gen.generate(length=12)  # "a7B9x2K4m8Lp"
```

#### `CollisionTracker`
Monitors and tracks hash collisions for security auditing.

```python
class CollisionTracker:
    def __init__(self, max_tracked: int = 1000) -> None
    def check_and_record(self, pseudonym: str, original: str) -> Optional[str]
    def get_collision_count(self) -> int
    def get_statistics(self) -> Dict[str, Any]
    def export_collisions(self, output_file: Path) -> None
```

**Usage Example:**
```python
tracker = CollisionTracker(max_tracked=10000)

# Check for collisions while generating pseudonyms
pseudonym = generate_hash(value)
collision = tracker.check_and_record(pseudonym, value)
if collision:
    logger.warning(f"Collision detected: {value} and {collision} -> {pseudonym}")

# Export collision report
tracker.export_collisions(Path("collision_report.json"))
```

### Utility Functions

#### `constant_time_compare`
Performs constant-time byte comparison to prevent timing attacks.

```python
def constant_time_compare(a: bytes, b: bytes) -> bool
```

#### `validate_key_size`
Validates encryption key size for security requirements.

```python
def validate_key_size(key: bytes, expected_bits: int = 256) -> None
```

#### `derive_key_from_password`
Derives encryption key from password using PBKDF2-HMAC-SHA256.

```python
def derive_key_from_password(password: str, salt: bytes,
                             iterations: int = 100000,
                             key_length: int = 32) -> bytes
```

**Usage Example:**
```python
# Derive AES-256 key from password
salt = secrets.token_bytes(16)
key = derive_key_from_password("strong_password", salt, iterations=150000)
```

#### `generate_deterministic_pseudonym`
Generates deterministic pseudonyms using HMAC for cross-system consistency.

```python
def generate_deterministic_pseudonym(identifier: str, domain: str,
                                     secret_key: bytes) -> str
```

**Usage Example:**
```python
secret_key = secrets.token_bytes(32)
# Same identifier+domain always produces same pseudonym
pseudo1 = generate_deterministic_pseudonym("user123", "system_a", secret_key)
pseudo2 = generate_deterministic_pseudonym("user123", "system_a", secret_key)
assert pseudo1 == pseudo2
```

### Exceptions

#### `PseudonymizationError`
Base exception for all pseudonymization-related errors.

#### `CryptoError`
Raised for cryptographic operation failures (e.g., decryption errors, invalid keys).

#### `HashCollisionError`
Raised when hash collisions are detected and collision strategy is set to "fail".

### Constants

- `CRYPTOGRAPHY_AVAILABLE`: Boolean indicating if cryptography package is available
- `BASE58_AVAILABLE`: Boolean indicating if base58 package is available

## Security Considerations

### Memory Security
- All sensitive data (keys, peppers, passwords) should be wrapped in `SecureBytes`
- Use context managers to ensure automatic cleanup
- Never log or print sensitive data

### Cryptographic Best Practices
- Always use salts of at least 32 bytes
- Generate new peppers for each session
- Use key derivation functions for password-based encryption
- Monitor collision rates in production environments

### Thread Safety
- All classes are designed to be thread-safe
- Use single instances across threads for better performance
- Collision tracking is centralized and thread-safe

## Performance Guidelines

### Optimization Strategies
1. **Enable Caching**: Use built-in caching for repeated pseudonymization of same values
2. **Batch Processing**: Process data in batches to amortize overhead
3. **Appropriate Tracking Limits**: Set `max_tracked` in CollisionTracker based on memory constraints
4. **Output Format Selection**: Hex encoding is fastest, Base58 is slowest but most compact

### Memory Management
- CollisionTracker automatically purges old entries when limit reached
- SaltManager caches loaded salts to avoid repeated file I/O
- Use context managers to ensure timely memory cleanup

## Complete Example

```python
from pathlib import Path
from pamola.pamola_core.utils.crypto_helpers.pseudonymization import (
    HashGenerator, SaltManager, PepperGenerator, 
    CollisionTracker, SecureBytes
)

# Initialize components
salt_manager = SaltManager(Path("config/salts.json"))
pepper_gen = PepperGenerator()
hash_gen = HashGenerator()
tracker = CollisionTracker(max_tracked=10000)

try:
    # Generate session pepper
    pepper = pepper_gen.generate()
    
    # Get or create field-specific salt
    salt = salt_manager.get_or_create_salt("email_field")
    
    # Process batch of emails
    emails = ["user1@example.com", "user2@example.com", "user1@example.com"]
    pseudonyms = {}
    
    for email in emails:
        # Check cache first
        if email in pseudonyms:
            continue
        
        # Generate pseudonym
        hash_bytes = hash_gen.hash_with_salt_and_pepper(email, salt, pepper)
        pseudonym = hash_gen.format_output(hash_bytes, "hex", 16)
        
        # Check for collisions
        collision = tracker.check_and_record(pseudonym, email)
        if collision and collision != email:
            print(f"Warning: Collision detected for {email}")
        
        pseudonyms[email] = pseudonym
    
    # Print results
    for email, pseudo in pseudonyms.items():
        print(f"{email} -> {pseudo}")
    
    # Export collision statistics
    print(f"Collision stats: {tracker.get_statistics()}")
    
finally:
    # Always clear pepper from memory
    pepper_gen.clear()
```

## Version History

- **1.1.0** (Current): Added context manager support, versioned salt storage, enhanced validation
- **1.0.0**: Initial release with core cryptographic functionality