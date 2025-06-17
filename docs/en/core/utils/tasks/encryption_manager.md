# EncryptionManager Module Documentation

## Overview

The `encryption_manager.py` module provides secure encryption key management and handling for the PAMOLA Core framework. It focuses on minimizing the attack surface when passing encryption capabilities from tasks to operations, ensuring sensitive data is properly protected. The module implements a security-focused approach to encryption management with memory protection for keys and integration with the rest of the PAMOLA infrastructure.

## Key Features

- **Secure Encryption Key Management**: Safely handles encryption keys with memory protection
- **Context-Based Encryption**: Provides encryption capabilities without directly exposing keys
- **Data Structure Redaction**: Ensures sensitive data is removed from logs and reports
- **Multiple Encryption Modes**: Supports different encryption methods (none, simple, age)
- **Memory Protection**: Implements secure handling of keys in memory
- **Integration with Progress Tracking**: Coordinates with progress manager for visible initialization
- **Automatic Key Management**: Loads, generates, and securely stores encryption keys
- **Path Security Validation**: Ensures all file paths meet security requirements

## Dependencies

- `base64`, `secrets`: Standard library modules for cryptographic operations
- `logging`: For log output
- `pathlib`: Path manipulation
- `cryptography.fernet`: Primary encryption backend (optional)
- `pyage`: Alternative encryption backend (optional)
- `pamola_core.utils.crypto_helpers.key_store`: For key retrieval (optional)
- `pamola_core.utils.tasks.path_security`: Path security validation
- `pamola_core.utils.tasks.progress_manager`: For progress tracking (optional)

## Enumerations

### EncryptionMode

Defines the encryption modes supported by the task framework.

```python
class EncryptionMode(Enum):
    NONE = "none"     # No encryption
    SIMPLE = "simple" # Simple symmetric encryption using Fernet
    AGE = "age"       # Age encryption (more secure, supports key rotation)

    @classmethod
    def from_string(cls, value: str) -> 'EncryptionMode':
        """Convert string to EncryptionMode enum value."""
```

## Exception Classes

- **EncryptionError**: Base exception for encryption-related errors
- **EncryptionInitializationError**: Raised when encryption initialization fails
- **KeyGenerationError**: Raised when key generation fails
- **KeyLoadingError**: Raised when key loading fails
- **DataRedactionError**: Raised when data redaction fails

## Classes

### EncryptionContext

#### Description

Secure encryption context that provides encryption capabilities without exposing the raw encryption key. This class encapsulates encryption functionality while protecting the actual key material, providing a safer interface for operations.

#### Constructor

```python
def __init__(self, mode: EncryptionMode, key_fingerprint: str)
```

**Parameters:**
- `mode`: Encryption mode to use
- `key_fingerprint`: Fingerprint of the encryption key (not the key itself)

#### Properties

- `can_encrypt`: Check if this context can perform encryption operations

#### Methods

##### to_dict

```python
def to_dict(self) -> Dict[str, Any]
```

Convert context to dictionary for serialization (contains no sensitive data).

**Returns:**
- Dictionary with context information

### MemoryProtectedKey

#### Description

Memory-protected encryption key container that implements secure handling of encryption keys in memory, with minimal exposure and prevention of unintended key leakage.

#### Constructor

```python
def __init__(self, key_material: bytes, key_id: Optional[str] = None)
```

**Parameters:**
- `key_material`: Raw key bytes
- `key_id`: Optional identifier for the key

#### Properties

- `fingerprint`: Get the key fingerprint (safe to expose)
- `key_id`: Get the key ID (safe to expose)
- `has_been_used`: Check if this key has been used

#### Context Manager Support

The class implements the context manager protocol (`__enter__` and `__exit__`) to provide secure, temporary access to the key material.

```python
with protected_key as key_material:
    # Use key_material safely within this context
    # Key reference count is tracked and cleanup attempted when context exits
```

### TaskEncryptionManager

#### Description

Main encryption manager for secure handling of encryption keys and sensitive data. This class encapsulates all encryption-related functionality, providing a secure interface for tasks to use encryption without exposing raw keys to operations or logs.

#### Constructor

```python
def __init__(
    self,
    task_config: Any,
    logger: Optional[logging.Logger] = None,
    progress_manager: Optional['TaskProgressManager'] = None
)
```

**Parameters:**
- `task_config`: Task configuration object containing encryption settings
- `logger`: Logger for encryption operations (optional)
- `progress_manager`: Progress manager for tracking initialization (optional)

#### Key Methods

##### initialize

```python
def initialize(self) -> bool
```

Initialize encryption based on configuration.

**Returns:**
- True if initialization successful, False otherwise

##### get_encryption_context

```python
def get_encryption_context(self) -> EncryptionContext
```

Get secure encryption context for operations. This method provides a safe way to pass encryption capabilities to operations without exposing the raw key.

**Returns:**
- EncryptionContext with necessary info for operations

##### encrypt_data

```python
def encrypt_data(self, data: bytes) -> bytes
```

Encrypt binary data using the configured encryption method.

**Parameters:**
- `data`: Data to encrypt

**Returns:**
- Encrypted data

**Raises:**
- `EncryptionError`: If encryption fails

##### decrypt_data

```python
def decrypt_data(self, encrypted_data: bytes) -> bytes
```

Decrypt binary data using the configured encryption method.

**Parameters:**
- `encrypted_data`: Data to decrypt

**Returns:**
- Decrypted data

**Raises:**
- `EncryptionError`: If decryption fails

##### add_sensitive_param_names

```python
def add_sensitive_param_names(self, param_names: Union[str, List[str]]) -> None
```

Add parameter names that should be treated as sensitive.

**Parameters:**
- `param_names`: Single name or list of parameter names

##### is_sensitive_param

```python
def is_sensitive_param(self, param_name: str) -> bool
```

Check if a parameter name should be treated as sensitive.

**Parameters:**
- `param_name`: Parameter name to check

**Returns:**
- True if parameter is sensitive, False otherwise

##### redact_sensitive_data

```python
def redact_sensitive_data(self, data: Any, redact_keys: bool = True) -> Any
```

Redact sensitive information from data structures. This method recursively processes dictionaries, lists, and other data structures to redact sensitive values based on key names.

**Parameters:**
- `data`: Data structure to redact
- `redact_keys`: Whether to redact dictionary keys (default: True)

**Returns:**
- Redacted copy of the data structure

**Raises:**
- `DataRedactionError`: If redaction fails

##### redact_config_dict

```python
def redact_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]
```

Redact sensitive information from a configuration dictionary. This is a convenience method specifically for configuration dictionaries.

**Parameters:**
- `config_dict`: Configuration dictionary to redact

**Returns:**
- Redacted copy of the configuration

##### get_encryption_info

```python
def get_encryption_info(self) -> Dict[str, Any]
```

Get information about the encryption configuration.

**Returns:**
- Dictionary with encryption information (no sensitive data)

##### check_dataset_encryption

```python
def check_dataset_encryption(self, data_source: Any) -> bool
```

Check if datasets in the data source are encrypted.

**Parameters:**
- `data_source`: Data source containing file paths

**Returns:**
- True if all datasets appear to be properly encrypted (when encryption is enabled)

##### is_file_encrypted

```python
def is_file_encrypted(self, file_path: Union[str, Path]) -> bool
```

Check if a file appears to be encrypted.

**Parameters:**
- `file_path`: Path to the file to check

**Returns:**
- True if the file appears to be encrypted, False otherwise

##### supports_encryption_mode

```python
def supports_encryption_mode(self, mode: Union[str, EncryptionMode]) -> bool
```

Check if the requested encryption mode is supported.

**Parameters:**
- `mode`: Encryption mode to check

**Returns:**
- True if mode is supported, False otherwise

##### cleanup

```python
def cleanup(self) -> None
```

Explicitly clean up resources. This should be called when the manager is no longer needed.

## Internal Methods

### _resolve_key_path

```python
def _resolve_key_path(self) -> Path
```

Resolve the encryption key path safely.

**Returns:**
- Path object for the encryption key

**Raises:**
- `PathSecurityError`: If the path fails security validation

### _get_key_from_store

```python
def _get_key_from_store(self) -> Optional[bytes]
```

Get encryption key from key store.

**Returns:**
- Key bytes if available, None otherwise

**Raises:**
- `KeyLoadingError`: If key store is available but returned an error

### _generate_encryption_key

```python
def _generate_encryption_key(self) -> bytes
```

Generate a new encryption key.

**Returns:**
- New key bytes

**Raises:**
- `KeyGenerationError`: If key generation fails

### _looks_like_key

```python
def _looks_like_key(self, text: str) -> bool
```

Check if a string looks like a key or sensitive data.

**Parameters:**
- `text`: String to check

**Returns:**
- True if string looks like a key, False otherwise

## Usage Examples

### Basic Encryption Setup

```python
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager, EncryptionMode
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create encryption manager with task configuration
class MockConfig:
    use_encryption = True
    encryption_mode = EncryptionMode.SIMPLE
    encryption_key_path = "path/to/key/file.key"
    task_id = "t_1A_encryption_example"

# Create encryption manager
encryption_manager = TaskEncryptionManager(
    task_config=MockConfig(),
    logger=logger
)

# Initialize encryption (this loads or generates the key)
success = encryption_manager.initialize()
if not success:
    print("Encryption initialization failed")
    # Proceed with encryption disabled
    
# Get encryption info (no sensitive data)
info = encryption_manager.get_encryption_info()
print(f"Encryption enabled: {info['enabled']}")
print(f"Encryption mode: {info['mode']}")
print(f"Key available: {info['key_available']}")

# Clean up when done
encryption_manager.cleanup()
```

### Encrypting and Decrypting Data

```python
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager, EncryptionError
from pathlib import Path

# Create encryption manager and initialize
encryption_manager = TaskEncryptionManager(
    task_config=mock_config,
    logger=logger
)
encryption_manager.initialize()

# Encrypt some data
try:
    secret_message = b"This is sensitive information"
    encrypted_data = encryption_manager.encrypt_data(secret_message)
    
    # Save to file
    with open("encrypted_message.bin", "wb") as f:
        f.write(encrypted_data)
    
    # Decrypt data
    decrypted_data = encryption_manager.decrypt_data(encrypted_data)
    assert decrypted_data == secret_message
    print("Encryption/decryption successful!")
    
except EncryptionError as e:
    print(f"Encryption error: {e}")
```

### Using Encryption Context

```python
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager

# Initialize encryption manager
encryption_manager = TaskEncryptionManager(
    task_config=mock_config,
    logger=logger
)
encryption_manager.initialize()

# Get encryption context (safe to pass to operations)
encryption_context = encryption_manager.get_encryption_context()

# Print context info (no sensitive data)
print(f"Mode: {encryption_context.mode.value}")
print(f"Key fingerprint: {encryption_context.key_fingerprint}")
print(f"Can encrypt: {encryption_context.can_encrypt}")

# Pass to an operation constructor
operation = ExampleOperation(
    encryption_context=encryption_context,
    other_params="value"
)
```

### Redacting Sensitive Data

```python
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager

# Initialize encryption manager
encryption_manager = TaskEncryptionManager(
    task_config=mock_config,
    logger=logger
)

# Data with sensitive information
config_with_secrets = {
    "api_endpoint": "https://api.example.com",
    "api_key": "sk-abcdefghijklmnopqrstuvwxyz12345678",
    "request_timeout": 30,
    "credentials": {
        "username": "user",
        "password": "super_secret_password"
    }
}

# Redact sensitive information
redacted_config = encryption_manager.redact_sensitive_data(config_with_secrets)

# Now safe to log or include in reports
print(redacted_config)
```

### Integration with Progress Manager

```python
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A",
    task_type="encryption_example",
    logger=logger,
    total_operations=3
)

# Create encryption manager with progress tracking
encryption_manager = TaskEncryptionManager(
    task_config=mock_config,
    logger=logger,
    progress_manager=progress_manager
)

# Initialize encryption (progress will be tracked)
encryption_manager.initialize()

# Check dataset encryption status (progress will be tracked)
data_source = mock_data_source  # Your data source object
encryption_manager.check_dataset_encryption(data_source)

# Clean up resources
encryption_manager.cleanup()
progress_manager.close()
```

### Checking File Encryption

```python
from pamola_core.utils.tasks.encryption_manager import TaskEncryptionManager
from pathlib import Path

# Create and initialize encryption manager
encryption_manager = TaskEncryptionManager(
    task_config=mock_config,
    logger=logger
)
encryption_manager.initialize()

# Check if files are encrypted
file_paths = [
    Path("data/file1.csv"),
    Path("data/file2.csv"),
    Path("data/file3.csv")
]

for path in file_paths:
    is_encrypted = encryption_manager.is_file_encrypted(path)
    print(f"File {path}: {'Encrypted' if is_encrypted else 'Not encrypted'}")
```

## Integration with BaseTask

The `encryption_manager.py` module is designed to integrate with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.initialize()
self.encryption_manager = TaskEncryptionManager(
    task_config=self.config,
    logger=self.logger,
    progress_manager=self.progress_manager
)

# Initialize encryption
self.encryption_manager.initialize()

# Get encryption info for backward compatibility
encryption_info = self.encryption_manager.get_encryption_info()
self.use_encryption = encryption_info["enabled"]
self.encryption_mode = EncryptionMode.from_string(encryption_info["mode"])

# In BaseTask.add_operation()
# Pass encryption context, not raw key
if self.use_encryption and 'use_encryption' in supported_params:
    operation_params['encryption_context'] = self.encryption_manager.get_encryption_context()

# In BaseTask.finalize()
# Redact sensitive data before logging or reporting
redacted_metrics = self.encryption_manager.redact_sensitive_data(self.metrics)
self.reporter.add_task_summary(metrics=redacted_metrics)

# Cleanup on task completion
self.encryption_manager.cleanup()
```

## Security Best Practices

1. **Never Store Raw Keys in Memory**: Always use `MemoryProtectedKey` for key storage to minimize exposure.

2. **Use Encryption Contexts**: Pass `EncryptionContext` to operations instead of raw keys.

3. **Redact Sensitive Data**: Always redact sensitive information before logging or including in reports.

4. **Validate Path Security**: Ensure all paths meet security requirements before use.

5. **Prefer External Key Stores**: Use key store integration for more secure key management.

6. **Safely Clean Up Keys**: Explicitly call `cleanup()` when encryption manager is no longer needed.

7. **Honor User Settings**: Respect the user's encryption preferences and don't enforce encryption where not requested.

8. **Add Additional Sensitive Params**: Use `add_sensitive_param_names()` to extend the default list of sensitive parameters for your application.

9. **Check File Encryption**: Use `check_dataset_encryption()` to ensure files are properly encrypted when encryption is enabled.

10. **Validate Dataset Compatibility**: Ensure operations support the active encryption mode to prevent data loss.

## Encryption Modes Comparison

| Mode | Security Level | Key Management | Suitability | Library Dependency |
|------|---------------|----------------|-------------|------------------|
| NONE | None | N/A | Non-sensitive data | None |
| SIMPLE | Medium | Simple key files | Basic encryption needs | cryptography (Fernet) |
| AGE | High | Advanced key management | Production/compliance scenarios | pyage |

## Key Generation and Storage Workflows

### Key Generation Workflow

```
1. Check if key is available in file (if path provided)
2. Check if key is available in key store (if configured)
3. Generate new key if none found (for simple mode only)
4. Optionally save generated key to file for reuse
5. Return False if no key available and encryption required
```

### Key Resolution Priority

```
1. Encryption key file (if exists and path provided)
2. Key store (if available and contains key for task)
3. Newly generated key (if SIMPLE mode)
4. Fall back to NONE mode if no key available
```

## Data Redaction Behavior

The data redaction functionality replaces sensitive information with placeholders according to these rules:

1. Dictionary keys that match sensitive pattern: `{"api_key": "value"}` → `{"<redacted:api>...": "<redacted>"}`

2. Dictionary values with sensitive keys: `{"password": "secret123"}` → `{"password": "<redacted>"}`

3. Long strings that look like keys: `"sk-abcdefghijklmnopqrstuvwxyz123456"` → `"<redacted:key-like>"`

4. Sensitive data in nested structures: Recursively processed to maintain structure while redacting sensitive parts

## Considerations and Limitations

1. **Memory Protection**: While the module attempts to securely clean up keys in memory, Python's memory management may retain copies temporarily.

2. **Performance Impact**: Encryption and decryption operations add processing overhead.

3. **Library Dependencies**: Full functionality requires the optional cryptography and/or pyage libraries.

4. **File Format Detection**: The file encryption detection is based on headers and may not be reliable for all file formats.

5. **Cross-Platform Compatibility**: Some security features may behave differently across operating systems.