# PAMOLA.CORE Pseudonymization Operations Software Requirements Sub-Specification

**Document Version:** 1.1.0  
**Parent Document:** PAMOLA.CORE Anonymization Package SRS v4.1.0  
**Last Updated:** 2025-01-20  
**Status:** Final

## 1. Introduction

### 1.1 Purpose

This Software Requirements Sub-Specification (Sub-SRS) defines the detailed requirements for pseudonymization operations within the PAMOLA.CORE anonymization package. Pseudonymization operations replace identifiers with pseudonyms while maintaining referential integrity and optional reversibility.

### 1.2 Scope

This document covers two pseudonymization operations for MVP:
- **Hash-Based Pseudonymization (HASH_SALT_PEPPER)**: Irreversible pseudonym generation using cryptographic hashing
- **Consistent Mapping Pseudonymization (CONSISTENT_MAPPING)**: Reversible pseudonym generation with encrypted mapping storage

All operations follow the base anonymization framework defined in the parent SRS.

### 1.3 Document Conventions

- **REQ-PSEUDO-XXX**: General pseudonymization requirements
- **REQ-HASH-XXX**: Hash-based pseudonymization specific requirements
- **REQ-MAP-XXX**: Consistent mapping specific requirements

### 1.4 Architecture Overview

```
pamola_core/anonymization/pseudonymization/
├── __init__.py
├── hash_based_op.py          # Hash-based pseudonymization operation
└── mapping_op.py             # Consistent mapping operation

pamola_core/anonymization/commons/    # Shared utilities (existing package)
├── pseudonymization_utils.py # Pseudonymization-specific utilities
└── mapping_storage.py        # Mapping file management

pamola_core/utils/crypto_helpers/    # Cryptographic infrastructure (existing)
└── pseudonymization.py       # Core crypto utilities
```

## 2. Common Pseudonymization Requirements

### 2.1 Base Class Inheritance

**REQ-PSEUDO-001 [MUST]** All pseudonymization operations SHALL inherit from `AnonymizationOperation` and follow the standard operation contract defined in the parent SRS (REQ-ANON-001).

### 2.2 Cryptographic Standards

**REQ-PSEUDO-002 [MUST]** Pseudonymization operations SHALL use industry-standard cryptographic algorithms provided by `crypto_helpers/pseudonymization.py`:
- **Hashing**: Keccak-256 (SHA-3 family) via `HashGenerator`
- **Encryption**: AES-256-GCM via `MappingEncryption`
- **Random Generation**: Cryptographically secure via `secrets` module

### 2.3 Referential Integrity

**REQ-PSEUDO-003 [MUST]** All pseudonymization operations SHALL maintain referential integrity:
- Same input values produce same pseudonyms within a task execution
- Consistency across multiple fields containing the same identifier
- Support for compound identifiers via `op_field_utils.create_composite_key()`

### 2.4 Framework Integration

**REQ-PSEUDO-004 [MUST]** Operations SHALL integrate with existing framework utilities:
- Use `op_field_utils.generate_output_field_name()` for output field naming
- Use `op_data_processing.get_dataframe_chunks()` for batch processing
- Use `op_data_processing.optimize_dataframe_dtypes()` for memory optimization
- Use `DataWriter` for all file operations

## 3. Hash-Based Pseudonymization (HASH_SALT_PEPPER)

### 3.1 Overview

**REQ-HASH-001 [MUST]** The `HashBasedPseudonymizationOperation` generates irreversible pseudonyms using cryptographic hashing with salt and pepper.

### 3.2 Constructor Interface

**REQ-HASH-002 [MUST]** Constructor signature:

```python
class HashBasedPseudonymizationOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field to pseudonymize
                 # Salt configuration
                 salt_source: str = "parameter",     # parameter, file
                 salt_value: Optional[Union[str, bytes]] = None,  # For parameter source
                 salt_file: Optional[Union[str, Path]] = None,    # For file source
                 # Pepper configuration
                 use_pepper: bool = True,            # Enable pepper for extra security
                 pepper_length: int = 32,            # Pepper length in bytes
                 # Output configuration
                 output_format: str = "hex",         # hex, base64, base58
                 output_length: Optional[int] = None, # Truncate output if specified
                 output_prefix: Optional[str] = None, # Optional prefix
                 output_field_name: Optional[str] = None,
                 # Collision handling
                 check_collisions: bool = True,      # Monitor for hash collisions
                 collision_strategy: str = "log",    # log, fail
                 # Standard parameters
                 mode: str = "REPLACE",
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,             # Cache hashes within execution
                 cache_size: int = 100000,           # Maximum cache entries
                 use_encryption: bool = False,        # Not applicable for hashing
                 encryption_key: Optional[Union[str, Path]] = None,
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 3.3 Salt Management (Simplified)

**REQ-HASH-003 [MUST]** Support simple salt configuration:

```python
# Use commons/pseudonymization_utils.py
def load_salt_configuration(config: Dict[str, Any], 
                          salt_file: Optional[Path] = None) -> bytes:
    """Load salt based on configuration."""
    source = config.get('source', 'parameter')
    
    if source == 'parameter':
        # Salt provided directly as hex string or bytes
        salt_value = config.get('value')
        if isinstance(salt_value, str):
            return bytes.fromhex(salt_value)
        return salt_value
    
    elif source == 'file':
        # Load from JSON file with field-specific salts
        with open(salt_file, 'r') as f:
            salts = json.load(f)
        field_name = config.get('field_name')
        return bytes.fromhex(salts[field_name])
```

### 3.4 Pepper Generation (Session-Based)

**REQ-HASH-004 [MUST]** Generate pepper once per task execution:

```python
# Use commons/pseudonymization_utils.py
def generate_session_pepper(length: int = 32) -> SecureBytes:
    """Generate pepper for current session."""
    pepper = secrets.token_bytes(length)
    return SecureBytes(pepper)  # Auto-cleanup on deletion
```

### 3.5 Hashing Algorithm

**REQ-HASH-005 [MUST]** Use `crypto_helpers.pseudonymization.HashGenerator`:

```python
def _generate_pseudonym(self, identifier: str) -> str:
    """Apply HASH_SALT_PEPPER algorithm."""
    # Use HashGenerator from crypto_helpers
    pepper_bytes = self._pepper.get() if self._pepper else b""
    hash_bytes = self._hash_generator.hash_with_salt_and_pepper(
        identifier, self._salt, pepper_bytes
    )
    
    # Format output using HashGenerator method
    pseudonym = self._hash_generator.format_output(
        hash_bytes, self.output_format, self.output_length
    )
    
    # Add prefix if configured
    if self.output_prefix:
        pseudonym = format_pseudonym_output(pseudonym, self.output_prefix)
    
    return pseudonym
```

### 3.6 Batch Processing with Framework Integration

**REQ-HASH-006 [MUST]** Process batches using framework utilities:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process batch with hash caching."""
    result = batch.copy()
    
    # Use op_field_utils for output field naming
    output_col = generate_output_field_name(
        self.field_name, self.mode, self.output_field_name,
        operation_suffix="hashed", column_prefix="_"
    )
    
    # Process with cache (from pseudonymization_utils)
    for idx, value in batch[self.field_name].items():
        if pd.isna(value) and self.null_strategy == "PRESERVE":
            continue
        
        str_value = str(value)
        
        # Check cache first
        pseudonym = self._cache.get(str_value) if self._cache else None
        
        if pseudonym is None:
            # Generate hash
            pseudonym = self._generate_pseudonym(str_value)
            
            # Check collisions using CollisionTracker
            if self._collision_tracker:
                collision = self._collision_tracker.check_and_record(
                    pseudonym, str_value
                )
                if collision:
                    self._handle_collision(str_value, pseudonym, collision)
            
            # Cache result
            if self._cache:
                self._cache.put(str_value, pseudonym)
        
        result.at[idx, output_col] = pseudonym
    
    return result
```

### 3.7 Memory Management

**REQ-HASH-007 [MUST]** Use framework utilities for memory optimization:

```python
def execute(self, data_source, task_dir, reporter=None, progress_tracker=None, **kwargs):
    """Execute with memory optimization."""
    # Optimize memory using op_data_processing
    df, _ = data_source.get_dataframe("main")
    if self.optimize_memory:
        df, info = optimize_dataframe_dtypes(df)
        
    # Adaptive batch sizing
    if self.adaptive_batch_size:
        memory_info = get_memory_usage(df)
        self.batch_size = self._calculate_optimal_batch_size(memory_info)
    
    # Process in chunks
    for chunk in get_dataframe_chunks(df, self.batch_size):
        processed_chunk = self.process_batch(chunk)
        # ... handle results ...
```

### 3.8 Metrics

**REQ-HASH-008 [MUST]** Collect these specific metrics:
- `values_pseudonymized`: Number of unique values hashed
- `cache_hit_rate`: From `PseudonymizationCache.get_statistics()`
- `collision_count`: From `CollisionTracker.get_statistics()`
- `hash_computation_time`: From `HashGenerator.get_statistics()`
- `salt_source`: Configuration used

## 4. Consistent Mapping Pseudonymization (CONSISTENT_MAPPING)

### 4.1 Overview

**REQ-MAP-001 [MUST]** The `ConsistentMappingPseudonymizationOperation` maintains an encrypted mapping between original values and generated pseudonyms, enabling reversibility.

### 4.2 Constructor Interface

**REQ-MAP-002 [MUST]** Constructor signature (simplified):

```python
class ConsistentMappingPseudonymizationOperation(AnonymizationOperation):
    def __init__(self,
                 field_name: str,                    # Field to pseudonymize
                 # Mapping configuration
                 mapping_file: Optional[Union[str, Path]] = None,  # Auto-generated if None
                 mapping_format: str = "csv",        # csv, json
                 # Pseudonym generation
                 pseudonym_type: str = "uuid",       # uuid, sequential, random_string
                 pseudonym_prefix: Optional[str] = None,  # Optional prefix
                 pseudonym_length: int = 36,         # For random_string type
                 # Encryption configuration
                 encryption_key: Union[str, bytes],  # Required hex string or bytes
                 # Mapping management
                 create_if_not_exists: bool = True,
                 backup_on_update: bool = True,      # Keep previous version
                 persist_frequency: int = 1000,      # Write every N new mappings
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,             # Not used (mapping is the cache)
                 use_encryption: bool = True,        # Always true for this operation
                 engine: str = "auto",
                 max_rows_in_memory: int = 1000000,
                 **kwargs):
```

### 4.3 Encryption via MappingStorage

**REQ-MAP-003 [MUST]** Use `commons/mapping_storage.py` with `crypto_helpers`:

```python
class MappingStorage:
    """Manages encrypted storage of pseudonymization mappings."""
    
    def __init__(self, mapping_file: Path, encryption_key: bytes,
                 format: str = "csv", backup_on_update: bool = True):
        self._encryptor = MappingEncryption(encryption_key)  # From crypto_helpers
        # ... rest of initialization ...
    
    def save(self, mapping: Dict[str, str]) -> None:
        """Encrypt and save mapping atomically."""
        # Serialize to CSV/JSON
        plaintext = self._serialize(mapping)
        
        # Encrypt using MappingEncryption
        encrypted = self._encryptor.encrypt(plaintext)
        
        # Atomic write
        temp_path = self.mapping_file.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            f.write(encrypted)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(self.mapping_file)
```

### 4.4 Pseudonym Generation

**REQ-MAP-004 [MUST]** Use `crypto_helpers.pseudonymization.PseudonymGenerator`:

```python
def _generate_unique_pseudonym(self) -> str:
    """Generate unique pseudonym using PseudonymGenerator."""
    existing = set(self._reverse_mapping.keys())
    return self._pseudonym_generator.generate_unique(
        existing, 
        prefix=self.pseudonym_prefix
    )
```

### 4.5 Batch Processing with Field Utils

**REQ-MAP-005 [MUST]** Process batches with proper field handling:

```python
def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
    """Process batch with consistent mapping."""
    result = batch.copy()
    
    # Use op_field_utils for output field
    output_col = generate_output_field_name(
        self.field_name, self.mode, self.output_field_name,
        operation_suffix="mapped", column_prefix="_"
    )
    
    # For compound identifiers
    if self.additional_fields:
        # Use op_field_utils.create_composite_key()
        composite_values = create_composite_key(
            batch, [self.field_name] + self.additional_fields,
            separator="|", hash_key=False
        )
        # Process composite values...
    
    # Standard processing with thread-safe mapping
    with self._mapping_lock:
        for idx, value in batch[self.field_name].items():
            # ... mapping logic ...
```

### 4.6 Atomic File Operations

**REQ-MAP-006 [MUST]** Ensure atomic persistence:

```python
def _persist_mappings(self) -> None:
    """Save mappings using MappingStorage."""
    with self._mapping_lock:
        self._mapping_storage.save(self._mapping)
        self.logger.info(f"Persisted {len(self._mapping)} mappings")
        self._new_mappings_count = 0
```

### 4.7 Thread Safety

**REQ-MAP-007 [MUST]** Ensure thread-safe operations:

```python
def __init__(self, ...):
    # Thread safety with standard threading
    self._mapping_lock = threading.RLock()
    
    # For Dask distributed processing
    if self.engine == "dask":
        # Note: Mapping operations may need centralized storage
        self.logger.warning("Dask support limited for mapping operations")
```

### 4.8 Metrics

**REQ-MAP-008 [MUST]** Collect these specific metrics:
- `total_mappings`: Current size of mapping dictionary
- `new_mappings_created`: New mappings in this execution
- `mapping_file_size`: From `MappingStorage.get_metadata()`
- `lookup_time_avg`: Average mapping lookup time
- `persistence_count`: Number of saves performed

## 5. Common Implementation Details

### 5.1 Simplified Key Management

**REQ-PSEUDO-005 [MUST]** Keys are provided as parameters:

```python
def _process_encryption_key(self, key: Union[str, bytes]) -> bytes:
    """Process encryption key parameter."""
    if isinstance(key, str):
        # Assume hex-encoded
        key_bytes = bytes.fromhex(key)
    else:
        key_bytes = key
    
    # Validate using crypto_helpers
    validate_key_size(key_bytes, 256)
    return key_bytes
```

### 5.2 Compound Identifiers

**REQ-PSEUDO-006 [SHOULD]** Support compound identifiers using field utils:

```python
# Use op_field_utils functions
composite_key = create_composite_key(
    df, 
    fields=['field1', 'field2'], 
    separator='|',
    hash_key=True  # For hash-based pseudonymization
)

# Or for reversible
reversible_key, info = create_reversible_composite_key(
    df,
    fields=['field1', 'field2'],
    encoding='base64'
)
```

### 5.3 Validation

**REQ-PSEUDO-007 [MUST]** Use existing validation framework:

```python
from pamola_core.anonymization.commons.validation import (
    check_field_exists,
    validate_strategy,
    validate_output_field_configuration
)

def validate_configuration(self) -> None:
    """Validate operation configuration."""
    # Use standard validators
    check_field_exists(self.df, self.field_name)
    
    # Validate output configuration
    validate_output_field_configuration(
        self.mode, self.output_field_name, self.field_name
    )
```

### 5.4 Visualization

**REQ-PSEUDO-008 [SHOULD]** Generate standard visualizations:

```python
def _generate_visualizations(self, original_data, processed_data, 
                           task_dir, writer):
    """Create visualization using commons utilities."""
    # Use standard visualization functions
    if isinstance(self, HashBasedPseudonymizationOperation):
        # Collision visualization if applicable
        if self._collision_tracker and self._collision_tracker.get_collision_count() > 0:
            create_metric_visualization(
                "collision_analysis",
                self._collision_tracker.get_statistics(),
                task_dir, self.field_name, "hash_pseudonymization"
            )
    
    elif isinstance(self, ConsistentMappingPseudonymizationOperation):
        # Mapping growth visualization
        create_metric_visualization(
            "mapping_growth",
            {"initial": 0, "final": len(self._mapping)},
            task_dir, self.field_name, "mapping_pseudonymization"
        )
```

## 6. Security Considerations

### 6.1 Memory Security

**REQ-PSEUDO-009 [MUST]** Use `SecureBytes` from crypto_helpers:

```python
def execute(self, ...):
    try:
        # Initialize pepper
        if self.use_pepper:
            self._pepper = generate_session_pepper(self.pepper_length)
        
        # Process data...
        
    finally:
        # Automatic cleanup via SecureBytes
        if hasattr(self, '_pepper') and self._pepper:
            self._pepper.clear()
```

### 6.2 Audit Logging

**REQ-PSEUDO-010 [SHOULD]** Log security-relevant events:

```python
# Standard logging, no complex audit infrastructure for MVP
self.logger.info(f"Generated {self.pepper_length}-byte pepper for session")
self.logger.info(f"Loaded salt from {self.salt_source}")
self.logger.warning(f"Hash collision detected: {collision_count} collisions")
```

## 7. Performance Optimization

### 7.1 Caching Strategy

**REQ-PSEUDO-011 [SHOULD]** Use `PseudonymizationCache` from commons:

```python
# In commons/pseudonymization_utils.py
class PseudonymizationCache:
    """Thread-safe LRU cache for pseudonyms."""
    def __init__(self, max_size: int = 100000):
        # Simple OrderedDict-based implementation
        # with thread safety and statistics
```

### 7.2 Batch Optimization

**REQ-PSEUDO-012 [SHOULD]** Use framework batch processing:

```python
# Use op_data_processing utilities
for chunk in get_dataframe_chunks(df, self.batch_size):
    # Process chunk
    processed = self.process_batch(chunk)
    
    # Memory cleanup for large batches
    if len(chunk) > 50000:
        force_garbage_collection()
```

## 8. Error Handling

### 8.1 Common Errors

**REQ-PSEUDO-013 [MUST]** Handle these error conditions:

1. **Cryptographic errors**: Invalid keys, missing crypto libraries
2. **File system errors**: Permission denied, disk full
3. **Memory errors**: Large mapping files, cache overflow
4. **Data errors**: Invalid identifiers, encoding issues

### 8.2 Error Recovery

**REQ-PSEUDO-014 [MUST]** Implement graceful error handling:

```python
def _handle_error(self, error: Exception, context: str) -> None:
    """Handle errors with appropriate recovery."""
    if isinstance(error, CryptoError):
        self.logger.error(f"Cryptographic error in {context}: {error}")
        raise  # Cannot recover from crypto errors
    
    elif isinstance(error, FileNotFoundError) and self.create_if_not_exists:
        self.logger.info(f"Creating new file for {context}")
        # Create new mapping file
    
    else:
        super()._handle_error(error, context)
```

## 9. Testing Requirements

### 9.1 Unit Tests

**REQ-PSEUDO-015 [MUST]** Test coverage must include:

1. **Hash-Based Tests**:
   - Salt loading from parameter and file
   - Pepper generation and cleanup
   - Hash computation with all output formats
   - Collision detection
   - Cache effectiveness

2. **Mapping Tests**:
   - Encryption/decryption roundtrip
   - Atomic file operations
   - Unique pseudonym generation
   - Thread safety
   - Persistence triggers

### 9.2 Integration Tests

**REQ-PSEUDO-016 [MUST]** Test framework integration:

```python
def test_field_utils_integration():
    """Test integration with op_field_utils."""
    # Test output field naming
    # Test composite key generation
    # Test conditional processing

def test_data_processing_integration():
    """Test integration with op_data_processing."""
    # Test chunk processing
    # Test memory optimization
    # Test adaptive batch sizing
```

## 10. Example Usage

### 10.1 Hash-Based Pseudonymization

```python
# Simple usage with parameter salt
operation = HashBasedPseudonymizationOperation(
    field_name="ssn",
    salt_source="parameter",
    salt_value="0123456789abcdef" * 4,  # 256-bit hex salt
    use_pepper=True,
    output_format="hex",
    output_length=16  # First 16 chars
)

# With salt file
operation = HashBasedPseudonymizationOperation(
    field_name="email",
    salt_source="file",
    salt_file="config/salts.json",
    output_format="base64",
    check_collisions=True
)
```

### 10.2 Consistent Mapping

```python
# Basic mapping with UUID
operation = ConsistentMappingPseudonymizationOperation(
    field_name="customer_id",
    encryption_key="0123456789abcdef" * 4,  # 256-bit key
    pseudonym_type="uuid",
    mapping_file="mappings/customers.csv.enc"
)

# Sequential with prefix
operation = ConsistentMappingPseudonymizationOperation(
    field_name="patient_id",
    encryption_key=encryption_key_bytes,
    pseudonym_type="sequential",
    pseudonym_prefix="PAT",
    persist_frequency=500  # Save every 500 new mappings
)
```

## 11. Summary

The updated pseudonymization operations provide privacy-preserving identifier replacement through:

- **Hash-Based (HASH_SALT_PEPPER)**: Simplified irreversible pseudonym generation using existing crypto infrastructure
- **Consistent Mapping (CONSISTENT_MAPPING)**: Streamlined reversible pseudonymization with secure storage

Key improvements in this version:
- Leverages existing `crypto_helpers/pseudonymization.py` module
- Uses framework utilities from `op_field_utils` and `op_data_processing`
- Simplified configuration with parameter-based key/salt management
- Maintains security while reducing complexity for MVP
- Full integration with PAMOLA.CORE framework standards