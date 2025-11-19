# PAMOLA.CORE Pseudonymization Package Implementation Plan (MVP)

## 1. Package Structure (Simplified)

```
pamola_core/anonymization/pseudonymization/
├── __init__.py
├── hash_based_op.py          # Hash-based pseudonymization operation
└── mapping_op.py             # Consistent mapping operation

pamola_core/anonymization/commons/    # Shared utilities (existing package)
├── pseudonymization_utils.py # Pseudonymization-specific utilities
└── mapping_storage.py        # Mapping file management
```

## 2. Dependencies and Integration

### 2.1 Crypto Integration
The implementation will use the existing `pamola_core/utils/crypto_helpers/pseudonymization.py` module which provides:
- `HashGenerator` - Keccak-256 hashing with salt and pepper
- `MappingEncryption` - AES-256-GCM for secure mapping storage
- `PseudonymGenerator` - UUID/sequential/random pseudonym generation
- `CollisionTracker` - Hash collision detection
- `SecureBytes` - Secure memory handling

### 2.2 Framework Integration
- Inherit from `base_anonymization_op.py`
- Use `op_field_utils.py` for field operations
- Use `op_data_processing.py` for memory optimization
- Use `DataWriter` for all file I/O operations
- Use existing validation and metrics utilities from commons

## 3. Module Implementation Details

### Phase 1: Commons Utilities

#### 3.1 `pseudonymization_utils.py` - Shared Pseudonymization Utilities
**Priority:** High  
**Location:** `pamola_core/anonymization/commons/`

**Core Functionality:**
```python
"""Shared utilities for pseudonymization operations."""

from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import json

from pamola_core.utils.crypto_helpers.pseudonymization import (
    HashGenerator,
    SecureBytes,
    constant_time_compare
)

logger = logging.getLogger(__name__)

class PseudonymizationCache:
    """Thread-safe LRU cache for pseudonyms."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[str]:
        """Get pseudonym from cache."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, value: str) -> None:
        """Add pseudonym to cache."""
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
            
            self._cache[key] = value
            self._access_order.append(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0
            }

def load_salt_configuration(config: Dict[str, Any], 
                          salt_file: Optional[Path] = None) -> bytes:
    """
    Load salt based on configuration.
    
    Args:
        config: Salt configuration dict with 'source' and 'value'
        salt_file: Optional path to salts file
        
    Returns:
        Salt bytes
    """
    source = config.get('source', 'parameter')
    
    if source == 'parameter':
        # Salt provided directly
        salt_value = config.get('value')
        if not salt_value:
            raise ValueError("salt_value required when source is 'parameter'")
        
        if isinstance(salt_value, str):
            # Assume hex encoded
            return bytes.fromhex(salt_value)
        elif isinstance(salt_value, bytes):
            return salt_value
        else:
            raise ValueError("salt_value must be hex string or bytes")
    
    elif source == 'file':
        # Load from JSON file
        if not salt_file or not salt_file.exists():
            raise ValueError(f"Salt file not found: {salt_file}")
        
        with open(salt_file, 'r') as f:
            salts = json.load(f)
        
        field_name = config.get('field_name')
        if field_name not in salts:
            raise ValueError(f"Salt for field '{field_name}' not found in {salt_file}")
        
        return bytes.fromhex(salts[field_name])
    
    else:
        raise ValueError(f"Unknown salt source: {source}")

def generate_session_pepper(length: int = 32) -> SecureBytes:
    """
    Generate pepper for current session.
    
    Args:
        length: Pepper length in bytes
        
    Returns:
        SecureBytes containing pepper
    """
    import secrets
    pepper = secrets.token_bytes(length)
    return SecureBytes(pepper)

def format_pseudonym_output(pseudonym: str, 
                          prefix: Optional[str] = None,
                          suffix: Optional[str] = None) -> str:
    """Format pseudonym with optional prefix/suffix."""
    if prefix:
        pseudonym = f"{prefix}{pseudonym}"
    if suffix:
        pseudonym = f"{pseudonym}{suffix}"
    return pseudonym
```

#### 3.2 `mapping_storage.py` - Mapping File Management
**Priority:** High  
**Location:** `pamola_core/anonymization/commons/`

**Core Functionality:**
```python
"""Secure storage and retrieval of pseudonymization mappings."""

import csv
import json
import io
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from pamola_core.utils.crypto_helpers.pseudonymization import MappingEncryption
from pamola_core.utils.ops.op_data_writer import DataWriter

logger = logging.getLogger(__name__)

class MappingStorage:
    """Manages encrypted storage of pseudonymization mappings."""
    
    def __init__(self, 
                 mapping_file: Path,
                 encryption_key: bytes,
                 format: str = "csv",
                 backup_on_update: bool = True):
        """
        Initialize mapping storage.
        
        Args:
            mapping_file: Path to mapping file
            encryption_key: 256-bit encryption key
            format: Storage format ("csv" or "json")
            backup_on_update: Whether to backup before updates
        """
        self.mapping_file = mapping_file
        self.format = format
        self.backup_on_update = backup_on_update
        self._encryptor = MappingEncryption(encryption_key)
        self._lock = threading.RLock()
        self.logger = logger
    
    def load(self) -> Dict[str, str]:
        """Load and decrypt mapping from file."""
        with self._lock:
            if not self.mapping_file.exists():
                return {}
            
            try:
                # Read encrypted file
                with open(self.mapping_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # Decrypt
                decrypted_data = self._encryptor.decrypt(encrypted_data)
                
                # Parse based on format
                if self.format == "csv":
                    reader = csv.DictReader(io.StringIO(decrypted_data.decode('utf-8')))
                    return {row['original']: row['pseudonym'] for row in reader}
                else:  # json
                    return json.loads(decrypted_data)
                    
            except Exception as e:
                self.logger.error(f"Failed to load mapping: {e}")
                raise
    
    def save(self, mapping: Dict[str, str]) -> None:
        """Encrypt and save mapping atomically."""
        with self._lock:
            # Create backup if requested
            if self.backup_on_update and self.mapping_file.exists():
                backup_path = self.mapping_file.with_suffix('.bak')
                shutil.copy2(self.mapping_file, backup_path)
            
            # Serialize mapping
            if self.format == "csv":
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=['original', 'pseudonym'])
                writer.writeheader()
                for orig, pseudo in mapping.items():
                    writer.writerow({'original': orig, 'pseudonym': pseudo})
                plaintext = output.getvalue().encode('utf-8')
            else:  # json
                plaintext = json.dumps(mapping, sort_keys=True).encode('utf-8')
            
            # Encrypt
            encrypted_data = self._encryptor.encrypt(plaintext)
            
            # Atomic write
            temp_path = self.mapping_file.with_suffix('.tmp')
            try:
                with open(temp_path, 'wb') as f:
                    f.write(encrypted_data)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename
                temp_path.replace(self.mapping_file)
                
            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get mapping file metadata."""
        if not self.mapping_file.exists():
            return {"exists": False}
        
        stat = self.mapping_file.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "format": self.format
        }
```

### Phase 2: Hash-Based Pseudonymization Operation

#### 3.1 `hash_based_op.py` - Irreversible Pseudonymization
**Priority:** High  
**Location:** `pamola_core/anonymization/pseudonymization/`

**Core Implementation:**
```python
"""
Hash-based pseudonymization operation using Keccak-256 with salt and pepper.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.pseudonymization_utils import (
    PseudonymizationCache,
    load_salt_configuration,
    generate_session_pepper,
    format_pseudonym_output
)
from pamola_core.utils.crypto_helpers.pseudonymization import (
    HashGenerator,
    CollisionTracker,
    SecureBytes
)

logger = logging.getLogger(__name__)

class HashBasedPseudonymizationOperation(AnonymizationOperation):
    """Irreversible pseudonymization using Keccak-256 with salt and pepper."""
    
    def __init__(self,
                 field_name: str,
                 # Salt configuration
                 salt_source: str = "parameter",  # parameter, file
                 salt_value: Optional[Union[str, bytes]] = None,
                 salt_file: Optional[Union[str, Path]] = None,
                 # Pepper configuration  
                 use_pepper: bool = True,
                 pepper_length: int = 32,
                 # Output configuration
                 output_format: str = "hex",
                 output_length: Optional[int] = None,
                 output_prefix: Optional[str] = None,
                 # Collision handling
                 check_collisions: bool = True,
                 collision_strategy: str = "log",  # log, fail
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 use_cache: bool = True,
                 cache_size: int = 100000,
                 **kwargs):
        """Initialize hash-based pseudonymization operation."""
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            null_strategy=null_strategy,
            batch_size=batch_size,
            use_cache=use_cache,
            **kwargs
        )
        
        # Store configuration
        self.salt_source = salt_source
        self.salt_value = salt_value
        self.salt_file = Path(salt_file) if salt_file else None
        self.use_pepper = use_pepper
        self.pepper_length = pepper_length
        self.output_format = output_format
        self.output_length = output_length
        self.output_prefix = output_prefix
        self.check_collisions = check_collisions
        self.collision_strategy = collision_strategy
        
        # Initialize components
        self._hash_generator = HashGenerator()
        self._cache = PseudonymizationCache(max_size=cache_size) if use_cache else None
        self._collision_tracker = CollisionTracker() if check_collisions else None
        
        # Will be initialized during execution
        self._salt: Optional[bytes] = None
        self._pepper: Optional[SecureBytes] = None
        
    def _initialize_crypto(self, task_dir: Path) -> None:
        """Initialize salt and pepper for operation."""
        # Load salt
        salt_config = {
            'source': self.salt_source,
            'value': self.salt_value,
            'field_name': self.field_name
        }
        
        # If salt file path is relative, make it relative to task_dir
        salt_file = self.salt_file
        if salt_file and not salt_file.is_absolute():
            salt_file = task_dir / salt_file
            
        self._salt = load_salt_configuration(salt_config, salt_file)
        
        # Generate pepper if enabled
        if self.use_pepper:
            self._pepper = generate_session_pepper(self.pepper_length)
            self.logger.info(f"Generated {self.pepper_length}-byte pepper for session")
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process batch with hash-based pseudonymization."""
        result = batch.copy()
        
        # Determine output column
        output_col = self.output_field_name if self.mode == "ENRICH" else self.field_name
        if self.mode == "ENRICH" and output_col not in result.columns:
            result[output_col] = None
        
        # Process each value
        for idx, value in batch[self.field_name].items():
            # Handle nulls
            if pd.isna(value) and self.null_strategy == "PRESERVE":
                if self.mode == "ENRICH":
                    result.at[idx, output_col] = value
                continue
            
            str_value = str(value)
            
            # Check cache first
            pseudonym = None
            if self._cache:
                pseudonym = self._cache.get(str_value)
            
            if pseudonym is None:
                # Generate hash
                pepper_bytes = self._pepper.get() if self._pepper else b""
                hash_bytes = self._hash_generator.hash_with_salt_and_pepper(
                    str_value, self._salt, pepper_bytes
                )
                
                # Format output
                pseudonym = self._hash_generator.format_output(
                    hash_bytes, self.output_format, self.output_length
                )
                
                # Add prefix if configured
                pseudonym = format_pseudonym_output(pseudonym, self.output_prefix)
                
                # Check for collisions
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
    
    def _handle_collision(self, new_value: str, pseudonym: str, 
                         existing_value: str) -> None:
        """Handle detected hash collision."""
        msg = (f"Hash collision detected: '{new_value}' and '{existing_value}' "
               f"both hash to {pseudonym[:16]}...")
        
        if self.collision_strategy == "fail":
            raise HashCollisionError(msg)
        else:  # log
            self.logger.warning(msg)
    
    def execute(self, data_source, task_dir: Path, 
                reporter=None, progress_tracker=None, **kwargs):
        """Execute with crypto initialization and cleanup."""
        try:
            # Initialize crypto components
            self._initialize_crypto(task_dir)
            
            # Execute operation
            result = super().execute(
                data_source, task_dir, reporter, progress_tracker, **kwargs
            )
            
            return result
            
        finally:
            # Secure cleanup
            if self._pepper:
                self._pepper.clear()
                self._pepper = None
    
    def _collect_specific_metrics(self, original_data: pd.Series,
                                 anonymized_data: pd.Series) -> Dict[str, Any]:
        """Collect hash-specific metrics."""
        metrics = {
            "hash_algorithm": "Keccak-256",
            "salt_source": self.salt_source,
            "pepper_enabled": self.use_pepper,
            "output_format": self.output_format,
        }
        
        if self._cache:
            cache_stats = self._cache.get_statistics()
            metrics.update({
                "cache_size": cache_stats["size"],
                "cache_hit_rate": cache_stats["hit_rate"]
            })
        
        if self._collision_tracker:
            collision_stats = self._collision_tracker.get_statistics()
            metrics.update({
                "collision_count": collision_stats["collision_count"],
                "collision_rate": collision_stats["collision_rate"]
            })
        
        # Add hash statistics
        hash_stats = self._hash_generator.get_statistics()
        metrics["total_hashes_computed"] = hash_stats["total_hashes"]
        
        return metrics
```

### Phase 3: Consistent Mapping Pseudonymization

#### 3.1 `mapping_op.py` - Reversible Pseudonymization
**Priority:** Medium  
**Location:** `pamola_core/anonymization/pseudonymization/`

**Core Implementation:**
```python
"""
Consistent mapping pseudonymization with encrypted storage.
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from pamola_core.anonymization.base_anonymization_op import AnonymizationOperation
from pamola_core.anonymization.commons.mapping_storage import MappingStorage
from pamola_core.utils.crypto_helpers.pseudonymization import (
    PseudonymGenerator,
    validate_key_size
)

logger = logging.getLogger(__name__)

class ConsistentMappingPseudonymizationOperation(AnonymizationOperation):
    """Reversible pseudonymization with encrypted mapping storage."""
    
    def __init__(self,
                 field_name: str,
                 # Mapping configuration
                 mapping_file: Optional[Union[str, Path]] = None,
                 mapping_format: str = "csv",
                 # Pseudonym generation
                 pseudonym_type: str = "uuid",
                 pseudonym_prefix: Optional[str] = None,
                 pseudonym_length: int = 36,  # For random_string type
                 # Encryption configuration
                 encryption_key: Union[str, bytes],  # Required
                 # Mapping management
                 create_if_not_exists: bool = True,
                 backup_on_update: bool = True,
                 persist_frequency: int = 1000,
                 # Standard parameters
                 mode: str = "REPLACE",
                 output_field_name: Optional[str] = None,
                 null_strategy: str = "PRESERVE",
                 batch_size: int = 10000,
                 **kwargs):
        """Initialize consistent mapping operation."""
        super().__init__(
            field_name=field_name,
            mode=mode,
            output_field_name=output_field_name,
            null_strategy=null_strategy,
            batch_size=batch_size,
            **kwargs
        )
        
        # Store configuration
        self.mapping_file = mapping_file
        self.mapping_format = mapping_format
        self.pseudonym_type = pseudonym_type
        self.pseudonym_prefix = pseudonym_prefix
        self.pseudonym_length = pseudonym_length
        self.create_if_not_exists = create_if_not_exists
        self.backup_on_update = backup_on_update
        self.persist_frequency = persist_frequency
        
        # Process encryption key
        if isinstance(encryption_key, str):
            # Assume hex-encoded
            self._encryption_key = bytes.fromhex(encryption_key)
        else:
            self._encryption_key = encryption_key
        
        validate_key_size(self._encryption_key, 256)
        
        # Initialize components
        self._pseudonym_generator = PseudonymGenerator(pseudonym_type)
        
        # Will be initialized during execution
        self._mapping_storage: Optional[MappingStorage] = None
        self._mapping: Dict[str, str] = {}
        self._reverse_mapping: Dict[str, str] = {}
        self._new_mappings_count = 0
        self._mapping_lock = threading.RLock()
    
    def _initialize_mapping(self, task_dir: Path) -> None:
        """Initialize mapping storage and load existing mappings."""
        # Determine mapping file path
        if not self.mapping_file:
            self.mapping_file = task_dir / f"{self.field_name}_mapping.{self.mapping_format}.enc"
        elif not Path(self.mapping_file).is_absolute():
            self.mapping_file = task_dir / self.mapping_file
        
        # Initialize storage
        self._mapping_storage = MappingStorage(
            mapping_file=Path(self.mapping_file),
            encryption_key=self._encryption_key,
            format=self.mapping_format,
            backup_on_update=self.backup_on_update
        )
        
        # Load existing mappings
        try:
            self._mapping = self._mapping_storage.load()
            self._reverse_mapping = {v: k for k, v in self._mapping.items()}
            self.logger.info(f"Loaded {len(self._mapping)} existing mappings")
        except FileNotFoundError:
            if self.create_if_not_exists:
                self.logger.info("Creating new mapping file")
                self._mapping = {}
                self._reverse_mapping = {}
            else:
                raise
    
    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process batch with consistent mapping."""
        result = batch.copy()
        
        # Determine output column
        output_col = self.output_field_name if self.mode == "ENRICH" else self.field_name
        if self.mode == "ENRICH" and output_col not in result.columns:
            result[output_col] = None
        
        new_mappings = 0
        
        # Process each value
        for idx, value in batch[self.field_name].items():
            # Handle nulls
            if pd.isna(value) and self.null_strategy == "PRESERVE":
                if self.mode == "ENRICH":
                    result.at[idx, output_col] = value
                continue
            
            str_value = str(value)
            
            # Thread-safe mapping lookup/creation
            with self._mapping_lock:
                if str_value in self._mapping:
                    pseudonym = self._mapping[str_value]
                else:
                    # Generate unique pseudonym
                    existing_pseudonyms = set(self._reverse_mapping.keys())
                    pseudonym = self._pseudonym_generator.generate_unique(
                        existing_pseudonyms,
                        prefix=self.pseudonym_prefix
                    )
                    
                    # Add to mappings
                    self._mapping[str_value] = pseudonym
                    self._reverse_mapping[pseudonym] = str_value
                    new_mappings += 1
            
            result.at[idx, output_col] = pseudonym
        
        # Update new mappings count
        if new_mappings > 0:
            with self._mapping_lock:
                self._new_mappings_count += new_mappings
                
                # Persist if threshold reached
                if self._new_mappings_count >= self.persist_frequency:
                    self._persist_mappings()
        
        return result
    
    def _persist_mappings(self) -> None:
        """Save current mappings to encrypted file."""
        with self._mapping_lock:
            self._mapping_storage.save(self._mapping)
            self.logger.info(f"Persisted {len(self._mapping)} mappings "
                           f"({self._new_mappings_count} new)")
            self._new_mappings_count = 0
    
    def execute(self, data_source, task_dir: Path,
                reporter=None, progress_tracker=None, **kwargs):
        """Execute with mapping initialization and final save."""
        try:
            # Initialize mapping storage
            self._initialize_mapping(task_dir)
            
            # Execute operation
            result = super().execute(
                data_source, task_dir, reporter, progress_tracker, **kwargs
            )
            
            # Final save of any remaining mappings
            if self._new_mappings_count > 0:
                self._persist_mappings()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in mapping pseudonymization: {e}")
            raise
    
    def _collect_specific_metrics(self, original_data: pd.Series,
                                 anonymized_data: pd.Series) -> Dict[str, Any]:
        """Collect mapping-specific metrics."""
        metadata = self._mapping_storage.get_metadata() if self._mapping_storage else {}
        
        return {
            "pseudonym_type": self.pseudonym_type,
            "total_mappings": len(self._mapping),
            "new_mappings_created": self._new_mappings_count,
            "mapping_file_size": metadata.get("size_bytes", 0),
            "encryption_algorithm": "AES-256-GCM",
            "persist_frequency": self.persist_frequency,
            "reversible": True
        }
```

### Phase 4: Package Initialization

#### 4.1 `__init__.py` - Package Initialization
**Location:** `pamola_core/anonymization/pseudonymization/`

```python
"""
PAMOLA.CORE Pseudonymization Operations

This package provides two pseudonymization operations:
- HashBasedPseudonymizationOperation: Irreversible using Keccak-256
- ConsistentMappingPseudonymizationOperation: Reversible with encrypted storage
"""

from .hash_based_op import HashBasedPseudonymizationOperation
from .mapping_op import ConsistentMappingPseudonymizationOperation

__all__ = [
    'HashBasedPseudonymizationOperation',
    'ConsistentMappingPseudonymizationOperation'
]

__version__ = '1.0.0'
```

## 4. Integration Points

### 4.1 Crypto Helpers Integration
```python
# Update pamola_core/utils/crypto_helpers/__init__.py
from .pseudonymization import (
    HashGenerator,
    MappingEncryption,
    PseudonymGenerator,
    CollisionTracker,
    SecureBytes,
    constant_time_compare
)
```

### 4.2 Commons Integration
The commons utilities will be imported by the operations as needed. No changes to existing commons structure required.

### 4.3 Framework Integration
Both operations inherit from `AnonymizationOperation` and follow standard patterns:
- Use `DataWriter` for file operations
- Use progress tracking
- Generate standard metrics and visualizations
- Support conditional processing and risk-based filtering

## 5. Key Simplifications for MVP

1. **No Complex Key Management**: Keys are passed as parameters
2. **No Salt Infrastructure**: Salt is provided via configuration
3. **Simple Pepper Management**: One pepper per session, auto-cleanup
4. **Basic Collision Handling**: Log or fail, no complex resolution
5. **Standard File Formats**: CSV/JSON for mappings
6. **Minimal Dependencies**: Only essential crypto libraries

## 6. Testing Strategy

### 6.1 Unit Tests
```python
# test_hash_based_op.py
- Test basic pseudonymization
- Test salt/pepper usage
- Test output formats
- Test collision detection
- Test cache effectiveness

# test_mapping_op.py
- Test mapping creation and persistence
- Test encryption/decryption roundtrip
- Test unique pseudonym generation
- Test concurrent access
- Test mapping file recovery
```

### 6.2 Integration Tests
- End-to-end pipeline with both operations
- Large dataset processing
- Memory usage validation
- Performance benchmarks

### 6.3 Security Tests
- No plaintext leakage in logs or files
- Proper memory cleanup
- Key handling security
- Collision resistance

## 7. Success Criteria

- [ ] Both operations process 100K records in <10 seconds
- [ ] Zero plaintext identifiers in outputs or logs
- [ ] Successful integration with existing pipeline
- [ ] All metrics properly collected and visualized
- [ ] Thread-safe concurrent processing
- [ ] Comprehensive error handling
- [ ] 80%+ test coverage
- [ ] Clear documentation and examples

## 8. Implementation Order

1. **STEP 1**: Commons utilities (pseudonymization_utils.py, mapping_storage.py)
2. **STEP 2**: Hash-based operation with full testing
3. **STEP 3**: Mapping operation with full testing
4. **STEP 4**: Integration testing, documentation, and examples

## 9. Post-MVP Enhancements

- Format-preserving encryption option
- Hierarchical pseudonym generation
- Distributed mapping storage (Redis/Database)
- Advanced collision resolution strategies
- Differential privacy integration
- Performance optimizations for Dask
- Multi-field pseudonymization support
- Audit trail integration