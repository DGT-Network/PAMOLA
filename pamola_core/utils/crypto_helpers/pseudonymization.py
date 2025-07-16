"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Cryptographic Pseudonymization Utilities
Package:       pamola_core.utils.crypto_helpers
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
    This module provides cryptographic utilities for pseudonymization operations,
    including secure hashing, salt/pepper management, mapping encryption, and
    pseudonym generation. It serves as the cryptographic foundation for the
    pseudonymization operations in the anonymization framework.

Key Features:
    - Keccak-256 (SHA-3) hashing with salt and pepper
    - Secure salt management with persistence and versioning
    - Session-based pepper generation with secure cleanup
    - AES-256-GCM encryption for mapping storage
    - Various pseudonym generation strategies (UUID, sequential, random)
    - Constant-time comparison for security
    - Memory-secure operations with explicit cleanup
    - Thread-safe implementations for concurrent processing
    - Context manager support for secure resources

Security Considerations:
    - All cryptographic operations use industry-standard algorithms
    - Sensitive data is securely cleared from memory after use
    - Constant-time comparisons prevent timing attacks
    - Keys and peppers are never logged or persisted in plaintext
    - Thread-safe operations for multi-threaded environments

Dependencies:
    - cryptography: For AES-GCM encryption
    - hashlib: For SHA-3 (Keccak) hashing
    - secrets: For cryptographically secure random generation
    - uuid: For UUID generation
    - base64/base58: For encoding options
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import string
import threading
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Try to import cryptography for AES-GCM
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    AESGCM = None  # type: ignore

# Try to import base58 for encoding
try:
    import base58

    BASE58_AVAILABLE = True
except ImportError:
    BASE58_AVAILABLE = False


    # Create stub for static analyzers
    class _Base58Stub:
        """Minimal stub for base58 API to satisfy static analyzers."""

        @staticmethod
        def b58encode(data: bytes) -> bytes:
            """Stub for b58encode."""
            raise ImportError("base58 package required for base58 encoding")

        @staticmethod
        def b58decode(data: Union[str, bytes]) -> bytes:
            """Stub for b58decode."""
            raise ImportError("base58 package required for base58 decoding")


    base58 = _Base58Stub()  # type: ignore[assignment]

# Configure module logger
logger = logging.getLogger(__name__)


class PseudonymizationError(Exception):
    """Base exception for pseudonymization errors."""
    pass


class CryptoError(PseudonymizationError):
    """Exception for cryptographic operation failures."""
    pass


class HashCollisionError(PseudonymizationError):
    """Exception for hash collision detection."""
    pass


class SecureBytes:
    """
    Secure byte array that overwrites memory on deletion.

    This class provides a secure container for sensitive byte data that
    ensures the memory is overwritten when the object is deleted.
    Supports context manager protocol for automatic cleanup.
    """

    def __init__(self, data: bytes):
        """
        Initialize secure bytes container.

        Args:
            data: Sensitive byte data to protect
        """
        self._data = bytearray(data)
        self._lock = threading.Lock()

    def get(self) -> bytes:
        """
        Get the byte data.

        Returns:
            The protected byte data
        """
        with self._lock:
            return bytes(self._data)

    def clear(self) -> None:
        """Securely clear the data from memory."""
        with self._lock:
            # Overwrite with random data
            for i in range(len(self._data)):
                self._data[i] = secrets.randbits(8)
            # Then overwrite with zeros
            for i in range(len(self._data)):
                self._data[i] = 0
            # Clear the reference
            self._data = bytearray()

    def __del__(self):
        """Ensure data is cleared on deletion."""
        self.clear()

    def __len__(self) -> int:
        """Return length of data."""
        return len(self._data)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clear data."""
        self.clear()
        return False


class HashGenerator:
    """
    Secure hash generator using Keccak-256 (SHA-3 family).

    This class provides methods for generating cryptographic hashes
    with salt and pepper for pseudonymization operations.
    """

    def __init__(self, algorithm: str = "sha3_256"):
        """
        Initialize hash generator.

        Args:
            algorithm: Hash algorithm to use (default: "sha3_256" for Keccak-256)
        """
        self.algorithm = algorithm
        self._hash_count = 0
        self._collision_count = 0
        self._lock = threading.Lock()

        # Validate algorithm
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"Hash algorithm '{algorithm}' not available")

    def hash_with_salt(self, data: Union[str, bytes], salt: bytes) -> bytes:
        """
        Generate hash with salt.

        Args:
            data: Data to hash (string or bytes)
            salt: Salt bytes

        Returns:
            Hash digest bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        with self._lock:
            self._hash_count += 1

        # Create hash object
        h = hashlib.new(self.algorithm)
        h.update(data)
        h.update(salt)

        return h.digest()

    def hash_with_salt_and_pepper(self, data: Union[str, bytes],
                                  salt: bytes, pepper: bytes) -> bytes:
        """
        Generate two-stage hash with salt and pepper.

        Args:
            data: Data to hash
            salt: Salt bytes
            pepper: Pepper bytes (session-specific)

        Returns:
            Final hash digest bytes
        """
        # Stage 1: Hash with salt
        intermediate = self.hash_with_salt(data, salt)

        # Stage 2: Hash intermediate with pepper
        if pepper:
            h2 = hashlib.new(self.algorithm)
            h2.update(intermediate)
            h2.update(pepper)
            return h2.digest()

        return intermediate

    def _encode_base58(self, data: bytes) -> str:
        """
        Encode data to base58 with availability check.

        Args:
            data: Data to encode

        Returns:
            Base58 encoded string

        Raises:
            ImportError: If base58 package not available
        """
        if not BASE58_AVAILABLE:
            raise ImportError("base58 package required for base58 encoding")
        return base58.b58encode(data).decode('ascii')

    def format_output(self, hash_bytes: bytes, output_format: str = "hex",
                      output_length: Optional[int] = None) -> str:
        """
        Format hash output according to specification.

        Args:
            hash_bytes: Raw hash bytes
            output_format: Output format ("hex", "base64", "base58")
            output_length: Optional truncation length

        Returns:
            Formatted hash string
        """
        if output_format == "hex":
            output = hash_bytes.hex()
        elif output_format == "base64":
            output = base64.urlsafe_b64encode(hash_bytes).decode('ascii').rstrip('=')
        elif output_format == "base58":
            output = self._encode_base58(hash_bytes)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

        # Truncate if requested
        if output_length and len(output) > output_length:
            output = output[:output_length]

        return output

    def get_statistics(self) -> Dict[str, int]:
        """Get hash generation statistics."""
        with self._lock:
            return {
                "total_hashes": self._hash_count,
                "collisions_detected": self._collision_count
            }


class SaltManager:
    """
    Manages salt generation, storage, and retrieval with versioning support.

    This class handles salt management for pseudonymization operations,
    including field-specific, global, and custom salts with file versioning.
    """

    SALT_FILE_VERSION = "1.0"

    def __init__(self, salts_file: Optional[Path] = None):
        """
        Initialize salt manager.

        Args:
            salts_file: Path to salts storage file (default: salts.json)
        """
        self.salts_file = salts_file or Path("salts.json")
        self._lock = threading.Lock()
        self._cache: Dict[str, bytes] = {}

    def generate_salt(self, length: int = 32) -> bytes:
        """
        Generate cryptographically secure salt.

        Args:
            length: Salt length in bytes (default: 32)

        Returns:
            Generated salt bytes
        """
        return secrets.token_bytes(length)

    def get_or_create_salt(self, identifier: str, length: int = 32) -> bytes:
        """
        Get existing salt or create new one for identifier.

        Args:
            identifier: Salt identifier (e.g., field name)
            length: Salt length for new salts

        Returns:
            Salt bytes
        """
        with self._lock:
            # Check cache first
            if identifier in self._cache:
                return self._cache[identifier]

            # Load from file
            salts = self._load_salts()

            if identifier in salts:
                salt = bytes.fromhex(salts[identifier])
            else:
                # Generate new salt
                salt = self.generate_salt(length)
                salts[identifier] = salt.hex()
                self._save_salts(salts)

            # Cache the salt
            self._cache[identifier] = salt
            return salt

    def _load_salts(self) -> Dict[str, str]:
        """Load salts from file with version check."""
        if self.salts_file.exists():
            try:
                with open(self.salts_file, 'r') as f:
                    data = json.load(f)

                # Check file version
                if isinstance(data, dict) and "_version" in data:
                    file_version = data.get("_version", "0.0")
                    if file_version != self.SALT_FILE_VERSION:
                        logger.warning(
                            f"Salt file version mismatch: {file_version} != {self.SALT_FILE_VERSION}"
                        )

                    # Extract only salts (ignore metadata)
                    salts = data.get("salts", {})
                else:
                    # Old format without version
                    salts = data
                    logger.info("Migrating salt file to versioned format")

                return salts

            except Exception as e:
                logger.warning(f"Failed to load salts file: {e}")
        return {}

    def _save_salts(self, salts: Dict[str, str]) -> None:
        """Save salts to file with metadata."""
        data = {
            "_version": self.SALT_FILE_VERSION,
            "_updated": datetime.now(timezone.utc).isoformat(),
            "_count": len(salts),
            "salts": salts
        }

        temp_file = self.salts_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_file.replace(self.salts_file)

        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise CryptoError(f"Failed to save salts: {e}")


class PepperGenerator:
    """
    Generates and manages session-specific pepper values.

    Pepper values are kept only in memory and are securely cleared
    when no longer needed.
    """

    def __init__(self):
        """Initialize pepper generator."""
        self._pepper: Optional[SecureBytes] = None
        self._lock = threading.Lock()

    def generate(self, length: int = 32) -> bytes:
        """
        Generate new pepper for session.

        Args:
            length: Pepper length in bytes

        Returns:
            Generated pepper bytes
        """
        with self._lock:
            # Clear any existing pepper
            if self._pepper:
                self._pepper.clear()

            # Generate new pepper
            pepper_bytes = secrets.token_bytes(length)
            self._pepper = SecureBytes(pepper_bytes)

            logger.info(f"Generated {length}-byte pepper for session")
            return pepper_bytes

    def get(self) -> Optional[bytes]:
        """Get current pepper if available."""
        with self._lock:
            if self._pepper:
                return self._pepper.get()
            return None

    def clear(self) -> None:
        """Securely clear pepper from memory."""
        with self._lock:
            if self._pepper:
                self._pepper.clear()
                self._pepper = None
                logger.info("Pepper cleared from memory")


class MappingEncryption:
    """
    Provides AES-256-GCM encryption for mapping storage.

    This class handles encryption and decryption of mapping files
    to ensure reversible pseudonymization mappings are protected.
    """

    def __init__(self, key: bytes):
        """
        Initialize mapping encryption.

        Args:
            key: 256-bit encryption key

        Raises:
            ImportError: If cryptography package not available
            ValueError: If key length is invalid
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for mapping encryption")

        if len(key) != 32:
            raise ValueError("Encryption key must be 32 bytes (256 bits)")

        self.cipher = AESGCM(key)
        self._secure_key = SecureBytes(key)

    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt data with AES-256-GCM.

        Args:
            plaintext: Data to encrypt
            associated_data: Optional authenticated data

        Returns:
            Encrypted data with prepended nonce
        """
        # Generate random 96-bit nonce
        nonce = os.urandom(12)

        # Encrypt with optional associated data
        ciphertext = self.cipher.encrypt(nonce, plaintext, associated_data)

        # Return nonce + ciphertext (includes auth tag)
        return nonce + ciphertext

    def decrypt(self, encrypted: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt AES-256-GCM encrypted data.

        Args:
            encrypted: Encrypted data with prepended nonce
            associated_data: Optional authenticated data

        Returns:
            Decrypted plaintext

        Raises:
            ValueError: If data format is invalid
            CryptoError: If decryption fails
        """
        if len(encrypted) < 28:  # 12 (nonce) + 16 (min ciphertext + tag)
            raise ValueError("Invalid encrypted data format")

        # Extract nonce and ciphertext
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]

        try:
            return self.cipher.decrypt(nonce, ciphertext, associated_data)
        except Exception as e:
            raise CryptoError(f"Decryption failed: {e}")

    def __del__(self):
        """Clear key on deletion."""
        if hasattr(self, '_secure_key'):
            self._secure_key.clear()


class PseudonymGenerator:
    """
    Generates various types of pseudonyms for consistent mapping.

    Supports UUID, sequential, and random string generation with
    optional prefixes and formatting.
    """

    def __init__(self, pseudonym_type: str = "uuid"):
        """
        Initialize pseudonym generator.

        Args:
            pseudonym_type: Type of pseudonym ("uuid", "sequential", "random_string")
        """
        self.pseudonym_type = pseudonym_type
        self._counter = 0
        self._counter_lock = threading.Lock()

        # Validate type
        valid_types = ["uuid", "sequential", "random_string"]
        if pseudonym_type not in valid_types:
            raise ValueError(f"Invalid pseudonym type. Must be one of: {valid_types}")

    def generate(self, prefix: Optional[str] = None, length: int = 36) -> str:
        """
        Generate a new pseudonym.

        Args:
            prefix: Optional prefix for the pseudonym
            length: Length for random_string type

        Returns:
            Generated pseudonym string
        """
        if self.pseudonym_type == "uuid":
            pseudonym = str(uuid.uuid4())

        elif self.pseudonym_type == "sequential":
            with self._counter_lock:
                self._counter += 1
                pseudonym = f"{self._counter:08d}"

        elif self.pseudonym_type == "random_string":
            # Alphanumeric characters
            chars = string.ascii_letters + string.digits
            pseudonym = ''.join(secrets.choice(chars) for _ in range(length))

        else:
            raise ValueError(f"Unknown pseudonym type: {self.pseudonym_type}")

        # Add prefix if provided
        if prefix:
            pseudonym = f"{prefix}{pseudonym}"

        return pseudonym

    def generate_unique(self, existing: set, prefix: Optional[str] = None,
                        max_attempts: int = 100) -> str:
        """
        Generate a unique pseudonym not in existing set.

        Args:
            existing: Set of existing pseudonyms
            prefix: Optional prefix
            max_attempts: Maximum generation attempts

        Returns:
            Unique pseudonym

        Raises:
            ValueError: If unique pseudonym cannot be generated
        """
        for attempt in range(max_attempts):
            pseudonym = self.generate(prefix)
            if pseudonym not in existing:
                return pseudonym

        raise ValueError(f"Failed to generate unique pseudonym after {max_attempts} attempts")


def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Perform constant-time comparison of byte strings.

    This prevents timing attacks by ensuring comparison time
    doesn't depend on the position of the first difference.

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        True if equal, False otherwise
    """
    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y

    return result == 0


def validate_key_size(key: bytes, expected_bits: int = 256) -> None:
    """
    Validate encryption key size.

    Args:
        key: Key bytes to validate
        expected_bits: Expected key size in bits

    Raises:
        ValueError: If key size is invalid
    """
    expected_bytes = expected_bits // 8
    if len(key) != expected_bytes:
        raise ValueError(f"Key must be {expected_bits} bits ({expected_bytes} bytes), "
                         f"got {len(key) * 8} bits")


def derive_key_from_password(password: str, salt: bytes,
                             iterations: int = 100000,
                             key_length: int = 32) -> bytes:
    """
    Derive encryption key from password using PBKDF2.

    Args:
        password: Password string
        salt: Salt bytes (should be at least 16 bytes)
        iterations: PBKDF2 iterations (min recommended: 100000)
        key_length: Desired key length in bytes (default: 32 for AES-256)

    Returns:
        Derived key bytes

    Raises:
        ValueError: If parameters are invalid
    """
    if len(salt) < 16:
        raise ValueError("Salt must be at least 16 bytes")

    if iterations < 10000:
        raise ValueError("Iterations should be at least 10000 for security")

    if key_length not in [16, 24, 32]:
        raise ValueError("Key length must be 16, 24, or 32 bytes")

    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'),
                               salt, iterations, dklen=key_length)


def generate_deterministic_pseudonym(identifier: str, domain: str,
                                     secret_key: bytes) -> str:
    """
    Generate deterministic pseudonym using HMAC.

    This creates the same pseudonym for the same identifier+domain
    combination, useful for cross-system consistency.

    Args:
        identifier: Original identifier
        domain: Domain/context for the pseudonym
        secret_key: Secret key for HMAC

    Returns:
        Deterministic pseudonym (hex string)
    """
    import hmac

    # Create domain-separated message
    message = f"{domain}:{identifier}".encode('utf-8')

    # Generate HMAC
    h = hmac.new(secret_key, message, hashlib.sha256)

    # Return first 16 bytes as hex (32 chars)
    return h.digest()[:16].hex()


class CollisionTracker:
    """
    Tracks hash collisions for monitoring and debugging.

    This class helps detect and handle hash collisions in
    pseudonymization operations.
    """

    def __init__(self, max_tracked: int = 1000):
        """
        Initialize collision tracker.

        Args:
            max_tracked: Maximum number of hashes to track
        """
        self.max_tracked = max_tracked
        self._tracker: Dict[str, str] = {}
        self._collision_count = 0
        self._lock = threading.Lock()

    def check_and_record(self, pseudonym: str, original: str) -> Optional[str]:
        """
        Check for collision and record mapping.

        Args:
            pseudonym: Generated pseudonym
            original: Original value

        Returns:
            Previous original value if collision detected, None otherwise
        """
        with self._lock:
            if len(self._tracker) >= self.max_tracked:
                # Clear oldest entries (simple strategy)
                to_remove = len(self._tracker) // 4
                for key in list(self._tracker.keys())[:to_remove]:
                    del self._tracker[key]

            if pseudonym in self._tracker:
                if self._tracker[pseudonym] != original:
                    self._collision_count += 1
                    return self._tracker[pseudonym]
            else:
                self._tracker[pseudonym] = original

            return None

    def get_collision_count(self) -> int:
        """Get total number of collisions detected."""
        with self._lock:
            return self._collision_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get collision tracking statistics."""
        with self._lock:
            return {
                "tracked_hashes": len(self._tracker),
                "collision_count": self._collision_count,
                "collision_rate": self._collision_count / len(self._tracker)
                if self._tracker else 0.0
            }

    def export_collisions(self, output_file: Path) -> None:
        """
        Export collision information for analysis.

        Args:
            output_file: Path to save collision data
        """
        with self._lock:
            collision_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tracked": len(self._tracker),
                "collision_count": self._collision_count,
                "collision_rate": self._collision_count / len(self._tracker) if self._tracker else 0.0,
                "collisions": []
            }

            # Find all collisions
            pseudonym_to_originals = {}
            for pseudonym, original in self._tracker.items():
                if pseudonym not in pseudonym_to_originals:
                    pseudonym_to_originals[pseudonym] = []
                pseudonym_to_originals[pseudonym].append(original)

            for pseudonym, originals in pseudonym_to_originals.items():
                if len(originals) > 1:
                    collision_data["collisions"].append({
                        "pseudonym": pseudonym,
                        "originals": originals,
                        "count": len(originals)
                    })

            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(collision_data, f, indent=2)


# Module metadata
__version__ = "1.1.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Define explicit exports
__all__ = [
    # Main classes
    'HashGenerator',
    'SaltManager',
    'PepperGenerator',
    'MappingEncryption',
    'PseudonymGenerator',
    'CollisionTracker',
    'SecureBytes',

    # Utility functions
    'constant_time_compare',
    'validate_key_size',
    'derive_key_from_password',
    'generate_deterministic_pseudonym',

    # Exceptions
    'PseudonymizationError',
    'CryptoError',
    'HashCollisionError',

    # Constants
    'CRYPTOGRAPHY_AVAILABLE',
    'BASE58_AVAILABLE'
]