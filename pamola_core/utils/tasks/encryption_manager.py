"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Encryption Manager
Description: Secure encryption key management and handling
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for securely managing encryption keys
and handling sensitive data, focusing on minimizing attack surface when
passing encryption capabilities from tasks to operations.

Key features:
- Secure encryption key initialization and handling
- Context-based encryption capabilities without direct key exposure
- Data structure redaction for logs and reports
- Support for multiple encryption modes (none, simple, age)
- Memory protection for encryption keys
- Integration with progress tracking
"""

import base64
import logging
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from pamola_core.utils.tasks.progress_manager import TaskProgressManager

# Conditional imports for encryption backends
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:  # pragma: no cover
    CRYPTOGRAPHY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug("Cryptography module not available")


    # Stub classes to satisfy static analyzers
    class _CryptoStub:
        # Minimal stub with no methods
        def __getattr__(self, item) -> Any:
            raise ImportError("Cryptography not installed")


    Fernet = _CryptoStub()  # type: ignore
    hashes = _CryptoStub()  # type: ignore
    PBKDF2HMAC = _CryptoStub()  # type: ignore

try:
    import pyage

    PYAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYAGE_AVAILABLE = False


    # Stub for pyage
    class _PyAgeStub:
        def __getattr__(self, item) -> Any:
            raise ImportError("PyAge not installed")


    pyage = _PyAgeStub()  # type: ignore

# Try to import key store at module level
try:
    from pamola_core.utils.crypto_helpers.key_store import (
        get_key_for_task,  # Function to get task key
        KeyStoreError  # Specific exception if available
    )

    KEY_STORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    KEY_STORE_AVAILABLE = False


    # Define stubs for IDE satisfaction
    def get_key_for_task(task_id: str):  # type: ignore
        raise ImportError("key-store unavailable")


    class KeyStoreError(Exception):
        pass

# Import path security validation
from pamola_core.common.enum.encryption_mode import EncryptionMode
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError

# Set up logger with a null handler (will be replaced by proper handler)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class EncryptionContext:
    """
    Secure encryption context that provides encryption capabilities
    without exposing the raw encryption key.

    This class encapsulates encryption functionality while protecting
    the actual key material, providing a safer interface for operations.
    """

    def __init__(self, mode: EncryptionMode, key_fingerprint: str):
        """
        Initialize encryption context.

        Args:
            mode: Encryption mode to use
            key_fingerprint: Fingerprint of the encryption key (not the key itself)
        """
        self.mode = mode
        self.key_fingerprint = key_fingerprint
        self._can_encrypt = mode != EncryptionMode.NONE

    @property
    def can_encrypt(self) -> bool:
        """Check if this context can perform encryption operations."""
        return self._can_encrypt

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary for serialization.

        Returns:
            Dict with context information (no sensitive data)
        """
        return {
            "mode": self.mode.value,
            "key_fingerprint": self.key_fingerprint,
            "can_encrypt": self._can_encrypt
        }


class EncryptionError(Exception):
    """Base exception for encryption-related errors."""
    pass


class EncryptionInitializationError(EncryptionError):
    """Exception raised when encryption initialization fails."""
    pass


class KeyGenerationError(EncryptionError):
    """Exception raised when key generation fails."""
    pass


class KeyLoadingError(EncryptionError):
    """Exception raised when key loading fails."""
    pass


class DataRedactionError(EncryptionError):
    """Exception raised when data redaction fails."""
    pass


class MemoryProtectedKey:
    """
    Memory-protected encryption key container.

    This class implements secure handling of encryption keys in memory,
    with minimal exposure and prevention of unintended key leakage.
    """

    def __init__(self, key_material: bytes, key_id: Optional[str] = None):
        """
        Initialize with key material.

        Args:
            key_material: Raw key bytes
            key_id: Optional identifier for the key
        """
        # Store key material securely
        self._key_material = key_material
        # Generate fingerprint immediately
        self._fingerprint = self._generate_fingerprint(key_material)
        # Set key ID or generate one
        self._key_id = key_id or f"key_{self._fingerprint[:8]}"
        # Mark if this key has been used
        self._used = False
        # Reference counter to track usage
        self._ref_count = 0

    def _generate_fingerprint(self, key_material: bytes) -> str:
        """
        Generate a fingerprint of the key for identification without exposing the key.

        Args:
            key_material: Key bytes

        Returns:
            String fingerprint
        """
        if CRYPTOGRAPHY_AVAILABLE:
            # Use cryptography for a proper fingerprint
            digest = hashes.Hash(hashes.SHA256())
            digest.update(key_material)
            return digest.finalize().hex()[:16]
        else:
            # Fallback to simple hash if cryptography not available
            # Not for cryptographic purposes, just for identification
            import hashlib
            return hashlib.sha256(key_material).hexdigest()[:16]

    def __enter__(self):
        """Context manager entry that provides the key material temporarily."""
        self._ref_count += 1
        self._used = True
        return self._key_material

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit that decrements reference counter.

        When reference count reaches zero, attempt secure cleanup.
        """
        self._ref_count -= 1
        if self._ref_count <= 0:
            try:
                # Attempt to securely clear key material
                self._secure_cleanup()
            except Exception as e:
                logger.debug(f"Key cleanup error (expected in some environments): {e}")

    def _secure_cleanup(self):
        """
        Attempt to securely clear key material from memory.

        This is a best-effort method as Python's garbage collection
        and memory management can make true secure erasure difficult.
        """
        if hasattr(self, '_key_material') and self._key_material:
            # Try to overwrite with zeros
            key_len = len(self._key_material)
            try:
                # For bytearrays, we can overwrite directly
                if isinstance(self._key_material, bytearray):
                    for i in range(key_len):
                        self._key_material[i] = 0
                # For bytes (immutable), create a new reference
                else:
                    self._key_material = bytes(key_len)
            except Exception:
                # Fallback - just remove the reference
                pass

    @property
    def fingerprint(self) -> str:
        """Get the key fingerprint (safe to expose)."""
        return self._fingerprint

    @property
    def key_id(self) -> str:
        """Get the key ID (safe to expose)."""
        return self._key_id

    @property
    def has_been_used(self) -> bool:
        """Check if this key has been used."""
        return self._used

    def __del__(self):
        """
        Destructor that attempts to securely clean up key material.
        """
        try:
            self._secure_cleanup()
        except Exception:
            # Ignore errors in __del__
            pass


class TaskEncryptionManager:
    """
    Encryption manager for secure handling of encryption keys and sensitive data.

    This class encapsulates all encryption-related functionality, providing a
    secure interface for tasks to use encryption without exposing raw keys to
    operations or logs.
    """

    # Class-level key cache to avoid repeated key store access
    _key_cache: Dict[str, bytes] = {}

    def __init__(
            self,
            task_config: Any,
            logger: Optional[logging.Logger] = None,
            progress_manager: Optional['TaskProgressManager'] = None
    ):
        """
        Initialize encryption manager with task configuration.

        Args:
            task_config: Task configuration object containing encryption settings
            logger: Logger for encryption operations (optional)
            progress_manager: Progress manager for tracking initialization (optional)
        """
        # Store references to config and logger
        self.config = task_config
        self.logger = logger or logging.getLogger(__name__)
        self.progress_manager = progress_manager

        # Initialize state variables
        self._protected_key = None
        self._encryption_mode = getattr(task_config, 'encryption_mode', EncryptionMode.NONE)
        self._use_encryption = getattr(task_config, 'use_encryption', False)
        self._encryption_key_path = getattr(task_config, 'encryption_key_path', None)
        self._initialized = False
        self._initialization_error = None

        # Define set of sensitive parameter names
        self._sensitive_param_names = {
            'encryption_key', 'key', 'password', 'secret', 'token', 'api_key',
            'access_key', 'private_key', 'master_key', 'passphrase', 'credentials'
        }

    def initialize(self) -> bool:
        """
        Initialize encryption based on configuration.

        Returns:
            True if initialization successful, False otherwise
        """
        # If already initialized, return cached result
        if self._initialized:
            return self._use_encryption and self._protected_key is not None

        try:
            # Create a progress tracking context if progress manager is available
            if self.progress_manager:
                progress_context = self.progress_manager.create_operation_context(
                    name="initialize_encryption",
                    total=5,  # steps: check config, load key, validate, check datasets, finalize
                    description=f"Initializing encryption ({self._encryption_mode.value})"
                )
            else:
                # Use a dummy context manager if progress manager isn't available
                from contextlib import nullcontext
                progress_context = nullcontext()

            with progress_context as progress:
                # Early exit if encryption is disabled
                if not self._use_encryption:
                    self.logger.debug("Encryption is disabled")
                    self._encryption_mode = EncryptionMode.NONE
                    self._initialized = True

                    # Update progress if available
                    if hasattr(progress, 'update'):
                        progress.update(5, {"status": "disabled"})

                    return False

                # Validate encryption mode
                if self._encryption_mode == EncryptionMode.NONE:
                    self.logger.warning(
                        f"Invalid encryption configuration: mode is 'none' but use_encryption is True, defaulting to 'simple'")
                    self._encryption_mode = EncryptionMode.SIMPLE

                self.logger.info(f"Initializing encryption with mode: {self._encryption_mode.value}")

                # Update progress if available
                if hasattr(progress, 'update'):
                    progress.update(1, {"status": "config_validated"})

                # Try to load or generate encryption key
                key_loaded = False

                # Try loading from file if path is provided
                if self._encryption_key_path:
                    try:
                        # Validate key path security
                        key_path = self._resolve_key_path()

                        if key_path.exists():
                            # Load key from file
                            with open(key_path, 'rb') as f:
                                key_material = f.read()

                            # Create protected key container
                            self._protected_key = MemoryProtectedKey(key_material)
                            self.logger.info(f"Loaded encryption key (fingerprint: {self._protected_key.fingerprint})")
                            key_loaded = True
                        else:
                            self.logger.warning(f"Encryption key file {key_path} not found")
                    except Exception as e:
                        self.logger.error(f"Error loading encryption key from file: {str(e)}")

                # Update progress if available
                if hasattr(progress, 'update'):
                    progress.update(1, {"status": "file_key_checked", "key_loaded": key_loaded})

                # If key not loaded from file, try key store
                if not key_loaded:
                    try:
                        # Try to get key from key store if available
                        key_material = self._get_key_from_store()
                        if key_material:
                            # Create protected key container
                            self._protected_key = MemoryProtectedKey(key_material)
                            self.logger.info(
                                f"Retrieved encryption key from key store (fingerprint: {self._protected_key.fingerprint})")
                            key_loaded = True
                        else:
                            self.logger.warning("No encryption key found in key store")
                    except KeyLoadingError as e:
                        self.logger.error(f"Error retrieving key from store: {str(e)}")

                # Update progress if available
                if hasattr(progress, 'update'):
                    progress.update(1, {"status": "store_key_checked", "key_loaded": key_loaded})

                # If still no key, try to generate one
                if not key_loaded and (self._encryption_mode == EncryptionMode.SIMPLE):
                    try:
                        # Generate new key for simple encryption
                        key_material = self._generate_encryption_key()
                        # Create protected key container
                        self._protected_key = MemoryProtectedKey(key_material)
                        self.logger.info(
                            f"Generated new encryption key (fingerprint: {self._protected_key.fingerprint})")
                        key_loaded = True

                        # Optionally save the key if path is provided
                        if self._encryption_key_path:
                            key_path = self._resolve_key_path()
                            if not key_path.exists():
                                # Create directory if it doesn't exist
                                key_path.parent.mkdir(parents=True, exist_ok=True)
                                # Save key to file
                                with open(key_path, 'wb') as f:
                                    f.write(key_material)
                                self.logger.info(f"Saved new encryption key to {key_path}")
                    except Exception as e:
                        self.logger.error(f"Error generating encryption key: {str(e)}")

                # Update progress if available
                if hasattr(progress, 'update'):
                    progress.update(1, {"status": "key_generation_attempted", "key_loaded": key_loaded})

                # If still no key and encryption is required, disable encryption
                if not key_loaded:
                    self.logger.warning("No encryption key available, disabling encryption")
                    self._use_encryption = False
                    self._encryption_mode = EncryptionMode.NONE
                    self._initialized = True

                    # Update progress if available
                    if hasattr(progress, 'update'):
                        progress.update(1, {"status": "disabled_due_to_no_key"})

                    return False

                # Log final encryption status
                if self._use_encryption:
                    self.logger.info(f"Encryption initialized successfully with mode: {self._encryption_mode.value}")
                else:
                    self.logger.info("Encryption is disabled")

                # Update progress if available
                if hasattr(progress, 'update'):
                    progress.update(1, {"status": "initialized", "mode": self._encryption_mode.value})

                # Mark as initialized
                self._initialized = True
                return self._use_encryption

        except Exception as e:
            # Handle initialization errors
            self._initialization_error = str(e)
            self.logger.exception(f"Encryption initialization failed: {e}")
            self._initialized = False
            return False

    def _resolve_key_path(self) -> Path:
        """
        Resolve the encryption key path safely.

        Returns:
            Path object for the encryption key

        Raises:
            PathSecurityError: If the path fails security validation
        """
        if isinstance(self._encryption_key_path, str):
            # Handle string path
            path_obj = Path(self._encryption_key_path)
            if not path_obj.is_absolute() and hasattr(self.config, 'resolve_legacy_path'):
                # Use config's path resolution if available
                path_obj = self.config.resolve_legacy_path(path_obj)
        elif isinstance(self._encryption_key_path, Path):
            # Already a Path object
            path_obj = self._encryption_key_path
        else:
            # Invalid path type
            raise ValueError(f"Invalid encryption key path type: {type(self._encryption_key_path)}")

        # Validate path security
        allowed_paths = getattr(self.config, 'allowed_external_paths', [])
        allow_external = getattr(self.config, 'allow_external', False)

        if not validate_path_security(
                path_obj,
                allowed_paths=allowed_paths,
                allow_external=allow_external
        ):
            raise PathSecurityError(f"Insecure encryption key path: {path_obj}")

        return path_obj

    def _get_key_from_store(self) -> Optional[bytes]:
        """
        Get encryption key from key store.

        Returns:
            Key bytes if available, None otherwise

        Raises:
            KeyLoadingError: If key store is available but returned an error
        """
        if not KEY_STORE_AVAILABLE:
            self.logger.debug("Key store module not available")
            return None

        task_id: str = getattr(self.config, "task_id", "unknown")

        # Check cache to avoid repeated backend calls
        if task_id in self._key_cache:
            return self._key_cache[task_id]

        try:
            key_obj = get_key_for_task(task_id)  # str | bytes | None
        except KeyStoreError as e:
            # Log without sensitive data
            self.logger.warning(
                "Key store error for task %s: %s", task_id, type(e).__name__
            )
            raise KeyLoadingError(str(e)) from e

        if key_obj is None:
            return None

        # Unify type: always return bytes
        if isinstance(key_obj, bytes):
            key_bytes = key_obj
        elif isinstance(key_obj, str):
            # Try base64-decode, otherwise utf-8
            try:
                key_bytes = base64.urlsafe_b64decode(key_obj)
            except Exception:
                key_bytes = key_obj.encode("utf-8")
        else:
            raise KeyLoadingError(f"Unsupported key type {type(key_obj)}")

        # Add to cache
        self._key_cache[task_id] = key_bytes
        return key_bytes

    def _generate_encryption_key(self) -> bytes:
        """
        Generate a new encryption key.

        Returns:
            New key bytes

        Raises:
            KeyGenerationError: If key generation fails
        """
        try:
            if self._encryption_mode == EncryptionMode.SIMPLE:
                if CRYPTOGRAPHY_AVAILABLE:
                    # Generate Fernet key
                    return Fernet.generate_key()
                else:
                    # Fallback to secrets for cryptographically secure random bytes
                    return base64.urlsafe_b64encode(secrets.token_bytes(32))
            elif self._encryption_mode == EncryptionMode.AGE:
                if PYAGE_AVAILABLE:
                    # Generate Age key
                    return pyage.generate_x25519_key_pair().private_key
                else:
                    raise KeyGenerationError("Cannot generate Age key - pyage module not available")
            else:
                raise KeyGenerationError(f"Cannot generate key for mode: {self._encryption_mode}")
        except Exception as e:
            raise KeyGenerationError(f"Failed to generate encryption key: {str(e)}") from e

    def get_encryption_context(self) -> EncryptionContext:
        """
        Get secure encryption context for operations.

        This method provides a safe way to pass encryption capabilities
        to operations without exposing the raw key.

        Returns:
            EncryptionContext with necessary info for operations
        """
        # Ensure encryption is initialized
        if not self._initialized:
            self.initialize()

        # Create context with fingerprint (not the raw key)
        if self._use_encryption and self._protected_key:
            return EncryptionContext(
                mode=self._encryption_mode,
                key_fingerprint=self._protected_key.fingerprint
            )
        else:
            return EncryptionContext(
                mode=EncryptionMode.NONE,
                key_fingerprint="none"
            )

    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt binary data using the configured encryption method.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data

        Raises:
            EncryptionError: If encryption fails
        """
        # Ensure encryption is initialized
        if not self._initialized:
            self.initialize()

        # Check if encryption is enabled
        if not self._use_encryption or not self._protected_key:
            raise EncryptionError("Encryption is not enabled or key is not available")

        try:
            # Use the protected key safely via context manager
            with self._protected_key as key_material:
                if self._encryption_mode == EncryptionMode.SIMPLE:
                    if CRYPTOGRAPHY_AVAILABLE:
                        # Use Fernet for encryption
                        f = Fernet(key_material)
                        return f.encrypt(data)
                    else:
                        raise EncryptionError("Cryptography module not available")
                elif self._encryption_mode == EncryptionMode.AGE:
                    if PYAGE_AVAILABLE:
                        # Use Age for encryption
                        return pyage.encrypt(data, pyage.x25519_recipient(key_material))
                    else:
                        raise EncryptionError("PyAge module not available")
                else:
                    raise EncryptionError(f"Unsupported encryption mode: {self._encryption_mode}")
        except Exception as e:
            if isinstance(e, EncryptionError):
                raise
            else:
                raise EncryptionError(f"Encryption failed: {str(e)}") from e

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt binary data using the configured encryption method.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data

        Raises:
            EncryptionError: If decryption fails
        """
        # Ensure encryption is initialized
        if not self._initialized:
            self.initialize()

        # Check if encryption is enabled
        if not self._use_encryption or not self._protected_key:
            raise EncryptionError("Encryption is not enabled or key is not available")

        try:
            # Use the protected key safely via context manager
            with self._protected_key as key_material:
                if self._encryption_mode == EncryptionMode.SIMPLE:
                    if CRYPTOGRAPHY_AVAILABLE:
                        # Use Fernet for decryption
                        f = Fernet(key_material)
                        return f.decrypt(encrypted_data)
                    else:
                        raise EncryptionError("Cryptography module not available")
                elif self._encryption_mode == EncryptionMode.AGE:
                    if PYAGE_AVAILABLE:
                        # Use Age for decryption
                        return pyage.decrypt(encrypted_data, key_material)
                    else:
                        raise EncryptionError("PyAge module not available")
                else:
                    raise EncryptionError(f"Unsupported encryption mode: {self._encryption_mode}")
        except Exception as e:
            if isinstance(e, EncryptionError):
                raise
            else:
                raise EncryptionError(f"Decryption failed: {str(e)}") from e

    def add_sensitive_param_names(self, param_names: Union[str, List[str]]) -> None:
        """
        Add parameter names that should be treated as sensitive.

        Args:
            param_names: Single name or list of parameter names
        """
        if isinstance(param_names, str):
            self._sensitive_param_names.add(param_names.lower())
        else:
            for name in param_names:
                self._sensitive_param_names.add(name.lower())

    def is_sensitive_param(self, param_name: str) -> bool:
        """
        Check if a parameter name should be treated as sensitive.

        Args:
            param_name: Parameter name to check

        Returns:
            True if parameter is sensitive, False otherwise
        """
        # Check in sensitive parameter set (case-insensitive)
        return param_name.lower() in self._sensitive_param_names

    def redact_sensitive_data(self, data: Any, redact_keys: bool = True) -> Any:
        """
        Redact sensitive information from data structures.

        This method recursively processes dictionaries, lists, and other data
        structures to redact sensitive values based on key names.

        Args:
            data: Data structure to redact
            redact_keys: Whether to redact dictionary keys (default: True)

        Returns:
            Redacted copy of the data structure

        Raises:
            DataRedactionError: If redaction fails
        """
        try:
            # Handle different types recursively
            if isinstance(data, dict):
                # Process dictionary
                result = {}
                for key, value in data.items():
                    # Check if key is sensitive
                    if redact_keys and isinstance(key, str) and self.is_sensitive_param(key):
                        # Use redacted key and value
                        redacted_key = f"<redacted:{key[:3]}...>"
                        result[redacted_key] = "<redacted>"
                    else:
                        # Keep key but maybe redact value
                        if isinstance(key, str) and self.is_sensitive_param(key):
                            # Redact sensitive value
                            result[key] = "<redacted>"
                        else:
                            # Recursively process value
                            result[key] = self.redact_sensitive_data(value, redact_keys)
                return result
            elif isinstance(data, list):
                # Process list elements recursively
                return [self.redact_sensitive_data(item, redact_keys) for item in data]
            elif isinstance(data, tuple):
                # Process tuple elements recursively
                return tuple(self.redact_sensitive_data(item, redact_keys) for item in data)
            elif isinstance(data, set):
                # Process set elements recursively
                return {self.redact_sensitive_data(item, redact_keys) for item in data}
            elif isinstance(data, (str, bytes)) and len(data) > 32:
                # Potential sensitive string/bytes - check for patterns
                data_str = data.decode('utf-8', errors='ignore') if isinstance(data, bytes) else data
                if self._looks_like_key(data_str):
                    return "<redacted:key-like>"
                else:
                    return data
            else:
                # Other types returned as-is
                return data
        except Exception as e:
            raise DataRedactionError(f"Failed to redact sensitive data: {str(e)}") from e

    def _looks_like_key(self, text: str) -> bool:
        """
        Check if a string looks like a key or sensitive data.

        Args:
            text: String to check

        Returns:
            True if string looks like a key, False otherwise
        """
        # Simple heuristics to identify potential keys
        # Base64 pattern - lots of alphanumeric with = padding
        if len(text) >= 32 and text.isalnum() and text.endswith('=='):
            return True

        # Check for common key prefixes
        key_prefixes = ['key-', 'sk-', 'xkey', 'pk-', 'token-']
        for prefix in key_prefixes:
            if text.startswith(prefix) and len(text) > 20:
                return True

        # Check for hex strings (common in fingerprints and keys)
        if all(c in '0123456789abcdefABCDEF' for c in text) and len(text) >= 32:
            return True

        return False

    def redact_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from a configuration dictionary.

        This is a convenience method specifically for configuration dictionaries.

        Args:
            config_dict: Configuration dictionary to redact

        Returns:
            Redacted copy of the configuration
        """
        return self.redact_sensitive_data(config_dict, redact_keys=False)

    def get_encryption_info(self) -> Dict[str, Any]:
        """
        Get information about the encryption configuration.

        Returns:
            Dictionary with encryption information (no sensitive data)
        """
        # Ensure encryption is initialized
        if not self._initialized:
            self.initialize()

        return {
            "enabled": self._use_encryption,
            "mode": self._encryption_mode.value,
            "key_available": self._protected_key is not None,
            "key_fingerprint": self._protected_key.fingerprint if self._protected_key else None,
            "initialization_error": self._initialization_error
        }

    def check_dataset_encryption(self, data_source: Any) -> bool:
        """
        Check if datasets in the data source are encrypted.

        Args:
            data_source: Data source containing file paths

        Returns:
            True if all datasets appear to be properly encrypted (when encryption is enabled)
        """
        # If encryption is disabled, no need to check
        if not self._use_encryption:
            return True

        # Create a progress tracking context if progress manager is available
        if self.progress_manager:
            # Get number of datasets to check
            datasets_count = len(data_source.get_file_paths()) if hasattr(data_source, 'get_file_paths') else 0

            progress_context = self.progress_manager.create_operation_context(
                name="check_encryption",
                total=datasets_count,
                description="Checking dataset encryption"
            )
        else:
            # Use a dummy context manager if progress manager isn't available
            from contextlib import nullcontext
            progress_context = nullcontext()

        with progress_context as progress:
            # Get all file paths from data source
            if hasattr(data_source, 'get_file_paths'):
                file_paths = data_source.get_file_paths()

                for i, (name, path) in enumerate(file_paths.items()):
                    try:
                        # Check if file exists
                        if not path.exists():
                            self.logger.warning(f"Dataset {name} does not exist: {path}")
                            continue

                        # Check if file is encrypted
                        encrypted = self.is_file_encrypted(path)

                        if not encrypted:
                            self.logger.warning(f"Dataset {name} appears to be unencrypted: {path}")

                        # Update progress if available
                        if hasattr(progress, 'update'):
                            progress.update(1, {"dataset": name, "encrypted": encrypted})

                    except Exception as e:
                        self.logger.error(f"Error checking encryption for dataset {name}: {str(e)}")
                        # Update progress if available
                        if hasattr(progress, 'update'):
                            progress.update(1, {"dataset": name, "error": str(e)})

        return True

    def is_file_encrypted(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file appears to be encrypted.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file appears to be encrypted, False otherwise
        """
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path

        try:
            # Validate path security
            if not validate_path_security(path_obj):
                raise PathSecurityError(f"Insecure file path: {path_obj}")

            # Check if file exists
            if not path_obj.exists():
                return False

            # Read the first few bytes of the file
            with open(path_obj, 'rb') as f:
                header = f.read(64)  # Read first 64 bytes

            # Simple heuristic check for encrypted content
            if self._encryption_mode == EncryptionMode.SIMPLE:
                # Fernet-encrypted files start with 'gAAAAA'
                return header.startswith(b'gAAAAA')
            elif self._encryption_mode == EncryptionMode.AGE:
                # Age-encrypted files start with "age-encryption.org/"
                return b"age-encryption.org/" in header
            else:
                # Cannot determine for unknown encryption mode
                return False

        except Exception as e:
            self.logger.error(f"Error checking if file is encrypted: {str(e)}")
            return False

    def supports_encryption_mode(self, mode: Union[str, EncryptionMode]) -> bool:
        """
        Check if the requested encryption mode is supported.

        Args:
            mode: Encryption mode to check

        Returns:
            True if mode is supported, False otherwise
        """
        # Convert string to enum if needed
        if isinstance(mode, str):
            try:
                mode = EncryptionMode.from_string(mode)
            except ValueError:
                return False

        # Check if mode is supported based on available libraries
        if mode == EncryptionMode.NONE:
            return True
        elif mode == EncryptionMode.SIMPLE:
            return CRYPTOGRAPHY_AVAILABLE
        elif mode == EncryptionMode.AGE:
            return PYAGE_AVAILABLE
        else:
            return False

    def cleanup(self) -> None:
        """
        Explicitly clean up resources.

        This should be called when the manager is no longer needed.
        """
        # Clear the protected key to trigger its cleanup
        self._protected_key = None

        if self.progress_manager:
            if hasattr(self.progress_manager, 'log_info'):
                self.progress_manager.log_info("Encryption manager resources released")

    def __del__(self):
        """
        Destructor to ensure protected key is cleaned up.
        """
        # Clear the protected key to trigger its cleanup
        self._protected_key = None