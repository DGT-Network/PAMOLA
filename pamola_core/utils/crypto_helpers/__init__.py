"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Package:       Cryptographic Helpers
Module:        pamola_core.utils.crypto_helpers
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2024-01-15
Modified:      2025-01-20
License:       BSD 3-Clause

Description:
    This package provides comprehensive cryptographic utilities for the PAMOLA.CORE
    framework, including encryption/decryption operations, key management,
    pseudonymization utilities, and audit logging for security operations.

Key Components:
    - Encryption Providers: Multiple encryption backends (Simple, Age, None)
    - Key Management: Secure storage and retrieval of encryption keys
    - Pseudonymization: Cryptographic utilities for data pseudonymization
    - Audit Logging: Security audit trail for all crypto operations
    - Legacy Migration: Support for migrating from older encryption formats

Security Features:
    - Industry-standard encryption algorithms (AES-256-GCM, ChaCha20-Poly1305)
    - Secure key derivation and storage
    - Memory-secure operations with automatic cleanup
    - Comprehensive audit logging
    - Thread-safe implementations

Usage:
    The crypto_helpers package is primarily used through the crypto_router
    in io_helpers, but direct access to specific components is available
    for advanced use cases.

    Example:
        # For encryption/decryption, use crypto_router
        from pamola_core.utils.io_helpers import crypto_router

        # For pseudonymization
        from pamola_core.utils.crypto_helpers.pseudonymization import (
            HashGenerator, SaltManager, PseudonymGenerator
        )

        # For key management
        from pamola_core.utils.crypto_helpers.key_store import EncryptedKeyStore
"""

import logging
from typing import List, Type, Any

# Configure package logger
logger = logging.getLogger(__name__)

# Version information
__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Import errors first (no dependencies)
from .errors import (
    CryptoError,
    EncryptionError,
    DecryptionError,
    KeyError,
    KeyStoreError,
    FormatError,
    ProviderError,
    ConfigurationError,
    ModeError,
    LegacyMigrationError,
    AgeToolError,
    MasterKeyError,
    TaskKeyError
)

# Import audit logging
from .audit import (
    setup_audit_logging,
    log_crypto_operation,
    log_key_access
)

# Import key store
from .key_store import (
    EncryptedKeyStore,
    get_key_for_task
)

# Import legacy migration
from .legacy_migration import (
    detect_legacy_format,
    migrate_legacy_file,
    auto_migrate_if_needed
)

# Import pseudonymization utilities
from pamola_core.utils.crypto_helpers.pseudonymization import (
    # Main classes
    HashGenerator,
    SaltManager,
    PepperGenerator,
    MappingEncryption,
    PseudonymGenerator,
    CollisionTracker,
    SecureBytes,

    # Utility functions
    constant_time_compare,
    validate_key_size,
    derive_key_from_password,
    generate_deterministic_pseudonym,

    # Exceptions
    PseudonymizationError,
    HashCollisionError,

    # Constants
    CRYPTOGRAPHY_AVAILABLE,
    BASE58_AVAILABLE
)

# Import providers package
from . import providers

# Lazy initialization flag
_initialized = False


def initialize_crypto_helpers():
    """
    Initialize the crypto_helpers package.

    This function performs one-time initialization tasks:
    - Sets up audit logging
    - Registers all available crypto providers
    - Validates system dependencies

    This is called automatically when needed, but can be called
    explicitly for early initialization.
    """
    global _initialized

    if _initialized:
        return

    try:
        # Set up audit logging
        setup_audit_logging()

        # Check for optional dependencies
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning(
                "cryptography package not available. "
                "Some encryption features will be limited."
            )

        if not BASE58_AVAILABLE:
            logger.debug(
                "base58 package not available. "
                "Base58 encoding will not be supported."
            )

        _initialized = True
        logger.info("Crypto helpers initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize crypto helpers: {e}")
        raise


# Auto-initialize on import if not disabled
import os

if os.environ.get("PAMOLA_DISABLE_AUTO_INIT", "0") != "1":
    try:
        initialize_crypto_helpers()
    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}")

# Define public API
__all__ = [
    # Initialization
    'initialize_crypto_helpers',

    # Error classes
    'CryptoError',
    'EncryptionError',
    'DecryptionError',
    'KeyError',
    'KeyStoreError',
    'FormatError',
    'ProviderError',
    'ConfigurationError',
    'ModeError',
    'LegacyMigrationError',
    'AgeToolError',
    'MasterKeyError',
    'TaskKeyError',

    # Audit logging
    'setup_audit_logging',
    'log_crypto_operation',
    'log_key_access',

    # Key management
    'EncryptedKeyStore',
    'get_key_for_task',

    # Legacy migration
    'detect_legacy_format',
    'migrate_legacy_file',
    'auto_migrate_if_needed',

    # Pseudonymization classes
    'HashGenerator',
    'SaltManager',
    'PepperGenerator',
    'MappingEncryption',
    'PseudonymGenerator',
    'CollisionTracker',
    'SecureBytes',

    # Pseudonymization utilities
    'constant_time_compare',
    'validate_key_size',
    'derive_key_from_password',
    'generate_deterministic_pseudonym',

    # Pseudonymization exceptions
    'PseudonymizationError',
    'HashCollisionError',

    # Constants
    'CRYPTOGRAPHY_AVAILABLE',
    'BASE58_AVAILABLE',

    # Provider registration
    'register_all_providers',

    # Sub-packages
    'providers'
]


# Convenience function for checking crypto support
def check_crypto_support() -> dict:
    """
    Check the status of cryptographic support.

    Returns:
        Dict with support status for various features:
        - cryptography: Whether cryptography package is available
        - base58: Whether base58 package is available
        - providers: List of registered crypto providers
        - age_tool: Whether age CLI tool is available
    """
    # Import here to avoid circular dependency
    from pamola_core.utils.io_helpers.crypto_router import get_all_providers

    # Check for age tool
    age_available = False
    try:
        from .providers.age_provider import AgeProvider
        age_available = AgeProvider.is_available()
    except Exception:
        pass

    return {
        'cryptography': CRYPTOGRAPHY_AVAILABLE,
        'base58': BASE58_AVAILABLE,
        'providers': list(get_all_providers().keys()),
        'age_tool': age_available,
        'audit_logging': _initialized
    }


# Add to exports
__all__.append('check_crypto_support')