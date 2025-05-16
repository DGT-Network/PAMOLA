"""
Exception hierarchy for the PAMOLA cryptographic subsystem.

This module defines all exceptions used throughout the cryptographic
components, providing a consistent error handling approach.
"""

class CryptoError(Exception):
    """Base class for all cryptographic subsystem exceptions."""
    pass


class EncryptionError(CryptoError):
    """Error during encryption operations."""
    pass


class DecryptionError(CryptoError):
    """Error during decryption operations."""
    pass


class KeyError(CryptoError):
    """Error related to cryptographic keys."""
    pass


class KeyStoreError(CryptoError):
    """Error related to key storage or retrieval."""
    pass


class FormatError(CryptoError):
    """Error related to encrypted data format."""
    pass


class ProviderError(CryptoError):
    """Error related to crypto provider operations."""
    pass


class ConfigurationError(CryptoError):
    """Error related to cryptographic configuration."""
    pass


class ModeError(CryptoError):
    """Error related to crypto mode selection or detection."""
    pass


class LegacyMigrationError(CryptoError):
    """Error during migration of legacy encrypted data."""
    pass


class AgeToolError(ProviderError):
    """Error when using the age CLI tool."""
    pass


class MasterKeyError(KeyStoreError):
    """Error related to master key operations."""
    pass


class TaskKeyError(KeyStoreError):
    """Error related to task-specific key operations."""
    pass