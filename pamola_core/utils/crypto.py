"""
Data Encryption and Decryption Utilities
----------------------------------------------
This module provides functionality for encrypting and decrypting data
using strong cryptographic algorithms.

Features:
- Encryption/decryption using AES-256-GCM or ChaCha20-Poly1305
- Key management utilities
- Support for file and in-memory data
- Stream encryption for large files
- Integrity verification

The module is designed to be used by other components that need to
store or transmit sensitive data securely.

(C) 2025 BDA

Author: Security Team
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Union, Optional, Dict, Tuple

from cryptography.hazmat.primitives import hashes
# Direct imports from cryptography library
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logger
logger = logging.getLogger("hhr.utils.crypto")

# Default settings
DEFAULT_ALGORITHM = "AES-GCM"  # Alternatives: "ChaCha20-Poly1305"
DEFAULT_KEY_LENGTH = 32  # 256 bits
DEFAULT_IV_LENGTH = 12  # 96 bits (recommended for GCM)
DEFAULT_ITERATIONS = 100000  # for key derivation


class CryptoError(Exception):
    """Base class for all crypto-related errors"""
    pass


class EncryptionError(CryptoError):
    """Error during encryption"""
    pass


class DecryptionError(CryptoError):
    """Error during decryption"""
    pass


class KeyError(CryptoError):
    """Error related to cryptographic keys"""
    pass


def derive_key(password: str, salt: Optional[bytes] = None,
               iterations: int = DEFAULT_ITERATIONS,
               key_length: int = DEFAULT_KEY_LENGTH) -> Tuple[bytes, bytes]:
    """
    Derive a cryptographic key from a password.

    Parameters:
    -----------
    password : str
        Password to derive key from
    salt : bytes, optional
        Salt for key derivation. If not provided, a random one is generated.
    iterations : int
        Number of iterations for PBKDF2 (default: 100000)
    key_length : int
        Length of the key in bytes (default: 32, which is 256 bits)

    Returns:
    --------
    Tuple[bytes, bytes]
        (key, salt) tuple

    Raises:
    -------
    KeyError
        If there's an error deriving the key
    """
    try:
        # Generate a random salt if not provided
        if salt is None:
            salt = os.urandom(16)

        # Convert password to bytes if it's a string
        if isinstance(password, str):
            password = password.encode('utf-8')

        # Derive the key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
        )

        key = kdf.derive(password)
        return key, salt

    except Exception as e:
        logger.error(f"Error deriving key: {e}")
        raise KeyError(f"Error deriving key: {e}")


def encrypt_data(data: Union[str, bytes],
                 key: Union[str, bytes],
                 algorithm: str = DEFAULT_ALGORITHM) -> Dict[str, str]:
    """
    Encrypt data using the specified algorithm.

    Parameters:
    -----------
    data : str or bytes
        Data to encrypt
    key : str or bytes
        Encryption key or password
    algorithm : str
        Encryption algorithm (default: "AES-GCM")

    Returns:
    --------
    Dict[str, str]
        Dictionary with encrypted data and metadata

    Raises:
    -------
    EncryptionError
        If there's an error during encryption
    """
    try:
        # Convert input data to bytes if it's a string
        if isinstance(data, str):
            data = data.encode('utf-8')

        # Prepare the key
        if isinstance(key, str):
            # If key is a string, treat it as a password and derive a key
            actual_key, salt = derive_key(key)
            salt_b64 = base64.b64encode(salt).decode('utf-8')
        else:
            # If key is bytes, use it directly
            actual_key = key
            salt_b64 = None

        # Generate a random IV/nonce
        iv = os.urandom(DEFAULT_IV_LENGTH)

        # Choose the encryption algorithm
        if algorithm == "AES-GCM":
            cipher = AESGCM(actual_key)
            ciphertext = cipher.encrypt(iv, data, None)
        elif algorithm == "ChaCha20-Poly1305":
            cipher = ChaCha20Poly1305(actual_key)
            ciphertext = cipher.encrypt(iv, data, None)
        else:
            raise EncryptionError(f"Unsupported algorithm: {algorithm}")

        # Encode as base64 for storage/transmission
        iv_b64 = base64.b64encode(iv).decode('utf-8')
        ciphertext_b64 = base64.b64encode(ciphertext).decode('utf-8')

        # Prepare the result dictionary
        result = {
            "algorithm": algorithm,
            "iv": iv_b64,
            "data": ciphertext_b64,
        }

        # Include salt if key was derived from a password
        if salt_b64:
            result["salt"] = salt_b64
            result["key_derivation"] = "PBKDF2-SHA256"

        return result

    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        raise EncryptionError(f"Error encrypting data: {e}")


def decrypt_data(encrypted_data: Dict[str, str],
                 key: Union[str, bytes]) -> bytes:
    """
    Decrypt data using the metadata in the encrypted data dictionary.

    Parameters:
    -----------
    encrypted_data : Dict[str, str]
        Dictionary with encrypted data and metadata
    key : str or bytes
        Decryption key or password

    Returns:
    --------
    bytes
        Decrypted data

    Raises:
    -------
    DecryptionError
        If there's an error during decryption
    """
    try:
        # Get algorithm and required fields
        algorithm = encrypted_data.get("algorithm", DEFAULT_ALGORITHM)
        iv_b64 = encrypted_data.get("iv")
        ciphertext_b64 = encrypted_data.get("data")

        if not iv_b64 or not ciphertext_b64:
            raise DecryptionError("Missing required fields in encrypted data")

        # Decode from base64
        iv = base64.b64decode(iv_b64)
        ciphertext = base64.b64decode(ciphertext_b64)

        # Prepare the key
        if isinstance(key, str):
            # If key derivation info is provided, derive the key
            if "salt" in encrypted_data and "key_derivation" in encrypted_data:
                salt = base64.b64decode(encrypted_data["salt"])
                actual_key, _ = derive_key(key, salt)
            else:
                # Treat the string as a raw key
                actual_key = key.encode('utf-8')
        else:
            # If key is bytes, use it directly
            actual_key = key

        # Choose the decryption algorithm
        if algorithm == "AES-GCM":
            cipher = AESGCM(actual_key)
            plaintext = cipher.decrypt(iv, ciphertext, None)
        elif algorithm == "ChaCha20-Poly1305":
            cipher = ChaCha20Poly1305(actual_key)
            plaintext = cipher.decrypt(iv, ciphertext, None)
        else:
            raise DecryptionError(f"Unsupported algorithm: {algorithm}")

        return plaintext

    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        raise DecryptionError(f"Error decrypting data: {e}")


def encrypt_file(input_path: Union[str, Path],
                 output_path: Union[str, Path],
                 key: Union[str, bytes],
                 algorithm: str = DEFAULT_ALGORITHM,
                 chunk_size: int = 1024 * 1024) -> None:
    """
    Encrypt a file using the specified algorithm.

    Parameters:
    -----------
    input_path : str or Path
        Path to the file to encrypt
    output_path : str or Path
        Path to save the encrypted file
    key : str or bytes
        Encryption key or password
    algorithm : str
        Encryption algorithm (default: "AES-GCM")
    chunk_size : int
        Size of chunks for large files (default: 1MB)

    Raises:
    -------
    EncryptionError
        If there's an error during encryption
    """
    # This is a placeholder for a full implementation
    # A real implementation would need to handle large files and streaming
    logger.warning("Full file encryption not implemented yet - using basic implementation")

    try:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Read the entire file
        with open(input_path, 'rb') as f:
            data = f.read()

        # Encrypt the data
        encrypted = encrypt_data(data, key, algorithm)

        # Write the encrypted data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted, f) # type: ignore

        logger.info(f"Encrypted {input_path} to {output_path}")

    except Exception as e:
        logger.error(f"Error encrypting file: {e}")
        raise EncryptionError(f"Error encrypting file: {e}")


def decrypt_file(input_path: Union[str, Path],
                 output_path: Union[str, Path],
                 key: Union[str, bytes]) -> None:
    """
    Decrypt a file.

    Parameters:
    -----------
    input_path : str or Path
        Path to the encrypted file
    output_path : str or Path
        Path to save the decrypted file
    key : str or bytes
        Decryption key or password

    Raises:
    -------
    DecryptionError
        If there's an error during decryption
    """
    # This is a placeholder for a full implementation
    # A real implementation would need to handle large files and streaming
    logger.warning("Full file decryption not implemented yet - using basic implementation")

    try:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Read the encrypted data
        with open(input_path, 'r', encoding='utf-8') as f:
            encrypted = json.load(f)

        # Decrypt the data
        decrypted = decrypt_data(encrypted, key)

        # Write the decrypted data
        with open(output_path, 'wb') as f:
            f.write(decrypted)

        logger.info(f"Decrypted {input_path} to {output_path}")

    except Exception as e:
        logger.error(f"Error decrypting file: {e}")
        raise DecryptionError(f"Error decrypting file: {e}")


def generate_key() -> bytes:
    """
    Generate a random encryption key.

    Returns:
    --------
    bytes
        Random key

    Raises:
    -------
    KeyError
        If there's an error generating the key
    """
    try:
        return os.urandom(DEFAULT_KEY_LENGTH)
    except Exception as e:
        logger.error(f"Error generating key: {e}")
        raise KeyError(f"Error generating key: {e}")


def save_key(key: bytes, file_path: Union[str, Path], password: Optional[str] = None) -> None:
    """
    Save an encryption key to a file, optionally encrypted with a password.

    Parameters:
    -----------
    key : bytes
        Encryption key to save
    file_path : str or Path
        Path to save the key
    password : str, optional
        Password to encrypt the key

    Raises:
    -------
    KeyError
        If there's an error saving the key
    """
    try:
        file_path = Path(file_path)

        # Encrypt the key if a password is provided
        if password:
            encrypted = encrypt_data(key, password)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted, f) # type: ignore
        else:
            # Save the raw key (not recommended for production)
            with open(file_path, 'wb') as f:
                f.write(key)

        logger.info(f"Saved key to {file_path}")

    except Exception as e:
        logger.error(f"Error saving key: {e}")
        raise KeyError(f"Error saving key: {e}")


def load_key(file_path: Union[str, Path], password: Optional[str] = None) -> bytes:
    """
    Load an encryption key from a file, optionally decrypting with a password.

    Parameters:
    -----------
    file_path : str or Path
        Path to the key file
    password : str, optional
        Password to decrypt the key

    Returns:
    --------
    bytes
        Encryption key

    Raises:
    -------
    KeyError
        If there's an error loading the key
    """
    try:
        file_path = Path(file_path)

        if password:
            # Try to load as an encrypted key
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    encrypted = json.load(f)
                key = decrypt_data(encrypted, password)
                return key
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not a JSON file, try as a raw key
                pass

        # Load as a raw key
        with open(file_path, 'rb') as f:
            key = f.read()

        logger.info(f"Loaded key from {file_path}")
        return key

    except Exception as e:
        logger.error(f"Error loading key: {e}")
        raise KeyError(f"Error loading key: {e}")