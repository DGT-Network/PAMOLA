"""
Utilities for encryption and decryption of file content.

This module provides helper functions for encrypting and decrypting
data when reading from or writing to files, isolating the crypto-specific
code from the main I/O module.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional

from pamola_core.utils import logging
from pamola_core.utils.crypto import encrypt_data, decrypt_data, EncryptionError, DecryptionError

# Configure module logger
logger = logging.get_logger("hhr.utils.io_helpers.crypto_utils")


def encrypt_file_content(data: Union[str, bytes],
                         encryption_key: str) -> Union[str, bytes, Dict]:
    """
    Encrypt data with proper error handling.

    Parameters:
    -----------
    data : str or bytes
        Data to encrypt
    encryption_key : str
        Encryption key

    Returns:
    --------
    Union[str, bytes, Dict]
        Encrypted data

    Raises:
    -------
    EncryptionError
        If encryption fails
    """
    if not encryption_key:
        return data

    try:
        if isinstance(data, bytes):
            return encrypt_data(data, encryption_key)
        elif isinstance(data, str):
            return encrypt_data(data, encryption_key)
        else:
            # Convert to string if not already string or bytes
            return encrypt_data(str(data), encryption_key)
    except EncryptionError as e:
        logger.error(f"Encryption failed: {e}")
        raise


def decrypt_file_content(data: Union[str, bytes, Dict],
                         encryption_key: str) -> Union[str, bytes]:
    """
    Decrypt data with proper error handling.

    Parameters:
    -----------
    data : str, bytes, or Dict
        Data to decrypt
    encryption_key : str
        Decryption key

    Returns:
    --------
    Union[str, bytes]
        Decrypted data

    Raises:
    -------
    DecryptionError
        If decryption fails
    """
    if not encryption_key:
        return data

    try:
        # Try to decrypt the data, handling different types appropriately in decrypt_data
        return decrypt_data(data, encryption_key)
    except (TypeError, ValueError, AttributeError) as e:
        # If we get here, the data format probably isn't what decrypt_data expects
        logger.warning(f"Unknown data format for decryption: {type(data)}, error: {e}")
        return data
    except DecryptionError as e:
        logger.error(f"Decryption failed: {e}")
        raise


def decrypt_file(file_path: Union[str, Path],
                 encryption_key: str) -> Union[str, bytes]:
    """
    Read and decrypt a file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the encrypted file
    encryption_key : str
        Decryption key

    Returns:
    --------
    Union[str, bytes]
        Decrypted content

    Raises:
    -------
    DecryptionError
        If decryption fails
    FileNotFoundError
        If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading and decrypting file: {file_path}")

    try:
        # Try reading as JSON first (for metadata-based encryption)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data and "algorithm" in data:
                # This looks like our encryption format
                return decrypt_file_content(data, encryption_key)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, try as binary
            pass

        # Read as binary if not a valid JSON
        with open(file_path, 'rb') as f:
            binary_data = f.read()

        return decrypt_file_content(binary_data, encryption_key)

    except Exception as e:
        logger.error(f"Error decrypting file {file_path}: {e}")
        raise


def encrypt_file(source_path: Union[str, Path],
                 destination_path: Union[str, Path],
                 encryption_key: str) -> Path:
    """
    Encrypt a file and save it to a new location.

    Parameters:
    -----------
    source_path : str or Path
        Path to the file to encrypt
    destination_path : str or Path
        Path where to save the encrypted file
    encryption_key : str
        Encryption key

    Returns:
    --------
    Path
        Path to the encrypted file

    Raises:
    -------
    EncryptionError
        If encryption fails
    FileNotFoundError
        If source file does not exist
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    logger.info(f"Encrypting file from {source_path} to {destination_path}")

    try:
        # Read source file
        with open(source_path, 'rb') as f:
            data = f.read()

        # Encrypt data
        encrypted_data = encrypt_file_content(data, encryption_key)

        # Save encrypted data
        if isinstance(encrypted_data, dict):
            # Save as JSON if it's a metadata structure
            with open(destination_path, 'w') as f:
                json.dump(encrypted_data, f) # type: ignore
        else:
            # Save as binary otherwise
            with open(destination_path, 'wb') as f:
                if isinstance(encrypted_data, str):
                    f.write(encrypted_data.encode('utf-8'))
                else:
                    f.write(encrypted_data)

        return destination_path

    except Exception as e:
        logger.error(f"Error encrypting file {source_path}: {e}")
        raise


def encrypt_content_to_file(content: Union[str, bytes],
                            file_path: Union[str, Path],
                            encryption_key: str) -> Path:
    """
    Encrypt content and save it to a file.

    Parameters:
    -----------
    content : str or bytes
        Content to encrypt
    file_path : str or Path
        Path where to save the encrypted file
    encryption_key : str
        Encryption key

    Returns:
    --------
    Path
        Path to the encrypted file

    Raises:
    -------
    EncryptionError
        If encryption fails
    """
    file_path = Path(file_path)

    logger.info(f"Encrypting content and saving to {file_path}")

    try:
        # Encrypt data
        encrypted_data = encrypt_file_content(content, encryption_key)

        # Save encrypted data
        if isinstance(encrypted_data, dict):
            # Save as JSON if it's a metadata structure
            with open(file_path, 'w') as f:
                json.dump(encrypted_data, f) # type: ignore
        else:
            # Save as binary otherwise
            with open(file_path, 'wb') as f:
                if isinstance(encrypted_data, str):
                    f.write(encrypted_data.encode('utf-8'))
                else:
                    f.write(encrypted_data)

        return file_path

    except Exception as e:
        logger.error(f"Error encrypting content to {file_path}: {e}")
        raise


def is_encrypted_data(data: Any) -> bool:
    """
    Check if data appears to be in the encrypted format.

    Parameters:
    -----------
    data : Any
        Data to check

    Returns:
    --------
    bool
        True if the data appears to be encrypted
    """
    if isinstance(data, dict):
        # Check for our encryption metadata format
        return "data" in data and "algorithm" in data

    # Could add additional heuristics here, but for now just check dict format
    return False


def get_encryption_metadata(data: Dict) -> Dict[str, Any]:
    """
    Extract metadata from encrypted data structure.

    Parameters:
    -----------
    data : Dict
        Encrypted data structure

    Returns:
    --------
    Dict[str, Any]
        Encryption metadata
    """
    if not is_encrypted_data(data):
        return {}

    metadata = {
        "algorithm": data.get("algorithm"),
        "timestamp": data.get("timestamp"),
    }

    # Include any other metadata fields except the actual encrypted data
    for key, value in data.items():
        if key not in ["data", "algorithm", "timestamp"]:
            metadata[key] = value

    return metadata


def safe_remove_temp_file(temp_file_path: Optional[Union[str, Path]], logger):
    """
    Safely removes a temporary file if it exists.

    Parameters:
    -----------
    temp_file_path : str, Path, or None
        Path to the temporary file to remove
    logger : logging.Logger
        Logger to record warnings
    """
    if temp_file_path is not None:
        try:
            import os
            os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {e}")