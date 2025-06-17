"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Crypto Utilities
Description: Pamola Core cryptographic operations supporting file and in-memory data encryption/decryption
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Pluggable provider-based crypto system with standardized interface
- File-level encryption and decryption with secure temporary handling
- In-memory data encryption for sensitive payloads
- Audit logging of cryptographic operations for compliance
- Extensible design for adding new encryption algorithms

"""


import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional

# Import register providers function to break circular imports
from pamola_core.utils.crypto_helpers.register_providers import register_all_providers

# Register all providers on module import
register_all_providers()

from pamola_core.utils.crypto_helpers.audit import log_crypto_operation
from pamola_core.utils.io_helpers.crypto_router import (
    encrypt_file_router,
    decrypt_file_router,
    encrypt_data_router,
    decrypt_data_router,
    detect_encryption_mode
)

# Configure logger
logger = logging.getLogger("pamola_core.utils.io_helpers.crypto_utils")


def encrypt_file(source_path: Union[str, Path],
                 destination_path: Union[str, Path],
                 key: Optional[str] = None,
                 mode: str = "simple",
                 task_id: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs) -> Path:
    """
    Encrypt a file and save it to a new location.

    Parameters:
    -----------
    source_path : str or Path
        Path to the file to encrypt
    destination_path : str or Path
        Path where to save the encrypted file
    key : str, optional
        Encryption key. If not provided, a task-specific key will be used
        or generated depending on the mode. Required for 'simple' mode,
        ignored for 'age' mode (which uses keypair).
    mode : str
        Encryption mode to use: "none", "simple", or "age"
    task_id : str, optional
        Identifier for the task associated with this operation
    description : str, optional
        Human-readable description of the file or operation
    **kwargs : dict
        Additional mode-specific parameters

    Returns:
    --------
    Path
        Path to the encrypted file

    Raises:
    -------
    EncryptionError
        If encryption fails
    """
    try:
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        # Add description to metadata if provided
        if description:
            kwargs["file_info"] = {"description": description}

        # Call the router to perform the encryption
        result = encrypt_file_router(
            source_path=source_path,
            destination_path=destination_path,
            key=key,
            mode=mode,
            **kwargs
        )

        # Log the operation for audit
        log_crypto_operation(
            operation="encrypt_file",
            mode=mode,
            status="success",
            source=source_path,
            destination=destination_path,
            task_id=task_id,
            metadata={"description": description} if description else None
        )

        return result

    except Exception as e:
        # Log the failure
        log_crypto_operation(
            operation="encrypt_file",
            mode=mode,
            status="failure",
            source=source_path,
            destination=destination_path,
            task_id=task_id,
            metadata={"error": str(e)}
        )

        logger.error(f"Error encrypting file {source_path}: {e}")
        raise


def decrypt_file(source_path: Union[str, Path],
                 destination_path: Union[str, Path],
                 key: Optional[str] = None,
                 mode: Optional[str] = None,
                 task_id: Optional[str] = None,
                 **kwargs) -> Path:
    """
    Decrypt a file and save it to a new location.

    Parameters:
    -----------
    source_path : str or Path
        Path to the encrypted file
    destination_path : str or Path
        Path where to save the decrypted file
    key : str, optional
        Decryption key. Required for 'simple' mode, ignored for 'age' mode
        (which uses keypair).
    mode : str, optional
        Encryption mode to use. If not provided, will be auto-detected.
    task_id : str, optional
        Identifier for the task associated with this operation
    **kwargs : dict
        Additional mode-specific parameters

    Returns:
    --------
    Path
        Path to the decrypted file

    Raises:
    -------
    DecryptionError
        If decryption fails
    """
    try:
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        # Auto-detect mode if not specified
        detected_mode = mode or detect_encryption_mode(source_path)

        # Call the router to perform the decryption
        result = decrypt_file_router(
            source_path=source_path,
            destination_path=destination_path,
            key=key,
            mode=detected_mode,
            **kwargs
        )

        # Log the operation for audit
        log_crypto_operation(
            operation="decrypt_file",
            mode=detected_mode,
            status="success",
            source=source_path,
            destination=destination_path,
            task_id=task_id
        )

        return result

    except Exception as e:
        # Try to detect the mode for logging purposes
        try:
            detected_mode = mode or detect_encryption_mode(source_path)
        except:
            detected_mode = "unknown"

        # Log the failure
        log_crypto_operation(
            operation="decrypt_file",
            mode=detected_mode,
            status="failure",
            source=source_path,
            destination=destination_path,
            task_id=task_id,
            metadata={"error": str(e)}
        )

        logger.error(f"Error decrypting file {source_path}: {e}")
        raise


def encrypt_data(data: Union[str, bytes],
                 key: Optional[str] = None,
                 mode: str = "simple",
                 task_id: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs) -> Union[str, bytes, Dict[str, Any]]:
    """
    Encrypt data in memory.

    Parameters:
    -----------
    data : str or bytes
        Data to encrypt
    key : str, optional
        Encryption key. Required for 'simple' mode, ignored for 'age' mode
        (which uses keypair).
    mode : str
        Encryption mode to use: "none", "simple", or "age"
    task_id : str, optional
        Identifier for the task associated with this operation
    description : str, optional
        Human-readable description of the data or operation
    **kwargs : dict
        Additional mode-specific parameters

    Returns:
    --------
    Union[str, bytes, Dict[str, Any]]
        Encrypted data

    Raises:
    -------
    EncryptionError
        If encryption fails
    """
    try:
        # Add description to metadata if provided
        if description:
            kwargs["data_info"] = {"description": description}

        # Call the router to perform the encryption
        result = encrypt_data_router(
            data=data,
            key=key,
            mode=mode,
            **kwargs
        )

        # Log the operation for audit
        log_crypto_operation(
            operation="encrypt_data",
            mode=mode,
            status="success",
            task_id=task_id,
            metadata={
                "description": description,
                "data_type": type(data).__name__,
                "result_type": type(result).__name__
            } if description else {
                "data_type": type(data).__name__,
                "result_type": type(result).__name__
            }
        )

        return result

    except Exception as e:
        # Log the failure
        log_crypto_operation(
            operation="encrypt_data",
            mode=mode,
            status="failure",
            task_id=task_id,
            metadata={
                "error": str(e),
                "data_type": type(data).__name__
            }
        )

        logger.error(f"Error encrypting data: {e}")
        raise


def decrypt_data(data: Union[str, bytes, Dict[str, Any]],
                 key: Optional[str] = None,
                 mode: Optional[str] = None,
                 task_id: Optional[str] = None,
                 **kwargs) -> Union[str, bytes]:
    """
    Decrypt data in memory.

    Parameters:
    -----------
    data : str, bytes, or Dict[str, Any]
        Data to decrypt
    key : str, optional
        Decryption key. Required for 'simple' mode, ignored for 'age' mode
        (which uses keypair).
    mode : str, optional
        Encryption mode to use. If not provided, will be auto-detected.
    task_id : str, optional
        Identifier for the task associated with this operation
    **kwargs : dict
        Additional mode-specific parameters

    Returns:
    --------
    Union[str, bytes]
        Decrypted data

    Raises:
    -------
    DecryptionError
        If decryption fails
    """
    try:
        # Call the router to perform the decryption
        result = decrypt_data_router(
            data=data,
            key=key,
            mode=mode,
            **kwargs
        )

        # Determine the mode that was used (for logging)
        detected_mode = mode
        if mode is None and isinstance(data, dict) and "mode" in data:
            detected_mode = data["mode"]
        elif mode is None:
            detected_mode = "auto"

        # Log the operation for audit
        log_crypto_operation(
            operation="decrypt_data",
            mode=detected_mode,
            status="success",
            task_id=task_id,
            metadata={
                "data_type": type(data).__name__,
                "result_type": type(result).__name__
            }
        )

        return result

    except Exception as e:
        # Determine the mode that was used (for logging)
        detected_mode = mode
        if mode is None and isinstance(data, dict) and "mode" in data:
            detected_mode = data["mode"]
        elif mode is None:
            detected_mode = "auto"

        # Log the failure
        log_crypto_operation(
            operation="decrypt_data",
            mode=detected_mode,
            status="failure",
            task_id=task_id,
            metadata={
                "error": str(e),
                "data_type": type(data).__name__
            }
        )

        logger.error(f"Error decrypting data: {e}")
        raise


def is_encrypted(data_or_path: Union[str, bytes, Dict[str, Any], Path]) -> bool:
    """
    Check if data or a file appears to be encrypted.

    Parameters:
    -----------
    data_or_path : str, bytes, Dict[str, Any], or Path
        Data or file path to check

    Returns:
    --------
    bool
        True if the data or file appears to be encrypted, False otherwise
    """
    # Check if it's a Path or string path
    if isinstance(data_or_path, (str, Path)) and Path(data_or_path).exists():
        try:
            mode = detect_encryption_mode(data_or_path)
            return mode != "none"
        except Exception:
            return False

    # If it's a dictionary, check for encryption indicators
    if isinstance(data_or_path, dict):
        return ("mode" in data_or_path and data_or_path["mode"] != "none") or \
            ("algorithm" in data_or_path and "data" in data_or_path)

    # For other types, we can't easily determine
    return False


def get_encryption_info(data_or_path: Union[str, bytes, Dict[str, Any], Path]) -> Dict[str, Any]:
    """
    Get information about encrypted data or file.

    Parameters:
    -----------
    data_or_path : str, bytes, Dict[str, Any], or Path
        Data or file path to check

    Returns:
    --------
    Dict[str, Any]
        Information about the encryption, empty if not encrypted
    """
    info = {}

    # Check if it's a Path or string path
    if isinstance(data_or_path, (str, Path)) and Path(data_or_path).exists():
        try:
            mode = detect_encryption_mode(data_or_path)
            info["mode"] = mode

            # If it's not encrypted, return minimal info
            if mode == "none":
                return info

            # Try to read file to get more info for 'simple' mode
            if mode == "simple":
                import json
                try:
                    with open(data_or_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract metadata while avoiding sensitive fields
                    for key, value in data.items():
                        if key not in ["data", "iv"]:
                            info[key] = value
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        except Exception:
            return info

    # If it's a dictionary, extract metadata
    elif isinstance(data_or_path, dict):
        # Extract metadata while avoiding sensitive fields
        for key, value in data_or_path.items():
            if key not in ["data", "iv"]:
                info[key] = value

    return info