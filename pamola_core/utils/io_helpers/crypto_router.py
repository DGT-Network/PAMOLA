"""
Crypto Router for PAMOLA.

This module handles routing encryption and decryption operations to the
appropriate provider based on the requested mode or file format detection.
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Type
import ijson

from pamola_core.utils.io_helpers.provider_interface import CryptoProvider
from pamola_core.utils.crypto_helpers.errors import ModeError, FormatError

# Configure logger
logger = logging.getLogger("pamola_core.utils.io_helpers.crypto_router")

# Provider registry
PROVIDERS: Dict[str, Type[CryptoProvider]] = {}

# Format detection constants
AGE_HEADER = b"age-encryption.org/"
SIMPLE_JSON_KEYS = ["algorithm", "iv", "data"]


def register_provider(provider_class: Type[CryptoProvider]) -> None:
    """
    Register a new crypto provider.

    Parameters:
    -----------
    provider_class : Type[CryptoProvider]
        Provider class to register
    """
    provider = provider_class()
    PROVIDERS[provider.mode] = provider_class
    logger.info(f"Registered crypto provider: {provider.mode}")


def get_provider(mode: str) -> CryptoProvider:
    """
    Get a provider instance for the specified mode.

    Parameters:
    -----------
    mode : str
        Encryption mode identifier

    Returns:
    --------
    CryptoProvider
        Provider instance

    Raises:
    -------
    ModeError
        If the mode is not supported
    """
    if mode not in PROVIDERS:
        logger.error(f"Unsupported encryption mode: {mode}")
        raise ModeError(f"Unsupported encryption mode: {mode}")

    return PROVIDERS[mode]()


def get_all_providers() -> List[CryptoProvider]:
    """
    Get instances of all registered providers.

    Returns:
    --------
    List[CryptoProvider]
        List of provider instances
    """
    return [provider_class() for provider_class in PROVIDERS.values()]


def is_likely_json(file_path: Path, chunk_size: int = 2048) -> bool:
    """
    Quickly checks whether a file is likely JSON by reading its first non-empty portion.

    Parameters
    ----------
    file_path : Path
        Path to the file to check.
    chunk_size : int, optional
        Number of characters to read from the beginning of the file (default is 2048).

    Returns
    -------
    bool
        True if the file likely starts with '{' or '[', indicating JSON. False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunk = f.read(chunk_size).lstrip()
            return chunk.startswith(('{', '['))
    except Exception as e:
        logger.debug(f"Failed to read file for JSON check: {e}")
        return False


def try_detect_json_structure(file_path: Path) -> Optional[str]:
    """
    Attempt to detect encryption mode or data structure from a JSON file using streaming parsing.

    The function first performs a lightweight check to see if the file likely contains JSON.
    If so, it uses `ijson` to stream through the top-level key-value pairs to detect:
    - A 'mode' key that matches a known provider.
    - A set of keys that match a known 'simple' data structure.

    Parameters
    ----------
    file_path : Path
        Path to the file to check.

    Returns
    -------
    Optional[str]
        Returns:
        - A provider mode string (e.g., 'aws', 'azure') if detected.
        - "simple" if the file matches the expected simple JSON key structure.
        - None if the file does not appear to be JSON or the structure is unrecognized.
    """
    if not is_likely_json(file_path):
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            parser = ijson.kvitems(f, '')  # Parse top-level JSON key-value pairs
            keys_found = set()

            for key, value in parser:
                if key == "mode" and value in PROVIDERS:
                    logger.info(f"Detected mode from metadata: {value}")
                    return value

                if key in SIMPLE_JSON_KEYS:
                    keys_found.add(key)

                if len(keys_found) == len(SIMPLE_JSON_KEYS):
                    logger.info("Detected 'simple' mode from JSON structure")
                    return "simple"

    except (ijson.JSONError, UnicodeDecodeError, ValueError, OSError) as e:
        logger.debug(f"Failed JSON detection using ijson: {e}")

    return None


def detect_encryption_mode(file_path: Union[str, Path]) -> str:
    """
    Detect the encryption mode of a file based on its content.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file to analyze.

    Returns
    -------
    str
        Detected encryption mode: one of 'simple', 'age', a custom provider's mode, or 'none'.

    Raises
    ------
    FormatError
        If the provided file path does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FormatError(f"File does not exist: {file_path}")

    # Step 1: Try to detect JSON-based encryption structure (regardless of file size)
    logger.info(f"Attempting to detect JSON structure in {file_path}")
    json_mode = try_detect_json_structure(file_path)
    if json_mode:
        logger.info(f"Detected JSON mode: {json_mode}")
        return json_mode

    # Step 2: Try to detect binary 'age' encryption mode from file header
    logger.info(f"Attempting to detect 'age' mode in {file_path}")
    try:
        with open(file_path, 'rb') as f:
            header = f.read(len(AGE_HEADER) + 10)
        if header.startswith(AGE_HEADER):
            logger.info("Detected 'age' mode from binary header")
            return "age"
    except Exception as e:
        logger.warning(f"Error reading binary header: {e}")

    # Step 3: Ask all available providers if they can decrypt this file
    logger.info(f"Attempting to detect decryption capabilities for {file_path}")
    for provider in get_all_providers():
        try:
            if provider.can_decrypt(file_path):
                logger.info(f"Provider {provider.mode} claims it can decrypt the file")
                return provider.mode
        except Exception as e:
            logger.debug(f"Provider {provider.mode} failed on can_decrypt: {e}")

    # Step 4: Fallback
    logger.warning(f"Could not detect encryption mode for {file_path}, defaulting to 'none'")
    return "none"


def encrypt_file_router(source_path: Union[str, Path],
                        destination_path: Union[str, Path],
                        key: Optional[str] = None,
                        mode: str = "simple",
                        **kwargs) -> Path:
    """
    Encrypt a file using the specified mode.

    Parameters:
    -----------
    source_path : str or Path
        Path to the file to encrypt
    destination_path : str or Path
        Path where to save the encrypted file
    key : str, optional
        Encryption key
    mode : str
        Encryption mode to use
    **kwargs : dict
        Additional provider-specific parameters

    Returns:
    --------
    Path
        Path to the encrypted file
    """
    provider = get_provider(mode)
    logger.info(f"Encrypting file using {mode} provider")

    return provider.encrypt_file(source_path, destination_path, key, **kwargs)


def decrypt_file_router(source_path: Union[str, Path],
                        destination_path: Union[str, Path],
                        key: Optional[str] = None,
                        mode: Optional[str] = None,
                        **kwargs) -> Path:
    """
    Decrypt a file, detecting the mode if not specified.

    Parameters:
    -----------
    source_path : str or Path
        Path to the encrypted file
    destination_path : str or Path
        Path where to save the decrypted file
    key : str, optional
        Decryption key
    mode : str, optional
        Encryption mode to use. If None, will be auto-detected.
    **kwargs : dict
        Additional provider-specific parameters

    Returns:
    --------
    Path
        Path to the decrypted file
    """
    # Detect mode if not specified
    if mode is None:
        mode = detect_encryption_mode(source_path)

    provider = get_provider(mode)
    logger.info(f"Decrypting file using {mode} provider")

    return provider.decrypt_file(source_path, destination_path, key, **kwargs)


def encrypt_data_router(data: Union[str, bytes],
                        key: Optional[str] = None,
                        mode: str = "simple",
                        **kwargs) -> Union[str, bytes, Dict[str, Any]]:
    """
    Encrypt data using the specified mode.

    Parameters:
    -----------
    data : str or bytes
        Data to encrypt
    key : str, optional
        Encryption key
    mode : str
        Encryption mode to use
    **kwargs : dict
        Additional provider-specific parameters

    Returns:
    --------
    Union[str, bytes, Dict[str, Any]]
        Encrypted data
    """
    provider = get_provider(mode)
    logger.info(f"Encrypting data using {mode} provider")

    encrypted = provider.encrypt_data(data, key, **kwargs)

    # If the result is a dictionary and doesn't have a mode field,
    # add it to help with format detection
    if isinstance(encrypted, dict) and "mode" not in encrypted:
        encrypted["mode"] = mode

    return encrypted


def decrypt_data_router(data: Union[str, bytes, Dict[str, Any]],
                        key: Optional[str] = None,
                        mode: Optional[str] = None,
                        **kwargs) -> Union[str, bytes]:
    """
    Decrypt data, detecting the mode if not specified.

    Parameters:
    -----------
    data : str, bytes, or Dict[str, Any]
        Data to decrypt
    key : str, optional
        Decryption key
    mode : str, optional
        Encryption mode to use. If None, will be auto-detected from the data.
    **kwargs : dict
        Additional provider-specific parameters

    Returns:
    --------
    Union[str, bytes]
        Decrypted data
    """
    # Detect mode if not specified
    if mode is None:
        # Check if data is a dictionary with mode info
        if isinstance(data, dict) and "mode" in data:
            mode = data["mode"]
        # Check if it's a simple encrypted dictionary
        elif isinstance(data, dict) and all(key in data for key in SIMPLE_JSON_KEYS):
            mode = "simple"
        # Default to none for anything else
        else:
            mode = "none"

    provider = get_provider(mode)
    logger.info(f"Decrypting data using {mode} provider")

    return provider.decrypt_data(data, key, **kwargs)