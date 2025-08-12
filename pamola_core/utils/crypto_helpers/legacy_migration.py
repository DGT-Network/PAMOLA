"""
Legacy format migration for PAMOLA cryptographic subsystem.

This module provides utilities for detecting and migrating files encrypted
with older versions of the cryptographic subsystem to the current format.
"""

import base64
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Union, Optional, Tuple

from pamola_core.utils.crypto_helpers.audit import log_crypto_operation
from pamola_core.utils.crypto_helpers.errors import LegacyMigrationError, FormatError
from pamola_core.utils.crypto_helpers.providers.simple_provider import SimpleProvider

# Configure logger
logger = logging.getLogger("pamola_core.utils.crypto_helpers.legacy_migration")

# Legacy format detection signatures
LEGACY_FORMATS = {
    "v1_base64": {
        "prefix": "PAMOLA_ENC_V1:",
        "suffix": ""
    },
    "v1_json": {
        "keys": ["data", "iv", "timestamp"]
    },
    "v0_simple": {
        "prefix": "ENC:",
        "suffix": ""
    }
}


def detect_legacy_format(file_path: Union[str, Path]) -> Optional[str]:
    """
    Detect if a file uses a legacy encryption format.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file to check

    Returns:
    --------
    Optional[str]
        The detected legacy format identifier, or None if not a legacy format
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    try:
        # First try to read as text (most legacy formats were text-based)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for text-based legacy formats
            if content.startswith(LEGACY_FORMATS["v1_base64"]["prefix"]):
                return "v1_base64"

            if content.startswith(LEGACY_FORMATS["v0_simple"]["prefix"]):
                return "v0_simple"

            # Try to parse as JSON
            try:
                data = json.loads(content)
                if (isinstance(data, dict) and
                        all(key in data for key in LEGACY_FORMATS["v1_json"]["keys"]) and
                        "version" not in data and "mode" not in data):
                    return "v1_json"
            except json.JSONDecodeError:
                pass

        except UnicodeDecodeError:
            # Not text, try binary formats
            pass

        # No legacy format detected
        return None

    except Exception as e:
        logger.warning(f"Error checking for legacy format: {e}")
        return None


def migrate_legacy_file(source_path: Union[str, Path],
                        destination_path: Union[str, Path],
                        key: str,
                        format_type: Optional[str] = None) -> Path:
    """
    Migrate a file from a legacy encryption format to the current format.

    Parameters:
    -----------
    source_path : str or Path
        Path to the legacy encrypted file
    destination_path : str or Path
        Path where to save the migrated file
    key : str
        Encryption key for both decryption and re-encryption
    format_type : str, optional
        Legacy format type if known, otherwise will be auto-detected

    Returns:
    --------
    Path
        Path to the migrated file

    Raises:
    -------
    LegacyMigrationError
        If migration fails
    """
    global temp_path
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    if not source_path.exists():
        raise LegacyMigrationError(f"Source file not found: {source_path}")

    try:
        # Detect format if not specified
        if format_type is None:
            format_type = detect_legacy_format(source_path)

        if not format_type:
            raise FormatError(f"Could not detect legacy format for {source_path}")

        # Create a temporary file for the decrypted content
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)
        temp_path = Path(temp_path)

        # Decrypt based on the legacy format
        if format_type == "v1_base64":
            _decrypt_v1_base64(source_path, temp_path, key)
        elif format_type == "v1_json":
            _decrypt_v1_json(source_path, temp_path, key)
        elif format_type == "v0_simple":
            _decrypt_v0_simple(source_path, temp_path, key)
        else:
            raise LegacyMigrationError(f"Unsupported legacy format: {format_type}")

        # Create destination directory if it doesn't exist
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        # Re-encrypt with the current format
        provider = SimpleProvider()
        provider.encrypt_file(
            source_path=temp_path,
            destination_path=destination_path,
            key=key,
            file_info={
                "description": f"Migrated from legacy format {format_type}"
            }
        )

        # Log the migration
        log_crypto_operation(
            operation="migrate_legacy",
            mode="simple",
            status="success",
            source=source_path,
            destination=destination_path,
            metadata={"legacy_format": format_type}
        )

        return destination_path

    except Exception as e:
        # Log the failure
        log_crypto_operation(
            operation="migrate_legacy",
            mode="simple",
            status="failure",
            source=source_path,
            destination=destination_path,
            metadata={"error": str(e), "legacy_format": format_type}
        )

        raise LegacyMigrationError(f"Error migrating legacy file: {e}")
    finally:
        # Clean up temporary file
        if 'temp_path' in locals() and Path(temp_path).exists():
            os.unlink(temp_path)


def _decrypt_v1_base64(source_path: Path, destination_path: Path, key: str) -> None:
    """
    Decrypt a file in v1 base64 format.

    Parameters:
    -----------
    source_path : Path
        Path to the encrypted file
    destination_path : Path
        Path where to save the decrypted file
    key : str
        Decryption key

    Raises:
    -------
    LegacyMigrationError
        If decryption fails
    """
    try:
        # Read the encrypted file
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Strip the prefix
        prefix = LEGACY_FORMATS["v1_base64"]["prefix"]
        if not content.startswith(prefix):
            raise LegacyMigrationError(f"File does not have expected prefix: {prefix}")

        # Get the encoded data
        encoded = content[len(prefix):]

        # Decode from base64
        try:
            encrypted = base64.b64decode(encoded)
        except Exception:
            raise LegacyMigrationError("Invalid base64 encoding")

        # Decrypt using a similar method to the original implementation
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        # The legacy format used a fixed salt and IV for simplicity
        # (this was a security issue that we're fixing with the migration)
        salt = b'PAMOLA_V1_SALT_0'
        iv = b'PAMOLA_V1_IV__'

        # Derive the key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,
        )
        derived_key = kdf.derive(key.encode('utf-8'))

        # Decrypt the data
        cipher = AESGCM(derived_key)
        decrypted = cipher.decrypt(iv, encrypted, None)

        # Save the decrypted data
        with open(destination_path, 'wb') as f:
            f.write(decrypted)

    except Exception as e:
        raise LegacyMigrationError(f"Error decrypting v1_base64 file: {e}")


def _decrypt_v1_json(source_path: Path, destination_path: Path, key: str) -> None:
    """
    Decrypt a file in v1 JSON format.

    Parameters:
    -----------
    source_path : Path
        Path to the encrypted file
    destination_path : Path
        Path where to save the decrypted file
    key : str
        Decryption key

    Raises:
    -------
    LegacyMigrationError
        If decryption fails
    """
    try:
        # Read the encrypted file
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check required fields
        required_keys = LEGACY_FORMATS["v1_json"]["keys"]
        if not all(k in data for k in required_keys):
            raise LegacyMigrationError(f"Missing required fields in JSON")

        # Extract encrypted data and IV
        encrypted_data = data["data"]
        iv_b64 = data["iv"]

        # Decode from base64
        try:
            encrypted = base64.b64decode(encrypted_data)
            iv = base64.b64decode(iv_b64)
        except Exception:
            raise LegacyMigrationError("Invalid base64 encoding")

        # Decrypt using a similar method to the original implementation
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        # The legacy format used a fixed salt
        salt = b'PAMOLA_V1_SALT_0'

        # Derive the key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256 bits
            salt=salt,
            iterations=100000,
        )
        derived_key = kdf.derive(key.encode('utf-8'))

        # Decrypt the data
        cipher = AESGCM(derived_key)
        decrypted = cipher.decrypt(iv, encrypted, None)

        # Save the decrypted data
        with open(destination_path, 'wb') as f:
            f.write(decrypted)

    except Exception as e:
        raise LegacyMigrationError(f"Error decrypting v1_json file: {e}")


def _decrypt_v0_simple(source_path: Path, destination_path: Path, key: str) -> None:
    """
    Decrypt a file in v0 simple format.

    Parameters:
    -----------
    source_path : Path
        Path to the encrypted file
    destination_path : Path
        Path where to save the decrypted file
    key : str
        Decryption key

    Raises:
    -------
    LegacyMigrationError
        If decryption fails
    """
    try:
        # Read the encrypted file
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Strip the prefix
        prefix = LEGACY_FORMATS["v0_simple"]["prefix"]
        if not content.startswith(prefix):
            raise LegacyMigrationError(f"File does not have expected prefix: {prefix}")

        # Get the encoded data
        encoded = content[len(prefix):]

        # The v0 format used a very simple XOR cipher (highly insecure)
        # This is included only for backward compatibility
        key_bytes = key.encode('utf-8')
        key_len = len(key_bytes)

        # Decode the simple hex encoding
        try:
            encrypted = bytes.fromhex(encoded)
        except ValueError:
            raise LegacyMigrationError("Invalid hex encoding in v0 format")

        # XOR decrypt (this was the original simplistic method)
        decrypted = bytearray(len(encrypted))
        for i in range(len(encrypted)):
            decrypted[i] = encrypted[i] ^ key_bytes[i % key_len]

        # Save the decrypted data
        with open(destination_path, 'wb') as f:
            f.write(decrypted)

    except Exception as e:
        raise LegacyMigrationError(f"Error decrypting v0_simple file: {e}")


def auto_migrate_if_needed(source_path: Union[str, Path],
                           destination_path: Union[str, Path],
                           key: str) -> Tuple[Path, bool]:
    """
    Automatically detect and migrate a legacy file if needed.

    Parameters:
    -----------
    source_path : str or Path
        Path to the possibly legacy encrypted file
    destination_path : str or Path
        Path where to save the migrated file (if migration is needed)
    key : str
        Encryption key

    Returns:
    --------
    Tuple[Path, bool]
        (Path to the output file, whether migration was performed)

    Raises:
    -------
    LegacyMigrationError
        If migration fails
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    # Detect if this is a legacy format
    format_type = detect_legacy_format(source_path)

    if format_type:
        # This is a legacy format, migrate it
        logger.info(f"Detected legacy format {format_type}, migrating...")
        migrated_path = migrate_legacy_file(source_path, destination_path, key, format_type)
        return migrated_path, True
    else:
        # Not a legacy format, just return the source path
        return source_path, False