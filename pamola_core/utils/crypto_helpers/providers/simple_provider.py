"""
Simple encryption provider for PAMOLA.

This provider implements AES-GCM encryption with JSON metadata format.
It's a balance between security and ease of use, suitable for most
applications within PAMOLA.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple

from pamola_core.utils.io_helpers.provider_interface import CryptoProvider
from pamola_core.utils.crypto_helpers.errors import EncryptionError, DecryptionError, FormatError

# Direct imports from cryptography library
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Constants
DEFAULT_KEY_LENGTH = 32  # 256 bits
DEFAULT_IV_LENGTH = 12  # 96 bits (recommended for GCM)
DEFAULT_ITERATIONS = 100000  # for key derivation
VERSION = "1.0"


class SimpleProvider(CryptoProvider):
    """
    Provider implementing AES-GCM encryption with JSON metadata.

    This provider uses AES-GCM for confidentiality and integrity
    protection, saving encrypted data with metadata in JSON format.
    """

    @property
    def mode(self) -> str:
        """Return the provider's encryption mode identifier."""
        return "simple"

    def derive_key(self,
                   password: str,
                   salt: Optional[bytes] = None,
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
            Number of iterations for PBKDF2
        key_length : int
            Length of the key in bytes

        Returns:
        --------
        Tuple[bytes, bytes]
            (key, salt) tuple

        Raises:
        -------
        EncryptionError
            If key derivation fails
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
            raise EncryptionError(f"Error deriving key: {e}")

    def encrypt_data(self,
                     data: Union[str, bytes],
                     key: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Encrypt data using AES-GCM.

        Parameters:
        -----------
        data : str or bytes
            Data to encrypt
        key : str, optional
            Encryption key
        **kwargs : dict
            Additional parameters:
            - data_info: Dict with metadata about the data

        Returns:
        --------
        Dict[str, Any]
            Dictionary with encrypted data and metadata

        Raises:
        -------
        EncryptionError
            If encryption fails
        """
        if not key:
            raise EncryptionError("Encryption key is required for simple mode")

        try:
            # Convert input data to bytes if it's a string
            if isinstance(data, str):
                data = data.encode('utf-8')

            # Derive a key from the password
            actual_key, salt = self.derive_key(key)

            # Generate a random IV
            iv = os.urandom(DEFAULT_IV_LENGTH)

            # Encrypt the data
            cipher = AESGCM(actual_key)
            ciphertext = cipher.encrypt(iv, data, None)

            # Encode binary data as base64 for JSON serialization
            import base64
            iv_b64 = base64.b64encode(iv).decode('utf-8')
            salt_b64 = base64.b64encode(salt).decode('utf-8')
            ciphertext_b64 = base64.b64encode(ciphertext).decode('utf-8')

            # Create metadata
            timestamp = datetime.now().isoformat()

            # Create the result dictionary
            result = {
                "mode": self.mode,
                "version": VERSION,
                "algorithm": "AES-GCM",
                "key_derivation": "PBKDF2-SHA256",
                "iterations": DEFAULT_ITERATIONS,
                "iv": iv_b64,
                "salt": salt_b64,
                "data": ciphertext_b64,
                "timestamp": timestamp,
                "format": "json",
            }

            # Add any additional metadata
            data_info = kwargs.get("data_info", {})
            if data_info:
                result["data_info"] = data_info

            return result

        except Exception as e:
            raise EncryptionError(f"Error encrypting data: {e}")

    def decrypt_data(self,
                     data: Union[str, bytes, Dict[str, Any]],
                     key: Optional[str] = None,
                     **kwargs) -> bytes:
        """
        Decrypt data encrypted with AES-GCM.

        Parameters:
        -----------
        data : str, bytes, or Dict[str, Any]
            Data to decrypt
        key : str, optional
            Decryption key
        **kwargs : dict
            Additional parameters (unused)

        Returns:
        --------
        bytes
            Decrypted data

        Raises:
        -------
        DecryptionError
            If decryption fails
        """
        if not key:
            raise DecryptionError("Decryption key is required for simple mode")

        try:
            # Handle different input types
            if isinstance(data, str):
                try:
                    # Try to parse as JSON
                    data_dict = json.loads(data)
                    if not isinstance(data_dict, dict):
                        raise DecryptionError("Invalid JSON format")
                    data = data_dict
                except json.JSONDecodeError:
                    raise DecryptionError("Data is not in valid JSON format")
            elif isinstance(data, bytes):
                try:
                    # Try to parse as JSON
                    data_dict = json.loads(data.decode('utf-8'))
                    if not isinstance(data_dict, dict):
                        raise DecryptionError("Invalid JSON format")
                    data = data_dict
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise DecryptionError("Data is not in valid JSON format")

            # Ensure we have a dictionary
            if not isinstance(data, dict):
                raise DecryptionError(f"Expected dictionary, got {type(data)}")

            # Check for required fields
            for field in ["algorithm", "iv", "salt", "data"]:
                if field not in data:
                    raise DecryptionError(f"Missing required field: {field}")

            # Verify algorithm
            if data["algorithm"] != "AES-GCM":
                raise DecryptionError(f"Unsupported algorithm: {data['algorithm']}")

            # Decode from base64
            import base64
            iv = base64.b64decode(data["iv"])
            salt = base64.b64decode(data["salt"])
            ciphertext = base64.b64decode(data["data"])

            # Derive the key
            actual_key, _ = self.derive_key(
                key,
                salt,
                iterations=data.get("iterations", DEFAULT_ITERATIONS)
            )

            # Decrypt the data
            cipher = AESGCM(actual_key)
            plaintext = cipher.decrypt(iv, ciphertext, None)

            return plaintext

        except Exception as e:
            raise DecryptionError(f"Error decrypting data: {e}")

    def encrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
                     **kwargs) -> Path:
        """
        Encrypt a file using AES-GCM and save with JSON metadata.

        Parameters:
        -----------
        source_path : str or Path
            Path to the file to encrypt
        destination_path : str or Path
            Path where to save the encrypted file
        key : str, optional
            Encryption key
        **kwargs : dict
            Additional parameters:
            - file_info: Dict with metadata about the file

        Returns:
        --------
        Path
            Path to the encrypted file

        Raises:
        -------
        EncryptionError
            If encryption fails
        """
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        if not source_path.exists():
            raise EncryptionError(f"Source file not found: {source_path}")

        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Read the source file
            with open(source_path, 'rb') as f:
                data = f.read()

            # Add file metadata
            file_info = kwargs.get("file_info", {})
            if not file_info:
                file_info = {}

            file_info["original_filename"] = source_path.name
            file_info["original_size"] = len(data)
            file_info["original_modified"] = datetime.fromtimestamp(
                source_path.stat().st_mtime).isoformat()

            # Encrypt the data
            encrypted = self.encrypt_data(
                data=data,
                key=key,
                data_info=file_info
            )

            # Save as JSON
            with open(destination_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted, f, indent=None) # type: ignore

            return destination_path

        except Exception as e:
            raise EncryptionError(f"Error encrypting file: {e}")

    def decrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
                     **kwargs) -> Path:
        """
        Decrypt a file encrypted with AES-GCM.

        Parameters:
        -----------
        source_path : str or Path
            Path to the encrypted file
        destination_path : str or Path
            Path where to save the decrypted file
        key : str, optional
            Decryption key
        **kwargs : dict
            Additional parameters (unused)

        Returns:
        --------
        Path
            Path to the decrypted file

        Raises:
        -------
        DecryptionError
            If decryption fails
        """
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        if not source_path.exists():
            raise DecryptionError(f"Source file not found: {source_path}")

        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Read the encrypted file
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise FormatError(f"Not a valid simple encrypted file: {source_path}")

            # Decrypt the data
            decrypted = self.decrypt_data(data, key)

            # Save to destination
            with open(destination_path, 'wb') as f:
                f.write(decrypted)

            # Try to restore file modification time if available
            if isinstance(data, dict) and "data_info" in data:
                file_info = data["data_info"]
                if "original_modified" in file_info:
                    try:
                        mtime = datetime.fromisoformat(file_info["original_modified"]).timestamp()
                        os.utime(destination_path, (time.time(), mtime))
                    except (ValueError, OSError):
                        # Ignore errors when setting mtime
                        pass

            return destination_path

        except FormatError:
            raise
        except Exception as e:
            raise DecryptionError(f"Error decrypting file: {e}")

    def can_decrypt(self,
                    source_path: Union[str, Path]) -> bool:
        """
        Check if this provider can decrypt the given file.

        Parameters:
        -----------
        source_path : str or Path
            Path to the file to check

        Returns:
        --------
        bool
            True if this provider can decrypt the file, False otherwise
        """
        source_path = Path(source_path)

        if not source_path.exists():
            return False

        try:
            # Try to read as JSON
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check for simple mode indicators
            if isinstance(data, dict):
                # Explicit mode check
                if "mode" in data and data["mode"] == self.mode:
                    return True

                # Check for required fields
                required_fields = ["algorithm", "iv", "salt", "data"]
                if all(field in data for field in required_fields):
                    # Check algorithm
                    if data.get("algorithm") == "AES-GCM":
                        return True

            return False

        except Exception:
            return False