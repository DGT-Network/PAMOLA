"""
Age encryption provider for PAMOLA.

This provider integrates with the external 'age' command-line tool for
encryption and decryption. It supports stream-based encryption for
large files and uses public/private key cryptography.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Dict, Any, Optional, List

from pamola_core.utils.crypto_helpers.errors import (
    EncryptionError,
    DecryptionError,
    ConfigurationError,
    AgeToolError
)
from pamola_core.utils.io_helpers.provider_interface import CryptoProvider

# Constants - only keep constant values at module level
AGE_HEADER = b"age-encryption.org/"
AGE_BINARY = os.environ.get("PAMOLA_AGE_BINARY", "age")


class AgeProvider(CryptoProvider):
    """
    Provider that integrates with the 'age' CLI tool for encryption.

    This provider uses public/private key cryptography for secure
    encryption and decryption of files and data.
    """

    def __init__(self):
        """
        Initialize the provider with lazy checking.

        The age tool availability is not checked until actually needed.
        """
        # Initialize state for lazy checking
        self._checked = False
        self.available = None  # Unknown until checked

    @property
    def mode(self) -> str:
        """Return the provider's encryption mode identifier."""
        return "age"

    def _ensure_available(self) -> None:
        """
        Ensure the age binary is available, checking only once.

        Raises:
        -------
        RuntimeError
            If age is not available when needed
        """
        # Skip if already checked
        if self.available is not None:
            if not self.available:
                raise RuntimeError("Age encryption requested, but age binary not found")
            return

        try:
            self._check_age_installed()
            self.available = True
        except Exception as e:
            self.available = False
            raise RuntimeError(f"Age encryption unavailable: {str(e)}")

    def _check_age_installed(self) -> None:
        """
        Check if the 'age' command-line tool is installed.

        Raises:
        -------
        ConfigurationError
            If 'age' is not installed or not found
        """
        try:
            result = subprocess.run(
                [AGE_BINARY, "--version"],
                capture_output=True,  # This captures stdout and stderr properly
                text=True,
                check=False  # Don't raise on non-zero exit
            )

            if result.returncode != 0:
                raise ConfigurationError(
                    f"age tool returned non-zero exit code: {result.returncode}\n"
                    f"stderr: {result.stderr}"
                )
        except FileNotFoundError:
            raise ConfigurationError(
                f"age command-line tool not found. Please install age or set PAMOLA_AGE_BINARY."
            )
        except Exception as e:
            raise ConfigurationError(f"Error checking age installation: {e}")

    def _get_recipients(self) -> List[str]:
        """
        Get the list of recipients for encryption.

        Returns:
        --------
        List[str]
            List of recipient arguments for age

        Raises:
        -------
        ConfigurationError
            If no recipients are available
        """
        # Read environment variables at method call time, not module import time
        age_recipients = os.environ.get("PAMOLA_AGE_RECIPIENTS", "").split(',')
        age_recipients_file = os.environ.get("PAMOLA_AGE_RECIPIENTS_FILE", "")

        recipients = []

        # Add recipients from environment
        for recipient in age_recipients:
            if recipient.strip():
                recipients.extend(["-r", recipient.strip()])

        # Add recipients file if specified
        if age_recipients_file:
            recipients.extend(["-R", age_recipients_file])

        # If no recipients are available, raise an error
        if not recipients:
            raise ConfigurationError(
                "No recipients specified for age encryption. "
                "Set PAMOLA_AGE_RECIPIENTS or PAMOLA_AGE_RECIPIENTS_FILE."
            )

        return recipients

    def _get_identity(self) -> List[str]:
        """
        Get the identity for decryption.

        Returns:
        --------
        List[str]
            Identity arguments for age

        Raises:
        -------
        ConfigurationError
            If no identity file is available
        """
        # Read environment variable at method call time
        age_identity_file = os.environ.get("PAMOLA_AGE_IDENTITY_FILE", "")

        # Use identity file
        if not age_identity_file:
            raise ConfigurationError(
                "No identity file specified for age decryption. "
                "Set PAMOLA_AGE_IDENTITY_FILE."
            )

        return ["-i", age_identity_file]

    def encrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
                     **kwargs) -> Path:
        """
        Encrypt a file using the age tool with public key cryptography.

        Parameters:
        -----------
        source_path : str or Path
            Path to the file to encrypt
        destination_path : str or Path
            Path where to save the encrypted file
        key : str, optional
            Ignored in keypair mode
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
        # First ensure age is available
        self._ensure_available()

        source_path = Path(source_path)
        destination_path = Path(destination_path)
        metadata_file = None

        if not source_path.exists():
            raise EncryptionError(f"Source file not found: {source_path}")

        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle file metadata (age doesn't support metadata directly)
            file_info = kwargs.get("file_info", {})

            if file_info:
                # Create a temp file to store metadata
                fd, metadata_file = tempfile.mkstemp(suffix=".json")
                os.close(fd)

                # Add standard metadata
                metadata = {
                    "mode": self.mode,
                    "timestamp": file_info.get("timestamp", ""),
                    "description": file_info.get("description", ""),
                    "original_filename": source_path.name,
                    "original_size": source_path.stat().st_size,
                    "original_modified": os.path.getmtime(source_path),
                }

                # Save metadata file
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f)  # type: ignore

            # Build the age command
            cmd = [AGE_BINARY]
            cmd.extend(self._get_recipients())
            cmd.extend(["-o", str(destination_path), str(source_path)])

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise AgeToolError(
                    f"age encryption failed with exit code {result.returncode}: {result.stderr}"
                )

            # If we have metadata, add it as an adjacent file
            if metadata_file:
                metadata_dest = str(destination_path) + ".meta"
                with open(metadata_file, 'r', encoding='utf-8') as src:
                    with open(metadata_dest, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())

                # Clean up temp file
                os.unlink(metadata_file)
                metadata_file = None

            return destination_path

        except Exception as e:
            # Clean up temp file if it exists
            if metadata_file and os.path.exists(metadata_file):
                os.unlink(metadata_file)

            raise EncryptionError(f"Error encrypting file with age: {e}")

    def decrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
                     **kwargs) -> Path:
        """
        Decrypt a file using the age tool with private key cryptography.

        Parameters:
        -----------
        source_path : str or Path
            Path to the encrypted file
        destination_path : str or Path
            Path where to save the decrypted file
        key : str, optional
            Ignored in keypair mode
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
        # First ensure age is available
        self._ensure_available()

        source_path = Path(source_path)
        destination_path = Path(destination_path)

        if not source_path.exists():
            raise DecryptionError(f"Source file not found: {source_path}")

        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Check for metadata file
            metadata_file = str(source_path) + ".meta"
            metadata = {}

            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Ignore errors reading metadata
                    pass

            # Build the age command
            cmd = [AGE_BINARY]
            cmd.extend(self._get_identity())
            cmd.extend(["-d", "-o", str(destination_path), str(source_path)])

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise AgeToolError(
                    f"age decryption failed with exit code {result.returncode}: {result.stderr}"
                )

            # Restore file timestamp if available
            if metadata and "original_modified" in metadata:
                try:
                    os.utime(
                        destination_path,
                        (os.path.getatime(destination_path), metadata["original_modified"])
                    )
                except (ValueError, OSError):
                    # Ignore errors when setting mtime
                    pass

            return destination_path

        except Exception as e:
            raise DecryptionError(f"Error decrypting file with age: {e}")

    def encrypt_data(self,
                     data: Union[str, bytes],
                     key: Optional[str] = None,
                     **kwargs) -> bytes:
        """
        Encrypt data using the age tool with public key cryptography.

        Parameters:
        -----------
        data : str or bytes
            Data to encrypt
        key : str, optional
            Ignored in keypair mode
        **kwargs : dict
            Additional parameters (unused)

        Returns:
        --------
        bytes
            Encrypted data

        Raises:
        -------
        EncryptionError
            If encryption fails
        """
        # First ensure age is available
        self._ensure_available()

        temp_in_path = None
        temp_out_path = None

        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(delete=False) as temp_in:
                # Write data to temp file
                if isinstance(data, str):
                    temp_in.write(data.encode('utf-8'))
                else:
                    temp_in.write(data)
                temp_in_path = temp_in.name

            # Create temporary output file
            fd, temp_out_path = tempfile.mkstemp()
            os.close(fd)

            # Build the age command
            cmd = [AGE_BINARY]
            cmd.extend(self._get_recipients())
            cmd.extend(["-o", temp_out_path, temp_in_path])

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise AgeToolError(
                    f"age encryption failed with exit code {result.returncode}: {result.stderr}"
                )

            # Read the encrypted data
            with open(temp_out_path, 'rb') as f:
                encrypted_data = f.read()

            return encrypted_data

        except Exception as e:
            raise EncryptionError(f"Error encrypting data with age: {e}")
        finally:
            # Clean up temporary files
            if temp_in_path and os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if temp_out_path and os.path.exists(temp_out_path):
                os.unlink(temp_out_path)

    def decrypt_data(self,
                     data: Union[str, bytes, Dict[str, Any]],
                     key: Optional[str] = None,
                     **kwargs) -> bytes:
        """
        Decrypt data using the age tool with private key cryptography.

        Parameters:
        -----------
        data : str, bytes, or Dict[str, Any]
            Data to decrypt
        key : str, optional
            Ignored in keypair mode
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
        # First ensure age is available
        self._ensure_available()

        temp_in_path = None
        temp_out_path = None

        try:
            # Ensure data is in bytes format
            if isinstance(data, str):
                binary_data = data.encode('utf-8')
            elif isinstance(data, dict) and "data" in data:
                # If it's a dictionary with data field, extract the data
                if isinstance(data["data"], str):
                    binary_data = data["data"].encode('utf-8')
                else:
                    binary_data = data["data"]
            else:
                binary_data = data

            # Create temporary input file
            with tempfile.NamedTemporaryFile(delete=False) as temp_in:
                temp_in.write(binary_data)
                temp_in_path = temp_in.name

            # Create temporary output file
            fd, temp_out_path = tempfile.mkstemp()
            os.close(fd)

            # Build the age command
            cmd = [AGE_BINARY]
            cmd.extend(self._get_identity())
            cmd.extend(["-d", "-o", temp_out_path, temp_in_path])

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise AgeToolError(
                    f"age decryption failed with exit code {result.returncode}: {result.stderr}"
                )

            # Read the decrypted data
            with open(temp_out_path, 'rb') as f:
                decrypted_data = f.read()

            return decrypted_data

        except Exception as e:
            raise DecryptionError(f"Error decrypting data with age: {e}")
        finally:
            # Clean up temporary files
            if temp_in_path and os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if temp_out_path and os.path.exists(temp_out_path):
                os.unlink(temp_out_path)

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
        # We don't need age to be installed just to check if a file is compatible
        source_path = Path(source_path)

        if not source_path.exists():
            return False

        try:
            # Check for metadata file
            metadata_file = str(source_path) + ".meta"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    if metadata.get("mode") == self.mode:
                        return True
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Ignore errors reading metadata
                    pass

            # Check file header
            with open(source_path, 'rb') as f:
                header = f.read(len(AGE_HEADER) + 10)

            return header.startswith(AGE_HEADER)

        except Exception:
            return False

    def is_available(self) -> bool:
        """
        Check if age is available without raising exceptions.

        Returns:
        --------
        bool
            True if age is available, False otherwise
        """
        if self.available is None:
            try:
                self._check_age_installed()
                self.available = True
            except Exception:
                self.available = False

        return self.available