"""
None encryption provider for PAMOLA.

This provider implements the CryptoProvider interface but performs
no actual encryption or decryption. It simply copies files as-is.
This mode is useful for testing and debugging.
"""

import shutil
from pathlib import Path
from typing import Union, Dict, Any, Optional

# Import from absolute path to avoid circular imports
from pamola_core.utils.io_helpers.provider_interface import CryptoProvider
from pamola_core.utils.crypto_helpers.errors import EncryptionError, DecryptionError


class NoneProvider(CryptoProvider):
    """
    A provider that performs no encryption.

    This provider implements the CryptoProvider interface but simply
    copies data without modification. Useful for development and testing.
    """

    @property
    def mode(self) -> str:
        """Return the provider's encryption mode identifier."""
        return "none"

    def encrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
                     **kwargs) -> Path:
        """
        'Encrypt' a file by simply copying it to the destination.

        Parameters:
        -----------
        source_path : str or Path
            Path to the file to 'encrypt'
        destination_path : str or Path
            Path where to save the 'encrypted' file
        key : str, optional
            Ignored in this provider
        **kwargs : dict
            Additional parameters (ignored)

        Returns:
        --------
        Path
            Path to the 'encrypted' file

        Raises:
        -------
        EncryptionError
            If the copy operation fails
        """
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file as-is
            shutil.copy2(source_path, destination_path)

            return destination_path
        except Exception as e:
            raise EncryptionError(f"Error copying file: {e}")

    def decrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
                     **kwargs) -> Path:
        """
        'Decrypt' a file by simply copying it to the destination.

        Parameters:
        -----------
        source_path : str or Path
            Path to the 'encrypted' file
        destination_path : str or Path
            Path where to save the 'decrypted' file
        key : str, optional
            Ignored in this provider
        **kwargs : dict
            Additional parameters (ignored)

        Returns:
        --------
        Path
            Path to the 'decrypted' file

        Raises:
        -------
        DecryptionError
            If the copy operation fails
        """
        source_path = Path(source_path)
        destination_path = Path(destination_path)

        try:
            # Create destination directory if it doesn't exist
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file as-is
            shutil.copy2(source_path, destination_path)

            return destination_path
        except Exception as e:
            raise DecryptionError(f"Error copying file: {e}")

    def encrypt_data(self,
                     data: Union[str, bytes],
                     key: Optional[str] = None,
                     **kwargs) -> Union[str, bytes]:
        """
        'Encrypt' data by simply returning it unchanged.

        Parameters:
        -----------
        data : str or bytes
            Data to 'encrypt'
        key : str, optional
            Ignored in this provider
        **kwargs : dict
            Additional parameters (ignored)

        Returns:
        --------
        Union[str, bytes]
            The original data
        """
        # Return data unchanged
        return data

    def decrypt_data(self,
                     data: Union[str, bytes, Dict[str, Any]],
                     key: Optional[str] = None,
                     **kwargs) -> Union[str, bytes]:
        """
        'Decrypt' data by simply returning it unchanged.

        Parameters:
        -----------
        data : str, bytes, or Dict[str, Any]
            Data to 'decrypt'
        key : str, optional
            Ignored in this provider
        **kwargs : dict
            Additional parameters (ignored)

        Returns:
        --------
        Union[str, bytes]
            The original data
        """
        # If we get a dictionary (which could happen when router auto-detects),
        # try to extract just the data
        if isinstance(data, dict) and "data" in data:
            return data["data"]

        # Otherwise return data unchanged
        return data

    def can_decrypt(self,
                    source_path: Union[str, Path]) -> bool:
        """
        Check if this provider can decrypt the given file.

        In 'none' mode, we assume we can handle any file that exists.

        Parameters:
        -----------
        source_path : str or Path
            Path to the file to check

        Returns:
        --------
        bool
            True if the file exists, False otherwise
        """
        return Path(source_path).exists()