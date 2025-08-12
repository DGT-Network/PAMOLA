"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Cryptographic Provider Interface
Description: Abstract interface for pluggable cryptographic providers in the PAMOLA system
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Abstract base class defining required encryption/decryption methods
- Standard interface for file and in-memory data operations
- Designed for consistent behavior across different crypto implementations
"""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional


# Import basic none_provider to avoid circular imports
class CryptoProvider(ABC):
    """
    Abstract base class for all cryptographic providers.

    All providers must implement the methods defined here to ensure
    consistent behavior across encryption modes.
    """

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return the provider's encryption mode identifier."""
        pass

    @abstractmethod
    def encrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
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
            Encryption key (may not be used in all providers)
        **kwargs : dict
            Additional provider-specific parameters

        Returns:
        --------
        Path
            Path to the encrypted file

        Raises:
        -------
        EncryptionError
            If encryption fails
        """
        pass

    @abstractmethod
    def decrypt_file(self,
                     source_path: Union[str, Path],
                     destination_path: Union[str, Path],
                     key: Optional[str] = None,
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
            Decryption key (may not be used in all providers)
        **kwargs : dict
            Additional provider-specific parameters

        Returns:
        --------
        Path
            Path to the decrypted file

        Raises:
        -------
        DecryptionError
            If decryption fails
        """
        pass

    @abstractmethod
    def encrypt_data(self,
                     data: Union[str, bytes],
                     key: Optional[str] = None,
                     **kwargs) -> Union[str, bytes, Dict[str, Any]]:
        """
        Encrypt data in memory.

        Parameters:
        -----------
        data : str or bytes
            Data to encrypt
        key : str, optional
            Encryption key (may not be used in all providers)
        **kwargs : dict
            Additional provider-specific parameters

        Returns:
        --------
        Union[str, bytes, Dict[str, Any]]
            Encrypted data, possibly with metadata

        Raises:
        -------
        EncryptionError
            If encryption fails
        """
        pass

    @abstractmethod
    def decrypt_data(self,
                     data: Union[str, bytes, Dict[str, Any]],
                     key: Optional[str] = None,
                     **kwargs) -> Union[str, bytes]:
        """
        Decrypt data in memory.

        Parameters:
        -----------
        data : str, bytes, or Dict[str, Any]
            Data to decrypt, possibly with metadata
        key : str, optional
            Decryption key (may not be used in all providers)
        **kwargs : dict
            Additional provider-specific parameters

        Returns:
        --------
        Union[str, bytes]
            Decrypted data

        Raises:
        -------
        DecryptionError
            If decryption fails
        """
        pass

    @abstractmethod
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
        pass