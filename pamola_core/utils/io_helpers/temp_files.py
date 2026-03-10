"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module: Temporary File Context Managers
Description: Shared helpers for encryption/decryption temp file handling
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union

import pamola_core.utils.io_helpers.crypto_utils as crypto_utils
import pamola_core.utils.io_helpers.directory_utils as directory_utils

logger = logging.getLogger(__name__)


@contextmanager
def temporary_decrypted_file(
    file_path: Union[str, Path],
    encryption_key: Optional[str],
    suffix: str = "",
    encryption_mode: str = "simple",
):
    """
    Context manager for handling temporary decrypted files.

    Parameters
    -----------
    file_path : Union[str, Path]
        Path to the encrypted file
    encryption_key : Optional[str]
        Decryption key (if None, yields original file)
    suffix : str
        File extension for temporary file

    Yields
    -------
    Path
        Path to the file to read (original or decrypted temporary)
    """
    if encryption_mode == "age":
        pass
    else:
        if not encryption_key:
            yield Path(file_path)
            return

        # encryption_key is NOT None => encryption_mode = 'simple'
        encryption_mode = "simple"

    temp_file_path = None
    try:
        logger.info("Decryption requested for file reading")
        temp_file_path = directory_utils.get_temp_file_for_decryption(
            file_path, suffix=suffix
        )

        crypto_utils.decrypt_file(
            source_path=file_path,
            destination_path=temp_file_path,
            key=encryption_key,
            mode=encryption_mode,
        )

        logger.debug(f"File decrypted to temporary location: {temp_file_path}")
        yield temp_file_path

    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise
    finally:
        if temp_file_path:
            directory_utils.safe_remove_temp_file(temp_file_path)


@contextmanager
def temporary_file_for_encryption(
    file_path: Union[str, Path],
    encryption_key: Optional[str],
    suffix: str = "",
    encryption_mode: str = "simple",
):
    """
    Context manager for handling temporary files before encryption.

    Parameters
    -----------
    file_path : Union[str, Path]
        Target path for the encrypted file
    encryption_key : Optional[str]
        Encryption key (if None, yields target path directly)
    suffix : str
        File extension for temporary file

    Yields
    -------
    Path
        Path to write to (temporary if encrypting, target if not)
    """
    if encryption_mode == "age":
        pass
    else:
        if not encryption_key:
            yield Path(file_path)
            return

        # encryption_key is NOT None => encryption_mode = 'simple'
        encryption_mode = "simple"

    temp_file_path = None
    try:
        logger.info("Encryption requested for file writing")
        temp_file_path = directory_utils.get_temp_file_for_encryption(
            file_path, suffix=suffix
        )

        yield temp_file_path

        # After writing, encrypt to final destination
        logger.info(f"Encrypting and saving to final destination: {file_path}")
        crypto_utils.encrypt_file(
            source_path=temp_file_path,
            destination_path=file_path,
            key=encryption_key,
            mode=encryption_mode,
        )

    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise
    finally:
        if temp_file_path:
            directory_utils.safe_remove_temp_file(temp_file_path)


@contextmanager
def temporary_decrypted_files(
    file_paths: List[Union[str, Path]],
    encryption_key: Optional[str],
    suffix: str = "dec",
    encryption_mode: str = "none",
):
    """
    Context manager for handling temporary decrypted files.

    Parameters
    -----------
    file_paths : List[Union[str, Path]]
        Path to the encrypted files
    encryption_key : Optional[str]
        Decryption key (if None, yields original file)
    suffix : str
        File extension for temporary files

    Yields
    -------
    List[Union[str, Path]]
        Path to the files to read (original or decrypted temporary)
    """
    temp_file_path = None
    temp_file_paths = []
    try:
        logger.info("Decryption requested for files reading")
        for file_path in file_paths:
            temp_file_path = directory_utils.get_temp_file_for_decryption(
                original_file=file_path, suffix=suffix
            )

            crypto_utils.decrypt_file(
                source_path=file_path,
                destination_path=temp_file_path,
                key=encryption_key,
                mode=encryption_mode,
            )

            temp_file_paths.append(temp_file_path)

        logger.debug(f"Files decrypted to temporary location: {temp_file_paths}")
        yield temp_file_paths

    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise
    finally:
        if temp_file_paths and False:
            for temp_file_path in temp_file_paths:
                directory_utils.safe_remove_temp_file(temp_file_path)


@contextmanager
def temporary_files_for_encryption(
    file_paths: List[Union[str, Path]],
    encryption_key: Optional[str],
    suffix: str = "enc",
    encryption_mode: str = "none",
):
    """
    Context manager for handling temporary files before encryption.

    Parameters
    -----------
    file_paths : List[Union[str, Path]]
        Target path for the encrypted files
    encryption_key : Optional[str]
        Encryption key (if None, yields target path directly)
    suffix : str
        File extension for temporary files

    Yields
    -------
    List[Union[str, Path]]
        Paths to write to (temporary if encrypting, target if not)
    """
    temp_file_path = None
    temp_file_paths = []
    try:
        logger.info("Encryption requested for files writing")
        for file_path in file_paths:
            temp_file_path = directory_utils.get_temp_file_for_decryption(
                original_file=file_path, suffix=suffix
            )

            temp_file_paths.append(temp_file_path)

        yield temp_file_paths

        # After writing, encrypt to final destination
        logger.info(f"Encrypting and saving to final destination: {file_paths}")
        for file_path, temp_file_path in list(zip(file_paths, temp_file_paths)):
            crypto_utils.encrypt_file(
                source_path=temp_file_path,
                destination_path=file_path,
                key=encryption_key,
                mode=encryption_mode,
            )

    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise
    finally:
        if temp_file_path:
            for temp_file_path in temp_file_paths:
                directory_utils.safe_remove_temp_file(temp_file_path)
