"""
Key storage for PAMOLA cryptographic subsystem.

This module provides secure storage and retrieval of cryptographic keys
for different tasks, encrypting task-specific keys with a master key.
"""

import base64
import json
import logging
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from pamola_core.utils.crypto_helpers.audit import log_key_access
from pamola_core.utils.crypto_helpers.errors import KeyStoreError, MasterKeyError, TaskKeyError
from pamola_core.utils.crypto_helpers.providers.simple_provider import SimpleProvider

# Configure logger
logger = logging.getLogger("pamola_core.utils.crypto_helpers.key_store")

# Constants
DEFAULT_KEYS_DB_PATH = "configs/keys.db"
DEFAULT_MASTER_KEY_PATH = "configs/master.key"
MASTER_KEY_LENGTH = 32  # 256 bits


class EncryptedKeyStore:
    """
    Secure storage for task-specific encryption keys.

    This class manages a collection of encryption keys for different tasks,
    storing them in an encrypted database using a master key.
    """

    def __init__(self,
                 keys_db_path: Optional[Union[str, Path]] = None,
                 master_key_path: Optional[Union[str, Path]] = None):
        """
        Initialize the key store.

        Parameters:
        -----------
        keys_db_path : str or Path, optional
            Path to the keys database file
        master_key_path : str or Path, optional
            Path to the master key file
        """
        self.keys_db_path = Path(keys_db_path or DEFAULT_KEYS_DB_PATH)
        self.master_key_path = Path(master_key_path or DEFAULT_MASTER_KEY_PATH)
        self.provider = SimpleProvider()

        # Ensure the keys database directory exists
        self.keys_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize the database if it doesn't exist
        if not self.keys_db_path.exists():
            self._initialize_keys_db()

    def _initialize_keys_db(self) -> None:
        """
        Initialize an empty keys database.

        Creates an empty, encrypted keys database file.
        """
        keys_db = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "keys": {}
        }

        # Get master key
        master_key = self._get_master_key()

        # Encrypt and save the database
        encrypted = self.provider.encrypt_data(
            data=json.dumps(keys_db),
            key=master_key
        )

        with open(self.keys_db_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted, f) # type: ignore

        # Set secure permissions
        os.chmod(self.keys_db_path, stat.S_IRUSR | stat.S_IWUSR)

        logger.info(f"Initialized new keys database at {self.keys_db_path}")

    def _get_master_key(self) -> str:
        """
        Get the master encryption key.

        Reads the master key from file or generates a new one if it doesn't exist.

        Returns:
        --------
        str
            The master key as a base64-encoded string

        Raises:
        -------
        MasterKeyError
            If there's an error accessing the master key
        """
        try:
            # Check if master key file exists
            if not self.master_key_path.exists():
                return self._generate_master_key()

            # Read the master key
            with open(self.master_key_path, 'r', encoding='utf-8') as f:
                master_key = f.read().strip()

            # Check if the key is a valid base64-encoded string
            try:
                decoded = base64.b64decode(master_key)
                if len(decoded) != MASTER_KEY_LENGTH:
                    logger.warning(
                        f"Master key has incorrect length: {len(decoded)} bytes"
                        f" (expected {MASTER_KEY_LENGTH}). Generating new key."
                    )
                    return self._generate_master_key()
            except Exception:
                logger.warning(
                    f"Master key is not a valid base64 string. Generating new key."
                )
                return self._generate_master_key()

            return master_key

        except Exception as e:
            raise MasterKeyError(f"Error accessing master key: {e}")

    def _generate_master_key(self) -> str:
        """
        Generate a new master key.

        Creates a new random master key and saves it to the master key file.

        Returns:
        --------
        str
            The generated master key as a base64-encoded string

        Raises:
        -------
        MasterKeyError
            If there's an error generating or saving the master key
        """
        try:
            # Generate a random key
            import os
            key_bytes = os.urandom(MASTER_KEY_LENGTH)
            master_key = base64.b64encode(key_bytes).decode('utf-8')

            # Ensure the directory exists
            self.master_key_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the key to file
            with open(self.master_key_path, 'w', encoding='utf-8') as f:
                f.write(master_key)

            # Set secure permissions
            os.chmod(self.master_key_path, stat.S_IRUSR | stat.S_IWUSR)

            logger.warning(
                f"Generated new master key at {self.master_key_path}. "
                f"This key should be securely backed up."
            )

            # Log the key generation
            log_key_access("generate", "master", "success")

            return master_key

        except Exception as e:
            log_key_access("generate", "master", "failure", metadata={"error": str(e)})
            raise MasterKeyError(f"Error generating master key: {e}")

    def _load_keys_db(self) -> Dict[str, Any]:
        """
        Load and decrypt the keys database.

        Returns:
        --------
        Dict[str, Any]
            The decrypted keys database

        Raises:
        -------
        KeyStoreError
            If there's an error loading or decrypting the database
        """
        try:
            # Check if the database exists
            if not self.keys_db_path.exists():
                self._initialize_keys_db()

            # Get master key
            master_key = self._get_master_key()

            # Read and decrypt the database
            with open(self.keys_db_path, 'r', encoding='utf-8') as f:
                encrypted = json.load(f)

            decrypted = self.provider.decrypt_data(encrypted, master_key)

            # Parse the JSON database
            if isinstance(decrypted, bytes):
                db = json.loads(decrypted.decode('utf-8'))
            else:
                db = json.loads(decrypted)

            return db

        except Exception as e:
            raise KeyStoreError(f"Error loading keys database: {e}")

    def _save_keys_db(self, db: Dict[str, Any]) -> None:
        """
        Encrypt and save the keys database.

        Parameters:
        -----------
        db : Dict[str, Any]
            The keys database to save

        Raises:
        -------
        KeyStoreError
            If there's an error encrypting or saving the database
        """
        try:
            # Get master key
            master_key = self._get_master_key()

            # Encrypt the database
            encrypted = self.provider.encrypt_data(
                data=json.dumps(db),
                key=master_key
            )

            # Save the encrypted database
            with open(self.keys_db_path, 'w', encoding='utf-8') as f:
                json.dump(encrypted, f) # type: ignore

            # Set secure permissions
            os.chmod(self.keys_db_path, stat.S_IRUSR | stat.S_IWUSR)

        except Exception as e:
            raise KeyStoreError(f"Error saving keys database: {e}")

    def store_task_key(self, task_id: str, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store an encryption key for a specific task.

        Parameters:
        -----------
        task_id : str
            Identifier for the task
        key : str
            Encryption key to store
        metadata : Dict[str, Any], optional
            Additional metadata about the key

        Raises:
        -------
        TaskKeyError
            If there's an error storing the key
        """
        try:
            # Load the keys database
            db = self._load_keys_db()

            # Add the new key with metadata
            db.setdefault("keys", {})[task_id] = {
                "key": key,
                "created": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "metadata": metadata or {}
            }

            # Save the updated database
            self._save_keys_db(db)

            # Log the key storage
            log_key_access("store", task_id, "success")

            logger.info(f"Stored encryption key for task '{task_id}'")

        except Exception as e:
            log_key_access("store", task_id, "failure", metadata={"error": str(e)})
            raise TaskKeyError(f"Error storing key for task '{task_id}': {e}")

    def load_task_key(self, task_id: str) -> str:
        """
        Load an encryption key for a specific task.

        Parameters:
        -----------
        task_id : str
            Identifier for the task

        Returns:
        --------
        str
            The encryption key

        Raises:
        -------
        TaskKeyError
            If the key doesn't exist or there's an error loading it
        """
        try:
            # Load the keys database
            db = self._load_keys_db()

            # Check if the key exists
            if task_id not in db.get("keys", {}):
                raise TaskKeyError(f"No key found for task '{task_id}'")

            # Update last used timestamp
            db["keys"][task_id]["last_used"] = datetime.now().isoformat()

            # Save the updated database
            self._save_keys_db(db)

            # Log the key access
            log_key_access("load", task_id, "success")

            return db["keys"][task_id]["key"]

        except TaskKeyError:
            log_key_access("load", task_id, "failure", metadata={"error": "Key not found"})
            raise
        except Exception as e:
            log_key_access("load", task_id, "failure", metadata={"error": str(e)})
            raise TaskKeyError(f"Error loading key for task '{task_id}': {e}")

    def delete_task_key(self, task_id: str) -> None:
        """
        Delete an encryption key for a specific task.

        Parameters:
        -----------
        task_id : str
            Identifier for the task

        Raises:
        -------
        TaskKeyError
            If the key doesn't exist or there's an error deleting it
        """
        try:
            # Load the keys database
            db = self._load_keys_db()

            # Check if the key exists
            if task_id not in db.get("keys", {}):
                raise TaskKeyError(f"No key found for task '{task_id}'")

            # Delete the key
            del db["keys"][task_id]

            # Save the updated database
            self._save_keys_db(db)

            # Log the key deletion
            log_key_access("delete", task_id, "success")

            logger.info(f"Deleted encryption key for task '{task_id}'")

        except TaskKeyError:
            log_key_access("delete", task_id, "failure", metadata={"error": "Key not found"})
            raise
        except Exception as e:
            log_key_access("delete", task_id, "failure", metadata={"error": str(e)})
            raise TaskKeyError(f"Error deleting key for task '{task_id}': {e}")

    def list_task_keys(self) -> List[Dict[str, Any]]:
        """
        List all stored task keys with metadata.

        Returns:
        --------
        List[Dict[str, Any]]
            List of task keys with metadata (excluding the actual keys)

        Raises:
        -------
        KeyStoreError
            If there's an error listing the keys
        """
        try:
            # Load the keys database
            db = self._load_keys_db()

            # Create a list of tasks with metadata (excluding the actual keys)
            tasks = []
            for task_id, task_data in db.get("keys", {}).items():
                tasks.append({
                    "task_id": task_id,
                    "created": task_data.get("created", ""),
                    "last_used": task_data.get("last_used", ""),
                    "metadata": task_data.get("metadata", {})
                })

            return tasks

        except Exception as e:
            raise KeyStoreError(f"Error listing task keys: {e}")

    def generate_task_key(self, task_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate and store a new random encryption key for a task.

        Parameters:
        -----------
        task_id : str
            Identifier for the task
        metadata : Dict[str, Any], optional
            Additional metadata about the key

        Returns:
        --------
        str
            The generated encryption key

        Raises:
        -------
        TaskKeyError
            If there's an error generating or storing the key
        """
        try:
            # Generate a random key
            import os
            key_bytes = os.urandom(MASTER_KEY_LENGTH)
            key = base64.b64encode(key_bytes).decode('utf-8')

            # Store the key
            self.store_task_key(task_id, key, metadata)

            return key

        except Exception as e:
            raise TaskKeyError(f"Error generating key for task '{task_id}': {e}")

    def is_master_key_exposed(self) -> bool:
        """
        Check if the master key has insecure permissions.

        Returns:
        --------
        bool
            True if the master key has insecure permissions, False otherwise
        """
        try:
            # Check if the master key file exists
            if not self.master_key_path.exists():
                return False

            # Check file permissions
            mode = os.stat(self.master_key_path).st_mode

            # Check if group or others have read permissions
            return bool(mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH))

        except Exception as e:
            logger.warning(f"Error checking master key permissions: {e}")
            return True  # Assume it's exposed if we can't check


def get_key_for_task(task_id: str) -> str:
    """
    Get an encryption key for a specific task.

    Loads the task key from the key store, or generates a new one if it doesn't exist.

    Parameters:
    -----------
    task_id : str
        Identifier for the task

    Returns:
    --------
    str
        The encryption key

    Raises:
    -------
    KeyStoreError
        If there's an error accessing the key store
    """
    key_store = EncryptedKeyStore()

    try:
        # Try to load the key
        return key_store.load_task_key(task_id)
    except TaskKeyError:
        # Generate a new key if it doesn't exist
        return key_store.generate_task_key(task_id)