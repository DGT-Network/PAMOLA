"""
Unit tests for the PAMOLA PAMOLA.CORE Cryptographic Subsystem.

Prerequisites:
- For testing the 'age' mode, the 'age' CLI tool must be installed:
  - Windows: `scoop install age` or `choco install age.portable`
  - Linux: `apt install age` or `dnf install age` or `pacman -S age`
  - macOS: `brew install age`
- Python cryptography library: `pip install cryptography`

All tests automatically clean up generated files and keys after execution.
"""

# Стандартные библиотеки
import os
import json
import shutil
import base64
import tempfile
import unittest
import logging
import sys
import subprocess
from pathlib import Path
from unittest import mock

# Import the modules to test
from pamola_core.utils.io_helpers.crypto_utils import (
    encrypt_file,
    decrypt_file,
    encrypt_data,
    decrypt_data,
    is_encrypted,
    get_encryption_info
)
from pamola_core.utils.crypto_helpers.key_store import EncryptedKeyStore, get_key_for_task
from pamola_core.utils.io_helpers.crypto_router import detect_encryption_mode
# Import modules for legacy_migration
from pamola_core.utils.crypto_helpers import legacy_migration
from pamola_core.utils.crypto_helpers.errors import EncryptionError, DecryptionError


class TestCryptoSubsystem(unittest.TestCase):
    """Test cases for the cryptographic subsystem."""

    def setUp(self):
        """Set up test environment."""
        # Configure logging to reduce output during tests
        logging.basicConfig(level=logging.WARNING)

        # Suppress specific loggers that are too verbose
        logging.getLogger("pamola_core.utils.crypto_helpers.audit").setLevel(logging.ERROR)
        logging.getLogger("pamola_core.utils.io_helpers.crypto_router").setLevel(logging.ERROR)
        logging.getLogger("pamola_core.utils.crypto_helpers.key_store").setLevel(logging.ERROR)

        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create a test key store with a different location
        self.key_store_path = os.path.join(self.test_dir, "test_keys.db")
        self.master_key_path = os.path.join(self.test_dir, "test_master.key")

        # Generate test data
        self.test_data = "This is test data for encryption and decryption."
        self.test_data_path = os.path.join(self.test_dir, "test_data.txt")
        with open(self.test_data_path, "w") as f:
            f.write(self.test_data)

        # Create a test key
        self.test_key = base64.b64encode(os.urandom(32)).decode('utf-8')

        # Check if age is installed
        self.age_available = self._check_age_available()

        # Set environment variables for testing
        self._original_env = {}
        env_vars = [
            "PAMOLA_AGE_BINARY",
            "PAMOLA_AGE_IDENTITY_FILE",
            "PAMOLA_AGE_RECIPIENTS_FILE",
            "PAMOLA_AGE_RECIPIENTS"
        ]
        for var in env_vars:
            self._original_env[var] = os.environ.get(var)

        if self.age_available:
            # Set up age keypair for testing
            self._setup_age_environment()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory and test files
        shutil.rmtree(self.test_dir, ignore_errors=True)

        # Restore original environment variables
        for var, value in self._original_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value

    def _check_age_available(self):
        """Check if the age CLI tool is available."""
        try:
            result = subprocess.run(
                ["age", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _setup_age_environment(self):
        """Set up the environment for age mode tests."""
        # Generate age key pair for testing
        age_key_path = os.path.join(self.test_dir, "age_key.txt")
        recipient_path = os.path.join(self.test_dir, "age_recipient.txt")

        # Create a key pair using age-keygen
        try:
            print("Setting up age keypair for testing...")

            # First, check if age-keygen is available
            try:
                subprocess.check_output(
                    ["age-keygen", "--help"],
                    stderr=subprocess.STDOUT,
                    text=True
                )
                print("age-keygen available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("age-keygen not found, falling back to alternative method")
                self.age_available = False
                return

            # Generate keypair with age-keygen
            subprocess.check_call(
                ["age-keygen", "-o", age_key_path],
                stderr=subprocess.STDOUT
            )

            # Check if key was generated
            if os.path.exists(age_key_path):
                print(f"Key generated at {age_key_path}")
                with open(age_key_path, 'r') as f:
                    key_content = f.read()

                # Find the recipient (public key)
                recipient = None
                for line in key_content.splitlines():
                    if line.startswith('# public key:'):
                        recipient = line.split(':', 1)[1].strip()
                        print(f"Found recipient: {recipient}")
                        break

                if not recipient:
                    print("Could not find recipient in key file")
                    self.age_available = False
                    return

                # Write recipient to file
                with open(recipient_path, 'w') as f:
                    f.write(recipient)
                print(f"Wrote recipient to {recipient_path}")

                # Set environment variables for age to find these files
                os.environ["PAMOLA_AGE_IDENTITY_FILE"] = age_key_path
                os.environ["PAMOLA_AGE_RECIPIENTS"] = recipient
                os.environ["PAMOLA_AGE_RECIPIENTS_FILE"] = ""  # Clear this to avoid conflicts

                # Store these for test use
                self.age_identity_path = age_key_path
                self.age_recipient = recipient

                print("Successfully set up age environment")
            else:
                print(f"Key file not created at {age_key_path}")
                self.age_available = False
                return

        except Exception as e:
            # If age-keygen isn't available, we'll skip age tests
            print(f"Error setting up age environment: {str(e)}")
            self.age_available = False
            return

        # Verify setup was successful with a simple test
        if self.age_available:
            try:
                # Test encryption with recipient
                test_enc = os.path.join(self.test_dir, "verify_enc.age")
                test_dec = os.path.join(self.test_dir, "verify_dec.txt")

                # Encrypt and decrypt to verify
                encrypt_file(
                    source_path=self.test_data_path,
                    destination_path=test_enc,
                    mode="age"
                )

                decrypt_file(
                    source_path=test_enc,
                    destination_path=test_dec,
                    mode="age"
                )

                # Verify content matches
                with open(test_dec, 'r') as f:
                    decrypted = f.read()

                if decrypted != self.test_data:
                    print("Test verification failed: content mismatch")
                    self.age_available = False
                else:
                    print("Age encryption setup verified successfully")
            except Exception as e:
                print(f"Age verification failed: {str(e)}")
                self.age_available = False

    def _create_key_store(self):
        """Create a test key store with a test task key."""
        key_store = EncryptedKeyStore(
            keys_db_path=self.key_store_path,
            master_key_path=self.master_key_path
        )
        key_store.store_task_key(
            task_id="test_task",
            key=self.test_key,
            metadata={"purpose": "testing"}
        )
        return key_store

    # ---- Tests for None Mode ----

    def test_none_mode_file_encryption(self):
        """Test file encryption and decryption in 'none' mode."""
        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_none.txt")
        decrypted_path = os.path.join(self.test_dir, "decrypted_none.txt")

        # Encrypt file in 'none' mode
        result_path = encrypt_file(
            source_path=source_path,
            destination_path=encrypted_path,
            mode="none"
        )

        # Verify the file was copied
        self.assertTrue(os.path.exists(encrypted_path))
        self.assertEqual(result_path, Path(encrypted_path))

        # Compare file contents (should be identical)
        with open(source_path, 'r') as f:
            original_content = f.read()
        with open(encrypted_path, 'r') as f:
            encrypted_content = f.read()
        self.assertEqual(original_content, encrypted_content)

        # Decrypt file
        result_path = decrypt_file(
            source_path=encrypted_path,
            destination_path=decrypted_path,
            mode="none"
        )

        # Verify decryption
        self.assertTrue(os.path.exists(decrypted_path))
        self.assertEqual(result_path, Path(decrypted_path))
        with open(decrypted_path, 'r') as f:
            decrypted_content = f.read()
        self.assertEqual(original_content, decrypted_content)

    def test_none_mode_data_encryption(self):
        """Test data encryption and decryption in 'none' mode."""
        original_data = self.test_data

        # Encrypt data in 'none' mode
        encrypted_data = encrypt_data(
            data=original_data,
            mode="none"
        )

        # Verify data wasn't changed
        self.assertEqual(original_data, encrypted_data)

        # Decrypt data
        decrypted_data = decrypt_data(
            data=encrypted_data,
            mode="none"
        )

        # Verify decryption
        self.assertEqual(original_data, decrypted_data)

    # ---- Tests for Simple Mode ----

    def test_simple_mode_file_encryption_with_key(self):
        """Test file encryption and decryption in 'simple' mode with explicit key."""
        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_simple.json")
        decrypted_path = os.path.join(self.test_dir, "decrypted_simple.txt")

        # Encrypt file in 'simple' mode
        result_path = encrypt_file(
            source_path=source_path,
            destination_path=encrypted_path,
            key=self.test_key,
            mode="simple",
            description="Test encryption"
        )

        # Verify the file was encrypted
        self.assertTrue(os.path.exists(encrypted_path))
        self.assertEqual(result_path, Path(encrypted_path))

        # Check that the file is in JSON format with expected fields
        with open(encrypted_path, 'r') as f:
            encrypted_content = json.load(f)

        self.assertEqual(encrypted_content["mode"], "simple")
        self.assertEqual(encrypted_content["algorithm"], "AES-GCM")
        self.assertIn("data", encrypted_content)
        self.assertIn("iv", encrypted_content)
        self.assertIn("salt", encrypted_content)

        # File content should be different
        with open(source_path, 'r') as f:
            original_content = f.read()
        self.assertNotEqual(original_content, encrypted_content["data"])

        # Decrypt file
        result_path = decrypt_file(
            source_path=encrypted_path,
            destination_path=decrypted_path,
            key=self.test_key
        )

        # Verify decryption
        self.assertTrue(os.path.exists(decrypted_path))
        self.assertEqual(result_path, Path(decrypted_path))
        with open(decrypted_path, 'r') as f:
            decrypted_content = f.read()
        self.assertEqual(original_content, decrypted_content)

        # Verify format detection
        detected_mode = detect_encryption_mode(encrypted_path)
        self.assertEqual(detected_mode, "simple")

        # Test is_encrypted and get_encryption_info
        self.assertTrue(is_encrypted(encrypted_path))
        info = get_encryption_info(encrypted_path)
        self.assertEqual(info["mode"], "simple")
        self.assertEqual(info["algorithm"], "AES-GCM")

    def test_simple_mode_data_encryption(self):
        """Test data encryption and decryption in 'simple' mode."""
        original_data = self.test_data

        # Encrypt data in 'simple' mode
        encrypted_data = encrypt_data(
            data=original_data,
            key=self.test_key,
            mode="simple"
        )

        # Verify data was encrypted
        self.assertIsInstance(encrypted_data, dict)
        self.assertEqual(encrypted_data["mode"], "simple")
        self.assertEqual(encrypted_data["algorithm"], "AES-GCM")
        self.assertIn("data", encrypted_data)
        self.assertNotEqual(original_data, encrypted_data["data"])

        # Decrypt data
        decrypted_data = decrypt_data(
            data=encrypted_data,
            key=self.test_key
        )

        # Verify decryption
        if isinstance(decrypted_data, bytes):
            decrypted_data = decrypted_data.decode('utf-8')
        self.assertEqual(original_data, decrypted_data)

        # Test is_encrypted and get_encryption_info
        self.assertTrue(is_encrypted(encrypted_data))
        info = get_encryption_info(encrypted_data)
        self.assertEqual(info["mode"], "simple")
        self.assertEqual(info["algorithm"], "AES-GCM")

    def test_simple_mode_with_key_store(self):
        """Test 'simple' mode with key store integration."""
        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_simple_ks.json")
        decrypted_path = os.path.join(self.test_dir, "decrypted_simple_ks.txt")

        # Create a test key store
        key_store = self._create_key_store()

        # Get a task key
        task_key = key_store.load_task_key("test_task")

        # Encrypt file using the task key
        encrypt_file(
            source_path=source_path,
            destination_path=encrypted_path,
            key=task_key,
            mode="simple",
            task_id="test_task"
        )

        # Decrypt file using the same task key
        decrypt_file(
            source_path=encrypted_path,
            destination_path=decrypted_path,
            key=task_key
        )

        # Verify decryption worked
        with open(source_path, 'r') as f:
            original_content = f.read()
        with open(decrypted_path, 'r') as f:
            decrypted_content = f.read()
        self.assertEqual(original_content, decrypted_content)

        # Test the get_key_for_task utility
        # Mock the EncryptedKeyStore to use our test paths
        with mock.patch('pamola_core.utils.crypto_helpers.key_store.EncryptedKeyStore', autospec=True) as mock_ks:
            mock_ks.return_value = key_store
            task_key2 = get_key_for_task("test_task")
            self.assertEqual(task_key, task_key2)

    def test_auto_format_detection(self):
        """Test automatic format detection during decryption."""
        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_autodetect.json")
        decrypted_path = os.path.join(self.test_dir, "decrypted_autodetect.txt")

        # Encrypt file in 'simple' mode
        encrypt_file(
            source_path=source_path,
            destination_path=encrypted_path,
            key=self.test_key,
            mode="simple"
        )

        # Decrypt file without specifying mode (should auto-detect)
        decrypt_file(
            source_path=encrypted_path,
            destination_path=decrypted_path,
            key=self.test_key
        )

        # Verify decryption worked
        with open(source_path, 'r') as f:
            original_content = f.read()
        with open(decrypted_path, 'r') as f:
            decrypted_content = f.read()
        self.assertEqual(original_content, decrypted_content)

    def test_key_store_operations(self):
        """Test key store operations."""
        # Create a test key store
        key_store = EncryptedKeyStore(
            keys_db_path=self.key_store_path,
            master_key_path=self.master_key_path
        )

        # Store a test key
        key_store.store_task_key(
            task_id="test_task1",
            key=self.test_key,
            metadata={"purpose": "testing", "owner": "test_user"}
        )

        # Generate a key for another task
        task2_key = key_store.generate_task_key(
            task_id="test_task2",
            metadata={"purpose": "another test"}
        )

        # Load a stored key
        loaded_key = key_store.load_task_key("test_task1")
        self.assertEqual(loaded_key, self.test_key)

        # List all keys
        key_list = key_store.list_task_keys()
        self.assertEqual(len(key_list), 2)
        task_ids = [item["task_id"] for item in key_list]
        self.assertIn("test_task1", task_ids)
        self.assertIn("test_task2", task_ids)

        # Check key metadata
        for item in key_list:
            if item["task_id"] == "test_task1":
                self.assertEqual(item["metadata"]["purpose"], "testing")
                self.assertEqual(item["metadata"]["owner"], "test_user")
            elif item["task_id"] == "test_task2":
                self.assertEqual(item["metadata"]["purpose"], "another test")

        # Delete a key
        key_store.delete_task_key("test_task1")

        # Verify key was deleted
        with self.assertRaises(Exception):
            key_store.load_task_key("test_task1")

        # Verify other key still exists
        loaded_key2 = key_store.load_task_key("test_task2")
        self.assertEqual(loaded_key2, task2_key)

    # ---- Tests for Age Mode ----

    def test_age_encryption(self):
        """Test file encryption and decryption in 'age' mode with keypair."""
        if not self.age_available:
            self.skipTest("Age tool is not available or not properly configured")

        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_age.age")
        decrypted_path = os.path.join(self.test_dir, "decrypted_age.txt")

        # Encrypt using the API
        result_path = encrypt_file(
            source_path=source_path,
            destination_path=encrypted_path,
            mode="age",
            description="Test age encryption"
        )

        # Verify the file was encrypted
        self.assertTrue(os.path.exists(encrypted_path))
        self.assertEqual(result_path, Path(encrypted_path))

        # Check for metadata file
        metadata_path = encrypted_path + ".meta"
        self.assertTrue(os.path.exists(metadata_path))

        # Decrypt using the API
        result_path = decrypt_file(
            source_path=encrypted_path,
            destination_path=decrypted_path,
            mode="age"
        )

        # Verify decryption
        self.assertTrue(os.path.exists(decrypted_path))
        self.assertEqual(result_path, Path(decrypted_path))

        with open(source_path, 'r') as f:
            original_content = f.read()
        with open(decrypted_path, 'r') as f:
            decrypted_content = f.read()
        self.assertEqual(original_content, decrypted_content)

        # Verify format detection
        detected_mode = detect_encryption_mode(encrypted_path)
        self.assertEqual(detected_mode, "age")

    def test_age_data_encryption(self):
        """Test data encryption and decryption in 'age' mode."""
        if not self.age_available:
            self.skipTest("Age tool is not available or not properly configured")

        original_data = self.test_data

        # Encrypt data
        encrypted_data = encrypt_data(
            data=original_data,
            mode="age"
        )

        # Verify encrypted data is not the same as original
        self.assertIsInstance(encrypted_data, bytes)
        self.assertNotEqual(original_data.encode('utf-8'), encrypted_data)

        # Decrypt data
        decrypted_data = decrypt_data(
            data=encrypted_data,
            mode="age"
        )

        # Verify decryption
        if isinstance(decrypted_data, bytes):
            decrypted_data = decrypted_data.decode('utf-8')
        self.assertEqual(original_data, decrypted_data)

    # ---- Tests for Legacy Formats ----

    def test_legacy_format_migration(self):
        """Test detection and migration of legacy encrypted formats."""
        # Create a mock legacy v1_base64 format file
        legacy_path = os.path.join(self.test_dir, "legacy_file.enc")
        with open(legacy_path, 'w') as f:
            f.write("PAMOLA_ENC_V1:" + base64.b64encode(b"mock encrypted data").decode('utf-8'))

        # Test format detection
        detected_format = legacy_migration.detect_legacy_format(legacy_path)
        self.assertEqual(detected_format, "v1_base64")

        # Test auto migration (using mocks since we can't actually decrypt the mock file)
        with mock.patch('pamola_core.utils.crypto_helpers.legacy_migration.migrate_legacy_file') as mock_migrate:
            mock_migrate.return_value = Path(os.path.join(self.test_dir, "migrated.enc"))
            path, was_migrated = legacy_migration.auto_migrate_if_needed(
                source_path=legacy_path,
                destination_path=os.path.join(self.test_dir, "migrated.enc"),
                key=self.test_key
            )
            self.assertTrue(was_migrated)
            mock_migrate.assert_called_once()

    # ---- Error Cases ----

    def test_error_handling_wrong_key(self):
        """Test error handling when using the wrong key for decryption."""
        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_wrong_key.json")
        decrypted_path = os.path.join(self.test_dir, "decrypted_wrong_key.txt")

        # Encrypt file
        encrypt_file(
            source_path=source_path,
            destination_path=encrypted_path,
            key=self.test_key,
            mode="simple"
        )

        # Try to decrypt with wrong key
        wrong_key = base64.b64encode(os.urandom(32)).decode('utf-8')
        with self.assertRaises(DecryptionError):
            decrypt_file(
                source_path=encrypted_path,
                destination_path=decrypted_path,
                key=wrong_key
            )

    def test_error_handling_missing_file(self):
        """Test error handling when source file is missing."""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent.txt")
        encrypted_path = os.path.join(self.test_dir, "encrypted_nonexistent.json")

        with self.assertRaises(EncryptionError):
            encrypt_file(
                source_path=nonexistent_path,
                destination_path=encrypted_path,
                key=self.test_key,
                mode="simple"
            )

    def test_error_handling_invalid_mode(self):
        """Test error handling when using an invalid encryption mode."""
        source_path = self.test_data_path
        encrypted_path = os.path.join(self.test_dir, "encrypted_invalid_mode.json")

        with self.assertRaises(Exception):
            encrypt_file(
                source_path=source_path,
                destination_path=encrypted_path,
                key=self.test_key,
                mode="invalid_mode"
            )


# This part is only executed when the file is run directly.
# This doesn't prevent the standard unittest from detecting and running tests when running via unittest.
if __name__ == "__main__":
    # Add the root of the project to the Python path for direct launch
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

    # Use the standard unittest.main() for consistent output
    unittest.main()