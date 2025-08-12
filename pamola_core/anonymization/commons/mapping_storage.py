"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Secure Mapping Storage for Pseudonymization
Package:       pamola_core.anonymization.commons
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-20
License:       BSD 3-Clause

Description:
    This module provides secure storage and retrieval of pseudonymization mappings
    with encryption support. It handles the persistence of original-to-pseudonym
    mappings for reversible pseudonymization operations, ensuring that mappings
    are protected at rest using AES-256-GCM encryption.

Key Features:
    - Encrypted storage using AES-256-GCM
    - Support for CSV and JSON formats
    - Atomic file operations to prevent corruption
    - Optional backup before updates
    - Thread-safe operations
    - Comprehensive error handling
    - Metadata tracking (creation time, last update, etc.)

Security Considerations:
    - All mappings are encrypted at rest
    - Encryption keys are never stored in plaintext
    - Atomic writes prevent partial updates
    - File permissions are set to owner-only access

Dependencies:
    - pamola_core.utils.crypto_helpers.pseudonymization: For AES-256-GCM encryption
    - pamola_core.utils.ops.op_data_writer: For consistent file operations
    - csv: For CSV format support
    - json: For JSON format support
    - threading: For thread safety
"""

import csv
import io
import json
import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from pamola_core.utils.crypto_helpers.pseudonymization import MappingEncryption

# Configure module logger
logger = logging.getLogger(__name__)


class MappingStorageError(Exception):
    """Base exception for mapping storage errors."""
    pass


class MappingStorage:
    """
    Manages encrypted storage of pseudonymization mappings.

    This class provides secure persistence for mappings between original
    values and their pseudonyms, supporting both CSV and JSON formats.
    All data is encrypted using AES-256-GCM before storage.

    Attributes:
        mapping_file: Path to the encrypted mapping file
        format: Storage format ("csv" or "json")
        backup_on_update: Whether to create backups before updates

    Thread Safety:
        All public methods are thread-safe and can be called concurrently.
    """

    def __init__(self,
                 mapping_file: Path,
                 encryption_key: bytes,
                 format: str = "csv",
                 backup_on_update: bool = True,
                 create_if_missing: bool = True):
        """
        Initialize mapping storage.

        Args:
            mapping_file: Path to mapping file
            encryption_key: 256-bit encryption key
            format: Storage format ("csv" or "json")
            backup_on_update: Whether to backup before updates
            create_if_missing: Create empty mapping if file doesn't exist

        Raises:
            ValueError: If format is invalid or key size is wrong
            MappingStorageError: If initialization fails
        """
        # Validate format
        if format not in ["csv", "json"]:
            raise ValueError(f"Invalid format: {format}. Must be 'csv' or 'json'")

        # Initialize attributes
        self.mapping_file = Path(mapping_file)
        self.format = format
        self.backup_on_update = backup_on_update
        self._lock = threading.RLock()
        self.logger = logger

        # Initialize encryption
        try:
            self._encryptor = MappingEncryption(encryption_key)
        except Exception as e:
            raise MappingStorageError(f"Failed to initialize encryption: {e}")

        # Ensure directory exists
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)

        # Create empty file if needed
        if create_if_missing and not self.mapping_file.exists():
            self._create_empty_mapping()

    def _create_empty_mapping(self) -> None:
        """Create an empty encrypted mapping file."""
        try:
            self.save({})
            self.logger.info(f"Created empty mapping file: {self.mapping_file}")
        except Exception as e:
            raise MappingStorageError(f"Failed to create empty mapping: {e}")

    def load(self) -> Dict[str, Any]:
        """
        Load and decrypt mapping from file.

        Returns:
            Dictionary mapping original values to pseudonyms

        Raises:
            MappingStorageError: If loading or decryption fails
        """
        with self._lock:
            if not self.mapping_file.exists():
                self.logger.warning(f"Mapping file not found: {self.mapping_file}")
                return {}

            try:
                # Read encrypted file
                with open(self.mapping_file, 'rb') as f:
                    encrypted_data = f.read()

                # Handle empty file
                if not encrypted_data:
                    self.logger.warning("Empty mapping file")
                    return {}

                # Decrypt
                try:
                    decrypted_data = self._encryptor.decrypt(encrypted_data)
                except Exception as e:
                    raise MappingStorageError(f"Decryption failed: {e}")

                # Parse based on format
                if self.format == "csv":
                    return self._parse_csv(decrypted_data)
                else:  # json
                    return self._parse_json(decrypted_data)

            except MappingStorageError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to load mapping: {e}")
                raise MappingStorageError(f"Failed to load mapping: {e}")

    def save(self, mapping: Dict[str, Any]) -> None:
        """
        Encrypt and save mapping atomically.

        Args:
            mapping: Dictionary mapping original values to pseudonyms

        Raises:
            MappingStorageError: If saving fails
        """
        with self._lock:
            try:
                # Create backup if requested and file exists
                if self.backup_on_update and self.mapping_file.exists():
                    self._create_backup()

                # Serialize mapping
                if self.format == "csv":
                    plaintext = self._serialize_csv(mapping)
                else:  # json
                    plaintext = self._serialize_json(mapping)

                # Encrypt
                encrypted_data = self._encryptor.encrypt(plaintext)

                # Atomic write
                temp_path = self.mapping_file.with_suffix('.tmp')
                try:
                    # Write to temporary file
                    with open(temp_path, 'wb') as f:
                        f.write(encrypted_data)
                        f.flush()
                        os.fsync(f.fileno())

                    # Set secure permissions (owner read/write only)
                    if os.name != 'nt':  # Unix-like systems
                        os.chmod(temp_path, 0o600)

                    # Atomic rename
                    temp_path.replace(self.mapping_file)

                    self.logger.debug(f"Saved {len(mapping)} mappings to {self.mapping_file}")

                except Exception as e:
                    # Clean up temporary file on error
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

            except Exception as e:
                raise MappingStorageError(f"Failed to save mapping: {e}")

    def update(self, new_mappings: Dict[str, str]) -> Dict[str, str]:
        """
        Update existing mappings with new entries.

        Args:
            new_mappings: New mappings to add/update

        Returns:
            Complete updated mapping dictionary

        Raises:
            MappingStorageError: If update fails
        """
        with self._lock:
            try:
                # Load existing mappings
                existing = self.load()

                # Check for conflicts
                conflicts = []
                for key, value in new_mappings.items():
                    if key in existing and existing[key] != value:
                        conflicts.append({
                            "key": key,
                            "existing": existing[key],
                            "new": value
                        })

                if conflicts:
                    self.logger.warning(
                        f"Found {len(conflicts)} mapping conflicts. "
                        "Existing values will be overwritten."
                    )

                # Update mappings
                existing.update(new_mappings)

                # Save updated mappings
                self.save(existing)

                return existing

            except Exception as e:
                raise MappingStorageError(f"Failed to update mappings: {e}")

    def _create_backup(self) -> None:
        """Create a backup of the current mapping file."""
        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.mapping_file.with_suffix(f'.bak.{timestamp}')

            # Copy file
            shutil.copy2(self.mapping_file, backup_path)

            # Set secure permissions on backup
            if os.name != 'nt':  # Unix-like systems
                os.chmod(backup_path, 0o600)

            self.logger.info(f"Created backup: {backup_path}")

            # Optional: Clean up old backups (keep last 5)
            self._cleanup_old_backups()

        except Exception as e:
            # Log but don't fail the operation
            self.logger.warning(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Clean up old backup files, keeping only the most recent ones."""
        try:
            # Find all backup files
            pattern = f"{self.mapping_file.stem}.bak.*"
            backups = list(self.mapping_file.parent.glob(pattern))

            # Sort by modification time
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old backups
            for backup in backups[keep_count:]:
                backup.unlink()
                self.logger.debug(f"Removed old backup: {backup}")

        except Exception as e:
            # Log but don't fail
            self.logger.debug(f"Failed to clean up old backups: {e}")

    def _parse_csv(self, data: bytes) -> Dict[str, str]:
        """Parse CSV format mapping data."""
        try:
            # Decode bytes to string
            text = data.decode('utf-8')

            # Parse CSV
            reader = csv.DictReader(io.StringIO(text))

            # Build mapping dictionary
            mapping = {}
            for row in reader:
                if 'original' in row and 'pseudonym' in row:
                    mapping[row['original']] = row['pseudonym']
                else:
                    self.logger.warning(f"Invalid CSV row: {row}")

            return mapping

        except Exception as e:
            raise MappingStorageError(f"Failed to parse CSV: {e}")

    def _parse_json(self, data: bytes) -> Dict[str, str]:
        """Parse JSON format mapping data."""
        try:
            # Decode and parse JSON
            text = data.decode('utf-8')
            parsed = json.loads(text)

            # Handle both direct mapping and metadata format
            if isinstance(parsed, dict):
                if '_metadata' in parsed and 'mappings' in parsed:
                    # Metadata format
                    return parsed['mappings']
                else:
                    # Direct mapping format
                    return parsed
            else:
                raise MappingStorageError("Invalid JSON structure")

        except json.JSONDecodeError as e:
            raise MappingStorageError(f"Invalid JSON: {e}")
        except Exception as e:
            raise MappingStorageError(f"Failed to parse JSON: {e}")

    def _serialize_csv(self, mapping: Dict[str, str]) -> bytes:
        """Serialize mapping to CSV format."""
        try:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['original', 'pseudonym'])
            writer.writeheader()

            for original, pseudonym in sorted(mapping.items()):
                writer.writerow({
                    'original': original,
                    'pseudonym': pseudonym
                })

            return output.getvalue().encode('utf-8')

        except Exception as e:
            raise MappingStorageError(f"Failed to serialize CSV: {e}")

    def _serialize_json(self, mapping: Dict[str, str]) -> bytes:
        """Serialize mapping to JSON format with metadata."""
        try:
            # Include metadata for better tracking
            data = {
                '_metadata': {
                    'version': '1.0',
                    'created': datetime.now().isoformat(),
                    'count': len(mapping),
                    'format': 'json'
                },
                'mappings': mapping
            }

            # Serialize with sorted keys for consistency
            return json.dumps(data, indent=2, sort_keys=True).encode('utf-8')

        except Exception as e:
            raise MappingStorageError(f"Failed to serialize JSON: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get mapping file metadata.

        Returns:
            Dictionary containing:
                - exists: Whether file exists
                - size_bytes: File size in bytes
                - modified: Last modification timestamp
                - format: Storage format
                - count: Number of mappings (requires loading)
        """
        metadata = {
            "exists": self.mapping_file.exists(),
            "format": self.format,
            "path": str(self.mapping_file)
        }

        if metadata["exists"]:
            try:
                stat = self.mapping_file.stat()
                metadata.update({
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })

                # Optionally load to get count (expensive)
                # Commented out to avoid unnecessary decryption
                # mapping = self.load()
                # metadata["count"] = len(mapping)

            except Exception as e:
                self.logger.warning(f"Failed to get file metadata: {e}")

        return metadata

    def validate_mappings(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate mapping consistency and detect issues.

        Args:
            mapping: Mapping dictionary to validate

        Returns:
            Dictionary with validation results:
                - valid: Whether mapping is valid
                - duplicate_values: List of pseudonyms mapped to multiple originals
                - empty_keys: Number of empty original values
                - empty_values: Number of empty pseudonyms
        """
        validation = {
            "valid": True,
            "duplicate_values": [],
            "empty_keys": 0,
            "empty_values": 0,
            "total_mappings": len(mapping)
        }

        # Check for empty keys/values
        for key, value in mapping.items():
            if not key:
                validation["empty_keys"] += 1
                validation["valid"] = False
            if not value:
                validation["empty_values"] += 1
                validation["valid"] = False

        # Check for duplicate pseudonyms (reverse mapping)
        reverse_mapping = {}
        for original, pseudonym in mapping.items():
            if pseudonym in reverse_mapping:
                if pseudonym not in validation["duplicate_values"]:
                    validation["duplicate_values"].append(pseudonym)
                validation["valid"] = False
            else:
                reverse_mapping[pseudonym] = original

        return validation


# Module metadata
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Define explicit exports
__all__ = [
    'MappingStorage',
    'MappingStorageError'
]