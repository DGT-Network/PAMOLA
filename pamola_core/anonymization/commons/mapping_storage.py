"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        Secure Mapping Storage for Pseudonymization
Package:       pamola_core.anonymization.commons
Version:       2.0.0
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
    - Support for CSV and JSON payload formats
    - Incremental (append-only) persistence: cost of a save is proportional to
      the number of NEW mappings, not to the size of the whole table
    - Fail-closed loading: a wrong key, a modified, truncated or reordered
      file raises ``MappingIntegrityError``; no partial mapping is ever returned
    - Atomic snapshot writes, fsync'd appends
    - Optional backup before the first modification
    - Thread-safe operations
    - Metadata tracking (e.g. last sequential pseudonym)

On-disk format (version 2)
    ::

        MAGIC (11 bytes: b"PAMOLA-MAP\\x02")
        frame*   where frame = uint32 big-endian length N, then N bytes of
                 MappingEncryption.encrypt(payload, associated_data=AAD)

        payload = <header JSON line> "\\n" <body>
        header  = {"seq": k, "format": "csv"|"json", "count": n, "metadata": {...}}
        body    = CSV rows ``original,pseudonym`` (with header row) or a JSON
                  object ``{original: pseudonym}``
        AAD     = MAGIC + seq as 8-byte big-endian

    Frame 0 is a snapshot (written atomically by :meth:`MappingStorage.save`);
    every later frame carries only the mappings added since the previous frame.
    The reader requires frames to be numbered 0, 1, 2, ... without gaps (the
    sequence number is authenticated, so frames cannot be dropped from the
    middle, duplicated or reordered undetected), a clean end of file on a frame
    boundary, no source value mapped to two pseudonyms and no pseudonym used by
    two source values. Anything else is reported as ``MappingIntegrityError``.

    Version 1 files (a single AES-GCM blob holding CSV or JSON, written by
    earlier releases) are still readable. The first incremental append to such
    a file converts it to version 2 (after taking a backup).

Consistency model
    * One writer per mapping file. Concurrent processes are not coordinated.
    * ``save()`` replaces the file atomically (write temp file, fsync, rename).
    * ``append()`` adds one frame and fsyncs. A crash during the append leaves a
      truncated final frame, which the next ``load()`` reports as an integrity
      error rather than guessing; restore the ``.bak.*`` copy taken before the
      run if that happens. Mappings of an append that did not return are not
      considered persisted.
    * Truncating the file exactly on a frame boundary (dropping the newest
      frames) is not detectable without external state, same as replacing the
      file with an older copy.

Security Considerations:
    - All mappings are encrypted at rest
    - Encryption keys are never stored in plaintext
    - Every frame uses a fresh random 96-bit nonce
    - File permissions are set to owner-only access where supported

Dependencies:
    - pamola_core.utils.crypto_helpers.pseudonymization: For AES-256-GCM encryption
    - csv, json, struct, threading
"""

import ast
import csv
import io
import json
import logging
import os
import shutil
import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from pamola_core.errors.exceptions import MappingStorageError, InvalidParameterError
from pamola_core.utils.crypto_helpers.pseudonymization import MappingEncryption

# Configure module logger
logger = logging.getLogger(__name__)

#: File signature + format version. Files without it are version 1 (single blob).
MAGIC = b"PAMOLA-MAP\x02"
_LEN = struct.Struct(">I")
_CSV_FIELDS = ("original", "pseudonym")


class MappingIntegrityError(MappingStorageError):
    """The mapping file cannot be trusted: wrong key, modified, truncated or inconsistent."""


def _aad(seq: int) -> bytes:
    return MAGIC + seq.to_bytes(8, "big")


class MappingStorage:
    """
    Manages encrypted storage of pseudonymization mappings.

    This class provides secure persistence for mappings between original
    values and their pseudonyms, supporting both CSV and JSON payload formats.
    All data is encrypted using AES-256-GCM before storage.

    Attributes
    ----------
        mapping_file: Path to the encrypted mapping file
        format: Payload format ("csv" or "json")
        backup_on_update: Whether to create a backup before the first
            modification of an existing file by this instance

    Thread Safety:
        All public methods are thread-safe and can be called concurrently
        within one process.
    """

    def __init__(
        self,
        mapping_file: Path,
        encryption_key: bytes,
        format: str = "csv",
        backup_on_update: bool = True,
        create_if_missing: bool = True,
    ):
        """
        Initialize mapping storage.

        Parameters
        ----------
            mapping_file: Path to mapping file
            encryption_key: 256-bit encryption key
            format: Payload format ("csv" or "json")
            backup_on_update: Whether to backup before the first update
            create_if_missing: Create empty mapping if file doesn't exist

        Raises
        ------
            InvalidParameterError: If format is invalid
            MappingStorageError: If initialization fails
        """
        # Validate format
        if format not in ["csv", "json"]:
            raise InvalidParameterError(
                param_name="format",
                param_value=format,
                reason="must be 'csv' or 'json'",
            )

        # Initialize attributes
        self.mapping_file = Path(mapping_file)
        self.format = format
        self.backup_on_update = backup_on_update
        self._lock = threading.RLock()
        self.logger = logger
        self._backup_done = False
        self._next_seq: Optional[int] = None  # known after load/save/append

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

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, str]:
        """
        Load and decrypt the mapping.

        Returns
        -------
            Dictionary mapping original values to pseudonyms

        Raises
        ------
            MappingIntegrityError: wrong key, or corrupted/modified/truncated file
            MappingStorageError: other I/O failures
        """
        return self.load_with_metadata()[0]

    def load_with_metadata(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Load the mapping together with the metadata stored with it.

        Returns
        -------
            ``(mapping, metadata)``. Metadata is the one stored with the newest
            frame that carried any; ``{}`` if none was stored.

        Raises
        ------
            MappingIntegrityError: wrong key, or corrupted/modified/truncated file
            MappingStorageError: other I/O failures
        """
        with self._lock:
            if not self.mapping_file.exists():
                self.logger.warning(f"Mapping file not found: {self.mapping_file}")
                return {}, {}

            try:
                with open(self.mapping_file, "rb") as f:
                    raw = f.read()
            except OSError as e:
                raise MappingStorageError(f"Failed to read mapping file: {e}")

            if not raw:
                self.logger.warning("Empty mapping file")
                self._next_seq = None
                return {}, {}

            try:
                if raw.startswith(MAGIC):
                    return self._load_v2(raw)
                return self._load_v1(raw)
            except MappingStorageError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to load mapping: {e}")
                raise MappingIntegrityError(
                    f"Failed to load mapping '{self.mapping_file.name}': {e}"
                )

    def _decrypt(self, blob: bytes, aad: Optional[bytes], what: str) -> bytes:
        try:
            return self._encryptor.decrypt(blob, aad)
        except Exception as e:
            raise MappingIntegrityError(
                f"Cannot decrypt {what} of '{self.mapping_file.name}': the "
                "encryption key is wrong, or the file was modified or corrupted "
                f"({e}). No mapping was loaded."
            )

    def _frames(self, raw: bytes) -> Iterator[Tuple[int, bytes]]:
        """Yield ``(seq, blob)`` for each frame; raise on a malformed layout."""
        pos = len(MAGIC)
        seq = 0
        end = len(raw)
        while pos < end:
            if pos + _LEN.size > end:
                raise MappingIntegrityError(
                    f"Mapping file '{self.mapping_file.name}' is truncated inside a "
                    f"frame header (frame {seq}). Restore the latest backup."
                )
            (length,) = _LEN.unpack_from(raw, pos)
            pos += _LEN.size
            if length < 28 or pos + length > end:
                raise MappingIntegrityError(
                    f"Mapping file '{self.mapping_file.name}' is truncated or its "
                    f"frame {seq} has an invalid length. Restore the latest backup."
                )
            yield seq, raw[pos : pos + length]
            pos += length
            seq += 1

    def _load_v2(self, raw: bytes) -> Tuple[Dict[str, str], Dict[str, Any]]:
        mapping: Dict[str, str] = {}
        metadata: Dict[str, Any] = {}
        reverse: Dict[str, str] = {}
        frames = 0
        for seq, blob in self._frames(raw):
            payload = self._decrypt(blob, _aad(seq), f"frame {seq}")
            header, body = self._split_payload(payload, seq)
            if header.get("seq") != seq:
                raise MappingIntegrityError(
                    f"Mapping file '{self.mapping_file.name}' has frames out of "
                    f"order (expected {seq}, found {header.get('seq')})."
                )
            entries = self._parse_body(header.get("format"), body, seq)
            if header.get("count") is not None and header["count"] != len(entries):
                raise MappingIntegrityError(
                    f"Frame {seq} of '{self.mapping_file.name}' announces "
                    f"{header['count']} mappings but holds {len(entries)}."
                )
            for original, pseudonym in entries.items():
                known = mapping.get(original)
                if known is not None and known != pseudonym:
                    raise MappingIntegrityError(
                        f"Source value maps to two different pseudonyms "
                        f"(frame {seq}); the mapping file is inconsistent."
                    )
                owner = reverse.get(pseudonym)
                if owner is not None and owner != original:
                    raise MappingIntegrityError(
                        f"A pseudonym is assigned to two source values "
                        f"(frame {seq}); reversal would be ambiguous."
                    )
                mapping[original] = pseudonym
                reverse[pseudonym] = original
            if header.get("metadata"):
                metadata = header["metadata"]
            frames += 1
        if frames == 0:
            raise MappingIntegrityError(
                f"Mapping file '{self.mapping_file.name}' has a header but no data "
                "frame; it is incomplete."
            )
        self._next_seq = frames
        return mapping, metadata

    def _load_v1(self, raw: bytes) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Read a version 1 file: one AES-GCM blob holding CSV or JSON."""
        data = self._decrypt(raw, None, "the file")
        parsed = (
            self._parse_csv(data) if self.format == "csv" else self._parse_json(data)
        )
        self._next_seq = None
        return self._unwrap_legacy_envelope(parsed)

    def _unwrap_legacy_envelope(
        self, parsed: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Release 1.0.0.dev4 passed ``{"mappings": ..., "_metadata": ...}`` to the
        storage as if it were a mapping. JSON files survived that (the wrapping
        happened to cancel out); CSV files hold the envelope as two rows whose
        second column is the ``repr`` of a dict. Recover both, because the
        table cannot be regenerated.
        """
        if set(parsed) == {"mappings", "_metadata"}:
            inner, meta = parsed["mappings"], parsed["_metadata"]
            try:
                if isinstance(inner, str):
                    inner = ast.literal_eval(inner)
                if isinstance(meta, str):
                    meta = ast.literal_eval(meta)
            except (ValueError, SyntaxError) as e:
                raise MappingIntegrityError(
                    f"Legacy mapping file '{self.mapping_file.name}' is not "
                    f"recoverable: {e}"
                )
            if isinstance(inner, dict) and isinstance(meta, dict):
                self.logger.warning(
                    "Recovered a mapping stored in the pre-2.0 envelope layout from "
                    f"'{self.mapping_file.name}'."
                )
                return {str(k): str(v) for k, v in inner.items()}, meta
        return parsed, {}

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(
        self, mapping: Dict[str, str], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Replace the stored mapping with ``mapping`` (atomic snapshot).

        Parameters
        ----------
            mapping: Dictionary mapping original values to pseudonyms. Keys and
                values must be strings.
            metadata: Optional JSON-serializable metadata stored with it

        Raises
        ------
            MappingStorageError: If the input is not a flat str->str mapping or
                saving fails
        """
        with self._lock:
            try:
                self._backup_once()
                frame = self._encode_frame(0, mapping, metadata)

                temp_path = self.mapping_file.with_suffix(".tmp")
                try:
                    with open(temp_path, "wb") as f:
                        f.write(MAGIC)
                        f.write(frame)
                        f.flush()
                        os.fsync(f.fileno())

                    if os.name != "nt":  # Unix-like systems
                        os.chmod(temp_path, 0o600)

                    temp_path.replace(self.mapping_file)
                except Exception:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

                self._next_seq = 1
                self.logger.debug(
                    f"Saved {len(mapping)} mappings to {self.mapping_file}"
                )
            except MappingStorageError:
                raise
            except Exception as e:
                raise MappingStorageError(f"Failed to save mapping: {e}")

    def append(
        self, new_mappings: Dict[str, str], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Persist only ``new_mappings`` by appending one encrypted frame.

        Cost is proportional to ``len(new_mappings)``. The caller guarantees the
        entries are new (a source value already stored keeps its pseudonym);
        conflicts are detected on the next load. A missing, empty or version 1
        file is first written as a version 2 snapshot of what it contained.

        Raises
        ------
            MappingIntegrityError: if the existing file is not trustworthy
            MappingStorageError: if the input is invalid or writing fails
        """
        with self._lock:
            try:
                exists = self.mapping_file.exists()
                size = self.mapping_file.stat().st_size if exists else 0
                if size == 0:
                    self.save(new_mappings, metadata)
                    return

                with open(self.mapping_file, "rb") as f:
                    head = f.read(len(MAGIC))
                if head != MAGIC:
                    # Version 1 file: convert once, preserving its content.
                    existing, existing_meta = self.load_with_metadata()
                    existing.update(new_mappings)
                    self.save(existing, metadata or existing_meta)
                    return

                seq = self._next_seq
                if seq is None:
                    seq = self._count_frames()
                frame = self._encode_frame(seq, new_mappings, metadata)

                self._backup_once()
                with open(self.mapping_file, "ab") as f:
                    f.write(frame)
                    f.flush()
                    os.fsync(f.fileno())
                self._next_seq = seq + 1
                self.logger.debug(
                    f"Appended {len(new_mappings)} mappings to {self.mapping_file}"
                )
            except MappingStorageError:
                raise
            except Exception as e:
                raise MappingStorageError(f"Failed to append mappings: {e}")

    def _count_frames(self) -> int:
        """Number of frames, walking length prefixes only (no decryption)."""
        size = self.mapping_file.stat().st_size
        pos = len(MAGIC)
        count = 0
        with open(self.mapping_file, "rb") as f:
            while pos < size:
                f.seek(pos)
                prefix = f.read(_LEN.size)
                if len(prefix) < _LEN.size:
                    raise MappingIntegrityError(
                        f"Mapping file '{self.mapping_file.name}' is truncated "
                        "inside a frame header. Restore the latest backup."
                    )
                (length,) = _LEN.unpack(prefix)
                pos += _LEN.size + length
                if length < 28 or pos > size:
                    raise MappingIntegrityError(
                        f"Mapping file '{self.mapping_file.name}' is truncated or "
                        f"frame {count} has an invalid length. Restore the latest "
                        "backup."
                    )
                count += 1
        return count

    def update(self, new_mappings: Dict[str, str]) -> Dict[str, str]:
        """
        Update existing mappings with new entries (full rewrite).

        Parameters
        ----------
            new_mappings: New mappings to add/update

        Returns
        -------
            Complete updated mapping dictionary

        Raises
        ------
            MappingStorageError: If update fails
        """
        with self._lock:
            try:
                existing, metadata = self.load_with_metadata()

                conflicts = [
                    key
                    for key, value in new_mappings.items()
                    if key in existing and existing[key] != value
                ]
                if conflicts:
                    self.logger.warning(
                        f"Found {len(conflicts)} mapping conflicts. "
                        "Existing values will be overwritten."
                    )

                existing.update(new_mappings)
                self.save(existing, metadata)
                return existing

            except MappingStorageError:
                raise
            except Exception as e:
                raise MappingStorageError(f"Failed to update mappings: {e}")

    # ------------------------------------------------------------------
    # Backups
    # ------------------------------------------------------------------

    def _backup_once(self) -> None:
        """Back up the existing file before this instance first modifies it."""
        if self._backup_done or not self.backup_on_update:
            return
        if self.mapping_file.exists() and self.mapping_file.stat().st_size > 0:
            self._backup_done = True
            self._create_backup()

    def _create_backup(self) -> None:
        """Create a backup of the current mapping file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.mapping_file.with_suffix(f".bak.{timestamp}")

            shutil.copy2(self.mapping_file, backup_path)

            if os.name != "nt":  # Unix-like systems
                os.chmod(backup_path, 0o600)

            self.logger.info(f"Created backup: {backup_path}")
            self._cleanup_old_backups()

        except Exception as e:
            # Log but don't fail the operation
            self.logger.warning(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Clean up old backup files, keeping only the most recent ones."""
        try:
            pattern = f"{self.mapping_file.stem}.bak.*"
            backups = list(self.mapping_file.parent.glob(pattern))
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            for backup in backups[keep_count:]:
                backup.unlink()
                self.logger.debug(f"Removed old backup: {backup}")

        except Exception as e:
            # Log but don't fail
            self.logger.debug(f"Failed to clean up old backups: {e}")

    # ------------------------------------------------------------------
    # Encoding / decoding of frames
    # ------------------------------------------------------------------

    def _encode_frame(
        self,
        seq: int,
        mapping: Dict[str, str],
        metadata: Optional[Dict[str, Any]],
    ) -> bytes:
        """Serialize, encrypt and length-prefix one frame."""
        for original, pseudonym in mapping.items():
            if not isinstance(original, str) or not isinstance(pseudonym, str):
                raise MappingStorageError(
                    "Mapping keys and values must be strings; got "
                    f"{type(original).__name__} -> {type(pseudonym).__name__}. "
                    "Pass the flat original->pseudonym mapping, not a wrapper."
                )
        header = json.dumps(
            {
                "seq": seq,
                "format": self.format,
                "count": len(mapping),
                "metadata": metadata or {},
            },
            sort_keys=True,
        ).encode("utf-8")
        body = (
            self._serialize_csv(mapping)
            if self.format == "csv"
            else self._serialize_json(mapping)
        )
        blob = self._encryptor.encrypt(header + b"\n" + body, _aad(seq))
        return _LEN.pack(len(blob)) + blob

    def _split_payload(self, payload: bytes, seq: int) -> Tuple[Dict[str, Any], bytes]:
        head, sep, body = payload.partition(b"\n")
        if not sep:
            raise MappingIntegrityError(f"Frame {seq} has no header.")
        try:
            header = json.loads(head.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MappingIntegrityError(f"Frame {seq} has an unreadable header: {e}")
        if not isinstance(header, dict):
            raise MappingIntegrityError(f"Frame {seq} has an invalid header.")
        return header, body

    def _parse_body(self, fmt: Any, body: bytes, seq: int) -> Dict[str, str]:
        if fmt == "csv":
            return self._parse_csv(body)
        if fmt == "json":
            parsed = self._parse_json(body)
            if not all(isinstance(v, str) for v in parsed.values()):
                raise MappingIntegrityError(f"Frame {seq} holds non-string pseudonyms.")
            return parsed
        raise MappingIntegrityError(f"Frame {seq} has unknown payload format {fmt!r}.")

    def _parse_csv(self, data: bytes) -> Dict[str, str]:
        """Parse CSV payload (``original,pseudonym`` rows with a header row)."""
        try:
            text = data.decode("utf-8")
            reader = csv.reader(io.StringIO(text, newline=""))
            header = next(reader, None)
            if header is None:
                return {}
            if tuple(header) != _CSV_FIELDS:
                raise MappingIntegrityError(
                    f"Unexpected CSV header {header!r}; expected {list(_CSV_FIELDS)}."
                )
            mapping: Dict[str, str] = {}
            for row in reader:
                if len(row) != 2:
                    raise MappingIntegrityError(f"Invalid CSV row: {row!r}")
                original, pseudonym = row
                if original in mapping and mapping[original] != pseudonym:
                    raise MappingIntegrityError(
                        "Source value appears with two different pseudonyms."
                    )
                mapping[original] = pseudonym
            return mapping
        except MappingStorageError:
            raise
        except Exception as e:
            raise MappingIntegrityError(f"Failed to parse CSV: {e}")

    def _parse_json(self, data: bytes) -> Dict[str, Any]:
        """Parse JSON payload (plain mapping, or the legacy metadata wrapper)."""
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MappingIntegrityError(f"Invalid JSON: {e}")
        if not isinstance(parsed, dict):
            raise MappingIntegrityError("Invalid JSON structure")
        if "_metadata" in parsed and "mappings" in parsed and len(parsed) == 2:
            return parsed["mappings"]
        return parsed

    def _serialize_csv(self, mapping: Dict[str, str]) -> bytes:
        """Serialize mapping to CSV rows."""
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(_CSV_FIELDS)
        writer.writerows(mapping.items())
        return output.getvalue().encode("utf-8")

    def _serialize_json(self, mapping: Dict[str, str]) -> bytes:
        """Serialize mapping to a JSON object."""
        return json.dumps(mapping, sort_keys=True).encode("utf-8")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get mapping file metadata.

        Returns
        -------
            Dictionary containing:
                - exists: Whether file exists
                - size_bytes: File size in bytes
                - modified: Last modification timestamp
                - format: Storage format
        """
        metadata = {
            "exists": self.mapping_file.exists(),
            "format": self.format,
            "path": str(self.mapping_file),
        }

        if metadata["exists"]:
            try:
                stat = self.mapping_file.stat()
                metadata.update(
                    {
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to get file metadata: {e}")

        return metadata

    def validate_mappings(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate mapping consistency and detect issues.

        Parameters
        ----------
            mapping: Mapping dictionary to validate

        Returns
        -------
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
            "total_mappings": len(mapping),
        }

        for key, value in mapping.items():
            if not key:
                validation["empty_keys"] += 1
                validation["valid"] = False
            if not value:
                validation["empty_values"] += 1
                validation["valid"] = False

        reverse_mapping = {}
        for original, pseudonym in mapping.items():
            if pseudonym in reverse_mapping:
                if pseudonym not in validation["duplicate_values"]:
                    validation["duplicate_values"].append(pseudonym)
                validation["valid"] = False
            else:
                reverse_mapping[pseudonym] = original

        return validation
